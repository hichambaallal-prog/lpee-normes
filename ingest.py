"""
Script d'ingestion — plateforme de recherche de normes LPEE  (v3)
==================================================================
Lit tous les PDF d'un dossier (ex: votre dossier OneDrive synchronisé), les découpe
en morceaux de texte ("chunks"), calcule leur embedding avec un modèle LOCAL (gratuit,
illimité) et les enregistre dans Supabase (pgvector).

Les PDF ne quittent jamais votre PC : l'embedding tourne entièrement sur votre machine.

Usage :
    python ingest.py              # indexe (reprend là où il s'est arrêté)
    python ingest.py --estimer    # ne fait QUE compter les extraits et estimer la taille en base

Variables d'environnement (fichier .env) :
    SUPABASE_URL   -> URL du projet Supabase
    SUPABASE_KEY   -> clé "service_role"
    PDF_FOLDER     -> dossier local contenant les PDF
  Optionnelles :
    SUPABASE_TABLE -> nom de la table (défaut : documents)
    LIMITE_MO      -> arrêt propre quand la base atteint cette taille (défaut : 450, limite gratuite = 500)
    OCR            -> "1" pour lire aussi les pages scannées (nécessite Tesseract installé + langue française)

Nouveautés v3 :
  • s'arrête proprement avant de dépasser le quota de la base (au lieu de la mettre en lecture seule)
  • chemins enregistrés au format "dossier/fichier.pdf" (compatible avec la recherche « normes principales »)
  • clé unique par chemin complet (deux PDF de même nom dans deux dossiers ne s'écrasent plus)
  • un fichier modifié est réindexé proprement (anciens extraits supprimés d'abord)
  • --estimer : savoir à l'avance si tout tient dans 500 Mo
⚠️ CHUNK_SIZE / CHUNK_OVERLAP / modèle / préfixe "passage: " doivent rester identiques à app.py.
"""

import argparse
import json
import os
import sys
import time
import unicodedata
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
PDF_FOLDER = os.environ.get("PDF_FOLDER", r"C:\Users\VotreNom\OneDrive\Normes_LPEE")
TABLE = os.environ.get("SUPABASE_TABLE", "documents")
LIMITE_MO = float(os.environ.get("LIMITE_MO", "450"))
OCR_ACTIF = os.environ.get("OCR", "0").strip() == "1"

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"  # local, 384 dimensions, bon en français
CHUNK_SIZE = 1200            # caractères par chunk (identique à TAILLE_CHUNK dans app.py)
CHUNK_OVERLAP = 150          # chevauchement (identique à CHEVAUCHEMENT dans app.py)
BATCH_SIZE = 64              # chunks encodés / envoyés ensemble
SEUIL_TEXTE = 20             # moins de caractères sur une page => page vide ou scannée
MIN_CHUNK = 40               # chunks plus courts ignorés (restes de découpage, numéros de page...)
OCTETS_PAR_CHUNK = 3200      # estimation : texte + halfvec + métadonnées + index HNSW
MANIFEST_PATH = Path("ingest_manifest.json")
MARQUEUR_VIDE = "|vide"      # fichier sans texte exploitable : inutile de le retraiter


class BaseSaturee(Exception):
    """Quota de la base atteint (ou base en lecture seule) : on arrête proprement."""


# ---------------------------------------------------------------- outils
def norm_chemin(chemin) -> str:
    """Chemin au format POSIX (/), accents normalisés : identique à celui du dépôt Hugging Face."""
    return unicodedata.normalize("NFC", str(chemin).replace("\\", "/"))


def charger_manifest() -> dict:
    if MANIFEST_PATH.exists():
        brut = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        return {norm_chemin(k): v for k, v in brut.items()}
    return {}


def sauver_manifest(manifest: dict):
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def empreinte(chemin: Path) -> str:
    """Empreinte rapide (taille + date de modification)."""
    st = chemin.stat()
    return f"{st.st_size}-{int(st.st_mtime)}"


# ---------------------------------------------------------------- extraction
def _ocr_page(page) -> str:
    try:
        import io
        import pytesseract
        from PIL import Image
        pix = page.get_pixmap(dpi=200)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return pytesseract.image_to_string(img, lang="fra").strip()
    except Exception:
        return ""


def extraire_pages(chemin: Path):
    """Liste de (numéro_page, texte). PyMuPDF si disponible (plus rapide et plus fiable), sinon pypdf."""
    pages = []
    try:
        import fitz  # PyMuPDF
        with fitz.open(str(chemin)) as doc:
            for i, page in enumerate(doc, start=1):
                texte = page.get_text("text").strip()
                if len(texte) < SEUIL_TEXTE and OCR_ACTIF:
                    texte = _ocr_page(page) or texte
                pages.append((i, texte))
        return pages
    except ImportError:
        pass
    except Exception as e:
        print(f"  ⚠️ PyMuPDF a échoué sur {chemin.name} ({e}), essai avec pypdf")
        pages = []
    try:
        from pypdf import PdfReader
        for i, page in enumerate(PdfReader(str(chemin)).pages, start=1):
            pages.append((i, (page.extract_text() or "").strip()))
    except Exception as e:
        print(f"  ⚠️ Erreur de lecture {chemin.name} : {e}")
    return pages


def decouper_en_chunks(texte: str, taille=CHUNK_SIZE, chevauchement=CHUNK_OVERLAP):
    """Découpage par caractères avec chevauchement, en coupant sur un espace si possible."""
    chunks, debut, n = [], 0, len(texte)
    while debut < n:
        fin = min(debut + taille, n)
        if fin < n:
            dernier_espace = texte.rfind(" ", debut, fin)
            if dernier_espace > debut:
                fin = dernier_espace
        morceau = texte[debut:fin].strip()
        if morceau:
            chunks.append(morceau)
        debut = fin - chevauchement if fin - chevauchement > debut else fin
    return chunks


def preparer_lignes(chemin: Path, chemin_rel: str) -> list:
    """Toutes les lignes (sans embedding) à insérer pour un PDF."""
    lignes, vus = [], set()
    for numero_page, texte in extraire_pages(chemin):
        if len(texte) < SEUIL_TEXTE:
            continue
        for idx, morceau in enumerate(decouper_en_chunks(texte)):
            if len(morceau) < MIN_CHUNK or morceau in vus:  # trop court, ou répété à l'identique dans ce PDF
                continue
            vus.add(morceau)
            cid = f"{numero_page}-{idx}"
            lignes.append({
                "content": morceau,
                "fichier": chemin_rel,          # clé unique = chemin complet
                "chunk_id": cid,
                "metadata": {"fichier": chemin.name, "chemin": chemin_rel, "page": numero_page, "chunk_id": cid},
            })
    return lignes


# ---------------------------------------------------------------- mode estimation
def estimer(racine: Path):
    fichiers = sorted(racine.rglob("*.pdf"))
    print(f"{len(fichiers)} PDF trouvé(s). Comptage des extraits (aucun envoi, aucun embedding)...\n")
    total, vides, par_fichier = 0, 0, []
    for i, f in enumerate(fichiers, 1):
        rel = norm_chemin(f.relative_to(racine).as_posix())
        n = len(preparer_lignes(f, rel))
        total += n
        vides += (n == 0)
        par_fichier.append((n, rel))
        if i % 25 == 0:
            print(f"  ... {i}/{len(fichiers)} fichiers, {total} extraits")
    mo = total * OCTETS_PAR_CHUNK / 1048576
    print(f"\n📊 {total} extraits au total ({vides} PDF sans texte, probablement scannés"
          f"{'' if OCR_ACTIF else ' — relancez avec OCR=1 pour les lire'}).")
    print(f"📦 Taille estimée en base : ≈ {mo:.0f} Mo (marge d'erreur ±25 %). Limite gratuite : 500 Mo.")
    if mo > 450:
        print("⚠️ Cela risque de dépasser la limite gratuite : indexez d'abord les dossiers prioritaires "
              "(normes principales), ou passez à l'offre Pro.")
    print("\nLes 10 PDF les plus volumineux :")
    for n, rel in sorted(par_fichier, reverse=True)[:10]:
        print(f"  {n:6d} extraits  {rel}")


# ---------------------------------------------------------------- indexation
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimer", action="store_true", help="compte les extraits et estime la taille, sans rien envoyer")
    args = parser.parse_args()

    racine = Path(PDF_FOLDER)
    if not racine.exists():
        raise SystemExit(f"Dossier introuvable : {racine}")
    if args.estimer:
        return estimer(racine)

    if not SUPABASE_URL or not SUPABASE_KEY:
        raise SystemExit("SUPABASE_URL et SUPABASE_KEY doivent être définis (fichier .env).")

    from supabase import create_client
    from sentence_transformers import SentenceTransformer

    print("Chargement du modèle d'embedding local (1-2 min la première fois)...")
    modele = SentenceTransformer(EMBEDDING_MODEL_NAME)
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    def taille_base_mo():
        try:
            return int(supabase.rpc("taille_base", {}).execute().data) / 1048576
        except Exception:
            return None

    def verifier_quota():
        mo = taille_base_mo()
        if mo is not None and mo >= LIMITE_MO:
            raise BaseSaturee(f"La base fait {mo:.0f} Mo (limite fixée à {LIMITE_MO:.0f} Mo).")

    def chemins_en_base():
        try:
            r = supabase.rpc("chemins_indexes", {}).execute()
            return {norm_chemin(l["chemin"]) for l in (r.data or []) if l.get("chemin")}
        except Exception:
            pass
        try:
            chemins, debut = set(), 0
            while True:
                r = supabase.table(TABLE).select("chemin:metadata->>chemin").range(debut, debut + 999).execute()
                lot = r.data or []
                chemins.update(norm_chemin(l["chemin"]) for l in lot if l.get("chemin"))
                if len(lot) < 1000:
                    break
                debut += 1000
            return chemins
        except Exception as e:
            print(f"⚠️ Vérification de l'index impossible ({e}) : manifeste laissé tel quel.")
            return None

    def reparer_manifest(manifest: dict):
        """Retire du manifeste les fichiers absents de Supabase, pour qu'ils soient (ré)indexés."""
        en_base = chemins_en_base()
        if en_base is None:
            return
        a_retirer = [k for k, v in manifest.items() if not str(v).endswith(MARQUEUR_VIDE) and k not in en_base]
        for k in a_retirer:
            del manifest[k]
        if a_retirer:
            print(f"🔧 {len(a_retirer)} fichier(s) absents de l'index seront (ré)indexés.\n")

    def supprimer_fichier(chemin_rel: str):
        supabase.table(TABLE).delete().eq("metadata->>chemin", chemin_rel).execute()

    def envoyer(lot: list):
        for tentative in range(3):
            try:
                supabase.table(TABLE).upsert(lot, on_conflict="fichier,chunk_id").execute()
                return
            except Exception as e:
                msg = str(e).lower()
                if "read-only" in msg or "read only" in msg:
                    raise BaseSaturee("Base en lecture seule (quota dépassé).")
                if tentative == 2:
                    raise
                time.sleep(2 * (tentative + 1))

    def indexer_fichier(chemin: Path, manifest: dict) -> int:
        chemin_rel = norm_chemin(chemin.relative_to(racine).as_posix())
        emp = empreinte(chemin)
        if str(manifest.get(chemin_rel, "")).split("|")[0] == emp:
            return 0  # déjà traité et inchangé

        verifier_quota()
        print(f"📄 {chemin_rel}")
        lignes = preparer_lignes(chemin, chemin_rel)
        if not lignes:
            print("  ⚠️ Aucun texte exploitable (document scanné ? essayez OCR=1).")
            manifest[chemin_rel] = emp + MARQUEUR_VIDE
            return 0

        supprimer_fichier(chemin_rel)  # repart d'une base propre pour ce fichier
        total = 0
        try:
            for i in range(0, len(lignes), BATCH_SIZE):
                lot = lignes[i:i + BATCH_SIZE]
                vecteurs = modele.encode([f"passage: {l['content']}" for l in lot],
                                         normalize_embeddings=True, show_progress_bar=False)
                for ligne, v in zip(lot, vecteurs):
                    ligne["embedding"] = v.tolist()
                envoyer(lot)
                total += len(lot)
            print(f"  → {total} extraits envoyés")
        except Exception:
            try:
                supprimer_fichier(chemin_rel)  # pas d'indexation à moitié
            except Exception:
                pass
            raise
        manifest[chemin_rel] = emp
        return total

    manifest = charger_manifest()
    reparer_manifest(manifest)
    fichiers = sorted(racine.rglob("*.pdf"))
    print(f"{len(fichiers)} fichier(s) PDF trouvé(s) dans {racine}")
    mo = taille_base_mo()
    if mo is not None:
        print(f"💾 Taille actuelle de la base : {mo:.0f} Mo (arrêt automatique à {LIMITE_MO:.0f} Mo)")
    print()

    debut, total, echecs = time.time(), 0, []
    try:
        for chemin in fichiers:
            try:
                total += indexer_fichier(chemin, manifest)
            except BaseSaturee:
                raise
            except Exception as e:
                echecs.append(chemin.name)
                print(f"  ❌ Échec sur {chemin.name} : {e}")
            sauver_manifest(manifest)  # reprise possible après interruption
    except BaseSaturee as e:
        sauver_manifest(manifest)
        print(f"\n⛔ Arrêt : {e}")
        print("   Libérez de l'espace (ou passez à Pro), puis relancez : l'indexation reprend où elle s'est arrêtée.")
        sys.exit(2)

    duree = (time.time() - debut) / 60
    print(f"\n✅ Terminé. {total} nouveaux extraits indexés en {duree:.1f} min.")
    if echecs:
        print(f"⚠️ {len(echecs)} fichier(s) en échec (relancez le script pour réessayer) : {', '.join(echecs[:10])}")
    mo = taille_base_mo()
    if mo is not None:
        print(f"💾 Taille finale de la base : {mo:.0f} Mo / 500 Mo")


if __name__ == "__main__":
    main()
