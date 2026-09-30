"""
Script d'ingestion — plateforme de recherche de normes LPEE
==============================================================
Lit tous les PDF d'un dossier (ex: votre dossier OneDrive synchronisé
localement), les découpe en morceaux de texte ("chunks"), calcule leur
embedding avec un modèle LOCAL (gratuit, illimité, aucun quota), et les
enregistre dans Supabase (pgvector).

Les PDF eux-mêmes ne quittent JAMAIS votre PC, et rien n'est envoyé à
Google pour cette étape : l'embedding tourne entièrement sur votre machine.

Usage :
    python ingest.py

Variables d'environnement requises (voir .env.example) :
    SUPABASE_URL -> URL de votre projet Supabase
    SUPABASE_KEY -> clé "service_role" de Supabase (Project Settings > API)
    PDF_FOLDER   -> chemin local vers le dossier OneDrive à indexer

Au premier lancement, le modèle (~470 Mo) est téléchargé automatiquement
depuis Hugging Face et mis en cache localement — ensuite tout fonctionne
hors-ligne pour l'embedding.

NOUVEAU : le manifeste se "répare" tout seul. Si une norme a été retirée de l'index
(par exemple supprimée automatiquement par l'application après sa suppression sur
Hugging Face) puis remise plus tard, elle est ré-indexée au prochain lancement.
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from pypdf import PdfReader
from supabase import create_client
from sentence_transformers import SentenceTransformer

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
PDF_FOLDER = os.environ.get("PDF_FOLDER", r"C:\Users\VotreNom\OneDrive\Normes_LPEE")

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"  # gratuit, local, 384 dimensions, bon en français
EMBEDDING_DIM = 384          # doit correspondre à la colonne vector(384) du schema.sql
CHUNK_SIZE = 1200            # caractères par chunk (≈ 250-300 mots)
CHUNK_OVERLAP = 150          # chevauchement entre chunks pour ne pas couper une idée en deux
BATCH_SIZE = 64              # nb de chunks encodés ensemble (purement local, peut être plus grand)
MANIFEST_PATH = Path("ingest_manifest.json")  # mémorise les fichiers déjà traités (reprise après interruption)
MARQUEUR_VIDE = "|vide"      # fichier lu mais sans texte exploitable (scanné) : inutile de le retraiter à chaque fois

print("Chargement du modèle d'embedding local (une seule fois, peut prendre 1-2 min la première fois)...")
modele = SentenceTransformer(EMBEDDING_MODEL_NAME)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def charger_manifest():
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {}


def sauver_manifest(manifest):
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def hash_fichier(chemin: Path) -> str:
    """Empreinte rapide basée sur taille + date de modification (évite de relire tout le fichier)."""
    stat = chemin.stat()
    return f"{stat.st_size}-{int(stat.st_mtime)}"


def chemins_en_base() -> set:
    """Chemins distincts actuellement présents dans Supabase (None si la lecture échoue)."""
    try:
        r = supabase.rpc("chemins_indexes", {}).execute()   # rapide si la fonction SQL existe
        return {l["chemin"] for l in (r.data or []) if l.get("chemin")}
    except Exception:
        pass
    try:
        chemins, debut = set(), 0
        while True:
            r = supabase.table("documents").select("chemin:metadata->>chemin").range(debut, debut + 999).execute()
            lot = r.data or []
            chemins.update(l["chemin"] for l in lot if l.get("chemin"))
            if len(lot) < 1000:
                break
            debut += 1000
        return chemins
    except Exception as e:
        print(f"⚠️ Vérification de l'index impossible ({e}) : manifeste laissé tel quel.")
        return None


def reparer_manifest(manifest: dict):
    """Retire du manifeste les fichiers qui ne sont plus dans Supabase, pour qu'ils soient ré-indexés."""
    en_base = chemins_en_base()
    if en_base is None:
        return
    en_base_norm = {c.replace("\\", "/") for c in en_base}
    a_retirer = [
        k for k, v in manifest.items()
        if not str(v).endswith(MARQUEUR_VIDE) and k.replace("\\", "/") not in en_base_norm
    ]
    for k in a_retirer:
        del manifest[k]
    if a_retirer:
        print(f"🔧 {len(a_retirer)} fichier(s) absents de l'index seront (ré)indexés.\n")


def extraire_pages(chemin: Path):
    """Retourne une liste de (numero_page, texte). Signale les pages probablement scannées (peu de texte)."""
    pages = []
    try:
        reader = PdfReader(str(chemin))
        for i, page in enumerate(reader.pages, start=1):
            texte = (page.extract_text() or "").strip()
            pages.append((i, texte))
    except Exception as e:
        print(f"  ⚠️ Erreur de lecture {chemin.name} : {e}")
    return pages


def decouper_en_chunks(texte: str, taille=CHUNK_SIZE, chevauchement=CHUNK_OVERLAP):
    """Découpage simple par caractères avec chevauchement, en essayant de couper sur un espace."""
    chunks = []
    debut = 0
    n = len(texte)
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


def embed_batch(textes: list[str]):
    """Calcule les embeddings localement (aucun appel réseau, aucun quota).
    Le modèle e5 attend un préfixe "passage: " pour les documents à indexer
    (et "query: " côté application pour la question posée par l'agent)."""
    textes_prefixes = [f"passage: {t}" for t in textes]
    vecteurs = modele.encode(textes_prefixes, normalize_embeddings=True, show_progress_bar=False)
    return vecteurs.tolist()


def indexer_fichier(chemin: Path, racine: Path, manifest: dict):
    chemin_relatif = str(chemin.relative_to(racine))
    empreinte = hash_fichier(chemin)

    if str(manifest.get(chemin_relatif, "")).split("|")[0] == empreinte:
        return 0  # déjà traité et inchangé depuis la dernière exécution

    print(f"📄 {chemin_relatif}")
    pages = extraire_pages(chemin)
    if not pages:
        return 0

    nb_pages_vides = sum(1 for _, t in pages if len(t) < 20)
    if nb_pages_vides > len(pages) * 0.7:
        print(f"  ⚠️ Ce document semble scanné (peu de texte extrait) — envisager un passage OCR séparé.")

    lignes_a_inserer = []
    for numero_page, texte in pages:
        if len(texte) < 20:
            continue
        for idx_chunk, morceau in enumerate(decouper_en_chunks(texte)):
            lignes_a_inserer.append({
                "content": morceau,
                "fichier": chemin.name,
                "chunk_id": f"{numero_page}-{idx_chunk}",
                "metadata": {
                    "fichier": chemin.name,
                    "chemin": chemin_relatif,
                    "page": numero_page,
                    "chunk_id": f"{numero_page}-{idx_chunk}",
                },
            })

    if not lignes_a_inserer:
        manifest[chemin_relatif] = empreinte + MARQUEUR_VIDE
        return 0

    total_insere = 0
    for i in range(0, len(lignes_a_inserer), BATCH_SIZE):
        lot = lignes_a_inserer[i:i + BATCH_SIZE]
        vecteurs = embed_batch([l["content"] for l in lot])
        for ligne, vecteur in zip(lot, vecteurs):
            ligne["embedding"] = vecteur
        supabase.table("documents").upsert(
            lot, on_conflict="fichier,chunk_id"
        ).execute()
        total_insere += len(lot)
        print(f"  → {total_insere}/{len(lignes_a_inserer)} chunks envoyés")

    manifest[chemin_relatif] = empreinte
    return total_insere


def main():
    racine = Path(PDF_FOLDER)
    if not racine.exists():
        raise SystemExit(f"Dossier introuvable : {racine}")

    manifest = charger_manifest()
    reparer_manifest(manifest)
    fichiers_pdf = sorted(racine.rglob("*.pdf"))
    print(f"{len(fichiers_pdf)} fichier(s) PDF trouvé(s) dans {racine}\n")

    debut = time.time()
    total = 0
    for chemin in fichiers_pdf:
        try:
            total += indexer_fichier(chemin, racine, manifest)
        except Exception as e:
            print(f"  ❌ Échec sur {chemin.name} : {e}")
        sauver_manifest(manifest)  # sauvegarde après chaque fichier -> reprise possible si interruption

    duree_min = (time.time() - debut) / 60
    print(f"\n✅ Terminé. {total} nouveaux chunks indexés en {duree_min:.1f} min.")


if __name__ == "__main__":
    main()
