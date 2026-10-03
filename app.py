"""
Plateforme de recherche des normes et fascicules techniques — LPEE
=====================================================================
Interface de type "chat" avec historique : les agents peuvent poser une
question, puis enchaîner des questions de suivi ("et pour le GNA ?") qui
tiennent compte de la conversation précédente.

À déployer sur Streamlit Community Cloud (gratuit), connecté à Supabase.

Secrets requis (Streamlit Cloud > Settings > Secrets, format .toml) :
    GOOGLE_API_KEY = "..."
    SUPABASE_URL = "..."
    SUPABASE_KEY = "..."   # doit autoriser la suppression (clé service_role ou policy DELETE)
    SUPABASE_TABLE = "documents"   # (optionnel) nom de la table des extraits
"""

import hashlib
import hmac
import io
import json
import os
import re
import threading
import time
import unicodedata

import streamlit as st
from datetime import datetime, timedelta
from streamlit_cookies_controller import CookieController
from supabase import create_client
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer
from postgrest.exceptions import APIError  # noqa: F401  (erreurs Supabase lisibles)

st.set_page_config(page_title="Recherche Normes LPEE", page_icon="📚", layout="wide")

# ==========================================
# GESTION DES UTILISATEURS (stockés sur Hugging Face, mots de passe hachés)
# ==========================================
FICHIER_UTILISATEURS = "utilisateurs.json"
COMPTES_PAR_DEFAUT = {
    "admin": {"password": "admin123", "nom": "Administrateur LPEE", "role": "admin"},
    "agent1": {"password": "lpee2026", "nom": "Agent Laboratoire Béton", "role": "agent"},
    "agent2": {"password": "lpee2026", "nom": "Agent Laboratoire Sols", "role": "agent"},
}


def hacher_mdp(mdp: str, sel: str = None) -> str:
    sel = sel or os.urandom(16).hex()
    h = hashlib.pbkdf2_hmac("sha256", mdp.encode(), bytes.fromhex(sel), 200_000).hex()
    return f"pbkdf2${sel}${h}"


def verifier_mdp(mdp: str, stocke: str) -> bool:
    if stocke.startswith("pbkdf2$"):
        _, sel, _h = stocke.split("$")
        return hmac.compare_digest(hacher_mdp(mdp, sel), stocke)
    return hmac.compare_digest(mdp.encode(), stocke.encode())  # ancien format en clair


def charger_utilisateurs():
    """Lit la liste des comptes sur Hugging Face. Retourne None si la lecture échoue
    (pour ne jamais écraser les vrais comptes par les comptes par défaut)."""
    repo, token = st.secrets.get("HF_REPO_ID", ""), st.secrets.get("HF_TOKEN", "")
    defaut = {k: {**v, "password": hacher_mdp(v["password"])} for k, v in COMPTES_PAR_DEFAUT.items()}
    if not (repo and token):
        return defaut
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError, RevisionNotFoundError
        try:
            chemin = hf_hub_download(repo_id=repo, filename=FICHIER_UTILISATEURS, repo_type="dataset",
                                     token=token, force_download=True)
        except (EntryNotFoundError, RevisionNotFoundError):
            return defaut  # première utilisation
        with open(chemin, encoding="utf-8") as f:
            return json.load(f) or defaut
    except Exception:
        return None


def sauver_utilisateurs(users: dict):
    repo, token = st.secrets.get("HF_REPO_ID", ""), st.secrets.get("HF_TOKEN", "")
    if not (repo and token):
        return False, "HF_REPO_ID / HF_TOKEN manquants."
    try:
        from huggingface_hub import HfApi
        HfApi(token=token).upload_file(
            path_or_fileobj=io.BytesIO(json.dumps(users, ensure_ascii=False, indent=1).encode("utf-8")),
            path_in_repo=FICHIER_UTILISATEURS, repo_id=repo, repo_type="dataset",
            commit_message="Mise à jour des comptes",
        )
        return True, ""
    except Exception as e:
        return False, str(e)[:300]


def modifier_comptes(fn):
    """Relit les comptes, applique fn(users) (retourne un message d'erreur ou None), puis sauvegarde."""
    users = charger_utilisateurs()
    if users is None:
        return False, "Lecture des comptes impossible, réessayez."
    err = fn(users)
    if err:
        return False, err
    ok, e = sauver_utilisateurs(users)
    if ok:
        st.session_state.utilisateurs = users
    return ok, e


# --- Connexion persistante (cookie signé) : reste connecté après un rafraîchissement ou un redéploiement ---
COOKIE_NOM = "lpee_auth"
DUREE_COOKIE_JOURS = 7


def _cle_signature() -> bytes:
    return (st.secrets.get("COOKIE_SECRET") or st.secrets.get("HF_TOKEN") or "lpee-secret").encode()


def creer_jeton(login: str, empreinte_mdp: str) -> str:
    expire = int(time.time()) + DUREE_COOKIE_JOURS * 86400
    base = f"{login}|{expire}"
    sig = hmac.new(_cle_signature(), f"{base}|{empreinte_mdp}".encode(), hashlib.sha256).hexdigest()
    return f"{base}|{sig}"


def lire_jeton(jeton: str, users: dict):
    """Retourne le login si le jeton est valide, non expiré et le mot de passe inchangé."""
    try:
        login, expire, sig = str(jeton).rsplit("|", 2)
        if int(expire) < time.time() or login not in users:
            return None
        attendu = hmac.new(_cle_signature(), f"{login}|{expire}|{users[login]['password']}".encode(),
                           hashlib.sha256).hexdigest()
        return login if hmac.compare_digest(sig, attendu) else None
    except Exception:
        return None


def poser_cookie(controller, login: str, empreinte_mdp: str):
    controller.set(
        COOKIE_NOM, creer_jeton(login, empreinte_mdp),
        expires=datetime.now() + timedelta(days=DUREE_COOKIE_JOURS),
        max_age=DUREE_COOKIE_JOURS * 86400, same_site="lax",
    )


if "authentifie" not in st.session_state:
    st.session_state.authentifie = False
if "username_courant" not in st.session_state:
    st.session_state.username_courant = ""
if "nom_utilisateur" not in st.session_state:
    st.session_state.nom_utilisateur = ""
if "role_utilisateur" not in st.session_state:
    st.session_state.role_utilisateur = ""

controller = CookieController()

if not st.session_state.authentifie and not st.session_state.get("deconnexion_volontaire"):
    _jeton = controller.get(COOKIE_NOM)
    if _jeton:
        _users = charger_utilisateurs()
        _login = lire_jeton(_jeton, _users) if _users else None
        if _login:
            st.session_state.utilisateurs = _users
            st.session_state.authentifie = True
            st.session_state.username_courant = _login
            st.session_state.nom_utilisateur = _users[_login]["nom"]
            st.session_state.role_utilisateur = _users[_login]["role"]
            st.rerun()

# --- ÉCRAN DE CONNEXION ---
if not st.session_state.authentifie:
    st.title("🔐 Connexion — Plateforme LPEE")
    st.markdown("Veuillez vous identifier pour accéder aux normes et fascicules techniques.")
    
    with st.form("form_login"):
        username_input = st.text_input("Nom d'utilisateur").strip().lower()
        password_input = st.text_input("Mot de passe", type="password")
        submit_login = st.form_submit_button("Se connecter")
        
        if submit_login:
            users = charger_utilisateurs()
            if users is None:
                st.error("Service d'authentification momentanément indisponible. Réessayez dans un instant.")
            elif username_input in users and verifier_mdp(password_input, users[username_input]["password"]):
                st.session_state.utilisateurs = users
                st.session_state.authentifie = True
                st.session_state.username_courant = username_input
                st.session_state.nom_utilisateur = users[username_input]["nom"]
                st.session_state.role_utilisateur = users[username_input]["role"]
                st.session_state.pop("deconnexion_volontaire", None)
                poser_cookie(controller, username_input, users[username_input]["password"])
                time.sleep(1)  # laisse le navigateur enregistrer le cookie
                st.rerun()
            else:
                st.error("Nom d'utilisateur ou mot de passe incorrect.")
    st.stop()


# ==========================================
# APPLICATION PRINCIPALE
# ==========================================

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"  # DOIT être le même modèle que dans ingest.py
MODELES_GENERATION = [
    "gemini-2.5-flash",          # rapide + précis (réflexion désactivée => réponse immédiate)
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
]
MODELE_REFORMULATION = "gemini-flash-lite-latest"
NB_CANDIDATS = 40          # extraits récupérés dans la base
NB_MIN_RESULTATS = 8       # minimum gardé même si la pertinence est faible
NB_MAX_RESULTATS = 20      # maximum envoyé au modèle (prompt plus court = réponse plus rapide)
ECART_PERTINENCE = 0.08    # on écarte les extraits trop éloignés du meilleur score
NB_ECHANGES_CONTEXTE = 4

client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])
supabase = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])


def _chemin_historique(username: str) -> str:
    return f"historiques/{re.sub(r'[^a-z0-9_-]', '_', username.lower())}.json"


def charger_historique(username: str):
    """Recharge les conversations de l'utilisateur depuis le dépôt Hugging Face (None si aucune / erreur)."""
    repo, token = st.secrets.get("HF_REPO_ID", ""), st.secrets.get("HF_TOKEN", "")
    if not (repo and token):
        st.session_state["erreur_historique"] = "HF_REPO_ID / HF_TOKEN manquants dans les Secrets."
        return None
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError, RevisionNotFoundError
        try:
            chemin = hf_hub_download(
                repo_id=repo, filename=_chemin_historique(username), repo_type="dataset",
                token=token, force_download=True,
            )
        except (EntryNotFoundError, RevisionNotFoundError):
            return None  # première connexion : pas encore d'historique
        with open(chemin, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("sessions"):
            return data["sessions"], data.get("session_courante")
    except Exception as e:
        st.session_state["erreur_historique"] = str(e)
    return None


def sauvegarder_historique():
    """Enregistre les conversations de l'utilisateur connecté dans le dépôt Hugging Face."""
    username = st.session_state.get("username_courant")
    repo, token = st.secrets.get("HF_REPO_ID", ""), st.secrets.get("HF_TOKEN", "")
    if not (username and repo and token):
        return
    allege = {}
    for nom, msgs in st.session_state.sessions.items():
        allege[nom] = [
            {**m, "sources": [{**c, "content": c["content"][:700]} for c in m["sources"]]} if m.get("sources") else m
            for m in msgs
        ]
    contenu = json.dumps(
        {"sessions": allege, "session_courante": st.session_state.session_courante},
        ensure_ascii=False,
    ).encode("utf-8")
    try:
        from huggingface_hub import HfApi
        HfApi(token=token).upload_file(
            path_or_fileobj=io.BytesIO(contenu),
            path_in_repo=_chemin_historique(username),
            repo_id=repo,
            repo_type="dataset",
            commit_message=f"Historique {username}",
        )
        st.session_state.pop("erreur_historique", None)
    except Exception as e:
        st.session_state["erreur_historique"] = str(e)


@st.cache_resource(show_spinner="Chargement du modèle de recherche (une seule fois)...")
def charger_modele():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texte(texte: str, prefixe: str):
    modele = charger_modele()
    vecteur = modele.encode([f"{prefixe}: {texte}"], normalize_embeddings=True, show_progress_bar=False)
    return vecteur[0].tolist()


def _erreur_temporaire(e) -> bool:
    msg = str(e)
    return any(m in msg for m in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500", "504", "INTERNAL", "DEADLINE"))


def _config(max_output_tokens, modele, temperature=0.1):
    """Température basse = réponses factuelles ; réflexion désactivée sur 2.5 = beaucoup plus rapide."""
    cfg = {"temperature": temperature}
    if max_output_tokens:
        cfg["max_output_tokens"] = max_output_tokens
    if "2.5" in modele:
        cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    return types.GenerateContentConfig(**cfg)


def reformuler_question(question: str, historique: list) -> str:
    """Rend la question autonome. Sautée quand ce n'est pas nécessaire (gain de 1 à 3 s)."""
    if not historique:
        return question
    mots = question.split()
    debut = question.lower().lstrip().split(" ")[0]
    if len(mots) >= 8 and debut not in ("et", "pour", "aussi", "alors", "donc", "mais", "même"):
        return question  # question déjà complète

    derniers = historique[-NB_ECHANGES_CONTEXTE:]
    fil = "\n".join(f"{'Agent' if h['role'] == 'user' else 'Assistant'} : {h['content'][:600]}" for h in derniers)
    prompt = f"""Conversation technique entre un agent LPEE et un assistant documentaire :

{fil}

Nouvelle question : "{question}"

Reformule cette question en une question AUTONOME et COMPLÈTE (remplace "il", "ça", "et pour X"...).
Réponds UNIQUEMENT par la question reformulée, sans guillemets ni commentaire."""
    try:
        resp = client.models.generate_content(
            model=MODELE_REFORMULATION, contents=prompt, config=_config(120, MODELE_REFORMULATION, 0.0)
        )
        return (resp.text or "").strip() or question
    except Exception:
        return question


# ==========================================
# SYNCHRONISATION HUGGING FACE -> INDEX (suppression automatique des normes retirées)
# ==========================================
INTERVALLE_SYNCHRO = 3 * 60 * 60  # ajouts + suppressions vérifiés toutes les 3 h (et au démarrage) : économise CPU et base


@st.cache_data(ttl=60, show_spinner=False)
def fichiers_pdf_existants():
    """Liste des PDF présents dans le dépôt Hugging Face (rafraîchie toutes les 60 s). None si erreur."""
    try:
        from huggingface_hub import HfApi
        fichiers = HfApi(token=st.secrets.get("HF_TOKEN", "")).list_repo_files(
            st.secrets.get("HF_REPO_ID", ""), repo_type="dataset")
        return frozenset(_norm_chemin(f) for f in fichiers if f.lower().endswith(".pdf"))
    except Exception:
        return None


def _norm_chemin(chemin) -> str:
    """Même forme pour les chemins Windows (\\) et Hugging Face (/), accents normalisés."""
    return unicodedata.normalize("NFC", str(chemin or "")).replace("\\", "/")


def _chemin_norm(chunk) -> str:
    return _norm_chemin(chunk["metadata"].get("chemin"))


@st.cache_data(ttl=600, show_spinner=False)
def layout_coherent() -> bool:
    """Vérifie (sur un échantillon de l'index) que les chemins de Supabase correspondent à ceux de
    Hugging Face. Si oui, on peut masquer sans risque les normes supprimées."""
    existants = fichiers_pdf_existants()
    if not existants:
        return False
    table = st.secrets.get("SUPABASE_TABLE", "documents")
    chemins = []
    for debut in (0, 3000, 10000):
        try:
            r = supabase.table(table).select("chemin:metadata->>chemin").range(debut, debut + 59).execute()
            chemins += [l["chemin"] for l in (r.data or []) if l.get("chemin")]
        except Exception:
            continue
    if not chemins:
        return False
    return sum(1 for c in chemins if _norm_chemin(c) in existants) / len(chemins) >= 0.5


def chemins_indexes(table: str) -> set:
    """Chemins distincts présents dans l'index Supabase.
    Utilise la fonction SQL chemins_indexes() si elle existe (rapide), sinon parcourt la table."""
    try:
        r = supabase.rpc("chemins_indexes", {}).execute()
        return {l["chemin"] for l in (r.data or []) if l.get("chemin")}
    except Exception as e:
        msg = str(e)
        if "PGRST202" not in msg and "Could not find the function" not in msg:
            raise RuntimeError(f"Fonction SQL chemins_indexes() en erreur : {msg[:250]}")  # vraie cause visible
    chemins, debut = set(), 0
    while True:
        r = supabase.table(table).select("chemin:metadata->>chemin").range(debut, debut + 999).execute()
        lot = r.data or []
        chemins.update(l["chemin"] for l in lot if l.get("chemin"))
        if len(lot) < 1000:
            break
        debut += 1000
    return chemins


def synchroniser_index(existants, table: str, chemins_db=None) -> dict:
    """Supprime de Supabase les extraits des PDF qui n'existent plus sur Hugging Face.
    'coherent' = les chemins de l'index correspondent bien à ceux du dépôt (sinon le filtrage reste désactivé)."""
    if not existants:
        return {"supprimes": [], "echecs": [], "coherent": False,
                "erreur": "Liste des PDF Hugging Face indisponible ou vide : rien supprimé."}
    try:
        chemins_db = chemins_db if chemins_db is not None else chemins_indexes(table)
        orphelins = [c for c in chemins_db if _norm_chemin(c) not in existants]
        if orphelins and len(orphelins) > max(3, len(chemins_db) // 2):
            return {"supprimes": [], "echecs": [], "coherent": False, "erreur":
                    f"Sécurité : {len(orphelins)} documents sur {len(chemins_db)} de l'index ne correspondent à aucun "
                    "fichier du dépôt Hugging Face (arborescence différente ?). Rien n'est supprimé ni filtré."}

        supprimes, echecs = [], []
        for c in orphelins:
            r = supabase.table(table).delete().eq("metadata->>chemin", c).execute()
            (supprimes if r.data else echecs).append(c)
        return {"supprimes": supprimes, "echecs": echecs, "coherent": True, "erreur": None}
    except Exception as e:
        return {"supprimes": [], "echecs": [], "coherent": None, "erreur": str(e)[:300]}


# ==========================================
# INDEXATION AUTOMATIQUE DES NOUVELLES NORMES (Hugging Face -> Supabase)
# ==========================================
# ⚠️ À aligner sur ingest.py (même découpage, même préfixe "passage: ", même modèle d'embedding).
TAILLE_CHUNK = 1200       # identique à CHUNK_SIZE d'ingest.py
CHEVAUCHEMENT = 150       # identique à CHUNK_OVERLAP d'ingest.py
SEUIL_TEXTE = 20          # moins de caractères sur une page => page scannée => OCR (même seuil qu'ingest.py)
TAILLE_LOT = 64


def _texte_page(page) -> str:
    """Texte d'une page PDF : couche texte si elle existe, sinon OCR Tesseract (français)."""
    texte = page.get_text("text").strip()
    if len(texte) >= SEUIL_TEXTE:
        return texte
    try:
        import pytesseract
        from PIL import Image
        pix = page.get_pixmap(dpi=200)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return pytesseract.image_to_string(img, lang="fra").strip()
    except Exception:
        return texte


def _decouper(texte: str, taille=TAILLE_CHUNK, chevauchement=CHEVAUCHEMENT) -> list:
    """Même découpage que decouper_en_chunks() d'ingest.py."""
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


def indexer_pdf(chemin: str, table: str, modele, repo: str, token: str, suivi=None) -> int:
    """Télécharge un PDF du dépôt, extrait le texte (OCR si besoin) et l'ajoute à Supabase,
    avec exactement le même format de lignes qu'ingest.py."""
    import fitz  # PyMuPDF
    from huggingface_hub import hf_hub_download
    nom = chemin.split("/")[-1]
    if suivi is not None:
        suivi["progression"] = f"{nom} — téléchargement depuis Hugging Face"
    local = hf_hub_download(repo_id=repo, filename=chemin, repo_type="dataset", token=token)
    lignes = []
    with fitz.open(local) as doc:
        for num, page in enumerate(doc, 1):
            if suivi is not None:
                suivi["progression"] = f"{nom} — lecture/OCR page {num}/{len(doc)}"
            texte = _texte_page(page)
            if len(texte) < 20:
                continue
            for idx, morceau in enumerate(_decouper(texte)):
                cid = f"{num}-{idx}"
                lignes.append({
                    "content": morceau, "fichier": chemin, "chunk_id": cid,  # clé unique = chemin complet (deux PDF de même nom ne s'écrasent plus)
                    "metadata": {"fichier": nom, "chemin": chemin, "page": num, "chunk_id": cid},
                })
    if not lignes:
        raise ValueError("aucun texte extrait (OCR indisponible ? vérifiez packages.txt)")
    if suivi is not None:
        suivi["progression"] = f"{nom} — calcul des embeddings et envoi ({len(lignes)} extraits)"
    supabase.table(table).delete().eq("metadata->>chemin", chemin).execute()  # repart d'une base propre
    try:
        for i in range(0, len(lignes), TAILLE_LOT):
            lot = lignes[i:i + TAILLE_LOT]
            vecteurs = modele.encode([f"passage: {l['content']}" for l in lot],
                                     normalize_embeddings=True, show_progress_bar=False)
            for l, v in zip(lot, vecteurs):
                l["embedding"] = v.tolist()
            supabase.table(table).upsert(lot, on_conflict="fichier,chunk_id").execute()
    except Exception:
        try:
            supabase.table(table).delete().eq("metadata->>chemin", chemin).execute()  # pas d'indexation à moitié
        except Exception:
            pass
        raise
    return len(lignes)


SEUIL_BASE_MO = 450            # au-delà, l'indexation s'arrête (limite gratuite Supabase : 500 Mo)
MAX_FICHIERS_PAR_PASSE = 5     # indexation automatique : au plus 5 PDF par passage (le CPU Streamlit Cloud est limité)


def _taille_base_directe():
    try:
        return int(supabase.rpc("taille_base", {}).execute().data) / 1048576
    except Exception:
        return None


def indexer_nouveaux(existants, table: str, modele, repo: str, token: str, chemins_db=None, suivi=None,
                     maximum=None) -> dict:
    """Indexe les PDF présents sur Hugging Face mais absents de l'index Supabase."""
    res = {"indexes": [], "echecs": {}, "erreur": None}
    if not existants:
        return res
    try:
        deja = {_norm_chemin(c) for c in (chemins_db if chemins_db is not None else chemins_indexes(table))}
    except Exception as e:
        res["erreur"] = str(e)[:300]
        return res
    a_faire = sorted(set(existants) - deja)
    if maximum:
        a_faire = a_faire[:maximum]
    for chemin in a_faire:
        mo = _taille_base_directe()
        if mo is not None and mo >= SEUIL_BASE_MO:
            res["erreur"] = (f"Indexation suspendue : la base fait {mo:.0f} Mo (seuil {SEUIL_BASE_MO} Mo sur 500). "
                             "Libérez de l'espace ou passez à l'offre Pro.")
            break
        try:
            n = indexer_pdf(chemin, table, modele, repo, token, suivi)
            res["indexes"].append(f"{chemin} ({n} extraits)")
        except Exception as e:
            res["echecs"][chemin] = str(e)[:200]
    if suivi is not None:
        suivi["progression"] = None
    if res["indexes"]:
        _candidats.clear()  # les recherches mises en cache (1 h) ne verraient pas la nouvelle norme
    return res


@st.cache_resource
def _etat_synchro():
    # filtre_sur : True seulement après avoir vérifié que l'index et Hugging Face utilisent les mêmes chemins
    return {"dernier": 0.0, "verrou": threading.Lock(), "resultat": None, "filtre_sur": False}


def _appliquer_resultat(etat, r):
    etat["resultat"] = r
    if r.get("coherent") is not None:
        etat["filtre_sur"] = bool(r["coherent"])


def lancer_synchro_auto():
    """Synchronisation en arrière-plan (ajouts + suppressions), sans ralentir l'utilisateur."""
    etat = _etat_synchro()
    if time.time() - etat["dernier"] < INTERVALLE_SYNCHRO:
        return
    existants = fichiers_pdf_existants()
    if not existants:
        return
    modele = charger_modele()  # chargé ici (thread principal), pas dans le thread d'arrière-plan
    if not etat["verrou"].acquire(blocking=False):
        return
    etat["dernier"] = time.time()
    table = st.secrets.get("SUPABASE_TABLE", "documents")
    repo, token = st.secrets.get("HF_REPO_ID", ""), st.secrets.get("HF_TOKEN", "")
    # INDEXATION_AUTO = "false" dans les secrets : pas d'indexation automatique (utile pendant une
    # indexation massive avec ingest.py sur votre PC, pour éviter que les deux travaillent en même temps)
    auto = str(st.secrets.get("INDEXATION_AUTO", "true")).strip().lower() not in ("false", "0", "non", "no")

    def _tache():
        try:
            etat["debut"] = time.time()
            etat["progression"] = "lecture de l'index Supabase"
            try:
                en_base = chemins_indexes(table)  # une seule lecture de l'index pour les deux étapes
            except Exception:
                en_base = None
            etat["progression"] = "nettoyage des normes supprimées"
            _appliquer_resultat(etat, synchroniser_index(existants, table, en_base))
            if auto:
                etat["indexation"] = indexer_nouveaux(existants, table, modele, repo, token, en_base, etat,
                                                      maximum=MAX_FICHIERS_PAR_PASSE)
            else:
                etat["indexation"] = {"indexes": [], "echecs": {}, "erreur":
                                      "Indexation automatique désactivée (secret INDEXATION_AUTO = false)."}
        finally:
            etat["progression"] = None
            etat["verrou"].release()

    threading.Thread(target=_tache, daemon=True).start()


# --- Normes principales : dossier "normes principales" du dépôt Hugging Face ---
BONUS_PRINCIPALE = 0.04          # avantage de classement donné aux normes principales
NB_CANDIDATS_PRINCIPALES = 25    # extraits cherchés spécifiquement dans ce dossier
MOTIF_SQL_PRINCIPALE = "normes_principal%/%"  # '_' = joker : accepte "normes principales" et "normes_principales"


def est_principale(chunk) -> bool:
    """Vrai si le document est dans le dossier « normes principales » (accents/majuscules/_ ignorés)."""
    chemin = _chemin_norm(chunk)
    if "/" not in chemin:
        return False
    premier = unicodedata.normalize("NFD", chemin.split("/")[0])
    premier = "".join(ch for ch in premier if unicodedata.category(ch) != "Mn")
    return premier.lower().replace("_", " ").replace("-", " ").strip().startswith("normes principal")


def _rpc(nom: str, params: dict, essais: int = 2):
    """Appel RPC Supabase : une 2e tentative en cas de délai dépassé / saturation,
    sinon une erreur LISIBLE (Streamlit Cloud masque le message d'origine de APIError)."""
    derniere = None
    for tentative in range(essais):
        try:
            return supabase.rpc(nom, params).execute()
        except Exception as e:
            derniere = e
            code = str(getattr(e, "code", "") or "")
            msg = str(getattr(e, "message", "") or e)
            temporaire = code in ("57014", "53300", "54000", "PGRST003") or "timeout" in msg.lower()
            if tentative + 1 < essais and temporaire:
                time.sleep(1.5)
                continue
            break
    code = str(getattr(derniere, "code", "") or "")
    details = " | ".join(str(x) for x in (getattr(derniere, "message", None) or derniere,
                                          getattr(derniere, "details", None),
                                          getattr(derniere, "hint", None)) if x)
    raise RuntimeError(f"Supabase · fonction « {nom} » · code {code or 'n/a'} · {details}"[:700])


@st.cache_data(ttl=60, show_spinner=False)
def nb_extraits():
    """Nombre de lignes dans la table des extraits, None si indisponible."""
    try:
        table = st.secrets.get("SUPABASE_TABLE", "documents")
        return supabase.table(table).select("id", count="exact").limit(1).execute().count
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def taille_base_mo():
    """Taille de la base en Mo (fonction SQL taille_base()), None si indisponible."""
    try:
        return int(_rpc("taille_base", {}, essais=1).data) / 1048576
    except Exception:
        return None


@st.cache_data(ttl=3600, max_entries=300, show_spinner=False)
def _vecteur_question(question: str):
    return embed_texte(question, prefixe="query")


@st.cache_data(ttl=3600, max_entries=300, show_spinner=False)
def _candidats(question: str):
    res = _rpc("match_documents", {
        "query_embedding": _vecteur_question(question),
        "match_count": NB_CANDIDATS,
    })
    if not res.data:
        # Une exception n'est PAS mise en cache par Streamlit (contrairement à une liste vide gardée 1 h)
        raise LookupError("La fonction match_documents n'a renvoyé aucun extrait : la table est vide, "
                          "ou l'index/la fonction SQL ne trouve rien (voir optimisation_supabase.sql).")
    return sorted(res.data, key=lambda c: c["similarity"], reverse=True)


@st.cache_data(ttl=3600, max_entries=300, show_spinner=False)
def _candidats_principales(question: str):
    res = _rpc("match_documents_prefixe", {
        "query_embedding": _vecteur_question(question),
        "match_count": NB_CANDIDATS_PRINCIPALES,
        "motif": MOTIF_SQL_PRINCIPALE,
    })
    return sorted(res.data or [], key=lambda c: c["similarity"], reverse=True)


def _principales(question: str, generaux: list) -> list:
    """Extraits des normes principales. Si la fonction SQL dédiée n'existe pas encore,
    on se rabat sur ceux déjà présents parmi les candidats généraux."""
    try:
        return _candidats_principales(question)
    except Exception:
        return [c for c in generaux if est_principale(c)]


def rechercher_chunks(question: str, seulement_principales: bool = False):
    """Recherche vectorielle en deux niveaux : les normes principales sont cherchées à part (et favorisées),
    puis complétées par le reste de la base (sauf en mode « normes principales uniquement »).
    Écarte aussi les normes supprimées de Hugging Face."""
    generaux = _candidats(question)
    principaux = _principales(question, generaux)

    if seulement_principales:
        candidats = list(principaux)
    else:
        candidats = list(generaux)
        deja = {(c["metadata"].get("fichier"), c["metadata"].get("page"), c["content"][:120]) for c in candidats}
        candidats += [c for c in principaux
                      if (c["metadata"].get("fichier"), c["metadata"].get("page"), c["content"][:120]) not in deja]

    existants = fichiers_pdf_existants()
    if existants and layout_coherent():
        candidats = [c for c in candidats if not _chemin_norm(c) or _chemin_norm(c) in existants]
    if not candidats:
        return []

    for c in candidats:
        c["principale"] = est_principale(c)
        c["score"] = c["similarity"] + (BONUS_PRINCIPALE if c["principale"] else 0.0)
    candidats.sort(key=lambda c: c["score"], reverse=True)

    meilleur = candidats[0]["score"]
    gardes, vus = [], set()
    for c in candidats:
        cle = (c["metadata"].get("fichier"), c["metadata"].get("page"), c["content"][:120])
        if cle in vus:
            continue  # doublon
        vus.add(cle)
        if len(gardes) < NB_MIN_RESULTATS or c["score"] >= meilleur - ECART_PERTINENCE:
            gardes.append(c)
        if len(gardes) >= NB_MAX_RESULTATS:
            break

    # Normes principales en premier, puis regroupement par document et page
    gardes.sort(key=lambda c: (not c["principale"], str(c["metadata"].get("fichier")), c["metadata"].get("page") or 0))
    return gardes


def construire_prompt(question: str, chunks: list, historique: list) -> str:
    contexte_docs = "\n\n---\n\n".join(
        f"[Source {i+1} — {'NORME PRINCIPALE' if c.get('principale') else 'document complémentaire'} — "
        f"{c['metadata'].get('fichier')}, page {c['metadata'].get('page')}]\n{c['content']}"
        for i, c in enumerate(chunks)
    )
    derniers = historique[-NB_ECHANGES_CONTEXTE:]
    fil = "\n".join(f"{'Agent' if h['role'] == 'user' else 'Assistant'} : {h['content'][:600]}" for h in derniers)
    bloc_historique = f"\nContexte de la conversation :\n{fil}\n" if derniers else ""

    return f"""Tu es un assistant technique pour les agents du LPEE (laboratoire d'essais de matériaux et travaux publics).

RÈGLES DE PRÉCISION (impératives) :
1. Réponds UNIQUEMENT à partir des extraits ci-dessous. N'utilise aucune connaissance extérieure.
2. Sois EXHAUSTIF : croise tous les extraits pertinents et fusionne ceux qui se complètent.
3. Reprends FIDÈLEMENT les valeurs chiffrées, unités, seuils, tolérances, formules, références de normes
   (ex. NF EN, NM, ASTM), numéros d'articles/paragraphes et conditions d'essai, exactement comme écrits.
   Ne les arrondis pas, ne les convertis pas.
4. Si des extraits se contredisent (versions/normes différentes), signale-le et cite chaque source.
5. Cite chaque affirmation avec (Source X) — fichier et page.
6. PRIORITÉ AUX NORMES PRINCIPALES : les extraits « NORME PRINCIPALE » font référence. Appuie-toi d'abord sur eux.
   N'utilise un « document complémentaire » que pour compléter, et précise-le (« d'après un document complémentaire »).
   En cas de divergence, la norme principale prévaut ; signale l'écart. Si aucun extrait de norme principale
   ne traite la question, dis-le dans ta première phrase.
7. Termine par une section « Points non couverts » listant ce que les extraits ne permettent pas de confirmer.
   Ne devine jamais.

FORMAT : commence directement par la réponse (pas d'introduction), puis sections courtes avec puces ;
tableau si tu compares des valeurs ; en dernier, « Points non couverts ».
{bloc_historique}
EXTRAITS :
{contexte_docs}

QUESTION : {question}

RÉPONSE :"""


def flux_reponse(prompt: str):
    """Générateur : affiche la réponse au fur et à mesure (1er mot en ~1 s) avec repli automatique de modèle."""
    derniere_erreur = None
    for modele in MODELES_GENERATION:
        emis = False
        try:
            for morceau in client.models.generate_content_stream(
                model=modele, contents=prompt, config=_config(4096, modele)
            ):
                if morceau.text:
                    emis = True
                    yield morceau.text
            if emis:
                return
        except Exception as e:
            derniere_erreur = e
            if emis:
                yield "\n\n⚠️ Réponse interrompue (saturation du service). Reposez la question pour la compléter."
                return
            if _erreur_temporaire(e):
                time.sleep(2)
    yield (
        "⚠️ Le service Gemini est momentanément saturé (côté Google). Les documents trouvés sont listés "
        "ci-dessous ; réessayez dans une minute.\n\n"
        f"Détail technique : `{derniere_erreur}`"
    )


# --- Téléchargement des PDF originaux (dépôt Hugging Face privé, gratuit) ---
HF_REPO_ID = st.secrets.get("HF_REPO_ID", "")
HF_TOKEN = st.secrets.get("HF_TOKEN", "")


@st.cache_data(show_spinner="Récupération du PDF...", max_entries=5)
def telecharger_pdf(chemin_relatif: str) -> bytes:
    from huggingface_hub import hf_hub_download
    chemin_local = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=chemin_relatif.replace("\\", "/"),
        repo_type="dataset",
        token=HF_TOKEN,
    )
    with open(chemin_local, "rb") as f:
        return f.read()


def afficher_sources(chunks: list, prefixe_cle: str):
    existants = fichiers_pdf_existants()
    verif = bool(existants) and layout_coherent()

    def _supprime(c) -> bool:
        return verif and bool(_chemin_norm(c)) and _chemin_norm(c) not in existants

    nb_retirees = sum(1 for c in chunks if _supprime(c))
    with st.expander("📎 Sources utilisées"):
        for i, c in enumerate(chunks, 1):
            if _supprime(c):
                continue
            meta = c["metadata"]
            etiquette = "⭐ Norme principale" if c.get("principale") else "Complémentaire"
            st.markdown(f"**Source {i} · {meta.get('fichier')}** — page {meta.get('page')} · {etiquette} (pertinence : {c['similarity']:.0%})")
            st.caption(c["content"])
            st.divider()

    if nb_retirees:
        st.caption(f"ℹ️ {nb_retirees} extrait(s) cité(s) proviennent de documents retirés de la base : masqués.")

    if not (HF_REPO_ID and HF_TOKEN):
        return

    fichiers_uniques = {}
    for c in chunks:
        if _supprime(c):
            continue
        meta = c["metadata"]
        if meta.get("chemin"):
            fichiers_uniques[meta["chemin"]] = meta.get("fichier")

    if fichiers_uniques:
        st.markdown("**📥 Télécharger les documents sources**")
        for i, (chemin, nom) in enumerate(fichiers_uniques.items()):
            cle = f"{prefixe_cle}_{i}"
            if st.button(f"📄 {nom}", key=f"prep_{cle}"):
                try:
                    st.download_button(
                        f"⬇️ Enregistrer {nom}",
                        data=telecharger_pdf(chemin),
                        file_name=nom,
                        mime="application/pdf",
                        key=f"dl_{cle}",
                    )
                except Exception as e:
                    st.error(f"PDF indisponible ({nom}) : {e}")


# --- Gestion multi-conversations & historique ---
if st.session_state.get("historique_charge_pour") != st.session_state.username_courant:
    chargé = charger_historique(st.session_state.username_courant)
    if chargé:
        st.session_state.sessions, cible = chargé
        st.session_state.session_courante = cible if cible in st.session_state.sessions else list(st.session_state.sessions)[-1]
    else:
        st.session_state.sessions = {"Conversation 1": []}
        st.session_state.session_courante = "Conversation 1"
    st.session_state.historique_charge_pour = st.session_state.username_courant

historique = st.session_state.sessions[st.session_state.session_courante]

# --- Barre latérale (Sidebar) ---
with st.sidebar:
    if os.path.exists("logo lpee.jpg"):
        st.image("logo lpee.jpg", use_container_width=True)
    elif os.path.exists("logo_lpee.jpg"):
        st.image("logo_lpee.jpg", use_container_width=True)
    
    st.write(f"Connecté : **{st.session_state.nom_utilisateur}**")
    if st.button("🚪 Déconnexion", use_container_width=True):
        st.session_state.authentifie = False
        st.session_state.username_courant = ""
        st.session_state.nom_utilisateur = ""
        st.session_state.role_utilisateur = ""
        for k in ("sessions", "session_courante", "historique_charge_pour"):
            st.session_state.pop(k, None)
        st.session_state.deconnexion_volontaire = True
        try:
            controller.remove(COOKIE_NOM)
        except Exception:
            pass
        time.sleep(0.7)
        st.rerun()

    with st.expander("🔑 Changer mon mot de passe"):
        with st.form("form_mdp", clear_on_submit=True):
            ancien = st.text_input("Mot de passe actuel", type="password")
            nouveau = st.text_input("Nouveau mot de passe (8 caractères min.)", type="password")
            confirmation = st.text_input("Confirmer le nouveau mot de passe", type="password")
            valider_mdp = st.form_submit_button("Modifier", use_container_width=True)
        if valider_mdp:
            login = st.session_state.username_courant
            actuels = charger_utilisateurs()
            if actuels is None or login not in actuels:
                st.error("Impossible de vérifier votre compte, réessayez.")
            elif not verifier_mdp(ancien, actuels[login]["password"]):
                st.error("Mot de passe actuel incorrect.")
            elif len(nouveau) < 8:
                st.error("Le nouveau mot de passe doit contenir au moins 8 caractères.")
            elif nouveau != confirmation:
                st.error("La confirmation ne correspond pas.")
            elif verifier_mdp(nouveau, actuels[login]["password"]):
                st.error("Le nouveau mot de passe doit être différent de l'ancien.")
            else:
                def _changer(users):
                    users[login]["password"] = hacher_mdp(nouveau)
                ok, err = modifier_comptes(_changer)
                if ok:
                    poser_cookie(controller, login, st.session_state.utilisateurs[login]["password"])
                    st.success("✅ Mot de passe modifié. Utilisez-le à la prochaine connexion.")
                else:
                    st.error(f"Échec de l'enregistrement : {err}")

    if st.session_state.role_utilisateur == "admin":
        with st.expander("🧹 Synchronisation des normes"):
            st.caption("Les normes ajoutées sur Hugging Face sont lues (OCR si scannées) et ajoutées à l'index ; "
                       "celles supprimées en sont retirées (vérification au démarrage puis toutes les 3 h).")
            table_idx = st.secrets.get("SUPABASE_TABLE", "documents")
            _n = nb_extraits()
            if _n is not None:
                (st.error if _n == 0 else st.caption)(f"📚 {_n:,} extraits indexés".replace(",", " ")
                                                       + (" — la base est VIDE : relancez l'indexation (ingest.py)." if _n == 0 else ""))
            _mo = taille_base_mo()
            if _mo is not None:
                _txt = f"💾 Taille de la base : {_mo:.0f} Mo / 500 Mo (offre gratuite)"
                if _mo >= 500:
                    st.error(_txt + " — base en lecture seule : supprimez des données.")
                elif _mo >= 450:
                    st.warning(_txt)
                else:
                    st.caption(_txt)
            _prog = _etat_synchro().get("progression")
            if _prog:
                _duree = int((time.time() - _etat_synchro().get("debut", time.time())) / 60)
                st.info(f"⏳ Indexation en cours (depuis {_duree} min) : " + _prog)
            if st.button("Vérifier l'état de l'index", use_container_width=True):
                fichiers_pdf_existants.clear()
                _ex = fichiers_pdf_existants()
                if not _ex:
                    st.error("Liste des PDF Hugging Face indisponible (vérifiez HF_REPO_ID / HF_TOKEN).")
                else:
                    try:
                        _en_base = {_norm_chemin(c) for c in chemins_indexes(table_idx)}
                        _manq = sorted(set(_ex) - _en_base)
                        st.write(f"{len(_ex)} PDF sur Hugging Face · {len(set(_ex) & _en_base)} indexés")
                        if _manq:
                            st.warning("Pas encore indexés : " + ", ".join(_manq))
                        else:
                            st.success("Tous les PDF du dépôt sont indexés.")
                    except Exception as _e:
                        st.error(f"Lecture de l'index impossible : {str(_e)[:300]}")
                        if "57014" in str(_e) or "timeout" in str(_e).lower():
                            st.info("L'index Supabase est trop lent à lire : exécutez le script "
                                    "optimisation_supabase.sql dans Supabase > SQL Editor.")
            if st.button("Synchroniser maintenant", use_container_width=True):
                fichiers_pdf_existants.clear()
                etat = _etat_synchro()
                if not etat["verrou"].acquire(blocking=False):
                    st.info("⏳ Une indexation est déjà en cours en arrière-plan"
                            + (f" : {etat['progression']}" if etat.get("progression") else "")
                            + ". La norme sera cherchable à la fin. Cliquez sur « Vérifier l'état » pour suivre.")
                else:
                    try:
                        with st.spinner("Synchronisation (l'OCR d'une norme scannée peut prendre plusieurs minutes)..."):
                            existants = fichiers_pdf_existants()
                            r = synchroniser_index(existants, table_idx)
                            _appliquer_resultat(etat, r)
                            ri = indexer_nouveaux(existants, table_idx, charger_modele(),
                                                  st.secrets.get("HF_REPO_ID", ""), st.secrets.get("HF_TOKEN", ""),
                                                  suivi=etat)
                            etat["indexation"] = ri
                    finally:
                        etat["verrou"].release()
                    if r["erreur"]:
                        st.error(r["erreur"])
                    if r["echecs"]:
                        st.warning("Suppression refusée par Supabase (droits insuffisants ?) : " + ", ".join(r["echecs"]))
                    if r["supprimes"]:
                        st.success("Retirées de l'index : " + ", ".join(r["supprimes"]))
                    if ri["erreur"]:
                        st.error(ri["erreur"])
                    if ri["indexes"]:
                        st.success("Ajoutées à l'index : " + ", ".join(ri["indexes"]))
                    for f, e in ri["echecs"].items():
                        st.warning(f"Échec d'indexation de {f} : {e}")
                    if not (r["supprimes"] or ri["indexes"] or ri["echecs"] or r["erreur"] or ri["erreur"]):
                        st.success("Index déjà à jour.")
            auto = _etat_synchro().get("resultat")
            if auto and auto["supprimes"]:
                st.caption("Dernier nettoyage auto : " + ", ".join(auto["supprimes"]))
            if auto and (auto["erreur"] or auto["echecs"]):
                st.caption("⚠️ Nettoyage auto : " + (auto["erreur"] or "suppression refusée (droits Supabase)"))
            ind = _etat_synchro().get("indexation")
            if ind and ind["indexes"]:
                st.caption("Dernière indexation auto : " + ", ".join(ind["indexes"]))
            if ind and (ind["erreur"] or ind["echecs"]):
                st.caption("⚠️ Indexation auto : " + (ind["erreur"] or "; ".join(f"{k} ({v})" for k, v in ind["echecs"].items())))

    st.markdown("---")

    # --- Section Gestion des Utilisateurs (Visible uniquement pour l'admin) ---
    if st.session_state.get("role_utilisateur") == "admin":
        with st.expander("👥 Gestion des Utilisateurs"):
            action_user = st.radio("Action", ["Ajouter", "Modifier / Supprimer"], label_visibility="collapsed")
            
            if action_user == "Ajouter":
                st.subheader("Nouvel agent")
                n_user = st.text_input("Identifiant (login)").strip().lower()
                n_pass = st.text_input("Mot de passe", type="password")
                n_nom = st.text_input("Nom complet / Rôle")
                n_role = st.selectbox("Rôle", ["agent", "admin"])

                if st.button("Enregistrer l'agent", use_container_width=True):
                    if n_user and n_pass and n_nom:
                        def _ajouter(users):
                            if n_user in users:
                                return "Cet identifiant existe déjà."
                            users[n_user] = {"password": hacher_mdp(n_pass), "nom": n_nom, "role": n_role}
                        ok, err = modifier_comptes(_ajouter)
                        if ok:
                            st.success(f"Agent {n_nom} ajouté avec succès !")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(err)
                    else:
                        st.warning("Veuillez remplir tous les champs.")

            else:
                st.subheader("Modifier ou Supprimer")
                liste_logins = list(st.session_state.utilisateurs.keys())
                sel_user = st.selectbox("Choisir un utilisateur", liste_logins)

                if sel_user:
                    fiche = st.session_state.utilisateurs[sel_user]
                    mod_nom = st.text_input("Nom complet", value=fiche["nom"])
                    mod_pass = st.text_input("Nouveau mot de passe (vide = inchangé)", type="password")
                    mod_role = st.selectbox("Rôle", ["agent", "admin"], index=0 if fiche["role"] == "agent" else 1)

                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        if st.button("Mettre à jour", use_container_width=True):
                            def _maj(users):
                                if sel_user not in users:
                                    return "Utilisateur introuvable."
                                users[sel_user]["nom"] = mod_nom
                                users[sel_user]["role"] = mod_role
                                if mod_pass:
                                    users[sel_user]["password"] = hacher_mdp(mod_pass)
                            ok, err = modifier_comptes(_maj)
                            if ok:
                                st.success("Modifications enregistrées !")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(err)
                    with col_m2:
                        if sel_user != "admin":
                            if st.button("Supprimer", use_container_width=True, type="primary"):
                                def _suppr(users):
                                    users.pop(sel_user, None)
                                ok, err = modifier_comptes(_suppr)
                                if ok:
                                    st.success("Utilisateur supprimé.")
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error(err)
                        else:
                            st.caption("Admin principal non supprimable.")

        st.markdown("---")

    st.header("🗂️ Historique & Sessions")
    
    if st.button("➕ Nouvelle conversation", use_container_width=True):
        nb = len(st.session_state.sessions) + 1
        nouvelle_cle = f"Conversation {nb}"
        st.session_state.sessions[nouvelle_cle] = []
        st.session_state.session_courante = nouvelle_cle
        sauvegarder_historique()
        st.rerun()

    st.markdown("---")
    st.subheader("Mes conversations")
    
    session_choisie = st.radio(
        "Sélectionner une session",
        list(st.session_state.sessions.keys()),
        index=list(st.session_state.sessions.keys()).index(st.session_state.session_courante),
        label_visibility="collapsed"
    )
    if session_choisie != st.session_state.session_courante:
        st.session_state.session_courante = session_choisie
        sauvegarder_historique()
        st.rerun()

    st.markdown("---")
    st.markdown("### 🔍 Recherches récentes (Session active)")
    questions_passees = [item["content"] for item in historique if item["role"] == "user"]
    if questions_passees:
        for q in reversed(questions_passees[-10:]):
            st.caption(f"• {q}")
    else:
        st.caption("Aucune question pour l'instant.")

if st.session_state.get("erreur_historique"):
    st.sidebar.warning("⚠️ Historique non sauvegardé. Détail de l'erreur ci-dessous :")
    st.sidebar.code(str(st.session_state["erreur_historique"])[:400], language=None)

lancer_synchro_auto()

# --- Interface Principale ---
st.title("📚 Recherche des normes et fascicules techniques — LPEE")
st.subheader(f"👋 Bonjour {st.session_state.nom_utilisateur}")
st.caption(f"Session active : **{st.session_state.session_courante}** — Posez une question, puis enchaînez des questions de suivi si besoin.")

col_titre, col_bouton = st.columns([5, 1])
with col_titre:
    seulement_principales = st.checkbox(
        "🎯 Chercher uniquement dans les normes principales",
        key="seulement_principales",
        help="Décochée : les normes principales sont consultées en priorité, puis complétées par les autres documents.",
    )
with col_bouton:
    if st.button("🗑️ Vider", use_container_width=True):
        st.session_state.sessions[st.session_state.session_courante] = []
        sauvegarder_historique()
        st.rerun()

for idx, echange in enumerate(historique):
    with st.chat_message("user" if echange["role"] == "user" else "assistant"):
        st.markdown(echange["content"])
        if echange["role"] == "assistant" and echange.get("sources"):
            afficher_sources(echange["sources"], prefixe_cle=f"{st.session_state.session_courante}_msg{idx}")

question = st.chat_input("Posez votre question ou enchaînez sur la précédente...")

if question:
    historique.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        t0 = time.time()
        try:
            with st.spinner("Recherche dans les documents indexés..."):
                question_recherche = reformuler_question(question, historique[:-1])
                chunks = rechercher_chunks(question_recherche, seulement_principales)
        except Exception as e:
            historique.pop()  # la question n'a pas reçu de réponse : on ne la garde pas dans l'historique
            detail = str(e)
            st.error("⚠️ La recherche dans la base a échoué. Réessayez dans un instant.")
            st.code(detail[:700], language=None)
            if "57014" in detail or "timeout" in detail.lower():
                st.info("Délai dépassé côté Supabase : exécutez `optimisation_supabase.sql` "
                        "(index + délai d'exécution) et vérifiez que la base reste sous 500 Mo.")
            elif "aucun extrait" in detail:
                st.info("Vérifiez dans Supabase : `select count(*), count(embedding) from documents;`")
            elif "PGRST202" in detail or "Could not find" in detail:
                st.info("Fonction SQL absente : exécutez `optimisation_supabase.sql` dans Supabase > SQL Editor.")
            elif "read-only" in detail.lower():
                st.info("Base en lecture seule (quota de 500 Mo dépassé) : libérez de l'espace.")
            st.stop()
        t_recherche = time.time() - t0

        if not chunks:
            reponse = (
                "Aucun passage trouvé dans les **normes principales** pour cette question. "
                "Décochez « Chercher uniquement dans les normes principales » pour élargir la recherche."
                if seulement_principales else
                "Aucun document pertinent trouvé pour cette question. Reformulez-la, ou vérifiez que l'indexation a bien été exécutée."
            )
            st.markdown(reponse)
            historique.append({"role": "assistant", "content": reponse, "sources": []})
            sauvegarder_historique()
        else:
            prompt = construire_prompt(question, chunks, historique[:-1])
            t1 = time.time()
            reponse = st.write_stream(flux_reponse(prompt))  # affichage progressif
            t_generation = time.time() - t1
            afficher_sources(chunks, prefixe_cle=f"{st.session_state.session_courante}_msg{len(historique)}")
            st.caption(f"⏱️ Recherche : {t_recherche:.1f} s · Génération de la réponse : {t_generation:.1f} s")
            historique.append({"role": "assistant", "content": reponse, "sources": chunks})
            sauvegarder_historique()
