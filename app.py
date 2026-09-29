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
    SUPABASE_KEY = "..."
"""

import os
import time

import streamlit as st
from supabase import create_client
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Recherche Normes LPEE", page_icon="📚", layout="wide")

# ==========================================
# GESTION DES UTILISATEURS (STOCKAGE EN SESSION)
# ==========================================
if "utilisateurs" not in st.session_state:
    st.session_state.utilisateurs = {
        "admin": {"password": "admin123", "nom": "Administrateur LPEE", "role": "admin"},
        "agent1": {"password": "lpee2026", "nom": "Agent Laboratoire Béton", "role": "agent"},
        "agent2": {"password": "lpee2026", "nom": "Agent Laboratoire Sols", "role": "agent"}
    }

if "authentifie" not in st.session_state:
    st.session_state.authentifie = False
if "username_courant" not in st.session_state:
    st.session_state.username_courant = ""
if "nom_utilisateur" not in st.session_state:
    st.session_state.nom_utilisateur = ""
if "role_utilisateur" not in st.session_state:
    st.session_state.role_utilisateur = ""

# --- ÉCRAN DE CONNEXION ---
if not st.session_state.authentifie:
    st.title("🔐 Connexion — Plateforme LPEE")
    st.markdown("Veuillez vous identifier pour accéder aux normes et fascicules techniques.")
    
    with st.form("form_login"):
        username_input = st.text_input("Nom d'utilisateur").strip().lower()
        password_input = st.text_input("Mot de passe", type="password")
        submit_login = st.form_submit_button("Se connecter")
        
        if submit_login:
            if username_input in st.session_state.utilisateurs and st.session_state.utilisateurs[username_input]["password"] == password_input:
                st.session_state.authentifie = True
                st.session_state.username_courant = username_input
                st.session_state.nom_utilisateur = st.session_state.utilisateurs[username_input]["nom"]
                st.session_state.role_utilisateur = st.session_state.utilisateurs[username_input]["role"]
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
NB_CANDIDATS = 30          # extraits récupérés dans la base
NB_MIN_RESULTATS = 8       # minimum gardé même si la pertinence est faible
NB_MAX_RESULTATS = 20      # maximum envoyé au modèle (prompt plus court = réponse plus rapide)
ECART_PERTINENCE = 0.08    # on écarte les extraits trop éloignés du meilleur score
NB_ECHANGES_CONTEXTE = 4

client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])
supabase = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

import io
import json
import re


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


@st.cache_data(ttl=3600, max_entries=300, show_spinner=False)
def rechercher_chunks(question: str):
    """Recherche vectorielle (mise en cache) + filtrage par pertinence + tri par document/page."""
    vecteur = embed_texte(question, prefixe="query")
    res = supabase.rpc("match_documents", {
        "query_embedding": vecteur,
        "match_count": NB_CANDIDATS,
    }).execute()
    candidats = sorted(res.data or [], key=lambda c: c["similarity"], reverse=True)
    if not candidats:
        return []

    meilleur = candidats[0]["similarity"]
    gardes, vus = [], set()
    for c in candidats:
        cle = (c["metadata"].get("fichier"), c["metadata"].get("page"), c["content"][:120])
        if cle in vus:
            continue  # doublon
        vus.add(cle)
        if len(gardes) < NB_MIN_RESULTATS or c["similarity"] >= meilleur - ECART_PERTINENCE:
            gardes.append(c)
        if len(gardes) >= NB_MAX_RESULTATS:
            break

    # Regroupe par document puis page : le modèle lit un texte plus cohérent
    gardes.sort(key=lambda c: (str(c["metadata"].get("fichier")), c["metadata"].get("page") or 0))
    return gardes


def construire_prompt(question: str, chunks: list, historique: list) -> str:
    contexte_docs = "\n\n---\n\n".join(
        f"[Source {i+1} — {c['metadata'].get('fichier')}, page {c['metadata'].get('page')}]\n{c['content']}"
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
6. Termine par une section « Points non couverts » listant ce que les extraits ne permettent pas de confirmer.
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
    with st.expander("📎 Sources utilisées"):
        for i, c in enumerate(chunks, 1):
            meta = c["metadata"]
            st.markdown(f"**Source {i} · {meta.get('fichier')}** — page {meta.get('page')} (pertinence : {c['similarity']:.0%})")
            st.caption(c["content"])
            st.divider()

    if not (HF_REPO_ID and HF_TOKEN):
        return

    fichiers_uniques = {}
    for c in chunks:
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
        st.rerun()

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
                        if n_user in st.session_state.utilisateurs:
                            st.error("Cet identifiant existe déjà.")
                        else:
                            st.session_state.utilisateurs[n_user] = {
                                "password": n_pass,
                                "nom": n_nom,
                                "role": n_role
                            }
                            st.success(f"Agent {n_nom} ajouté avec succès !")
                            time.sleep(1)
                            st.rerun()
                    else:
                        st.warning("Veuillez remplir tous les champs.")
            
            else:
                st.subheader("Modifier ou Supprimer")
                liste_logins = list(st.session_state.utilisateurs.keys())
                sel_user = st.selectbox("Choisir un utilisateur", liste_logins)
                
                if sel_user:
                    mod_nom = st.text_input("Nom complet", value=st.session_state.utilisateurs[sel_user]["nom"])
                    mod_pass = st.text_input("Nouveau mot de passe", value=st.session_state.utilisateurs[sel_user]["password"], type="password")
                    mod_role = st.selectbox("Rôle", ["agent", "admin"], index=0 if st.session_state.utilisateurs[sel_user]["role"] == "agent" else 1)
                    
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        if st.button("Mettre à jour", use_container_width=True):
                            st.session_state.utilisateurs[sel_user]["nom"] = mod_nom
                            st.session_state.utilisateurs[sel_user]["password"] = mod_pass
                            st.session_state.utilisateurs[sel_user]["role"] = mod_role
                            st.success("Modifications enregistrées !")
                            time.sleep(1)
                            st.rerun()
                    with col_m2:
                        if sel_user != "admin":
                            if st.button("Supprimer", use_container_width=True, type="primary"):
                                del st.session_state.utilisateurs[sel_user]
                                st.success("Utilisateur supprimé.")
                                time.sleep(1)
                                st.rerun()
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

# --- Interface Principale ---
st.title("📚 Recherche des normes et fascicules techniques — LPEE")
st.caption(f"Session active : **{st.session_state.session_courante}** — Posez une question, puis enchaînez des questions de suivi si besoin.")

col_titre, col_bouton = st.columns([5, 1])
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
        with st.spinner("Recherche dans les documents indexés..."):
            question_recherche = reformuler_question(question, historique[:-1])
            chunks = rechercher_chunks(question_recherche)

        if not chunks:
            reponse = "Aucun document pertinent trouvé pour cette question. Reformulez-la, ou vérifiez que l'indexation a bien été exécutée."
            st.markdown(reponse)
            historique.append({"role": "assistant", "content": reponse, "sources": []})
            sauvegarder_historique()
        else:
            prompt = construire_prompt(question, chunks, historique[:-1])
            reponse = st.write_stream(flux_reponse(prompt))  # affichage progressif
            afficher_sources(chunks, prefixe_cle=f"{st.session_state.session_courante}_msg{len(historique)}")
            historique.append({"role": "assistant", "content": reponse, "sources": chunks})
            sauvegarder_historique()
