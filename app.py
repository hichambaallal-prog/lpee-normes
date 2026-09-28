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
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Recherche Normes LPEE", page_icon="📚", layout="wide")

# --- Authentification multi-utilisateurs simple ---
if "authentifie" not in st.session_state:
    st.session_state.authentifie = False
    st.session_state.nom_utilisateur = ""

if not st.session_state.authentifie:
    st.title("🔐 Connexion — Plateforme LPEE")
    st.markdown("Veuillez vous identifier pour accéder aux normes et fascicules techniques.")
    
    with st.form("form_login"):
        username_input = st.text_input("Nom d'utilisateur")
        password_input = st.text_input("Mot de passe", type="password")
        submit_login = st.form_submit_button("Se connecter")
        
        if submit_login:
            # Comptes autorisés (Vous pouvez en ajouter facilement ici)
            if username_input == "admin" and password_input == "admin123":
                st.session_state.authentifie = True
                st.session_state.nom_utilisateur = "Administrateur LPEE"
                st.rerun()
            elif username_input == "agent" and password_input == "lpee2026":
                st.session_state.authentifie = True
                st.session_state.nom_utilisateur = "Agent LPEE"
                st.rerun()
            elif username_input == "ingenieur" and password_input == "pass123":
                st.session_state.authentifie = True
                st.session_state.nom_utilisateur = "Ingénieur LPEE"
                st.rerun()
            else:
                st.error("Nom d'utilisateur ou mot de passe incorrect.")
    st.stop()

# --- APPLICATION PRINCIPALE (Si connecté) ---

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"  # DOIT être le même modèle que dans ingest.py
# Modèles essayés dans l'ordre : si le premier est saturé (erreur 503), on bascule sur le suivant.
# Un nom inexistant chez vous est simplement ignoré (le suivant est essayé), sans faire planter l'appli.
MODELES_GENERATION = [
    "gemini-flash-latest",        # modèle principal
    "gemini-flash-lite-latest",   # secours 1 : plus léger, moins souvent saturé
    "gemini-2.5-flash",           # secours 2
]
NB_RESULTATS = 25  # + de chunks récupérés = réponses plus riches/complètes (au prix d'un peu de vitesse)
NB_ECHANGES_CONTEXTE = 4  # nb de questions/réponses précédentes gardées comme contexte de conversation

client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])
supabase = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])


@st.cache_resource(show_spinner="Chargement du modèle de recherche (une seule fois)...")
def charger_modele():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texte(texte: str, prefixe: str):
    modele = charger_modele()
    vecteur = modele.encode([f"{prefixe}: {texte}"], normalize_embeddings=True, show_progress_bar=False)
    return vecteur[0].tolist()


def _erreur_temporaire(e) -> bool:
    """Erreurs côté Google qui se résolvent d'elles-mêmes (surcharge, quota par minute...)."""
    msg = str(e)
    return any(m in msg for m in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500", "504", "INTERNAL", "DEADLINE"))


def appeler_gemini(prompt: str, max_output_tokens=None, modeles=None, essais_par_modele=2, attente=5):
    """Appelle Gemini. En cas de surcharge temporaire, réessaie après une pause, puis bascule
    sur le modèle suivant de la liste. Retourne la réponse, ou lève la dernière erreur si tout échoue."""
    config = {"max_output_tokens": max_output_tokens} if max_output_tokens else None
    derniere_erreur = None
    for modele in (modeles or MODELES_GENERATION):
        for essai in range(essais_par_modele):
            try:
                return client.models.generate_content(model=modele, contents=prompt, config=config)
            except Exception as e:
                derniere_erreur = e
                if _erreur_temporaire(e) and essai < essais_par_modele - 1:
                    time.sleep(attente * (essai + 1))
                    continue
                break  # erreur non temporaire (ex: nom de modèle inconnu) ou essais épuisés -> modèle suivant
    raise derniere_erreur


def reformuler_question(question: str, historique: list) -> str:
    """Transforme une question de suivi ("et pour le GNA ?") en question autonome
    et complète ("Quelle est l'exigence de teneur en eau pour une GNA ?"), en
    s'appuyant sur les derniers échanges — uniquement pour améliorer la RECHERCHE,
    la question affichée à l'agent reste inchangée."""
    if not historique:
        return question

    derniers = historique[-NB_ECHANGES_CONTEXTE:]
    fil = "\n".join(f"{'Agent' if h['role'] == 'user' else 'Assistant'} : {h['content']}" for h in derniers)
    prompt = f"""Voici le début d'une conversation technique entre un agent LPEE et un assistant documentaire :

{fil}

Nouvelle question de l'agent : "{question}"

Reformule cette nouvelle question en une question AUTONOME et COMPLÈTE, compréhensible sans le
reste de la conversation (remplace "il", "ça", "et pour X" etc. par ce à quoi ça fait référence).
Ne réponds pas à la question, donne UNIQUEMENT la question reformulée, sans guillemets ni commentaire."""
    try:
        resp = appeler_gemini(prompt, modeles=MODELES_GENERATION[:2], essais_par_modele=1)
        reformulee = (resp.text or "").strip()
        return reformulee if reformulee else question
    except Exception:
        return question  # en cas d'erreur, on retombe sur la question telle quelle


def rechercher_chunks(question: str, k=NB_RESULTATS):
    vecteur = embed_texte(question, prefixe="query")
    res = supabase.rpc("match_documents", {
        "query_embedding": vecteur,
        "match_count": k,
    }).execute()
    return res.data or []


def generer_reponse(question: str, chunks: list, historique: list):
    contexte_docs = "\n\n---\n\n".join(
        f"[Source {i+1} — {c['metadata'].get('fichier')}, page {c['metadata'].get('page')}]\n{c['content']}"
        for i, c in enumerate(chunks)
    )

    derniers = historique[-NB_ECHANGES_CONTEXTE:]
    fil_conversation = "\n".join(f"{'Agent' if h['role'] == 'user' else 'Assistant'} : {h['content']}" for h in derniers)
    bloc_historique = f"\nDébut de la conversation (pour le contexte) :\n{fil_conversation}\n" if derniers else ""

    prompt = f"""Tu es un assistant technique pour les agents du LPEE (laboratoire d'essais de matériaux et travaux publics).
Réponds à la question de façon COMPLÈTE et DÉTAILLÉE, en t'appuyant sur TOUS les extraits pertinents fournis
ci-dessous (ne te limite pas au premier extrait venu : croise et synthétise l'information de plusieurs sources
quand elles se complètent). Structure ta réponse avec des sections/puces si le sujet s'y prête.
Réponds UNIQUEMENT à partir de ces extraits. Si un point précis n'est pas couvert par les extraits, dis-le
clairement plutôt que d'inventer, plutôt que de raccourcir artificiellement la réponse.
Cite systématiquement le document et la page source de chaque affirmation, au format (Source X).
{bloc_historique}
Extraits disponibles pour cette nouvelle question :
{contexte_docs}

Question de l'agent : {question}

Réponse (complète, détaillée, structurée, avec citations des sources) :"""

    try:
        resp = appeler_gemini(prompt, max_output_tokens=4096)
    except Exception as e:
        return (
            "⚠️ Le service Gemini est momentanément saturé (problème côté Google, pas côté application). "
            "Les documents trouvés pour votre question sont listés ci-dessous : vous pouvez les consulter "
            "ou les télécharger dès maintenant, puis reposer la question dans une minute.\n\n"
            f"Détail technique : `{e}`"
        )

    if resp.text:
        return resp.text
    # Réponse vide (souvent un blocage par les filtres de sécurité Gemini)
    motif = getattr(resp.candidates[0], "finish_reason", None) if resp.candidates else None
    return f"⚠️ Le modèle n'a renvoyé aucun texte (motif : {motif}). Reformulez la question et réessayez."


# --- Téléchargement des PDF originaux (dépôt Hugging Face privé, gratuit) ---
HF_REPO_ID = st.secrets.get("HF_REPO_ID", "")
HF_TOKEN = st.secrets.get("HF_TOKEN", "")


@st.cache_data(show_spinner="Récupération du PDF...", max_entries=5)
def telecharger_pdf(chemin_relatif: str) -> bytes:
    """Télécharge un PDF depuis le dépôt privé Hugging Face (à la demande seulement)."""
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
        for c in chunks:
            meta = c["metadata"]
            st.markdown(f"**{meta.get('fichier')}** — page {meta.get('page')} (pertinence : {c['similarity']:.0%})")
            st.caption(c["content"])
            st.divider()

    if not (HF_REPO_ID and HF_TOKEN):
        return  # téléchargement non configuré : on n'affiche que les extraits

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
if "sessions" not in st.session_state:
    st.session_state.sessions = {"Conversation 1": []}
if "session_courante" not in st.session_state:
    st.session_state.session_courante = "Conversation 1"

# Raccourci vers l'historique actif
historique = st.session_state.sessions[st.session_state.session_courante]

# --- Barre latérale (Sidebar) : Logo, Déconnexion & Historique des recherches ---
with st.sidebar:
    # Affichage du logo LPEE s'il est présent au même niveau que app.py
    if os.path.exists("logo lpee.jpg"):
        st.image("logo lpee.jpg", use_container_width=True)
    elif os.path.exists("logo_lpee.jpg"):
        st.image("logo_lpee.jpg", use_container_width=True)
    
    st.write(f"Connecté : **{st.session_state.nom_utilisateur}**")
    if st.button("🚪 Déconnexion", use_container_width=True):
        st.session_state.authentifie = False
        st.rerun()

    st.markdown("---")

    st.header("🗂️ Historique & Sessions")
    
    if st.button("➕ Nouvelle conversation", use_container_width=True):
        nb = len(st.session_state.sessions) + 1
        nouvelle_cle = f"Conversation {nb}"
        st.session_state.sessions[nouvelle_cle] = []
        st.session_state.session_courante = nouvelle_cle
        st.rerun()

    st.markdown("---")
    st.subheader("Mes conversations")
    
    # Sélecteur de session active
    session_choisie = st.radio(
        "Sélectionner une session",
        list(st.session_state.sessions.keys()),
        index=list(st.session_state.sessions.keys()).index(st.session_state.session_courante),
        label_visibility="collapsed"
    )
    if session_choisie != st.session_state.session_courante:
        st.session_state.session_courante = session_choisie
        st.rerun()

    st.markdown("---")
    st.markdown("### 🔍 Recherches récentes (Session active)")
    # Extraction des questions posées par l'utilisateur dans la session en cours
    questions_passees = [item["content"] for item in historique if item["role"] == "user"]
    if questions_passees:
        for q in reversed(questions_passees[-10:]):
            st.caption(f"• {q}")
    else:
        st.caption("Aucune question pour l'instant.")

# --- Interface Principale ---
st.title("📚 Recherche des normes et fascicules techniques — LPEE")
st.caption(f"Session active : **{st.session_state.session_courante}** — Posez une question, puis enchaînez des questions de suivi si besoin.")

col_titre, col_bouton = st.columns([5, 1])
with col_bouton:
    if st.button("🗑️ Vider", use_container_width=True):
        st.session_state.sessions[st.session_state.session_courante] = []
        st.rerun()

# Affiche tout l'historique de la conversation active
for idx, echange in enumerate(historique):
    with st.chat_message("user" if echange["role"] == "user" else "assistant"):
        st.markdown(echange["content"])
        if echange["role"] == "assistant" and echange.get("sources"):
            afficher_sources(echange["sources"], prefixe_cle=f"{st.session_state.session_courante}_msg{idx}")

# Zone de saisie de la nouvelle question, en bas de page (comportement chat classique)
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
            else:
                with st.spinner("Génération de la réponse..."):
                    reponse = generer_reponse(question, chunks, historique[:-1])

                st.markdown(reponse)
                afficher_sources(chunks, prefixe_cle=f"{st.session_state.session_courante}_msg{len(historique)}")

                historique.append({"role": "assistant", "content": reponse, "sources": chunks})
