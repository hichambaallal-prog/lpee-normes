"""
Plateforme de recherche des normes et fascicules techniques — LPEE
=====================================================================
À déployer sur Streamlit Community Cloud (gratuit), connecté à Supabase.

Secrets requis (Streamlit Cloud > Settings > Secrets, format .toml) :
    GOOGLE_API_KEY = "..."
    SUPABASE_URL = "..."
    SUPABASE_KEY = "..."
"""

import streamlit as st
from supabase import create_client
from google import genai
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Recherche Normes LPEE", page_icon="📚", layout="wide")

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"  # DOIT être le même modèle que dans ingest.py
GENERATION_MODEL = "gemini-2.5-flash"  # Modèle standard performant pour la génération
NB_RESULTATS = 25  # + de chunks récupérés = réponses plus riches/complètes

client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])
supabase = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])


@st.cache_resource(show_spinner="Chargement du modèle de recherche (une seule fois)...")
def charger_modele():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_question(question: str):
    modele = charger_modele()
    # Préfixe "query: " requis par le modèle e5 côté question (symétrique du "passage: " de ingest.py)
    vecteur = modele.encode([f"query: {question}"], normalize_embeddings=True, show_progress_bar=False)
    return vecteur[0].tolist()


def rechercher_chunks(question: str, k=NB_RESULTATS):
    vecteur = embed_question(question)
    res = supabase.rpc("match_documents", {
        "query_embedding": vecteur,
        "match_count": k,
    }).execute()
    return res.data or []


def generer_reponse(question: str, chunks: list, historique: list):
    contexte = "\n\n---\n\n".join(
        f"[Source {i+1} — {c['metadata'].get('fichier')}, page {c['metadata'].get('page')}]\n{c['content']}"
        for i, c in enumerate(chunks)
    )
    
    # Formatage de l'historique récent de la discussion (excluant la question actuelle)
    historique_str = ""
    if len(historique) > 1:
        historique_str = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in historique[:-1]])

    prompt = f"""Tu es un assistant technique pour les agents du LPEE (laboratoire d'essais de matériaux et travaux publics).
Réponds à la question de façon COMPLÈTE et DÉTAILLÉE, en t'appuyant sur TOUS les extraits pertinents fournis
ci-dessous et sur l'historique de la conversation si nécessaire. Ne te limite pas au premier extrait venu : croise et synthétise l'information de plusieurs sources quand elles se complètent. 
Structure ta réponse avec des sections/puces si le sujet s'y prête.
Réponds UNIQUEMENT à partir de ces extraits et du contexte. Si un point précis n'est pas couvert par les extraits, dis-le clairement plutôt que d'inventer ou de raccourcir artificiellement la réponse.
Cite systématiquement le document et la page source de chaque affirmation, au format (Source X).

Historique récent de la conversation :
{historique_str}

Extraits documentaires disponibles :
{contexte}

Question actuelle de l'agent : {question}

Réponse (complète, détaillée, structurée, avec citations des sources) :"""

    resp = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
        config={"max_output_tokens": 4096},
    )
    return resp.text


# --- Interface Chat ---
st.title("📚 Recherche des normes et fascicules techniques — LPEE")
st.caption("Posez vos questions en langage naturel. L'assistant mémorise l'historique pour vous permettre d'affiner vos recherches au fil de la discussion.")

# Initialisation de l'historique de discussion dans la session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage de l'historique des messages précédents
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📎 Sources utilisées"):
                for src in message["sources"]:
                    meta = src["metadata"]
                    st.write(f"- **{meta.get('fichier')}** — page {meta.get('page')} (pertinence : {src['similarity']:.0%})")

# Zone de saisie du chat en bas de page
if question := st.chat_input("Ex: Quelle est la teneur en eau OPN exigée pour une GNF 0/40 ?"):
    # Ajouter la question de l'utilisateur à l'historique et l'afficher
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Recherche des documents dans Supabase
    with st.spinner("Recherche dans les documents indexés..."):
        chunks = rechercher_chunks(question)

    if not chunks:
        reponse = "Aucun document pertinent trouvé. Vérifiez que l'indexation (`ingest.py`) a bien été exécutée."
        sources_list = []
    else:
        with st.spinner("Génération de la réponse détaillée..."):
            reponse = generer_reponse(question, chunks, st.session_state.messages)
        sources_list = chunks

    # Affichage de la réponse de l'assistant
    with st.chat_message("assistant"):
        st.markdown(reponse)
        if sources_list:
            with st.expander("📎 Sources utilisées"):
                for c in sources_list:
                    meta = c["metadata"]
                    st.write(f"- **{meta.get('fichier')}** — page {meta.get('page')} (pertinence : {c['similarity']:.0%})")

    # Enregistrer la réponse et les sources dans l'historique de session
    st.session_state.messages.append({
        "role": "assistant",
        "content": reponse,
        "sources": sources_list
    })
