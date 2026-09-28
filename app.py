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
GENERATION_MODEL = "gemini-flash-latest"  # vérifiez le nom exact disponible dans votre AI Studio
NB_RESULTATS = 25  # + de chunks récupérés = réponses plus riches/complètes (au prix d'un peu de vitesse)

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


def generer_reponse(question: str, chunks: list):
    contexte = "\n\n---\n\n".join(
        f"[Source {i+1} — {c['metadata'].get('fichier')}, page {c['metadata'].get('page')}]\n{c['content']}"
        for i, c in enumerate(chunks)
    )
    prompt = f"""Tu es un assistant technique pour les agents du LPEE (laboratoire d'essais de matériaux et travaux publics).
Réponds à la question de façon COMPLÈTE et DÉTAILLÉE, en t'appuyant sur TOUS les extraits pertinents fournis
ci-dessous (ne te limite pas au premier extrait venu : croise et synthétise l'information de plusieurs sources
quand elles se complètent). Structure ta réponse avec des sections/puces si le sujet s'y prête.
Réponds UNIQUEMENT à partir de ces extraits. Si un point précis n'est pas couvert par les extraits, dis-le
clairement plutôt que d'inventer, plutôt que de raccourcir artificiellement la réponse.
Cite systématiquement le document et la page source de chaque affirmation, au format (Source X).

Extraits disponibles :
{contexte}

Question de l'agent : {question}

Réponse (complète, détaillée, structurée, avec citations des sources) :"""

    resp = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
        config={"max_output_tokens": 4096},
    )
    return resp.text


# --- Interface ---
st.title("📚 Recherche des normes et fascicules techniques — LPEE")
st.caption("Posez une question en langage naturel ; la réponse s'appuie sur les documents indexés et cite ses sources.")

question = st.text_input("Votre question", placeholder="Ex: Quelle est la teneur en eau OPN exigée pour une GNF 0/40 ?")

col_a, col_b = st.columns([1, 4])
with col_a:
    lancer = st.button("🔍 Rechercher", type="primary", use_container_width=True)

if lancer and question.strip():
    with st.spinner("Recherche dans les documents indexés..."):
        chunks = rechercher_chunks(question)

    if not chunks:
        st.warning("Aucun document pertinent trouvé. Vérifiez que l'indexation (`ingest.py`) a bien été exécutée.")
    else:
        with st.spinner("Génération de la réponse..."):
            reponse = generer_reponse(question, chunks)

        st.markdown("### Réponse")
        st.markdown(reponse)

        st.markdown("### 📎 Sources utilisées")
        for c in chunks:
            meta = c["metadata"]
            with st.expander(f"{meta.get('fichier')} — page {meta.get('page')} (pertinence : {c['similarity']:.0%})"):
                st.write(c["content"])
elif lancer:
    st.info("Veuillez saisir une question.")
