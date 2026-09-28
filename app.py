"""
Plateforme de recherche des normes et fascicules techniques — LPEE
=====================================================================
Interface de type "chat" avec historique : les agents peuvent poser une
question, puis enchaîner des questions de suivi ("et pour le GNA ?") qui
tiennent compte de la conversation précédente.
"""

import streamlit as st
from supabase import create_client
from google import genai
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Recherche Normes LPEE", page_icon="📚", layout="wide")

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"  # DOIT être le même modèle que dans ingest.py
GENERATION_MODEL = "gemini-2.5-flash"  # Modèle stable et performant
NB_RESULTATS = 25  # + de chunks récupérés = réponses plus riches/complètes
NB_ECHANGES_CONTEXTE = 4  # nb de questions/réponses précédentes gardées comme contexte

client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])
supabase = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])


@st.cache_resource(show_spinner="Chargement du modèle de recherche (une seule fois)...")
def charger_modele():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texte(texte: str, prefixe: str):
    modele = charger_modele()
    vecteur = modele.encode([f"{prefixe}: {texte}"], normalize_embeddings=True, show_progress_bar=False)
    return vecteur[0].tolist()


def reformuler_question(question: str, historique: list) -> str:
    if not historique:
        return question

    derniers = historique[-NB_ECHANGES_CONTEXTE:]
    fil = "\n".join(f"{'Agent' if h['role'] == 'user' else 'Assistant'} : {h['content']}" for h in derniers)
    prompt = f"""Voici le début d'une conversation technique entre un agent LPEE et un assistant documentaire :

{fil}

Nouvelle question de l'agent : "{question}"

Reformule cette nouvelle question en une question AUTONOME et COMPLÈTE, compréhensible sans le
reste de la conversation. Ne réponds pas à la question, donne UNIQUEMENT la question reformulée, sans guillemets ni commentaire."""
    try:
        resp = client.models.generate_content(model=GENERATION_MODEL, contents=prompt)
        reformulee = (resp.text or "").strip()
        return reformulee if reformulee else question
    except Exception:
        return question


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
ci-dessous. Structure ta réponse avec des sections/puces si le sujet s'y prête.
Réponds UNIQUEMENT à partir de ces extraits. Si un point précis n'est pas couvert, dis-le clairement.
Cite systématiquement le document et la page source de chaque affirmation, au format (Source X).
{bloc_historique}
Extraits disponibles pour cette nouvelle question :
{contexte_docs}

Question de l'agent : {question}

Réponse (complète, détaillée, structurée, avec citations des sources) :"""

    resp = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
        config={"max_output_tokens": 4096},
    )
    return resp.text


# --- État de la conversation ---
if "historique" not in st.session_state:
    st.session_state.historique = []

# --- Interface ---
st.title("📚 Recherche des normes et fascicules techniques — LPEE")
st.caption("Discutez avec l'assistant : posez une question, puis enchaînez des questions de suivi si besoin.")

col_titre, col_bouton = st.columns([5, 1])
with col_bouton:
    if st.button("🗑️ Nouvelle conversation", use_container_width=True):
        st.session_state.historique = []
        st.rerun()

# Affiche tout l'historique
for echange in st.session_state.historique:
    with st.chat_message("user" if echange["role"] == "user" else "assistant"):
        st.markdown(echange["content"])
        if echange["role"] == "assistant" and echange.get("sources"):
            with st.expander("📎 Sources utilisées"):
                for c in echange["sources"]:
                    meta = c["metadata"]
                    st.markdown(f"**{meta.get('fichier')}** — page {meta.get('page')} (pertinence : {c['similarity']:.0%})")
                    st.caption(c["content"])
                    st.divider()

# Zone de saisie de la nouvelle question
question = st.chat_input("Posez votre question ou enchaînez sur la précédente...")

if question:
    st.session_state.historique.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Recherche dans les documents indexés..."):
            question_recherche = reformuler_question(question, st.session_state.historique[:-1])
            chunks = rechercher_chunks(question_recherche)

        if not chunks:
            reponse = "Aucun document pertinent trouvé pour cette question. Reformulez-la, ou vérifiez que l'indexation a bien été exécutée."
            st.markdown(reponse)
            st.session_state.historique.append({"role": "assistant", "content": reponse, "sources": []})
        else:
            with st.spinner("Génération de la réponse..."):
                reponse = generer_reponse(question, chunks, st.session_state.historique[:-1])

            st.markdown(reponse)
            with st.expander("📎 Sources utilisées"):
                for c in chunks:
                    meta = c["metadata"]
                    st.markdown(f"**{meta.get('fichier')}** — page {meta.get('page')} (pertinence : {c['similarity']:.0%})")
                    st.caption(c["content"])
                    st.divider()

            st.session_state.historique.append({"role": "assistant", "content": reponse, "sources": chunks})
