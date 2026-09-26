import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Configuration de la page Streamlit
st.set_page_config(page_title="Plateforme Normes LPEE - Enrobés", page_icon="🏗️", layout="wide")

st.title("🏗️ Assistant Technique LPEE - Spécial Enrobés & Compactage")
st.markdown("Posez vos questions sur les guides techniques et normes LCPC/LPEE indexés.")

# Configuration sécurisée de la clé API Gemini via Streamlit Secrets
if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
else:
    st.error("⚠️ Veuillez configurer la clé GOOGLE_API_KEY dans les secrets de Streamlit Cloud.")

@st.cache_resource
def charger_moteur():
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    # Connexion à la base de données vectorielle locale
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="base_normes_lpee_cloud")
    return encoder, collection

encoder, collection = charger_moteur()

# Zone de discussion (Chat)
question = st.chat_input("Ex: Quelles sont les recommandations de compactage des enrobés ?")

if question:
    with st.chat_message("user"):
        st.write(question)
        
    with st.chat_message("assistant"):
        with st.spinner("Recherche dans les guides techniques LPEE..."):
            # Recherche des passages les plus pertinents dans la base chroma_db
            query_vector = encoder.encode(question).tolist()
            resultats = collection.query(
                query_embeddings=[query_vector],
                n_results=3
            )
            
            contextes = resultats['documents'][0]
            sources = set([meta['source'] for meta in resultats['metadatas'][0]])
            contexte_global = "\n---\n".join(contextes)
            
            # Génération de la réponse via Gemini
            modele_gemini = genai.GenerativeModel('gemini-2.5-flash')
            prompt = f"""
            Tu es un assistant technique expert en génie civil et normes du LPEE. 
            Réponds à la question suivante en te basant UNIQUEMENT sur les extraits de documents techniques fournis ci-dessous. 
            Cite précisément les sources (fichiers PDF) utilisées. Si la réponse n'est pas dans les documents, dis-le clairement.

            Extraits des documents :
            {contexte_global}

            Question : {question}
            """
            
            reponse = modele_gemini.generate_content(prompt)
            st.markdown(reponse.text)
            
            st.markdown("---")
            st.caption(f"**Documents LPEE sources :** {', '.join(list(sources))}")