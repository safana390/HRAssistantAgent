# app/main.py
import os
import streamlit as st
from dotenv import load_dotenv
from google import genai
from chromadb import Client
from chromadb.config import Settings
from utils.prompts import QA_PROMPT

load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
if not GEMINI_KEY:
    st.error("GEMINI_API_KEY not found in .env")
    st.stop()

# clients
gclient = genai.Client(api_key=GEMINI_KEY)
chromaclient = Client(Settings(persist_directory=CHROMA_DIR, chroma_db_impl="duckdb+parquet"))
collection_name = "hr_policies"
collection = chromaclient.get_or_create_collection(name=collection_name)

st.set_page_config(page_title="HR Policy Assistant", layout="centered")
st.title("HR Policy Assistant — Document Q&A (Gemini + Chroma)")
st.write("Upload HR documents, index them, and ask policy questions. Answers include source citations.")

# Upload area
uploaded = st.file_uploader("Upload documents (PDF, DOCX, TXT)", accept_multiple_files=True)
if uploaded:
    os.makedirs("uploaded_docs", exist_ok=True)
    for f in uploaded:
        path = os.path.join("uploaded_docs", f.name)
        with open(path, "wb") as out:
            out.write(f.getbuffer())
    st.success(f"Saved {len(uploaded)} files to uploaded_docs/. Now click 'Index uploaded_docs' to embed & index them.")

if st.button("Index uploaded_docs"):
    st.info("Indexing started... this may take a minute.")
    import subprocess, sys
    subprocess.run([sys.executable, "ingest/ingest.py", "--folder", "uploaded_docs", "--collection", collection_name])
    st.success("Indexing finished. You can now ask questions.")

# Query UI
question = st.text_input("Ask a question about policies", "")
top_k = st.slider("How many source chunks to retrieve", 1, 6, 3)

if st.button("Get Answer") and question.strip():
    with st.spinner("Retrieving relevant policy snippets..."):
        # embed query
        emb_resp = gclient.models.embed_content(
            model="models/text-embedding-004",
            contents=question
        )
        q_vec = emb_resp.embedding.values

        # query chroma
        results = collection.query(query_embeddings=[q_vec], n_results=top_k, include=["documents","metadatas","distances"])
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]

        # prepare context
        context_blocks = []
        for i, d in enumerate(docs):
            meta = metas[i] if i < len(metas) else {}
            fname = meta.get("source", "unknown")
            chunk_idx = meta.get("chunk_index", i)
            context_blocks.append(f"[source: {fname} | chunk: {chunk_idx}]\n{d}")

        context = "\n\n".join(context_blocks)

        # build prompt
        prompt = QA_PROMPT.format(question=question, context=context)

        # call Gemini
        gen = gclient.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=prompt
        )
        # print answer
        try:
            answer = gen.text
        except:
            answer = str(gen)

    st.markdown("### Answer")
    st.write(answer)

    st.markdown("### Sources (top results)")
    for i, d in enumerate(docs):
        meta = metas[i] if i < len(metas) else {}
        st.write(f"- **{meta.get('source','unknown')}** — chunk {meta.get('chunk_index', i)}")
        snippet = d if len(d) < 600 else d[:600] + "..."
        st.code(snippet, language="text")
