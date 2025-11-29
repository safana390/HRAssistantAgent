#!/usr/bin/env python3
"""
answer_generator_hybrid.py

Provides:
 - answer(question: str, use_hybrid: bool=True) -> str
 - main() to run as a script (reads question from stdin)

Designed to work with:
 - google-genai client (genai.Client)
 - chromadb.PersistentClient (new Chroma API)
 - .env containing GEMINI_API_KEY and CHROMA_DIR
"""

import os
import traceback
from typing import Optional, List

from dotenv import load_dotenv

# Load env
load_dotenv()
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "hr_policies")

# Default models
EMBED_MODEL = "models/text-embedding-004"         # 768-dim embeddings
GEN_MODEL = "models/gemini-2.5-flash"             # text generation

# Lazy imports so module import doesn't crash Streamlit if env missing
def _create_clients():
    """Return (gclient, chroma_client, collection) or raise Exception."""
    try:
        from google import genai
        import chromadb
        from chromadb.config import Settings
    except Exception as e:
        raise RuntimeError(f"Missing package import: {e}")

    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY missing in .env")

    gclient = genai.Client(api_key=GEMINI_API_KEY)

    # Use PersistentClient (the project uses this earlier)
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings())
    # Get collection (raise if not found)
    try:
        coll = chroma_client.get_collection(name=COLLECTION_NAME)
    except Exception:
        # Try create/get_or_create in case API differs
        try:
            coll = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
        except Exception as e:
            raise RuntimeError(f"Could not open or create collection '{COLLECTION_NAME}': {e}")
    return gclient, chroma_client, coll


def _embed_text(gclient, text: str, model: str = EMBED_MODEL) -> List[float]:
    """Return embedding vector for `text`. Tries multiple response shapes for robustness."""
    resp = gclient.models.embed_content(model=model, contents=text)
    # try multiple extraction patterns (depends on SDK version)
    # 1) resp.embedding.values
    if hasattr(resp, "embedding") and getattr(resp.embedding, "values", None) is not None:
        return list(resp.embedding.values)
    # 2) resp.embeddings -> list
    if hasattr(resp, "embeddings"):
        emb_list = getattr(resp, "embeddings")
        if isinstance(emb_list, (list, tuple)) and len(emb_list) > 0:
            first = emb_list[0]
            if hasattr(first, "values"):
                return list(first.values)
            # maybe simple list
            return list(first)
    # 3) resp.raw or dict-like
    try:
        d = dict(resp.__dict__)
    except Exception:
        d = {}
    # try to find a numeric list anywhere
    for v in d.values():
        if isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], (float, int)):
            return list(v)
    raise RuntimeError("Could not extract embedding vector from embed_content response.")


def _generate_from_gemini(gclient, prompt: str, model: str = GEN_MODEL) -> str:
    """Call Gemini generative API and return text (best candidate)."""
    # generate_content signature expects model and contents=...
    resp = gclient.models.generate_content(model=model, contents=prompt)
    # response contains candidates (SDK object). Try to extract candidate text robustly.
    try:
        # typical path: resp.candidates[0].content.parts[0].text
        cand = resp.candidates[0]
        if hasattr(cand, "content") and getattr(cand.content, "parts", None):
            part = cand.content.parts[0]
            if hasattr(part, "text"):
                return part.text
        # try simpler
        if hasattr(cand, "text"):
            return cand.text
    except Exception:
        pass

    # fallback: convert to str
    return str(resp)


def _query_chroma(collection, query_embedding: List[float], n_results: int = 5):
    """
    Query Chroma using precomputed embedding to avoid dimension mismatches.
    Return results dict from collection.query(...) or None on error.
    We request documents, metadatas and distances.
    """
    try:
        res = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        return res
    except Exception as e:
        # Surface the error to caller
        raise RuntimeError(f"Chroma query failed: {e}")


def answer(question: str, use_hybrid: bool = True) -> str:
    """
    Primary callable. Returns a human-readable string answer.
    - If use_hybrid=True: try Chroma (PDF docs) first, if no good hits, fallback to Gemini.
    - If use_hybrid=False and mode == "PDF-only": only query Chroma and summarize results.
    - If Gemini-only: call Gemini only.
    """
    if not question or not question.strip():
        return "Please provide a non-empty question."

    # Try to create clients
    try:
        gclient, chroma_client, collection = _create_clients()
    except Exception as e:
        # If clients can't be created, fall back to Gemini-only generation if possible
        try:
            if GEMINI_API_KEY:
                from google import genai  # ensure import
                gclient = genai.Client(api_key=GEMINI_API_KEY)
                out = _generate_from_gemini(gclient, f"You are an HR assistant. Answer concisely: {question}")
                return f"(Gemini fallback — clients init failed)\n\n{out}"
        except Exception:
            return f"Error creating clients: {e}\n\nTrace:\n{traceback.format_exc()}"

    # If user requested Gemini only
    if not use_hybrid:
        try:
            gen = _generate_from_gemini(gclient, f"You are an HR assistant. Answer concisely: {question}")
            return gen
        except Exception as e:
            return f"Gemini generation failed: {e}\n\n{traceback.format_exc()}"

    # HYBRID: first try to retrieve from documents
    try:
        # compute query embedding using same model used to index the collection
        qvec = _embed_text(gclient, question, model=EMBED_MODEL)
    except Exception as e:
        return f"Embedding failed: {e}\n\nTrace:\n{traceback.format_exc()}"

    # Query Chroma using embedding
    try:
        query_res = _query_chroma(collection, qvec, n_results=5)
        # query_res format depends on chromadb client implementation:
        # It often has keys: 'documents', 'metadatas', 'distances' each being lists-of-lists.
        docs = []
        try:
            # If query_res is dict-like
            if isinstance(query_res, dict):
                docs_lists = query_res.get("documents", [])
                if docs_lists and len(docs_lists) > 0:
                    # flatten first result list
                    docs = docs_lists[0]
                metas_lists = query_res.get("metadatas", [])
            else:
                # some chroma returns an object with .get or attributes
                # try attribute access
                if hasattr(query_res, "documents"):
                    docs_attr = getattr(query_res, "documents")
                    if isinstance(docs_attr, (list, tuple)) and len(docs_attr) > 0:
                        docs = docs_attr[0]
        except Exception:
            docs = []

        # If we found useful text chunks, synthesize an answer using those
        if docs and any(docs):
            # Combine top few retrieved chunks into context
            top_docs = [d for d in docs if d and isinstance(d, str)][:4]
            context = "\n\n---\n\n".join(top_docs)
            prompt = (
                "You are an HR assistant. Use the provided company document excerpts below to answer the user's question. "
                "If the documents do not contain the answer, say you don't know and offer a general guideline.\n\n"
                f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer concisely and cite where relevant."
            )
            try:
                gen = _generate_from_gemini(gclient, prompt)
                return f"{gen}\n\n(Used PDF context — {len(top_docs)} chunks.)"
            except Exception as ge:
                # If generation fails, at least return the retrieved chunks
                retrieved_text = "\n\n-----\n\n".join(top_docs)
                return f"(Gemini generation failed: {ge})\n\nTop retrieved chunks:\n\n{retrieved_text}"
        else:
            # No retrieval results — fallback to Gemini generic generation
            prompt = f"You are an HR assistant. Answer concisely: {question}"
            gen = _generate_from_gemini(gclient, prompt)
            return f"(No document hits — Gemini fallback)\n\n{gen}"

    except Exception as e:
        # Something failed with Chroma query — fallback to Gemini
        try:
            gen = _generate_from_gemini(gclient, f"You are an HR assistant. Answer concisely: {question}")
            return f"(Chroma failed: {e})\n\nGemini answer:\n\n{gen}"
        except Exception as e2:
            return f"Both Chroma and Gemini failed:\nChroma error: {e}\nGemini error: {e2}\n\nTrace:\n{traceback.format_exc()}"


def main():
    """Allow running the script from command line (reads question via input())."""
    try:
        q = input("Enter question: ").strip()
    except Exception:
        q = ""
    out = answer(q, use_hybrid=True)
    print(out)


if __name__ == "__main__":
    main()
