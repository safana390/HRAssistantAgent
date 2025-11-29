# --- Begin replacement code for retrieval + synthesis ---

from typing import List
import uuid
# ensure you already have: gclient (genai.Client) and client (chromadb.PersistentClient) initialized above

COLLECTION_NAME = "hr_policies"
EMBED_MODEL = "models/text-embedding-004"
GEN_MODEL = "models/gemini-2.5-flash"   # keep the model you know works

def embed_question(question: str) -> List[float]:
    """Embed the question, handling different response shapes."""
    resp = gclient.models.embed_content(model=EMBED_MODEL, contents=[question])
    # resp shape can vary by SDK version — handle a few possibilities
    # 1) resp.embedding.values
    # 2) resp.embeddings[0].values
    # 3) resp['embeddings'][0]['values'] (dict)
    try:
        # prefer resp.embedding.values
        vec = getattr(resp, "embedding", None)
        if vec is not None:
            return vec.values
    except Exception:
        pass

    try:
        embeds = getattr(resp, "embeddings", None)
        if embeds:
            return embeds[0].values
    except Exception:
        pass

    # fallback if it's a dict-like
    if isinstance(resp, dict) and "embeddings" in resp and resp["embeddings"]:
        e0 = resp["embeddings"][0]
        if isinstance(e0, dict) and "values" in e0:
            return e0["values"]

    # if we get here, raise helpful error
    raise RuntimeError("Could not parse embedding response. Inspect `resp` to see its shape.")

def query_collection_by_vector(vec: List[float], top_k: int = 5):
    """Query Chroma collection using a raw vector and request distances."""
    coll = client.get_collection(name=COLLECTION_NAME)

    # call Chroma's query: request documents, metadatas and distances
    # different versions of chromadb may give dict or object; handle both below
    res = coll.query(
        query_embeddings=[vec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    # normalize result into Python lists
    if isinstance(res, dict):
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
    else:
        # object-like return
        # some versions return attributes .documents, .metadatas, .distances as lists-of-lists
        docs = getattr(res, "documents", [[]])[0]
        metas = getattr(res, "metadatas", [[]])[0]
        dists = getattr(res, "distances", [[]])[0]

    return docs, metas, dists

def dedupe_chunks_preserve_order(chunks: List[str], metas: List[dict], dists: List[float]):
    """Remove duplicate chunk texts while preserving the first occurrence and keeping associated metadata & distance."""
    seen = set()
    out_chunks, out_metas, out_dists = [], [], []
    for c, m, d in zip(chunks, metas, dists):
        key = c.strip()[:500]  # use prefix as identity (trim long text)
        if key in seen:
            continue
        seen.add(key)
        out_chunks.append(c)
        out_metas.append(m)
        out_dists.append(d)
    return out_chunks, out_metas, out_dists

def build_answer_with_gemini(question: str, top_chunks: List[str], top_metas: List[dict], top_dists: List[float]):
    """Construct a clear prompt and call Gemini to synthesize."""
    # Create a compact context block from retrieved chunks
    context_parts = []
    for i, (c, m, d) in enumerate(zip(top_chunks, top_metas, top_dists), start=1):
        src = m.get("source", "unknown")
        context_parts.append(f"[{i}] source={src} distance={d:.4f}\n{c.strip()}")
    context = "\n\n".join(context_parts)

    prompt = f"""
You are an HR assistant. Use ONLY the information in the retrieved context to answer the user's question.
Question: {question}

Context (top retrieved chunks):
{context}

Write a short, clear, and accurate answer (2-6 sentences). If the answer is not present in the context, say "I don't know — not in the provided documents."
Do not hallucinate policies.
"""
    # Call Gemini (the same form that worked in your test_gemini.py)
    resp = gclient.models.generate_content(
        model=GEN_MODEL,
        # the SDK variants differ; this worked for you earlier when test_gemini printed "Gemini says: ...".
        input=prompt
    )

    # resp shape may be object-like or dict-like. Normalize:
    if hasattr(resp, "candidates"):
        # older SDK style
        text = resp.candidates[0].content if resp.candidates else ""
    elif isinstance(resp, dict) and "candidates" in resp and resp["candidates"]:
        text = resp["candidates"][0].get("content", "")
    else:
        # try other common properties
        text = getattr(resp, "output", "") or getattr(resp, "text", "") or str(resp)

    return text.strip()

# --- Main orchestration used in answer_generator_local.py ---
def answer_question_local(question: str, top_k: int = 5):
    # 1) embed
    vec = embed_question(question)

    # 2) vector search
    docs, metas, dists = query_collection_by_vector(vec, top_k=top_k)

    # 3) dedupe
    docs, metas, dists = dedupe_chunks_preserve_order(docs, metas, dists)

    # 4) debug print (so Streamlit/CLI shows distances)
    print("\nTop retrieved chunks (with distances):\n")
    for i, (m, d) in enumerate(zip(metas, dists), start=1):
        src = m.get("source", "unknown")
        print(f"[{i}] {src} — distance {d}")

    # 5) ask Gemini to synthesize a concise answer
    answer = build_answer_with_gemini(question, docs[:top_k], metas[:top_k], dists[:top_k])
    return answer

# --- End replacement code ---
