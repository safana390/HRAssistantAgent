# answer_generator.py
from dotenv import load_dotenv
import os, sys
from google import genai
import chromadb
from chromadb.config import Settings
import json

load_dotenv()

def extract_vector_from_resp(resp):
    # best-effort extraction for different client shapes
    try:
        # google-genai embed_content() often returns .embedding.values or .embeddings...
        if hasattr(resp, "embedding") and hasattr(resp.embedding, "values"):
            return list(resp.embedding.values)
        if hasattr(resp, "embeddings") and resp.embeddings:
            first = resp.embeddings[0]
            if hasattr(first, "values"):
                return list(first.values)
            return list(first)
        # dict fallback
        if isinstance(resp, dict):
            if "embedding" in resp:
                return list(resp["embedding"])
            if "embeddings" in resp and resp["embeddings"]:
                return list(resp["embeddings"][0])
    except Exception:
        pass
    return None

def build_context_string(docs, metadatas):
    parts = []
    for i, (d, m) in enumerate(zip(docs, metadatas)):
        src = m.get("source", "unknown")
        idx = m.get("chunk_index", i)
        header = f"[Source: {src} | chunk: {idx}]"
        parts.append(f"{header}\n{d}\n")
    return "\n---\n".join(parts)

def try_generate_with_variations(gclient, model_name, prompt):
    # Try several possible call styles so script works across genai versions
    attempts = []
    # 1) try models.generate_content(input=prompt)
    try:
        resp = gclient.models.generate_content(model=model_name, input=prompt)
        return resp
    except Exception as e:
        attempts.append(("models.generate_content(model, input=...)", str(e)))
    # 2) try models.generate_content(prompt=prompt)
    try:
        resp = gclient.models.generate_content(model=model_name, prompt=prompt)
        return resp
    except Exception as e:
        attempts.append(("models.generate_content(model, prompt=...)", str(e)))
    # 3) try models.generate_content(messages=[...])
    try:
        msgs = [{"role":"user","content": prompt}]
        resp = gclient.models.generate_content(model=model_name, messages=msgs)
        return resp
    except Exception as e:
        attempts.append(("models.generate_content(model, messages=[...])", str(e)))
    # 4) try responses.create (newer style)
    try:
        if hasattr(gclient, "responses"):
            resp = gclient.responses.create(model=model_name, input=prompt)
            return resp
    except Exception as e:
        attempts.append(("responses.create(model, input=...)", str(e)))
    # 5) last resort: return attempts for debugging
    return {"_error_attempts": attempts}

def print_generation_text(resp):
    # Try a few heuristics to extract text from response object
    try:
        # if dict-like
        if isinstance(resp, dict):
            if "output" in resp:
                outs = resp["output"]
                # collect text parts
                texts = []
                for o in outs:
                    if isinstance(o, dict) and "content" in o:
                        for c in o["content"]:
                            if isinstance(c, dict) and "text" in c:
                                texts.append(c["text"])
                if texts:
                    print("\n".join(texts))
                    return
            if "_error_attempts" in resp:
                print("Generation failed for multiple call styles. Debugs:")
                print(json.dumps(resp["_error_attempts"], indent=2))
                return
        # object with .text or .output_text
        if hasattr(resp, "text"):
            print(resp.text)
            return
        if hasattr(resp, "output_text"):
            print(resp.output_text)
            return
        # print full resp as last resort
        print("Raw response object:")
        print(resp)
    except Exception as e:
        print("Could not parse generation response:", e)
        print(resp)

def main():
    GEMINI_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_KEY:
        print("Set GEMINI_API_KEY in .env")
        return

    gclient = genai.Client(api_key=GEMINI_KEY)
    client = chromadb.PersistentClient(path=os.getenv("CHROMA_DIR","./chroma_db"), settings=Settings())

    query = input("Enter question: ").strip()
    if not query:
        return

    # 1) embed query
    try:
        emb_resp = gclient.models.embed_content(model="models/text-embedding-004", contents=query)
    except Exception as e:
        print("Embedding call failed:", e)
        return

    qvec = extract_vector_from_resp(emb_resp)
    if not qvec:
        print("Could not extract embedding vector from response; printing raw response for debug:")
        print(emb_resp)
        return

    # 2) retrieve top-k
    coll = client.get_collection(name="hr_policies")
    result = coll.query(query_embeddings=[qvec], n_results=4, include=["documents","metadatas","distances"])
    docs = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]

    if not docs:
        print("No docs retrieved for query; ensure collection has data.")
        return

    # 3) build prompt with context and instruction
    context = build_context_string(docs, metadatas)
    prompt = (
        "You are an HR Policy Assistant. Use ONLY the following context excerpts from official HR policy documents "
        "to answer the user's question. Quote sources in square brackets after each factual claim (e.g. [Leave_Policy.pdf - chunk 0]).\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\nAnswer concisely, include citations and, if the answer is not contained in the context, say 'I don't know; please contact HR.'\n"
    )

    # 4) call Gemini (try several patterns)
    model_name = "models/gemini-2.5-flash"  # adjust if you prefer gemini-flash-latest
    print("\n=== Sending prompt to Gemini (attempting generation) ===\n")
    resp = try_generate_with_variations(gclient, model_name, prompt)

    # 5) print generation or debug
    print_generation_text(resp)

    # if resp parsing failed, print the context and suggestions
    print("\n=== Retrieved context (for reference) ===\n")
    print(context[:4000] + ("\n...[truncated]" if len(context) > 4000 else ""))

if __name__ == "__main__":
    main()
