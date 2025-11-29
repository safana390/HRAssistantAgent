# answer_generator_fix.py
from dotenv import load_dotenv
import os, json
from google import genai
import chromadb
from chromadb.config import Settings

load_dotenv()

def extract_vector_from_resp(resp):
    try:
        if hasattr(resp, "embedding") and hasattr(resp.embedding, "values"):
            return list(resp.embedding.values)
        if hasattr(resp, "embeddings") and resp.embeddings:
            first = resp.embeddings[0]
            if hasattr(first, "values"):
                return list(first.values)
            return list(first)
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

def try_generate_with_more_variations(gclient, model_name, prompt):
    attempts = []
    # 1) try models.generate_content with 'text' / 'content' variations
    for kwargs in [
        {"model": model_name, "text": prompt},
        {"model": model_name, "content": prompt},
        {"model": model_name, "input_text": prompt},
        {"model": model_name, "instance": {"input": prompt}},
    ]:
        try:
            resp = gclient.models.generate_content(**kwargs)
            return resp
        except Exception as e:
            attempts.append((f"models.generate_content{list(kwargs.keys())}", str(e)))

    # 2) try models.generate (older/newer libs)
    for kwargs in [
        {"model": model_name, "content": prompt},
        {"model": model_name, "messages": [{"role":"user","content":prompt}]},
        {"model": model_name, "input": prompt},
    ]:
        try:
            if hasattr(gclient.models, "generate"):
                resp = gclient.models.generate(**kwargs)
                return resp
        except Exception as e:
            attempts.append((f"models.generate{list(kwargs.keys())}", str(e)))

    # 3) try top-level generate / predict
    for fn_call in [
        lambda: getattr(gclient, "generate")(model=model_name, input=prompt) if hasattr(gclient, "generate") else (_ for _ in ()).throw(Exception("no generate")),
        lambda: getattr(gclient, "predict")(model=model_name, input=prompt) if hasattr(gclient, "predict") else (_ for _ in ()).throw(Exception("no predict")),
        lambda: gclient.responses.create(model=model_name, input=prompt) if hasattr(gclient, "responses") else (_ for _ in ()).throw(Exception("no responses")),
    ]:
        try:
            resp = fn_call()
            return resp
        except Exception as e:
            attempts.append(("top-level-call", str(e)))

    # 4) try models.generate_content with 'messages' shaped differently
    try:
        resp = gclient.models.generate_content(model=model_name, content=[{"type":"message","text":prompt}])
        return resp
    except Exception as e:
        attempts.append(("models.generate_content(content=[message])", str(e)))

    return {"_error_attempts": attempts}

def print_generation_text(resp):
    try:
        if isinstance(resp, dict):
            if "_error_attempts" in resp:
                print("Generation failed for multiple call styles. Debugs:")
                print(json.dumps(resp["_error_attempts"], indent=2))
                return
            # heuristic: find text fields
            json_str = json.dumps(resp, default=str)
            print("Raw dict response (truncated):")
            print(json_str[:4000])
            return
        if hasattr(resp, "text"):
            print(resp.text)
            return
        if hasattr(resp, "output_text"):
            print(resp.output_text)
            return
        # google genai objects: try common attributes
        if hasattr(resp, "candidates"):
            # some older clients return candidates with .content or .text
            try:
                cand = resp.candidates[0]
                if hasattr(cand, "content"):
                    print(cand.content[0].get("text") if isinstance(cand.content, (list,tuple)) else cand.content)
                    return
            except Exception:
                pass
        # last resort
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

    # embed query
    try:
        emb_resp = gclient.models.embed_content(model="models/text-embedding-004", contents=query)
    except Exception as e:
        print("Embedding call failed:", e)
        return

    qvec = extract_vector_from_resp(emb_resp)
    if not qvec:
        print("Could not extract embedding vector from response; raw embedding response:")
        print(emb_resp)
        return

    coll = client.get_collection(name="hr_policies")
    result = coll.query(query_embeddings=[qvec], n_results=4, include=["documents","metadatas","distances"])
    docs = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]

    if not docs:
        print("No docs retrieved for query; ensure collection has data.")
        return

    context = build_context_string(docs, metadatas)
    prompt = (
        "You are an HR Policy Assistant. Use ONLY the following context excerpts to answer the user's question. "
        "Cite sources inline like [Leave_Policy.pdf - chunk 0].\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer concisely with citations."
    )

    model_name = "models/gemini-2.5-flash"  # adjust if needed
    print("\n=== Sending prompt to Gemini (attempting generation) ===\n")
    resp = try_generate_with_more_variations(gclient, model_name, prompt)
    print_generation_text(resp)
    print("\n=== Retrieved context (for reference) ===\n")
    print(context[:4000] + ("\n...[truncated]" if len(context) > 4000 else ""))

if __name__ == "__main__":
    main()
