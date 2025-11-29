
from google import genai
from dotenv import load_dotenv
import os
load_dotenv()
gclient = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

models = [
    "models/gemini-embedding-001",
    "models/text-embedding-004",
    "models/gemini-embedding-exp-03-07",
    "models/gemini-embedding-001"  # duplicate safe entry — add or remove as you like
]

for m in models:
    try:
        resp = gclient.models.embed_content(model=m, contents="hello world")
    except Exception as e:
        print(f"{m} -> ERROR calling embed_content: {e!r}")
        continue

    # Try a few ways to extract vector values (different SDK versions return slightly different shapes)
    vec = None
    # common attr in your earlier runs: resp.embedding.values
    if hasattr(resp, "embedding"):
        ent = resp.embedding
        if hasattr(ent, "values"):
            vec = ent.values
        elif isinstance(ent, (list, tuple)) and len(ent) and hasattr(ent[0], "values"):
            vec = ent[0].values
    # some versions may return resp.embeddings or resp.embeddings[0].values
    if vec is None and hasattr(resp, "embeddings"):
        ent = resp.embeddings
        if isinstance(ent, (list,tuple)) and len(ent):
            first = ent[0]
            if hasattr(first, "values"):
                vec = first.values
            elif isinstance(first, (list,tuple)):
                vec = first
    # fallback: resp['embedding'] or resp.get(...)
    try:
        if vec is None:
            # dict-like fallback
            d = dict(resp.__dict__) if hasattr(resp, "__dict__") else {}
            if "embedding" in d:
                maybe = d["embedding"]
                if hasattr(maybe, "values"):
                    vec = maybe.values
    except Exception:
        pass

    if vec is None:
        # Final attempt: stringify response small sample
        print(f"{m} -> Could not extract vector; resp repr (truncated): {str(resp)[:300]}")
        continue

    print(f"{m} -> embedding length: {len(vec)}")

