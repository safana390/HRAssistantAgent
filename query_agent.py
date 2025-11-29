# snippet to embed the user query and search the hr_policies_v2 collection
from google import genai
from chromadb import PersistentClient
from chromadb.config import Settings
import os
from dotenv import load_dotenv

load_dotenv()
gclient = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
client = PersistentClient(path=os.getenv("CHROMA_DIR", "./chroma_db"), settings=Settings())

def search_query(user_query, collection_name="hr_policies_v2", top_k=5):
    resp = gclient.models.embed_content(model="models/text-embedding-004", contents=user_query)
    # extract vector - same logic as ingest
    if hasattr(resp, "embedding") and getattr(resp.embedding, "values", None) is not None:
        qvec = resp.embedding.values
    elif isinstance(resp, dict) and "embeddings" in resp:
        qvec = resp["embeddings"][0]["values"]
    else:
        qvec = getattr(resp, "embeddings", [None])[0]
        if hasattr(qvec, "values"): qvec = qvec.values

    collection = client.get_collection(name=collection_name)
    # use vector query so dims align
    results = collection.query(query_embeddings=[qvec], n_results=top_k, include=["documents","metadatas","distances"])
    return results
