from chromadb import PersistentClient
from chromadb.config import Settings
import os, sys

try:
    client = PersistentClient(
        path=os.getenv("CHROMA_DIR", "./chroma_db"),
        settings=Settings()
    )
except Exception as e:
    print("Failed to open Chroma client:", e)
    sys.exit(1)

cols = client.list_collections()
names = []
for c in cols:
    try:
        name = getattr(c, "name", None)
        if not name:
            try:
                name = c['name']
            except Exception:
                name = str(c)
        names.append(name)
    except Exception:
        names.append(str(c))

print("Collections found:", names)

name = "hr_policies"
try:
    coll = client.get_collection(name=name)
except Exception as e:
    print(f"Could not get collection '{name}':", e)
    sys.exit(0)

# Try several safe ways to get a vector count
count = None
# 1) try count() if available
try:
    count = coll.count()
    print(f"Collection '{name}' .count() ->", count)
except Exception:
    pass

# 2) try get() with allowed includes and infer length
if count is None:
    try:
        # allowed include keys usually: documents, embeddings, metadatas, distances, uris, data
        data = coll.get(include=['documents', 'metadatas'])
        # data['documents'] is a list of lists (one per query), but for get it is list of documents
        # For Chroma Python client when storing one collection, data['documents'] is list of stored documents
        docs = data.get('documents', [])
        if isinstance(docs, list):
            # If nested (e.g. docs is [ [doc1, doc2, ...] ]) flatten accordingly
            if len(docs) and isinstance(docs[0], list):
                flat = docs[0]
            else:
                flat = docs
            count = len(flat)
            print(f"Collection '{name}' document count (from get include documents) ->", count)
    except Exception as e:
        print("Fallback get(include=['documents','metadatas']) failed:", e)

# 3) if still None, try to query small sample and measure ids
if count is None:
    try:
        res = coll.query(query_embeddings=[[0.0]*1], n_results=1)
        # if query works, inspect returned ids/documents
        ids = res.get('ids', [])
        # ids may be nested per query, so try to infer
        if isinstance(ids, list):
            if len(ids) and isinstance(ids[0], list):
                # hard to infer full count from query; we'll just report sample present
                print(f"Collection '{name}' appears to have data (sample query returned {len(ids[0])} id(s)).")
            else:
                print(f"Collection '{name}' query returned ids: {len(ids)}")
    except Exception as e:
        print("Sample query fallback failed (this is okay):", e)

if count is None:
    print(f"Could not determine exact vector count for collection '{name}', but it exists.")
