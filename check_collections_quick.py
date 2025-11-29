# check_collections_quick.py
from chromadb import PersistentClient
from chromadb.config import Settings
import os

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
client = PersistentClient(path=CHROMA_DIR, settings=Settings())

cols = client.list_collections()
print("Collections found:", [c.name for c in cols])

coll = client.get_collection("hr_policies_v2")
print("Collection 'hr_policies_v2' .count() ->", coll.count())
