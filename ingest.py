# ingest/ingest.py
import os
import uuid
import argparse
from dotenv import load_dotenv

# third-party clients
from google import genai
import chromadb
from chromadb.config import Settings

# local utils
from utils.docs import load_document, clean_text, chunk_text

# load env
load_dotenv()

# constants / config
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
EMBED_MODEL = "models/text-embedding-004"   # 768-d
COLLECTION_DEFAULT = "hr_policies_text768"

# create chroma persistent client (new-style)
client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings())

# genai client (Gemini)
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment")
gclient = genai.Client(api_key=GEMINI_KEY)


def extract_vec_from_resp(resp):
    """
    Try multiple ways to pull vector list from SDK response object.
    """
    # SDK sometimes returns embeddings list, sometimes embedding; check both
    vec = None
    try:
        if hasattr(resp, "embeddings") and resp.embeddings:
            vec = resp.embeddings[0].values
        elif hasattr(resp, "embedding") and resp.embedding:
            vec = resp.embedding.values
        else:
            # fallback: convert to dict-like and search
            rdict = getattr(resp, "__dict__", None)
            if rdict:
                for v in rdict.values():
                    if isinstance(v, list) and len(v) > 0 and isinstance(v[0], (float, int)):
                        vec = v
                        break
    except Exception:
        vec = None
    return vec


def index_document(file_path: str, collection_name: str = COLLECTION_DEFAULT, chunk_size: int = 1200, chunk_overlap: int = 200):
    raw = load_document(file_path)
    if not raw or not raw.strip():
        print(f"Skipping empty file: {file_path}")
        return

    text = clean_text(raw)
    chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    docs, metadatas, ids, embeddings = [], [], [], []

    for i, (chunk, start, end) in enumerate(chunks):
        try:
            # create embedding for this chunk
            resp = gclient.models.embed_content(model=EMBED_MODEL, contents=[chunk])
        except Exception as e:
            print(f"Embedding API error for file {file_path} chunk {i}: {e}")
            continue

        vec = extract_vec_from_resp(resp)
        if not vec:
            print(f"Embedding extraction error for {file_path} chunk {i}: response did not contain vector")
            continue

        doc_id = str(uuid.uuid4())
        docs.append(chunk)
        metadatas.append({
            "source": os.path.basename(file_path),
            "chunk_index": i,
            "start": start,
            "end": end
        })
        ids.append(doc_id)
        embeddings.append(vec)

    if not docs:
        print(f"No chunks indexed for {file_path}")
        return

    # get or create collection
    coll = client.get_or_create_collection(name=collection_name)
    coll.add(documents=docs, metadatas=metadatas, ids=ids, embeddings=embeddings)
    # NOTE: PersistentClient persists automatically — no client.persist() call here
    print(f"Indexed {len(docs)} chunks from {file_path} into collection '{collection_name}'")


def index_folder(folder_path: str, collection_name: str = COLLECTION_DEFAULT):
    for fname in os.listdir(folder_path):
        path = os.path.join(folder_path, fname)
        if os.path.isfile(path):
            index_document(path, collection_name=collection_name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="Folder containing docs to index (PDF/DOCX/TXT)")
    ap.add_argument("--collection", default=COLLECTION_DEFAULT)
    args = ap.parse_args()
    index_folder(args.folder, collection_name=args.collection)
