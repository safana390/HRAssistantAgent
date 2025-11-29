# test_gemini.py
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise SystemExit("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)

print("=== Text generation test using models/gemini-2.5-flash ===")
try:
    resp = client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents="Say: Hello Safana! This is a quick Gemini test from models/gemini-2.5-flash."
    )
    # try printing the convenient .text property if present
    try:
        print("Gemini says:", resp.text)
    except Exception:
        print("Raw generate response:", resp)
except Exception as e:
    print("Generation error:", repr(e))

print("\n=== Embedding test using models/text-embedding-004 ===")
try:
    emb = client.models.embed_content(
        model="models/text-embedding-004",
        contents="This is a test embedding for the HR Policy Assistant project."
    )

    try:
        vec = emb.embedding.values
        print("Embedding length:", len(vec))
    except Exception:
        print("Raw embed response:", emb)

except Exception as e:
    print("Embedding error:", repr(e))
