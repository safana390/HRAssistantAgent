print("Loading .env...")

import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
print("API KEY FOUND?:", api_key is not None)

if not api_key:
    print("❌ ERROR: GEMINI_API_KEY not found in .env")
    raise SystemExit()

client = genai.Client(api_key=api_key)

print("\nRequesting model list...")
models = client.models.list()

print("\n=== MODELS AVAILABLE ===\n")
for m in models:
    print(m)
