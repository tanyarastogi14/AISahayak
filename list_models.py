import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

KEY = os.getenv("GEMINI_API_KEY")
if not KEY:
    raise RuntimeError("GEMINI_API_KEY not found in .env")

genai.configure(api_key=KEY)

print("Fetching models...\n")

try:
    models = genai.list_models()  # <-- this is a generator
    for m in models:
        print(m.name)
        # If you want more details:
        # print(m)
except Exception as e:
    print("Error listing models:", e)
