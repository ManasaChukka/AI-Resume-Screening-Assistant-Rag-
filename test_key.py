import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=API_KEY)

try:
    models = genai.list_models()
    print("✅ API Key is valid!")
    for model in models:
        print(model.name)
except Exception as e:
    print("❌ Error:", e)