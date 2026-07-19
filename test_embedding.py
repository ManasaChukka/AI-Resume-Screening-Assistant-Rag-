from dotenv import load_dotenv
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

print("API Key:", os.getenv("GOOGLE_API_KEY")[:10] + "...")

emb = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

print(emb.embed_query("Hello"))