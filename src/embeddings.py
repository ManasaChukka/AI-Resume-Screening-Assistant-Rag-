import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def get_embedding_model():
    api_key = os.getenv("GOOGLE_API_KEY")

    print("=" * 60)
    print("Current Working Directory:", os.getcwd())
    print("Using .env file:", find_dotenv())
    print("API Key Exists:", api_key is not None)
    print("API Key Length:", len(api_key) if api_key else 0)
    print("API Key Prefix:", api_key[:10] if api_key else "None")
    print("=" * 60)

    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found.")

    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )