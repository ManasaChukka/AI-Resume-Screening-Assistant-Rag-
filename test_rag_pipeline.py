"""
Quick manual end-to-end test script (Modules 2-9).

Usage:
1. Place one or more resume PDFs in data/resumes/
2. Set GOOGLE_API_KEY in your .env file
3. Run: python test_rag_pipeline.py
"""

from pathlib import Path

from src.pdf_processor import extract_text_from_pdf
from src.text_splitter import chunk_resume_text
from src.vector_store import create_vector_store, save_vector_store
from src.rag_pipeline import run_rag_query

if __name__ == "__main__":
    resume_dir = Path("data/resumes")
    pdf_files = list(resume_dir.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in data/resumes/. Add at least one and re-run.")
        raise SystemExit(1)

    all_chunks = []
    all_metadatas = []

    for pdf_path in pdf_files:
        text = extract_text_from_pdf(str(pdf_path))
        chunks = chunk_resume_text(text)
        metadatas = [{"source": pdf_path.name, "chunk_id": i} for i in range(len(chunks))]
        all_chunks.extend(chunks)
        all_metadatas.extend(metadatas)
        print(f"Processed {pdf_path.name}: {len(chunks)} chunks")

    vector_store = create_vector_store(all_chunks, all_metadatas)
    save_vector_store(vector_store)
    print("\nVector store built and saved.\n")

    test_query = "What programming languages and technical skills does this candidate have?"
    result = run_rag_query(vector_store, test_query, top_k=5)

    print(f"Query: {test_query}\n")
    print(f"Answer:\n{result['answer']}\n")
    print("Sources:")
    for src in result["sources"]:
        print(f"  - {src['source']} (relevance: {src['relevance_score']})")
