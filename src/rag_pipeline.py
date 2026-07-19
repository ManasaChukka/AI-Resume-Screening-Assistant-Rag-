from langchain_community.vectorstores import FAISS
from src.retriever import retrieve_relevant_chunks
from src.llm import ask_llm

SYSTEM_INSTRUCTIONS = """You are a professional resume screening assistant.

Rules you must follow strictly:
1. Answer ONLY using the resume excerpts provided below. Do not use any
   outside knowledge or make assumptions beyond what is written.
2. If the excerpts do not contain enough information to answer the
   question, respond exactly with: "Not found in the provided resume excerpts."
3. Always mention which source file(s) your answer is based on.
4. Be concise, factual, and avoid speculation.
"""


def build_augmented_prompt(query: str, retrieved_chunks: list) -> str:
    """
    Build a full RAG prompt by combining system instructions,
    retrieved resume excerpts, and the user's question.

    Args:
        query: The user's original natural language question.
        retrieved_chunks: List of (Document, score) tuples from retrieval.

    Returns:
        A single formatted prompt string ready for the LLM.
    """
    excerpt_blocks = []
    for i, (doc, score) in enumerate(retrieved_chunks):
        source = doc.metadata.get("source", "unknown")
        excerpt_blocks.append(
            f"EXCERPT {i + 1} (source: {source}):\n{doc.page_content}"
        )

    excerpts_text = "\n\n".join(excerpt_blocks)

    prompt = (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"{excerpts_text}\n\n"
        f"QUESTION: {query}"
    )
    return prompt


def run_rag_query(
    vector_store: FAISS,
    query: str,
    top_k: int = 5
) -> dict:
    """
    Run the full RAG pipeline: retrieve, augment, generate.

    Args:
        vector_store: A loaded FAISS vector store.
        query: The user's natural language question.
        top_k: Number of chunks to retrieve.

    Returns:
        A dictionary containing the answer, the retrieved chunks,
        and their relevance scores (for source citation display).
    """
    retrieved_chunks = retrieve_relevant_chunks(vector_store, query, top_k)

    if not retrieved_chunks:
        return {
            "answer": "No relevant resume information found.",
            "sources": [],
        }

    prompt = build_augmented_prompt(query, retrieved_chunks)
    answer = ask_llm(prompt, temperature=0.0)

    sources = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "text": doc.page_content,
            "relevance_score": round(score, 3),
        }
        for doc, score in retrieved_chunks
    ]

    return {
        "answer": answer,
        "sources": sources,
    }
