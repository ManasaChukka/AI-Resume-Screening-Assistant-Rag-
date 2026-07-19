from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


def get_retriever(vector_store: FAISS, top_k: int = 5):
    """
    Create a retriever from a FAISS vector store.

    Args:
        vector_store: A loaded/created FAISS vector store.
        top_k: Number of top results to retrieve per query.

    Returns:
        A LangChain retriever object.
    """
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )
    return retriever


def retrieve_relevant_chunks(
    vector_store: FAISS,
    query: str,
    top_k: int = 5
) -> list[tuple[Document, float]]:
    """
    Retrieve the most relevant chunks for a query, with similarity scores.

    Args:
        vector_store: A loaded/created FAISS vector store.
        query: The natural language search query.
        top_k: Number of top results to return.

    Returns:
        List of (Document, score) tuples, ranked by relevance.
        Score is a similarity value where higher = more relevant.
    """
    results = vector_store.similarity_search_with_relevance_scores(
        query, k=top_k
    )
    return results


def format_retrieved_chunks(results: list[tuple[Document, float]]) -> str:
    """
    Format retrieved chunks into a readable string for display or
    for injecting into an LLM prompt.

    Args:
        results: List of (Document, score) tuples from retrieval.

    Returns:
        A formatted string summarizing each result.
    """
    formatted = []
    for i, (doc, score) in enumerate(results):
        source = doc.metadata.get("source", "unknown")
        formatted.append(
            f"[Result {i+1}] Source: {source} | Relevance: {score:.2f}\n"
            f"{doc.page_content}\n"
        )
    return "\n".join(formatted)
