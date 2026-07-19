from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from src.embeddings import get_embedding_model

FAISS_INDEX_PATH = "data/faiss_index"


def create_vector_store(
    chunks: list[str],
    metadatas: list[dict]
) -> FAISS:
    """
    Create a new FAISS vector store from text chunks.

    Args:
        chunks: List of text chunks to embed and store.
        metadatas: List of metadata dicts, one per chunk
            (e.g., {"source": "resume_1.pdf", "chunk_id": 0}).

    Returns:
        A FAISS vector store instance.
    """
    embedding_model = get_embedding_model()

    documents = [
        Document(page_content=chunk, metadata=metadata)
        for chunk, metadata in zip(chunks, metadatas)
    ]

    vector_store = FAISS.from_documents(documents, embedding_model)
    return vector_store


def save_vector_store(vector_store: FAISS, path: str = FAISS_INDEX_PATH) -> None:
    """
    Persist a FAISS vector store to disk.

    Args:
        vector_store: The FAISS vector store to save.
        path: Directory path to save the index files.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(path)


def load_vector_store(path: str = FAISS_INDEX_PATH) -> FAISS:
    """
    Load a previously saved FAISS vector store from disk.

    Args:
        path: Directory path where the index was saved.

    Returns:
        A loaded FAISS vector store instance.

    Raises:
        FileNotFoundError: If no saved index exists at the path.
    """
    if not Path(path).exists():
        raise FileNotFoundError(
            f"No FAISS index found at '{path}'. Create one first."
        )

    embedding_model = get_embedding_model()
    vector_store = FAISS.load_local(
        path,
        embedding_model,
        allow_dangerous_deserialization=True,
    )
    return vector_store
