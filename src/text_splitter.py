from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_resume_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> list[str]:
    """
    Split resume text into overlapping chunks for embedding.

    Args:
        text: Cleaned resume text to split.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of characters shared between
            consecutive chunks to preserve context.

    Returns:
        List of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_text(text)
    return chunks
