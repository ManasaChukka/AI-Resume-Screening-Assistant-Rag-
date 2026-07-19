import fitz  # PyMuPDF
from pathlib import Path


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all text content from a PDF file.

    Args:
        pdf_path: Path to the PDF file on disk.

    Returns:
        A single string containing text from all pages,
        separated by newlines.

    Raises:
        FileNotFoundError: If the PDF path does not exist.
        ValueError: If the PDF contains no extractable text.
    """
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"No file found at: {pdf_path}")

    document = fitz.open(pdf_path)

    all_text = []
    for page_number in range(len(document)):
        page = document[page_number]
        page_text = page.get_text()
        all_text.append(page_text)

    document.close()

    full_text = "\n".join(all_text)

    if not full_text.strip():
        raise ValueError(f"No extractable text found in: {pdf_path}")

    return clean_text(full_text)


def clean_text(text: str) -> str:
    """
    Clean raw extracted PDF text by removing excess whitespace
    and blank lines.

    Args:
        text: Raw text extracted from a PDF.

    Returns:
        Cleaned text with normalized spacing.
    """
    lines = text.split("\n")
    cleaned_lines = [line.strip() for line in lines if line.strip()]
    return "\n".join(cleaned_lines)
