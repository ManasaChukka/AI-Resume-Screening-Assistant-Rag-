"""
Quick manual test script for PDF extraction (Module 2).
Usage: place a resume PDF at data/resumes/sample_resume.pdf, then run:
    python test_pdf.py
"""

from src.pdf_processor import extract_text_from_pdf

if __name__ == "__main__":
    text = extract_text_from_pdf("data/resumes/sample_resume.pdf")
    print(text)
    print(f"\nTotal characters: {len(text)}")
