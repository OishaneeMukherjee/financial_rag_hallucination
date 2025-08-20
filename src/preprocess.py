# src/preprocess.py
"""
Preprocess raw financial/legal documents:
- Extracts text & tables from PDFs, DOCX, TXT, HTML
- Cleans and normalizes text
- Splits into chunks for embedding
- Saves processed JSON files in data/processed/
"""

import os
import re
import json
import fitz  # PyMuPDF for PDF
import docx
from bs4 import BeautifulSoup

RAW_DATA_DIR = "data/raw/"
PROCESSED_DATA_DIR = "data/processed/"
CHUNK_SIZE = 500  # words per chunk


def clean_text(text: str) -> str:
    """Clean and normalize extracted text."""
    text = re.sub(r'\s+', ' ', text)  # collapse multiple spaces/newlines
    text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)  # remove page numbers
    text = text.strip()
    return text


def extract_from_pdf(filepath: str) -> str:
    """Extract text from PDF using PyMuPDF."""
    doc = fitz.open(filepath)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return clean_text(text)


def extract_from_docx(filepath: str) -> str:
    """Extract text from DOCX."""
    doc = docx.Document(filepath)
    text = "\n".join([para.text for para in doc.paragraphs])
    return clean_text(text)


def extract_from_txt(filepath: str) -> str:
    """Extract text from TXT."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return clean_text(text)


def extract_from_html(filepath: str) -> str:
    """Extract text from HTML files (e.g., MCA filings in HTML format)."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, "html.parser")
    text = soup.get_text(separator=" ")
    return clean_text(text)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE):
    """Split text into chunks of ~chunk_size words."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


def preprocess_documents():
    """Main function to preprocess all raw docs."""
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    processed = []

    for fname in os.listdir(RAW_DATA_DIR):
        fpath = os.path.join(RAW_DATA_DIR, fname)
        if not os.path.isfile(fpath):
            continue

        try:
            if fname.lower().endswith(".pdf"):
                text = extract_from_pdf(fpath)
            elif fname.lower().endswith(".docx"):
                text = extract_from_docx(fpath)
            elif fname.lower().endswith(".txt"):
                text = extract_from_txt(fpath)
            elif fname.lower().endswith(".html") or fname.lower().endswith(".htm"):
                text = extract_from_html(fpath)
            else:
                print(f"Skipping unsupported file: {fname}")
                continue

            chunks = chunk_text(text)

            # Save each file’s processed chunks into a JSON
            out_file = os.path.join(PROCESSED_DATA_DIR, f"{os.path.splitext(fname)[0]}.json")
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump({"filename": fname, "chunks": chunks}, f, indent=2)

            processed.append(out_file)
            print(f"Processed {fname} -> {out_file}")

        except Exception as e:
            print(f"Error processing {fname}: {e}")

    return processed


if __name__ == "__main__":
    processed_files = preprocess_documents()
    print("\nPreprocessing complete. Processed files saved in:", PROCESSED_DATA_DIR)
