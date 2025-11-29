# utils/docs.py
from typing import List, Tuple
import pdfplumber
from docx import Document
import re
import os

def extract_text_from_pdf(path: str) -> str:
    text_parts = []
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            page_text = p.extract_text() or ""
            text_parts.append(page_text)
    return "\n".join(text_parts)

def extract_text_from_docx(path: str) -> str:
    doc = Document(path)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text_from_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def clean_text(s: str) -> str:
    s = s.replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def load_document(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    elif ext in [".docx", ".doc"]:
        return extract_text_from_docx(path)
    else:
        return extract_text_from_txt(path)

def chunk_text(text: str, chunk_size: int = 1200, chunk_overlap: int = 200) -> List[Tuple[str,int,int]]:
    """
    Split into chunks (by characters). Returns list of (chunk_text, start, end)
    """
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append((chunk.strip(), start, min(end, text_len)))
        start = end - chunk_overlap
        if start < 0:
            start = 0
    return chunks
