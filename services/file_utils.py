import os
from PyPDF2 import PdfReader
import docx

def extract_paragraphs(file_path: str):
    paragraphs = []
    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            paragraphs = text.split("\n\n")

    elif file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        paragraphs = text.split("\n\n")

    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]

    # Assign indexes
    indexed_paragraphs = [{"id": idx, "text": para.strip()} for idx, para in enumerate(paragraphs) if para.strip()]
    return indexed_paragraphs
