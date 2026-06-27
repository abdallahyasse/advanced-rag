import io
from pypdf import PdfReader

def load_pdf(source, filename: str = "") -> list[dict]:
    if isinstance(source, (str,)):
        with open(source, "rb") as f:
            raw = f.read()
    else:
        raw = source

    reader = PdfReader(io.BytesIO(raw))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"text": text, "page": i + 1})
    return pages


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


def ingest_pdf(source, filename: str = "") -> list[dict]:
    pages = load_pdf(source, filename)
    if not pages:
        raise ValueError("No text found in PDF.")

    chunks = []
    for page_data in pages:
        page_chunks = chunk_text(page_data["text"])
        for chunk in page_chunks:
            chunks.append({
                "text": chunk,
                "source": filename or "unknown.pdf",
                "page": page_data["page"],
            })
    return chunks