from pypdf import PdfReader

from app.config.settings import settings


def ingest_pdf(
    pdf_path: str,
    filename: str = "",
) -> list[dict]:

    reader = PdfReader(pdf_path)

    documents = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text()

        if not text:
            continue

        start = 0

        while start < len(text):

            chunk = text[
                start:start + settings.chunk_size
            ]

            documents.append(
                {
                    "text": chunk,
                    "page": page_number,
                    "source": filename,
                }
            )

            start += (
                settings.chunk_size
                - settings.chunk_overlap
            )

    return documents