from pypdf import PdfReader

from app.config.settings import settings


def _split_text(text: str) -> list[str]:
    """
    Split PDF text into retrieval-friendly chunks.

    The splitter tries to preserve:
    - paragraphs
    - lines
    - complete sentences

    while respecting the configured chunk size.
    """

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove excessive blank lines
    lines = [
        line.strip()
        for line in text.split("\n")
        if line.strip()
    ]

    chunks = []
    current_lines = []
    current_length = 0

    for line in lines:

        line_length = len(line)

        # If adding this line exceeds the chunk size,
        # finalize the current chunk.
        if (
            current_lines
            and current_length + line_length + 1
            > settings.chunk_size
        ):
            chunks.append(
                "\n".join(current_lines)
            )

            # Keep a small overlap using the last line.
            overlap_lines = []

            overlap_length = 0

            for previous_line in reversed(current_lines):

                if (
                    overlap_length
                    + len(previous_line)
                    + 1
                    > settings.chunk_overlap
                ):
                    break

                overlap_lines.insert(
                    0,
                    previous_line,
                )

                overlap_length += (
                    len(previous_line) + 1
                )

            current_lines = overlap_lines
            current_length = overlap_length

        current_lines.append(line)

        current_length += line_length + 1

    # Add remaining text.
    if current_lines:
        chunks.append(
            "\n".join(current_lines)
        )

    return chunks


def ingest_pdf(
    pdf_path: str,
    filename: str = "",
) -> list[dict]:

    reader = PdfReader(pdf_path)

    documents = []

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):

        text = page.extract_text()

        if not text:
            continue

        chunks = _split_text(text)

        for chunk in chunks:

            documents.append(
                {
                    "text": chunk,
                    "page": page_number,
                    "source": filename,
                }
            )

    return documents