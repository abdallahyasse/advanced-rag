# Data Directory

This folder is used to store PDF documents for the RAG system.

## Usage

1. Place your PDF file inside this folder.

Example:

```
data/
    my_document.pdf
```

2. Update the `.env` file:

```env
PDF_PATH=data/my_document.pdf
```

3. Start the API:

```bash
python -m uvicorn app.main:app --reload
```

or

```bash
docker compose up
```

The system will automatically ingest the PDF, build or load the retrieval indexes, and answer questions based on the document.