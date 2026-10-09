"""
RAG extension for ArXiv Paper Suggester.

Pipeline:
  download PDFs → convert to Markdown → chunk → embed (EmbeddingGemma 2)
  → retrieve → answer with Gemini + citations

CLI:   python -m src.rag.cli ...
Web:   streamlit run src/rag/app.py
Eval:  python -m src.rag.cli eval
"""

__all__ = [
    "download_paper",
    "pdf_to_markdown",
    "chunk_markdown",
    "ingest_papers",
    "ask",
]
