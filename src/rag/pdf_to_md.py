"""Convert arXiv PDFs to Markdown. Prefer text layer; OCR only when needed."""

from __future__ import annotations

from pathlib import Path

from .config import MARKDOWN_DIR
from .download import normalise_arxiv_id


def markdown_path_for(arxiv_id: str) -> Path:
    safe = arxiv_id.replace("/", "_")
    return MARKDOWN_DIR / f"{safe}.md"


def _has_usable_text_layer(pdf_path: Path, sample_pages: int = 3) -> bool:
    """True if the PDF already has a reasonable text layer (skip OCR)."""
    try:
        import pymupdf
    except ImportError:
        return True  # let pymupdf4llm decide
    try:
        doc = pymupdf.open(str(pdf_path))
        n = min(sample_pages, len(doc))
        chars = 0
        for i in range(n):
            chars += len(doc[i].get_text("text") or "")
        doc.close()
        # ~200 chars/page average → digital text layer present
        return chars >= 150 * n
    except Exception:
        return True


def pdf_to_markdown(
    pdf_path: str | Path,
    arxiv_id: str | None = None,
    force: bool = False,
) -> Path:
    """
    Convert a PDF to Markdown and cache it under data/markdown/.
    Uses text layer when available to avoid slow/noisy OCR on arXiv PDFs.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    if arxiv_id is None:
        stem = pdf_path.stem.replace("_", "/")
        arxiv_id = normalise_arxiv_id(stem) or pdf_path.stem

    md_path = markdown_path_for(arxiv_id)
    if (
        not force
        and md_path.exists()
        and md_path.stat().st_mtime >= pdf_path.stat().st_mtime
        and md_path.stat().st_size > 200
    ):
        return md_path

    try:
        import pymupdf4llm
    except ImportError:
        raise SystemExit(
            "pymupdf4llm is required for PDF→Markdown conversion.\n"
            "  pip install pymupdf4llm"
        )

    # Prefer text layer — avoids Tesseract on born-digital arXiv PDFs
    kwargs = {}
    if _has_usable_text_layer(pdf_path):
        # force_ocr=False is the default; explicit for clarity when supported
        try:
            md_text = pymupdf4llm.to_markdown(str(pdf_path), force_ocr=False)
        except TypeError:
            md_text = pymupdf4llm.to_markdown(str(pdf_path))
    else:
        md_text = pymupdf4llm.to_markdown(str(pdf_path))

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md_text or "", encoding="utf-8")
    return md_path


def ensure_markdown(arxiv_id: str, force: bool = False) -> Path:
    """Download if needed, then convert. Returns path to .md."""
    from .download import download_paper

    pdf = download_paper(arxiv_id, force=force)
    return pdf_to_markdown(pdf, arxiv_id=normalise_arxiv_id(arxiv_id) or arxiv_id, force=force)
