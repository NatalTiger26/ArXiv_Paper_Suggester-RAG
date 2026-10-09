"""Download arXiv PDFs by ID with retries and a small local cache."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Iterable

import requests
from tqdm import tqdm

from .config import DOWNLOAD_RETRIES, DOWNLOAD_TIMEOUT, PAPERS_DIR

ARXIV_PDF_URL = "https://arxiv.org/pdf/{id}.pdf"
ARXIV_ABS_URL = "https://export.arxiv.org/api/query?id_list={id}"
ARXIV_ID_RE = re.compile(
    r"(?P<id>(?:\d{4}\.\d{4,5}(?:v\d+)?)|(?:[a-z\-]+/\d{7}))",
    re.IGNORECASE,
)


def normalise_arxiv_id(raw: str) -> str | None:
    """Extract a clean arXiv ID from free text or a URL (version suffix stripped)."""
    if not raw:
        return None
    raw = raw.strip()
    raw = re.sub(r"^https?://arxiv\.org/(abs|pdf)/", "", raw, flags=re.I)
    raw = raw.replace(".pdf", "").strip("/")
    m = ARXIV_ID_RE.search(raw)
    if not m:
        return None
    aid = m.group("id")
    aid = re.sub(r"v\d+$", "", aid, flags=re.I)
    return aid


def pdf_path_for(arxiv_id: str) -> Path:
    safe = arxiv_id.replace("/", "_")
    return PAPERS_DIR / f"{safe}.pdf"


def fetch_arxiv_title(arxiv_id: str) -> str | None:
    """Fetch the canonical title from the arXiv Atom API."""
    aid = normalise_arxiv_id(arxiv_id)
    if not aid:
        return None
    url = ARXIV_ABS_URL.format(id=aid)
    try:
        resp = requests.get(url, timeout=20)
        if resp.status_code != 200:
            return None
        # Minimal XML parse — title is the first <title> after the feed title
        titles = re.findall(r"<title[^>]*>(.*?)</title>", resp.text, flags=re.S | re.I)
        if len(titles) >= 2:
            title = re.sub(r"\s+", " ", titles[1]).strip()
            title = re.sub(r"^arXiv:", "", title, flags=re.I).strip()
            return title[:300] if title else None
    except Exception:
        return None
    return None


def download_paper(arxiv_id: str, force: bool = False) -> Path:
    """
    Download a single paper PDF with retries.
    Skips download if the file already exists (unless force=True).
    """
    aid = normalise_arxiv_id(arxiv_id)
    if not aid:
        raise ValueError(f"Could not parse arXiv ID from: {arxiv_id!r}")

    dest = pdf_path_for(aid)
    if dest.exists() and dest.stat().st_size > 1000 and not force:
        return dest

    url = ARXIV_PDF_URL.format(id=aid)
    last_err: Exception | None = None
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=DOWNLOAD_TIMEOUT, stream=True)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code} for {url}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            total = int(resp.headers.get("content-length", 0))
            tmp = dest.with_suffix(".pdf.part")
            with open(tmp, "wb") as f, tqdm(
                total=total or None,
                unit="B",
                unit_scale=True,
                desc=f"Downloading {aid} (try {attempt})",
                leave=False,
            ) as bar:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
            tmp.replace(dest)
            if dest.stat().st_size < 1000:
                raise RuntimeError(f"Downloaded file too small: {dest}")
            return dest
        except Exception as e:
            last_err = e
            if attempt < DOWNLOAD_RETRIES:
                time.sleep(1.5 * attempt)
    raise RuntimeError(
        f"Failed to download {aid} after {DOWNLOAD_RETRIES} attempts: {last_err}"
    )


def download_papers(ids: Iterable[str], force: bool = False) -> list[Path]:
    """Download multiple papers. Returns list of successful local paths."""
    paths = []
    for raw in ids:
        try:
            paths.append(download_paper(raw, force=force))
        except Exception as e:
            print(f"[download] skip {raw}: {e}")
    return paths
