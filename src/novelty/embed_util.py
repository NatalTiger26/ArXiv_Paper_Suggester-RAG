"""Embed texts using the project's local embedding path (model+dim tagged)."""

from __future__ import annotations

from typing import Any

import numpy as np


def model_and_dim() -> tuple[str, int]:
    try:
        from settings import EMBEDDING_MODEL, EMBEDDING_DIM
    except ImportError:
        from src.settings import EMBEDDING_MODEL, EMBEDDING_DIM

    try:
        from src.rag import embed_store as es
        if getattr(es, "_model", None) is not None:
            dim = int(es._active_dim())
            name = getattr(es, "_model_name_loaded", EMBEDDING_MODEL)
            return str(name), dim
    except Exception:
        pass

    dim = int(EMBEDDING_DIM) if EMBEDDING_DIM else 768
    return str(EMBEDDING_MODEL), dim


def embed_texts(texts: list[str], model=None) -> np.ndarray:
    try:
        from src.rag.embed_store import embed_texts as _embed, load_embedding_model
    except ImportError:
        from rag.embed_store import embed_texts as _embed, load_embedding_model  # type: ignore

    if model is None:
        model = load_embedding_model()
    vecs = _embed(texts, model=model)
    return np.asarray(vecs, dtype=np.float32)


def text_for_paper(title: str | None, abstract: str | None) -> str | None:
    title = (title or "").strip()
    abstract = (abstract or "").strip()
    if abstract and title:
        return f"{title}\n{title}\n{abstract}"
    if abstract:
        return abstract
    if title:
        return title
    return None
