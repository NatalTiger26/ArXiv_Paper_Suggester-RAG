"""Local EmbeddingGemma 2 + Chroma for paper chunks."""

from __future__ import annotations

import os
from typing import Any

from .config import (
    CHROMA_RAG_DIR,
    COLLECTION_NAME,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    HF_TOKEN,
    PROJECT_ROOT,
)
from .chunk import Chunk

_model = None
_model_name_loaded: str | None = None
_model_full_dim: int | None = None


def _get_device() -> str:
    try:
        import torch

        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _apply_dim(vectors, dim: int):
    """Matryoshka truncate + L2 re-normalize."""
    import numpy as np

    arr = np.asarray(vectors, dtype="float32")
    if not dim or dim <= 0 or dim >= arr.shape[-1]:
        return arr
    arr = arr[..., :dim]
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


def load_embedding_model(model_name: str | None = None):
    """Load EmbeddingGemma 2 only (no MiniLM fallback)."""
    global _model, _model_name_loaded, _model_full_dim
    if _model is not None:
        return _model

    if HF_TOKEN:
        os.environ.setdefault("HF_TOKEN", HF_TOKEN)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", HF_TOKEN)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise SystemExit("uv add sentence-transformers")

    name = model_name or EMBEDDING_MODEL
    device = _get_device()
    print(f"[embed] Loading {name} on {device} …")
    try:
        _model = SentenceTransformer(name, device=device)
    except Exception as e:
        raise RuntimeError(
            f"Could not load {name!r}: {e}\n"
            "  uv add torchvision   # required by EmbeddingGemma 2\n"
            "  Optional: HF_TOKEN=... in .env"
        ) from e

    full = (
        _model.get_embedding_dimension()
        if hasattr(_model, "get_embedding_dimension")
        else _model.get_sentence_embedding_dimension()
    )
    _model_full_dim = int(full)
    _model_name_loaded = name
    used = EMBEDDING_DIM if EMBEDDING_DIM and 0 < EMBEDDING_DIM < full else full
    print(f"[embed] Ready. full_dim={full}  active_dim={used}  model={name}")
    return _model


def get_collection(create: bool = True):
    import chromadb
    from chromadb.config import Settings

    CHROMA_RAG_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(CHROMA_RAG_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    meta_model = _model_name_loaded or EMBEDDING_MODEL
    if create:
        return client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={
                "hnsw:space": "cosine",
                "model": meta_model,
                "embedding_dim": str(EMBEDDING_DIM or "full"),
            },
        )
    return client.get_collection(COLLECTION_NAME)


def embed_texts(texts: list[str], model=None) -> list[list[float]]:
    if model is None:
        model = load_embedding_model()
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 8,
        convert_to_numpy=True,
    )
    vectors = _apply_dim(vectors, EMBEDDING_DIM)
    return [v.tolist() for v in vectors]


def upsert_chunks(chunks: list[Chunk], model=None) -> int:
    if not chunks:
        return 0
    collection = get_collection(create=True)
    if model is None:
        model = load_embedding_model()

    ids = [c.chunk_id for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [c.to_metadata() for c in chunks]
    embeddings = embed_texts(documents, model=model)

    batch = 64
    for i in range(0, len(ids), batch):
        collection.upsert(
            ids=ids[i : i + batch],
            documents=documents[i : i + batch],
            metadatas=metadatas[i : i + batch],
            embeddings=embeddings[i : i + batch],
        )
    return len(ids)


def query_chunks(
    query: str,
    top_k: int = 6,
    paper_ids: list[str] | None = None,
    model=None,
) -> list[dict[str, Any]]:
    collection = get_collection(create=False)
    if model is None:
        model = load_embedding_model()

    qvec = embed_texts([query], model=model)[0]
    where = None
    if paper_ids:
        if len(paper_ids) == 1:
            where = {"paper_id": paper_ids[0]}
        else:
            where = {"paper_id": {"$in": paper_ids}}

    res = collection.query(
        query_embeddings=[qvec],
        n_results=min(top_k, max(collection.count(), 1)),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    out = []
    if not res["ids"] or not res["ids"][0]:
        return out

    for i, cid in enumerate(res["ids"][0]):
        dist = float(res["distances"][0][i])
        score = max(0.0, 1.0 - dist)
        meta = res["metadatas"][0][i] or {}
        out.append(
            {
                "chunk_id": cid,
                "score": round(score, 4),
                "text": res["documents"][0][i],
                "paper_id": meta.get("paper_id", ""),
                "title": meta.get("title", ""),
                "section": meta.get("section", ""),
                "source_md": meta.get("source_md", ""),
            }
        )
    return out


def list_ingested_papers() -> list[dict[str, Any]]:
    try:
        collection = get_collection(create=False)
    except Exception:
        return []
    if collection.count() == 0:
        return []
    n = min(collection.count(), 5000)
    res = collection.get(limit=n, include=["metadatas"])
    by_id: dict[str, str] = {}
    for meta in res.get("metadatas") or []:
        if not meta:
            continue
        pid = meta.get("paper_id") or ""
        if pid and pid not in by_id:
            by_id[pid] = meta.get("title") or ""
    return [{"paper_id": k, "title": v} for k, v in sorted(by_id.items())]


def index_stats() -> dict[str, Any]:
    try:
        collection = get_collection(create=False)
        count = collection.count()
    except Exception:
        count = 0
    papers = list_ingested_papers() if count else []
    full = _model_full_dim
    active = (
        EMBEDDING_DIM
        if EMBEDDING_DIM and full and 0 < EMBEDDING_DIM < full
        else full
    )
    return {
        "chunk_count": count,
        "paper_count": len(papers),
        "papers": papers,
        "embedding_model": _model_name_loaded or EMBEDDING_MODEL,
        "embedding_dim": active,
        "persist_dir": str(CHROMA_RAG_DIR.relative_to(PROJECT_ROOT))
        if CHROMA_RAG_DIR.is_relative_to(PROJECT_ROOT)
        else str(CHROMA_RAG_DIR),
    }
