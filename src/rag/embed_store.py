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
    try:
        from settings import IS_STREAMLIT_CLOUD
    except Exception:
        try:
            from src.settings import IS_STREAMLIT_CLOUD
        except Exception:
            IS_STREAMLIT_CLOUD = False
    heavy = any(x in (name or "").lower() for x in ("embeddinggemma", "gemma-2", "e5-large"))
    if IS_STREAMLIT_CLOUD and heavy and os.getenv("ALLOW_LARGE_EMBED") != "1":
        name = "sentence-transformers/all-MiniLM-L6-v2"
        print(
            f"[embed] Cloud guard: switched heavy model → {name}",
            flush=True,
        )
    device = _get_device()
    # Prefer CPU on Cloud to avoid huge CUDA wheels if present
    if IS_STREAMLIT_CLOUD:
        device = "cpu"
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


def _active_dim() -> int:
    """Resolved embedding dimension after Matryoshka truncate."""
    if _model_full_dim:
        full = int(_model_full_dim)
    else:
        # Before the model is loaded, infer from configured name (avoid false 768 vs 384 resets).
        name = (_model_name_loaded or EMBEDDING_MODEL or "").lower()
        if "minilm" in name:
            full = 384
        elif "embeddinggemma" in name or "gemma" in name:
            full = 768
        else:
            full = 768
    if EMBEDDING_DIM and 0 < EMBEDDING_DIM < full:
        return int(EMBEDDING_DIM)
    return int(full)


def reset_rag_collection() -> None:
    """Delete the RAG Chroma collection (call after changing EMBEDDING_DIM)."""
    import chromadb
    from chromadb.config import Settings

    CHROMA_RAG_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(CHROMA_RAG_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"[embed] Deleted collection {COLLECTION_NAME!r} (dim change / reset)")
    except Exception:
        pass


def get_collection(create: bool = True):
    """
    Open or create the RAG collection.

    If EMBEDDING_DIM changed vs metadata stored in the collection, recreate it
    so Chroma does not reject vectors with the wrong width.
    """
    import chromadb
    from chromadb.config import Settings

    CHROMA_RAG_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(CHROMA_RAG_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    meta_model = _model_name_loaded or EMBEDDING_MODEL
    want_dim = str(_active_dim())
    want_meta = {
        "hnsw:space": "cosine",
        "model": meta_model,
        "embedding_dim": want_dim,
    }

    existing = None
    try:
        existing = client.get_collection(COLLECTION_NAME)
    except Exception:
        existing = None

    if existing is not None:
        meta = existing.metadata or {}
        got_dim = str(meta.get("embedding_dim") or "")
        got_model = str(meta.get("model") or "")
        dim_mismatch = bool(got_dim and got_dim not in ("full", "") and got_dim != want_dim)
        # Gemma vs MiniLM etc.
        model_mismatch = False
        if got_model and meta_model:
            g = got_model.lower()
            w = meta_model.lower()
            if ("minilm" in w and "gemma" in g) or ("gemma" in w and "minilm" in g):
                model_mismatch = True
        if dim_mismatch or model_mismatch:
            print(
                f"[embed] Index incompatible (collection dim={got_dim or '?'}, model={got_model or '?'}; "
                f"want dim={want_dim}, model={meta_model}). Recreating collection.",
                flush=True,
            )
            try:
                client.delete_collection(COLLECTION_NAME)
            except Exception:
                pass
            existing = None
        else:
            return existing

    if not create and existing is None:
        raise RuntimeError(
            f"Collection {COLLECTION_NAME!r} not found. Run ingest first."
        )

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata=want_meta,
    )


def embed_texts(texts: list[str], model=None) -> list[list[float]]:
    if model is None:
        model = load_embedding_model()
    batch = 32
    try:
        batch = int(__import__("os").environ.get("RAG_EMBED_BATCH") or 32)
    except ValueError:
        batch = 32
    batch = max(4, min(batch, 64))
    vectors = model.encode(
        texts,
        batch_size=batch,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 8,
        convert_to_numpy=True,
    )
    vectors = _apply_dim(vectors, EMBEDDING_DIM)
    return [v.tolist() for v in vectors]


def upsert_chunks(chunks: list[Chunk], model=None) -> int:
    if not chunks:
        return 0
    if model is None:
        model = load_embedding_model()

    ids = [c.chunk_id for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [c.to_metadata() for c in chunks]
    embeddings = embed_texts(documents, model=model)

    def _write(collection) -> None:
        batch = 64
        for i in range(0, len(ids), batch):
            collection.upsert(
                ids=ids[i : i + batch],
                documents=documents[i : i + batch],
                metadatas=metadatas[i : i + batch],
                embeddings=embeddings[i : i + batch],
            )

    collection = get_collection(create=True)
    try:
        _write(collection)
    except Exception as e:
        if "dimension" not in str(e).lower():
            raise
        print(
            f"[embed] Upsert dim mismatch ({e}). Resetting RAG collection and retrying…",
            flush=True,
        )
        reset_rag_collection()
        collection = get_collection(create=True)
        _write(collection)
    return len(ids)


def query_chunks(
    query: str,
    top_k: int = 6,
    paper_ids: list[str] | None = None,
    model=None,
    query_vector: list[float] | None = None,
) -> list[dict[str, Any]]:
    if query_vector is not None:
        qvec = query_vector
    else:
        if model is None:
            model = load_embedding_model()
        qvec = embed_texts([query], model=model)[0]
    where = None
    if paper_ids:
        if len(paper_ids) == 1:
            where = {"paper_id": paper_ids[0]}
        else:
            where = {"paper_id": {"$in": paper_ids}}

    def _run(coll):
        n = max(coll.count(), 1)
        return coll.query(
            query_embeddings=[qvec],
            n_results=min(top_k, n),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

    try:
        collection = get_collection(create=False)
        res = _run(collection)
    except Exception as e:
        msg = str(e).lower()
        if "dimension" in msg or "not found" in msg:
            print(
                f"[embed] Query failed ({e}). Resetting incompatible RAG collection.",
                flush=True,
            )
            reset_rag_collection()
            return []
        raise

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



# Process-level caches (survive Streamlit reruns in the same worker)
_list_papers_cache: list[dict[str, Any]] | None = None
_list_papers_cache_key: tuple | None = None
_chunk_count_cache: int | None = None


def _progress_done_ids() -> list[str]:
    """Fast path: ids from ingest progress files (no Chroma scan)."""
    candidates = [
        PROJECT_ROOT / "data" / "rag_ingest_progress" / "done.txt",
        PROJECT_ROOT / "data" / "rag_ingest_progress" / "indexed_ids.txt",
    ]
    for path in candidates:
        try:
            if not path.is_file():
                continue
            ids: list[str] = []
            seen: set[str] = set()
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                pid = (line or "").strip().split()[0] if line.strip() else ""
                if not pid or pid.startswith("#") or pid in seen:
                    continue
                seen.add(pid)
                ids.append(pid)
            if ids:
                return ids
        except Exception:
            continue
    return []


def _titles_from_corpus(ids: list[str]) -> dict[str, str]:
    """Map arxiv id → title from corpus.json (cheap file read)."""
    out: dict[str, str] = {}
    if not ids:
        return out
    try:
        import json

        corpus_path = PROJECT_ROOT / "data" / "corpus.json"
        if not corpus_path.is_file():
            return out
        data = json.loads(corpus_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return out
        want = set(ids)
        for row in data:
            if not isinstance(row, dict):
                continue
            pid = str(
                row.get("arxiv_id") or row.get("id") or row.get("paper_id") or ""
            ).strip()
            if "v" in pid:
                base, _, ver = pid.rpartition("v")
                if ver.isdigit():
                    pid = base
            if pid in want and row.get("title"):
                out[pid] = str(row.get("title") or "").strip()
    except Exception:
        pass
    return out


def list_ingested_papers(*, force_chroma: bool = False) -> list[dict[str, Any]]:
    """Return unique papers in the RAG index.

    Prefer progress files + corpus titles (fast). Full Chroma metadata scan only
    when forced or when progress files are empty — that scan is expensive on
    large indexes and must not run on every Streamlit rerun.
    """
    global _list_papers_cache, _list_papers_cache_key

    try:
        collection = get_collection(create=False)
        total = int(collection.count() or 0)
    except Exception as e:
        print(f"[embed] list_ingested: no collection ({e})", flush=True)
        return []
    if total == 0:
        _list_papers_cache = []
        return []

    cache_key = (total, force_chroma)
    if (
        not force_chroma
        and _list_papers_cache is not None
        and _list_papers_cache_key == cache_key
    ):
        return _list_papers_cache

    # Fast path: done.txt / indexed_ids.txt
    if not force_chroma:
        done_ids = _progress_done_ids()
        if done_ids:
            titles = _titles_from_corpus(done_ids)
            papers = [
                {
                    "paper_id": pid,
                    "title": titles.get(pid, ""),
                    "arxiv_id": pid,
                }
                for pid in done_ids
            ]
            papers.sort(key=lambda x: (x.get("title") or x["paper_id"]).lower())
            _list_papers_cache = papers
            _list_papers_cache_key = cache_key
            return papers

    # Slow path: scan Chroma metadatas (only when needed)
    by_id: dict[str, str] = {}
    batch = 2000
    offset = 0

    def _pid(meta: dict) -> str:
        if not meta:
            return ""
        return str(
            meta.get("paper_id")
            or meta.get("arxiv_id")
            or meta.get("id")
            or ""
        ).strip()

    print(f"[embed] listing papers from {total} chunks (Chroma scan)…", flush=True)
    while offset < total:
        try:
            res = collection.get(limit=batch, offset=offset, include=["metadatas"])
        except TypeError:
            res = collection.get(limit=min(total, batch), include=["metadatas"])
            for meta in res.get("metadatas") or []:
                pid = _pid(meta)
                if pid and pid not in by_id:
                    by_id[pid] = (meta or {}).get("title") or ""
            break
        except Exception as e:
            print(f"[embed] list_ingested get failed at offset={offset}: {e}", flush=True)
            break
        metas = res.get("metadatas") or []
        if not metas:
            break
        for meta in metas:
            pid = _pid(meta)
            if pid and pid not in by_id:
                by_id[pid] = (meta or {}).get("title") or ""
        offset += len(metas)
        if offset % 20000 < batch or offset >= total:
            print(
                f"[embed] list progress chunks={min(offset, total)}/{total} "
                f"unique_papers={len(by_id)}",
                flush=True,
            )
        if len(metas) < batch:
            break
    print(f"[embed] list done: {len(by_id)} papers", flush=True)
    papers = [
        {"paper_id": k, "title": v, "arxiv_id": k} for k, v in sorted(by_id.items())
    ]
    _list_papers_cache = papers
    _list_papers_cache_key = cache_key
    return papers


def invalidate_paper_list_cache() -> None:
    global _list_papers_cache, _list_papers_cache_key, _chunk_count_cache
    _list_papers_cache = None
    _list_papers_cache_key = None
    _chunk_count_cache = None


def index_stats(*, include_papers: bool = False) -> dict[str, Any]:
    """Lightweight by default: chunk count only. Set include_papers=True for full list.

    Streamlit calls this often; never scan all chunks unless requested.
    """
    global _chunk_count_cache
    try:
        collection = get_collection(create=False)
        count = int(collection.count() or 0)
        _chunk_count_cache = count
    except Exception:
        count = int(_chunk_count_cache or 0)

    papers: list[dict[str, Any]] = []
    paper_count = 0
    if include_papers and count:
        papers = list_ingested_papers()
        paper_count = len(papers)
    else:
        # Prefer progress file length (O(lines)), not Chroma scan
        done_ids = _progress_done_ids()
        if done_ids:
            paper_count = len(done_ids)
        elif count and _list_papers_cache is not None:
            paper_count = len(_list_papers_cache)
        else:
            paper_count = 0  # unknown without scan; UI may load catalog on demand

    full = _model_full_dim
    active = (
        EMBEDDING_DIM
        if EMBEDDING_DIM and full and 0 < EMBEDDING_DIM < full
        else full
    )
    return {
        "chunk_count": count,
        "paper_count": paper_count,
        "papers": papers,
        "embedding_model": _model_name_loaded or EMBEDDING_MODEL,
        "embedding_dim": active,
        "persist_dir": str(CHROMA_RAG_DIR.relative_to(PROJECT_ROOT))
        if CHROMA_RAG_DIR.is_relative_to(PROJECT_ROOT)
        else str(CHROMA_RAG_DIR),
    }
