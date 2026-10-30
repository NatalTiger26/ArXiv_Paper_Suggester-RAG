#!/usr/bin/env python3
"""
Corpus index — Embedding & Retrieval Backbone

  python src/corpus_index.py build
  python src/corpus_index.py query "mechanistic interpretability of transformers"
  python src/corpus_index.py demo
  python src/corpus_index.py interactive
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

# Resolve paths relative to the project root so the scripts work no matter
# which directory you invoke them from (as long as the repo layout is intact).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = PROJECT_ROOT / "data" / "corpus.json"
CACHE_DIR = PROJECT_ROOT / "data" / "embeddings"
EMB_CACHE = CACHE_DIR / "corpus_embeddings.json"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_corpus"
COLLECTION_NAME = "arxiv_corpus"
try:
    from settings import EMBEDDING_MODEL as DEFAULT_MODEL, EMBEDDING_DIM
except ImportError:
    try:
        from src.settings import EMBEDDING_MODEL as DEFAULT_MODEL, EMBEDDING_DIM
    except ImportError:
        DEFAULT_MODEL = "google/embeddinggemma-2"
        EMBEDDING_DIM = 0


_CACHED_CORPUS: list[dict] | None = None
_CACHED_CORPUS_MTIME: float = 0.0


def load_corpus(path: Path = CORPUS_PATH) -> list[dict]:
    global _CACHED_CORPUS, _CACHED_CORPUS_MTIME
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Corpus not found: {p}\n"
            "  Place data/corpus.json (or legacy corpus_100.json), then optionally\n"
            "  rebuild the index with:  python src/corpus_index.py build"
        )
    mtime = p.stat().st_mtime
    if _CACHED_CORPUS is not None and mtime == _CACHED_CORPUS_MTIME:
        return _CACHED_CORPUS
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("corpus.json must be a non-empty JSON array")
    _CACHED_CORPUS = data
    _CACHED_CORPUS_MTIME = mtime
    return data


def paper_id(p: dict) -> str:
    if p.get("arxiv_id"):
        return str(p["arxiv_id"])
    if p.get("doi"):
        return f"doi:{p['doi']}"
    if p.get("internal_id"):
        return str(p["internal_id"])
    t = re.sub(r"\s+", " ", (p.get("title") or "")[:60])
    return f"title:{t}"


def text_for_embedding(p: dict) -> str:
    title = (p.get("title") or "").strip()
    abstract = (p.get("abstract") or "").strip()
    if abstract:
        return f"{title}\n{title}\n{abstract}"
    return title


def get_device() -> str:
    try:
        import torch
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


_CORPUS_MODEL = None
_CORPUS_MODEL_NAME = None


def load_model(model_name: str | None = None):
    """Load EmbeddingGemma 2 (or EMBEDDING_MODEL). Cached in-process singleton."""
    global _CORPUS_MODEL, _CORPUS_MODEL_NAME
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise SystemExit("uv add sentence-transformers chromadb torchvision")
    device = get_device()
    name = model_name or DEFAULT_MODEL
    try:
        from settings import IS_STREAMLIT_CLOUD
    except Exception:
        try:
            from src.settings import IS_STREAMLIT_CLOUD
        except Exception:
            IS_STREAMLIT_CLOUD = False
    if IS_STREAMLIT_CLOUD and any(
        x in (name or "").lower() for x in ("embeddinggemma", "gemma-2", "e5-large")
    ) and __import__("os").getenv("ALLOW_LARGE_EMBED") != "1":
        name = "sentence-transformers/all-MiniLM-L6-v2"
        print(f"[corpus_index] Cloud guard: using {name}", flush=True)

    if _CORPUS_MODEL is not None and _CORPUS_MODEL_NAME == name:
        return _CORPUS_MODEL

    try:
        from src.rag import embed_store as es
        if getattr(es, "_model", None) is not None and getattr(es, "_model_name_loaded", None) == name:
            _CORPUS_MODEL = es._model
            _CORPUS_MODEL_NAME = name
            return _CORPUS_MODEL
    except Exception:
        pass

    print(f"Loading model: {name}  (device={device})")
    t0 = time.time()
    try:
        model = SentenceTransformer(name, device=device)
    except Exception as e:
        raise SystemExit(
            f"Could not load embedding model {name!r}: {e}\n"
            "  uv add torchvision   # required by EmbeddingGemma 2\n"
            "  Optional: HF_TOKEN=... in .env"
        ) from e
    print(f"  loaded in {time.time() - t0:.1f}s")
    _CORPUS_MODEL = model
    _CORPUS_MODEL_NAME = name

    try:
        from src.rag import embed_store as es
        if getattr(es, "_model", None) is None:
            es._model = model
            es._model_name_loaded = name
    except Exception:
        pass

    return model



def _apply_matryoshka(vectors, dim: int):
    import numpy as np
    arr = np.asarray(vectors, dtype="float32")
    if not dim or dim <= 0 or dim >= arr.shape[-1]:
        return arr
    arr = arr[..., :dim]
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)

def embed_texts(model, texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Encode texts in batches. Prefer smaller batches on MPS to avoid silent kills."""
    import numpy as np

    if not texts:
        return []
    device = get_device()
    # MPS / unified memory: large batches of Gemma often OOM and kill the process
    if device == "mps" and batch_size > 16:
        batch_size = 16
        print(f"[corpus_index] MPS: capping encode batch_size={batch_size}", flush=True)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 8,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    vectors = _apply_matryoshka(vectors, EMBEDDING_DIM)
    return [v.tolist() for v in vectors]


def _load_embedding_map(force: bool = False) -> dict[str, list[float]]:
    """id → vector from JSON cache and/or npz checkpoint (partial OK)."""
    out: dict[str, list[float]] = {}
    if force:
        return out
    # Primary JSON cache (may be large)
    if EMB_CACHE.exists():
        try:
            cached = json.loads(EMB_CACHE.read_text(encoding="utf-8"))
            c_ids = cached.get("ids") or []
            c_vecs = cached.get("embeddings") or []
            for i, pid in enumerate(c_ids):
                if i < len(c_vecs) and c_vecs[i]:
                    out[str(pid)] = c_vecs[i]
            if out:
                print(
                    f"[corpus_index] loaded {len(out)} vectors from {EMB_CACHE.name}",
                    flush=True,
                )
        except Exception as e:
            print(f"[corpus_index] JSON cache load skipped: {e}", flush=True)
    # Incremental npz checkpoint (survives mid-run crashes)
    npz_path = CACHE_DIR / "corpus_embeddings_checkpoint.npz"
    meta_path = CACHE_DIR / "corpus_embeddings_checkpoint_ids.json"
    if npz_path.exists() and meta_path.exists():
        try:
            import numpy as np

            arr = np.load(npz_path)["vectors"]
            c_ids = json.loads(meta_path.read_text(encoding="utf-8"))
            for i, pid in enumerate(c_ids):
                if i < len(arr) and str(pid) not in out:
                    out[str(pid)] = arr[i].tolist()
            print(
                f"[corpus_index] checkpoint has {len(c_ids)} ids "
                f"(map now {len(out)} unique)",
                flush=True,
            )
        except Exception as e:
            print(f"[corpus_index] checkpoint load skipped: {e}", flush=True)
    return out


def _save_checkpoint(id_to_vec: dict[str, list[float]], model_name: str) -> None:
    """Write npz + ids so a killed run can resume."""
    import numpy as np

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ids = list(id_to_vec.keys())
    if not ids:
        return
    arr = np.asarray([id_to_vec[i] for i in ids], dtype="float32")
    npz_path = CACHE_DIR / "corpus_embeddings_checkpoint.npz"
    meta_path = CACHE_DIR / "corpus_embeddings_checkpoint_ids.json"
    np.savez_compressed(npz_path, vectors=arr)
    meta_path.write_text(json.dumps(ids), encoding="utf-8")
    # Also refresh full JSON when not huge (optional; skip if very large)
    if len(ids) <= 8000:
        EMB_CACHE.write_text(
            json.dumps(
                {
                    "model": model_name,
                    "dim": int(arr.shape[1]),
                    "ids": ids,
                    "embeddings": [id_to_vec[i] for i in ids],
                }
            ),
            encoding="utf-8",
        )


def build(args: argparse.Namespace) -> None:
    corpus = load_corpus()
    ids = [paper_id(p) for p in corpus]
    texts = [text_for_embedding(p) for p in corpus]
    model_name = args.model or DEFAULT_MODEL
    force = bool(getattr(args, "force_embed", False))

    id_to_vec = _load_embedding_map(force=force)
    missing_idx = [i for i, pid in enumerate(ids) if pid not in id_to_vec]

    if not missing_idx:
        print(
            f"Reusing embeddings for all {len(ids)} papers "
            f"(dim={len(next(iter(id_to_vec.values())))})",
            flush=True,
        )
    else:
        print(
            f"Need to embed {len(missing_idx)} / {len(ids)} papers "
            f"({len(ids) - len(missing_idx)} already cached)",
            flush=True,
        )
        model = load_model(args.model)
        batch_size = int(getattr(args, "batch_size", None) or 16)
        # Encode in chunks and checkpoint often (crash-safe)
        chunk = max(32, min(128, batch_size * 4))
        t0 = time.time()
        done = 0
        for start in range(0, len(missing_idx), chunk):
            sl = missing_idx[start : start + chunk]
            sub_texts = [texts[i] for i in sl]
            sub_ids = [ids[i] for i in sl]
            try:
                sub_vecs = embed_texts(model, sub_texts, batch_size=batch_size)
            except Exception as e:
                print(f"[corpus_index] encode failed at +{done}: {e}", flush=True)
                _save_checkpoint(id_to_vec, model_name)
                print(
                    f"[corpus_index] saved checkpoint ({len(id_to_vec)} vectors). "
                    "Re-run the same command to resume.",
                    flush=True,
                )
                raise SystemExit(1) from e
            for pid, vec in zip(sub_ids, sub_vecs):
                id_to_vec[pid] = vec
            done += len(sl)
            _save_checkpoint(id_to_vec, model_name)
            print(
                f"  embedded {done}/{len(missing_idx)} new  "
                f"total_cached={len(id_to_vec)}  "
                f"elapsed={time.time() - t0:.0f}s",
                flush=True,
            )
        print(
            f"  embedding finished in {time.time() - t0:.1f}s  "
            f"dim={len(next(iter(id_to_vec.values())))}",
            flush=True,
        )
        print(f"Cached embeddings -> {EMB_CACHE} (+ checkpoint)", flush=True)

    # Align vectors to corpus order
    missing_final = [pid for pid in ids if pid not in id_to_vec]
    if missing_final:
        raise SystemExit(
            f"Still missing {len(missing_final)} embeddings after encode — "
            "re-run build."
        )
    vectors = [id_to_vec[pid] for pid in ids]

    import chromadb
    from chromadb.config import Settings

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))
    # Drop existing collection if present (Cloud demos rebuild often).
    for _ in range(2):
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    meta_model = args.model or DEFAULT_MODEL
    want_dim = len(vectors[0]) if vectors else 0
    try:
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={
                "hnsw:space": "cosine",
                "model": str(meta_model),
                "embedding_dim": str(want_dim),
            },
        )
    except Exception as e:
        # Race / residual collection on Cloud
        print(f"[corpus_index] create_collection retry after: {e}", flush=True)
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={
                "hnsw:space": "cosine",
                "model": str(meta_model),
                "embedding_dim": str(want_dim),
            },
        )
    metadatas, documents = [], []
    for p, text in zip(corpus, texts):
        metadatas.append({
            "arxiv_id": str(p.get("arxiv_id") or ""),
            "title": (p.get("title") or "")[:300],
            "year": str((p.get("published_date") or "")[:4]),
            "doi": str(p.get("doi") or ""),
            "categories": ",".join(p.get("categories") or [])[:200],
            "citation_count": str(p.get("citation_count") if p.get("citation_count") is not None else ""),
        })
        documents.append(text[:2000])
    # Chroma enforces a max upsert batch (often ~5461). Chunk large corpora.
    chroma_batch = 2000
    try:
        chroma_batch = min(2000, int(client.get_max_batch_size()))
    except Exception:
        pass
    n = len(ids)
    for start in range(0, n, chroma_batch):
        end = min(start + chroma_batch, n)
        collection.add(
            ids=ids[start:end],
            embeddings=vectors[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
        print(f"  chroma add {end}/{n}", flush=True)
    print(f"Chroma index ready  n={collection.count()}")
    print('Try: python src/corpus_index.py query "mechanistic interpretability circuits"')


def get_collection():
    import chromadb
    from chromadb.config import Settings
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"No Chroma index at {CHROMA_DIR}\n"
            "  Run:  python src/corpus_index.py build"
        )
    client = chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))
    return client.get_collection(COLLECTION_NAME)


def retrieve(query: str, k: int = 10, model_name: str | None = None, model=None) -> list[dict[str, Any]]:
    collection = get_collection()
    if model is None:
        model = load_model(model_name or DEFAULT_MODEL)
    qvec = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    qvec = _apply_matryoshka(qvec, EMBEDDING_DIM)[0].tolist()
    res = collection.query(
        query_embeddings=[qvec],
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    out = []
    for i, pid in enumerate(res["ids"][0]):
        score = max(0.0, 1.0 - float(res["distances"][0][i]))
        meta = res["metadatas"][0][i] or {}
        out.append({
            "paper_id": pid,
            "score": round(score, 4),
            "title": meta.get("title") or "",
            "arxiv_id": meta.get("arxiv_id") or "",
            "year": meta.get("year") or "",
            "citation_count": meta.get("citation_count") or "",
            "snippet": (res["documents"][0][i] or "")[:240].replace("\n", " "),
        })
    return out


def print_results(query: str, results: list[dict]) -> None:
    print("\n" + "=" * 70)
    print(f"Query: {query}")
    print("=" * 70)
    for i, r in enumerate(results, 1):
        aid = r.get("arxiv_id") or r.get("paper_id") or ""
        print(f"{i:2}. [{r['score']:.3f}]  {aid:16}  {r['title'][:58]}")
        if r.get("year") or r.get("citation_count"):
            print(f"     year={r.get('year') or '?'}  cites={r.get('citation_count') or '?'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument(
        "--batch-size",
        type=int,
        default=12,
        help="Encode batch size (default 12; keep small on Mac MPS to avoid OOM kills)",
    )
    b.add_argument(
        "--force-embed",
        action="store_true",
        help="Ignore embedding caches and re-embed everything",
    )

    q = sub.add_parser("query")
    q.add_argument("query")
    q.add_argument("-k", type=int, default=10)

    d = sub.add_parser("demo")
    d.add_argument("-k", type=int, default=8)

    i = sub.add_parser("interactive")
    i.add_argument("-k", type=int, default=8)

    args = parser.parse_args()
    if args.cmd == "build":
        build(args)
    elif args.cmd == "query":
        print_results(args.query, retrieve(args.query, k=args.k, model_name=args.model))
    elif args.cmd == "demo":
        model = load_model(args.model)
        for qtext in [
            "mechanistic interpretability of large language models and circuit discovery",
            "statistical mechanics of deep learning and generalization",
            "random matrix theory applied to neural network loss landscapes",
            "sparse autoencoders for feature interpretability in transformers",
            "grokking and phase transitions in neural network training",
        ]:
            print_results(qtext, retrieve(qtext, k=args.k, model=model))
            print()
    elif args.cmd == "interactive":
        model = load_model(args.model)
        print("Empty line or q to quit.\n")
        while True:
            try:
                qtext = input("interest> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not qtext or qtext.lower() in {"q", "quit", "exit"}:
                break
            print_results(qtext, retrieve(qtext, k=args.k, model=model))
            print()


if __name__ == "__main__":
    main()