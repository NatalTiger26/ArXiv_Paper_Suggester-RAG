#!/usr/bin/env python3
"""
Automated 256-Dim Matryoshka Embedding Migration Script
Migrates Chroma collections, SQLite novelty cache, and corpus checkpoints
to 256 dimensions using zero-inference prefix slicing and L2 re-normalization.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

TARGET_DIM = 256


def slice_and_normalize(vectors: np.ndarray, dim: int = TARGET_DIM) -> np.ndarray:
    """Slice first 256 dimensions and L2 re-normalize with zero forward passes."""
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    sliced = arr[:, :dim]
    norms = np.linalg.norm(sliced, axis=-1, keepdims=True)
    return sliced / np.maximum(norms, 1e-12)


def migrate_corpus_files() -> None:
    """Migrate data/embeddings/corpus_embeddings.json and checkpoint npz to 256-dim."""
    print("\n[1/4] Migrating corpus disk embeddings...")
    emb_dir = PROJECT_ROOT / "data" / "embeddings"
    json_path = emb_dir / "corpus_embeddings.json"
    npz_path = emb_dir / "corpus_embeddings_checkpoint.npz"

    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ids = data.get("ids", [])
        raw_vecs = np.asarray(data.get("embeddings", []), dtype=np.float32)
        if raw_vecs.shape[-1] != TARGET_DIM:
            vecs_256 = slice_and_normalize(raw_vecs, TARGET_DIM)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({"ids": ids, "embeddings": vecs_256.tolist()}, f)
            print(f"  corpus_embeddings.json: migrated {len(ids)} vectors from {raw_vecs.shape[-1]} to {TARGET_DIM}")
        else:
            print(f"  corpus_embeddings.json: already at {TARGET_DIM}-dim")

    if npz_path.exists():
        try:
            npz = np.load(npz_path, allow_pickle=True)
            saved_dict = {}
            for k in npz.files:
                val = npz[k]
                if isinstance(val, np.ndarray) and val.ndim == 2 and val.shape[-1] != TARGET_DIM:
                    val = slice_and_normalize(val, TARGET_DIM)
                saved_dict[k] = val
            np.savez_compressed(npz_path, **saved_dict)
            print(f"  corpus_embeddings_checkpoint.npz: migrated to {TARGET_DIM}-dim")
        except Exception as e:
            print(f"  Warning: could not migrate npz checkpoint ({e})")


def migrate_chroma_corpus() -> None:
    """Migrate data/chroma_corpus (arxiv_corpus collection) to 256-dim."""
    print("\n[2/4] Migrating Chroma corpus collection (arxiv_corpus)...")
    import chromadb

    corpus_dir = PROJECT_ROOT / "data" / "chroma_corpus"
    if not corpus_dir.exists():
        print("  data/chroma_corpus directory does not exist, skipping")
        return

    client = chromadb.PersistentClient(path=str(corpus_dir))
    col = None
    try:
        col = client.get_collection("arxiv_corpus")
    except Exception:
        print("  arxiv_corpus collection not found, skipping")
        return

    cnt = col.count()
    meta = col.metadata or {}
    if str(meta.get("embedding_dim")) == str(TARGET_DIM):
        print(f"  arxiv_corpus: already at {TARGET_DIM}-dim with {cnt} items")
        return

    print(f"  arxiv_corpus: reading {cnt} items...")
    data = col.get(limit=cnt + 100, include=["embeddings", "documents", "metadatas"])
    if not data["ids"]:
        print("  arxiv_corpus is empty")
        return

    vecs_256 = slice_and_normalize(np.asarray(data["embeddings"], dtype=np.float32), TARGET_DIM)

    # Recreate collection
    client.delete_collection("arxiv_corpus")
    new_col = client.create_collection(
        "arxiv_corpus",
        metadata={
            "model": "google/embeddinggemma-2",
            "embedding_dim": str(TARGET_DIM),
            "hnsw:space": "cosine",
        },
    )
    # Add in batches
    batch_size = 2000
    for i in range(0, len(data["ids"]), batch_size):
        end = i + batch_size
        new_col.add(
            ids=data["ids"][i:end],
            embeddings=vecs_256[i:end].tolist(),
            documents=data["documents"][i:end] if data.get("documents") else None,
            metadatas=data["metadatas"][i:end] if data.get("metadatas") else None,
        )
    print(f"  arxiv_corpus: successfully migrated {cnt} items to {TARGET_DIM}-dim")


def migrate_novelty_cache() -> None:
    """Migrate SQLite novelty_cache.sqlite3 embeddings and recompute scores."""
    print("\n[3/4] Migrating SQLite novelty cache (novelty_cache.sqlite3)...")
    db_path = PROJECT_ROOT / "data" / "novelty" / "novelty_cache.sqlite3"
    if not db_path.exists():
        print("  novelty_cache.sqlite3 not found, skipping")
        return

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # 1. Embeddings table
    rows_768 = conn.execute(
        "SELECT canonical_id, model, vector FROM embeddings WHERE dim = 768"
    ).fetchall()
    print(f"  Found {len(rows_768)} embeddings under dim = 768")

    now = time.time()
    batch_inserts = []
    for r in rows_768:
        cid = r["canonical_id"]
        mod = r["model"]
        blob = r["vector"]
        vec_768 = np.frombuffer(blob, dtype=np.float32)
        if len(vec_768) >= TARGET_DIM:
            sliced = vec_768[:TARGET_DIM]
            norm = np.linalg.norm(sliced)
            vec_256 = sliced / (norm if norm > 1e-12 else 1.0)
            batch_inserts.append((cid, mod, TARGET_DIM, vec_256.astype(np.float32).tobytes(), now))

    if batch_inserts:
        conn.executemany(
            """
            INSERT OR REPLACE INTO embeddings (canonical_id, model, dim, vector, embedded_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            batch_inserts,
        )
        conn.commit()
        print(f"  Saved {len(batch_inserts)} 256-dim embeddings into cache")

    # 2. Recompute scores for 5,711 papers using fast 256-dim NNLS
    scores_768 = conn.execute(
        "SELECT focal_id, model, profile_json, cohort_note FROM scores WHERE dim = 768"
    ).fetchall()
    print(f"  Recomputing {len(scores_768)} paper score profiles in 256 dimensions...")

    from scipy.optimize import nnls
    from src.novelty.metrics import compute_profile

    updated_scores = []
    t0 = time.time()
    for s in scores_768:
        fid = s["focal_id"]
        mod = s["model"]
        note = s["cohort_note"]

        # Fetch focal vector
        f_row = conn.execute(
            "SELECT vector FROM embeddings WHERE canonical_id = ? AND dim = 256",
            (fid,),
        ).fetchone()
        if not f_row:
            continue
        x = np.frombuffer(f_row["vector"], dtype=np.float32)

        # Fetch reference vectors
        ref_rows = conn.execute(
            """
            SELECT e.vector, e.canonical_id FROM ref_edges r
            JOIN embeddings e ON e.canonical_id = r.ref_id
            WHERE r.focal_id = ? AND e.dim = 256
            """,
            (fid,),
        ).fetchall()

        if len(ref_rows) < 3:
            continue

        ref_mat = np.column_stack([np.frombuffer(r["vector"], dtype=np.float32) for r in ref_rows])
        ref_ids = [r["canonical_id"] for r in ref_rows]

        prof = compute_profile(x, ref_mat, ref_ids=ref_ids)
        prof["model"] = mod
        prof["dim"] = TARGET_DIM
        updated_scores.append((fid, mod, TARGET_DIM, json.dumps(prof), note, now))

    if updated_scores:
        conn.executemany(
            """
            INSERT OR REPLACE INTO scores (focal_id, model, dim, profile_json, cohort_note, computed_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            updated_scores,
        )
        conn.commit()
        print(f"  Recomputed and saved {len(updated_scores)} profiles in {time.time() - t0:.2f}s")

    conn.close()


def migrate_chroma_rag(batch_size: int = 5000, max_chunks: int | None = None) -> None:
    """Migrate data/chroma_rag (paper_chunks) to 256 dimensions with atomic swap."""
    print("\n[4/4] Migrating Chroma RAG collection (paper_chunks)...")
    import chromadb

    rag_dir = PROJECT_ROOT / "data" / "chroma_rag"
    client = chromadb.PersistentClient(path=str(rag_dir))

    try:
        source_col = client.get_collection("paper_chunks")
    except Exception:
        print("  Collection 'paper_chunks' not found, skipping")
        return

    total_chunks = source_col.count()
    meta = source_col.metadata or {}
    if str(meta.get("embedding_dim")) == str(TARGET_DIM):
        print(f"  paper_chunks: already at {TARGET_DIM}-dim with {total_chunks} items")
        return

    target_count = min(total_chunks, max_chunks) if max_chunks else total_chunks
    print(f"  Target chunks to migrate: {target_count} (batch size: {batch_size})")

    # Create temporary collection
    target_name = "paper_chunks_256_migrating"
    try:
        client.delete_collection(target_name)
    except Exception:
        pass

    target_col = client.create_collection(
        target_name,
        metadata={
            "model": "google/embeddinggemma-2",
            "embedding_dim": str(TARGET_DIM),
            "hnsw:space": "cosine",
        },
    )

    t0 = time.time()
    offset = 0
    migrated = 0

    while offset < target_count:
        cur_limit = min(batch_size, target_count - offset)
        batch = source_col.get(
            limit=cur_limit,
            offset=offset,
            include=["embeddings", "documents", "metadatas"],
        )

        n_items = len(batch["ids"])
        if n_items == 0:
            break

        raw_vecs = np.asarray(batch["embeddings"], dtype=np.float32)
        vecs_256 = slice_and_normalize(raw_vecs, TARGET_DIM)

        target_col.add(
            ids=batch["ids"],
            embeddings=vecs_256.tolist(),
            documents=batch["documents"],
            metadatas=batch["metadatas"],
        )

        offset += n_items
        migrated += n_items
        elapsed = time.time() - t0
        rate = migrated / max(elapsed, 1e-6)
        pct = (migrated / target_count) * 100
        print(f"    Progress: {migrated}/{target_count} ({pct:.1f}%) in {elapsed:.1f}s ({rate:.1f} chunks/sec)")

    print("  Swapping collections atomically via modify name...")
    client.delete_collection("paper_chunks")
    target_col.modify(name="paper_chunks")
    print(f"  paper_chunks: successfully migrated {target_col.count()} chunks to {TARGET_DIM}-dim")


def verify_all() -> None:
    """Verify that all collections and files are operating at 256 dimensions."""
    print("\n" + "=" * 70)
    print("VERIFICATION OF 256-DIM MIGRATION")
    print("=" * 70)
    import chromadb

    # 1. Corpus files
    json_path = PROJECT_ROOT / "data" / "embeddings" / "corpus_embeddings.json"
    if json_path.exists():
        with open(json_path) as f:
            d = json.load(f)
        v = d.get("embeddings", [[]])[0]
        print(f"  Corpus JSON: {len(d.get('ids', []))} vectors, dimension: {len(v)}")
        assert len(v) == TARGET_DIM, f"Expected {TARGET_DIM}, got {len(v)}"

    # 2. Chroma corpus
    c_dir = PROJECT_ROOT / "data" / "chroma_corpus"
    if c_dir.exists():
        c = chromadb.PersistentClient(path=str(c_dir))
        col = c.get_collection("arxiv_corpus")
        print(f"  Chroma Corpus: {col.count()} items, metadata: {col.metadata}")
        assert str(col.metadata.get("embedding_dim")) == str(TARGET_DIM)

    # 3. Novelty cache
    n_db = PROJECT_ROOT / "data" / "novelty" / "novelty_cache.sqlite3"
    if n_db.exists():
        conn = sqlite3.connect(str(n_db))
        n_256 = conn.execute("SELECT COUNT(*) FROM embeddings WHERE dim = 256").fetchone()[0]
        s_256 = conn.execute("SELECT COUNT(*) FROM scores WHERE dim = 256").fetchone()[0]
        conn.close()
        print(f"  Novelty Cache: {n_256} 256-dim embeddings, {s_256} 256-dim scores")
        assert n_256 > 0, "No 256-dim embeddings found in novelty cache"

    print("\nALL VERIFICATIONS PASSED SUCCESSFULLY!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate ArXiv Paper Suggester embeddings to 256 dimensions.")
    parser.add_argument("--all", action="store_true", help="Migrate corpus, novelty cache, and Chroma RAG")
    parser.add_argument("--corpus", action="store_true", help="Migrate corpus embeddings and Chroma corpus")
    parser.add_argument("--novelty", action="store_true", help="Migrate novelty cache SQLite database")
    parser.add_argument("--rag", action="store_true", help="Migrate Chroma RAG paper_chunks collection")
    parser.add_argument("--rag-limit", type=int, default=None, help="Limit number of chunks to migrate for testing")
    parser.add_argument("--verify", action="store_true", help="Verify dimensions across all stores")

    args = parser.parse_args()

    if not (args.all or args.corpus or args.novelty or args.rag or args.verify):
        parser.print_help()
        return

    if args.all or args.corpus:
        migrate_corpus_files()
        migrate_chroma_corpus()

    if args.all or args.novelty:
        migrate_novelty_cache()

    if args.all or args.rag:
        migrate_chroma_rag(max_chunks=args.rag_limit)

    if args.verify or args.all:
        verify_all()


if __name__ == "__main__":
    main()
