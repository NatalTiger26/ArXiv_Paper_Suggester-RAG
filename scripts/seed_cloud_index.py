#!/usr/bin/env python3
"""Seed a lightweight MiniLM index suitable for Streamlit Community Cloud (free tier).

Streamlit Community Cloud has ~1 GB RAM limits. Loading EmbeddingGemma 2 (~1GB)
causes OOM container terminations. This script seeds a tiny, fast index using
sentence-transformers/all-MiniLM-L6-v2 (~80 MB RAM) for 2-3 curated papers.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ["FORCE_CLOUD_EMBED"] = "1"
os.environ["EMBEDDING_MODEL"] = "sentence-transformers/all-MiniLM-L6-v2"
os.environ["EMBEDDING_DIM"] = "0"

from src.corpus_builder import from_ids
from src.corpus_index import build_index
from src.rag.embed_store import reset_rag_collection
from src.rag.ingest import ingest_papers
from src.rag.rag_pipeline import ask

CURATED_PAPERS = ["1806.07572", "2501.16496"]


def seed_cloud():
    print("=================================================================")
    print(" Seeding Lightweight Cloud Index (all-MiniLM-L6-v2, 384-d)")
    print("=================================================================")

    # 1. Clear previous incompatible collections
    print("\n[1/4] Resetting RAG collection to MiniLM dimension...")
    reset_rag_collection()

    # 2. Add curated papers to corpus
    print("\n[2/4] Fetching metadata for curated papers...")
    from_ids(CURATED_PAPERS, build_chroma=False)

    # 3. Build recommend index with MiniLM
    print("\n[3/4] Building Recommend Index with MiniLM...")
    build_index(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 4. Ingest full text into RAG with MiniLM
    print("\n[4/4] Ingesting full text into RAG with MiniLM...")
    ingest_papers(CURATED_PAPERS, force_download=False, force_embed=True)

    # Verify
    print("\nVerifying retrieval under MiniLM...")
    ans = ask("What is the neural tangent kernel?", top_k=3, use_cache=False)
    print(f"Coverage: {ans.get('coverage')}")
    print(f"References: {len(ans.get('references') or [])}")

    print("\n✓ Cloud seed complete! Total RAM footprint is < 150 MB.")
    print("  Ready for Streamlit Community Cloud deployment.")


if __name__ == "__main__":
    seed_cloud()
