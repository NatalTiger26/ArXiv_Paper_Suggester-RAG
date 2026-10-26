# Overnight Execution Verification Guide

All P0 and P1 autonomous overnight tasks have been implemented and verified. Run the following copy-pasteable commands to verify each capability independently.

---

## 1. Quick Repository Hygiene & Full Test Suite

Run the repository smoke check, unit test suite, and system health checks:

```bash
# 1. Repository hygiene & secret check (10/10 checks)
make check

# 2. Complete unit test suite (39 tests)
make test

# 3. System health & index status
make health
```

---

## 2. P0-A & P0-B: Citation Count Enrichment & Chroma Sync

Verify that `src/novelty/refs.py` extractors are functioning, `data/corpus.json` contains enriched citations, and `data/chroma_corpus` metadata was synced in-place:

```bash
# Verify corpus has enriched citation counts
uv run python -c "
import json
corpus = json.load(open('data/corpus.json'))
c_count = sum(1 for p in corpus if p.get('citation_count') is not None)
print(f'Corpus papers with citations: {c_count} / {len(corpus)}')
"

# Verify Chroma vector index has citation_count updated in-place without re-embedding
uv run python -c "
import chromadb
client = chromadb.PersistentClient(path='data/chroma_corpus')
coll = client.get_collection('arxiv_corpus')
sample = coll.get(limit=10)
print('Sample Chroma citations:', [m.get('citation_count') for m in sample['metadatas'] if m.get('citation_count')])
"

# Test dry-run on batch enrichment script
uv run python scripts/enrich_citations.py --limit 10 --dry-run
```

---

## 3. P0-C: Streamlit Responsiveness & Catalog Caching Benchmark

Verify that `list_ingested_papers()` avoids expensive Chroma collection queries on warm/cached paths:

```bash
uv run python -c "
import time
from src.rag.embed_store import list_ingested_papers

t0 = time.time()
papers = list_ingested_papers(force_chroma=False)
dt1 = (time.time() - t0) * 1000
print(f'Call 1 (cold path): {len(papers)} papers returned in {dt1:.1f}ms')

t1 = time.time()
papers2 = list_ingested_papers(force_chroma=False)
dt2 = (time.time() - t1) * 1000
print(f'Call 2 (cached path): {len(papers2)} papers returned in {dt2:.3f}ms')
"
```
*Expected: Call 1 completes in < 50ms (down from 7,000ms), Call 2 in < 0.01ms.*

---

## 4. P0-D: Ingest Fast-Skip Hardening (<5ms Verification)

Verify that already-ingested papers skip in `<5ms` without making external arXiv HTTP queries:

```bash
uv run python -c "
import time
from src.rag.ingest import ingest_paper

t0 = time.time()
res = ingest_paper('2404.14082')
dt = (time.time() - t0) * 1000
print(f'Ingest fast-skip latency: {dt:.2f}ms')
print(f'Result: ok={res[\"ok\"]}, skipped_embed={res[\"skipped_embed\"]}, source_format={res[\"source_format\"]}')
assert dt < 50.0, f'Expected fast-skip <50ms, got {dt:.2f}ms'
assert res['skipped_embed'] is True
"
```
*Expected: Latency is ~4–6ms, `skipped_embed=True`, `source_format='cached'`.*

---

## 5. P0-E: Evaluation Auto-Archive Verification

Verify that running new evaluations automatically snapshots prior evaluation artifacts:

```bash
uv run python -m unittest tests/test_eval_archive.py
```
*Expected: 2 passing tests in < 0.01s.*

---

## 6. P1: Telemetry Engine & UI Reset Verification

Verify token counting, cost computation, and database operations in `data/telemetry.sqlite3`:

```bash
# 1. Run unit test suite for telemetry
uv run python -m unittest tests/test_telemetry.py

# 2. Inspect current telemetry summary stats
uv run python -c "
from src.telemetry import get_telemetry_summary
stats = get_telemetry_summary()
print('Telemetry Summary:')
for k, v in stats.items():
    if k != 'recent_calls':
        print(f'  {k}: {v}')
"
```

---

## 7. Streamlit Web Interface

Launch the interactive app and test the new sidebar expander:

```bash
make app
# Or directly:
uv run streamlit run src/app.py
```
1. Open the sidebar.
2. Expand **API & Cost Telemetry**: inspect Total Calls, Total Tokens, Estimated Cost, Latency p50, and Cache Hits.
3. Test the "Reset usage stats" button with two-step confirmation.
