# Reference-conditional profiles (`src/novelty`)

Offline analytics package. **Does not** modify recommend, RAG, Streamlit, or Chroma.

**Framing:** numbers describe *geometry relative to cited works in one embedding space*.  
Not a validated scientific “novelty score.” Prefer the name **reference-conditional profile**.

---

## What was implemented

| Layer | Behavior |
|-------|----------|
| **Refs** | Semantic Scholar → OpenAlex backup |
| **Cache** | SQLite under `data/novelty/` (abstracts + vectors + edges + scores) |
| **Ids** | Canonical `arxiv:` / `s2:` / `openalex:` / `doi:`; version stripped |
| **Dedupe** | Merge duplicate canonical ids; drop self-citations |
| **Cap** | Deterministic seeded cap (prefer refs with abstracts), default 80 |
| **Coverage gate** | **Refuse** scoring unless ≥5 refs with abstracts **and** ≥**70%** of refs have abstracts |
| **Encode** | Same path as RAG (`embed_store.embed_texts` + title-title-abstract text format) |
| **Centering** | Cohort mean of cached vectors for `(model, dim)`, then re-unit (if ≥20 vectors) |
| **Metrics** | \(m, a, c, D, R^2_+, \Delta\), attribution \(\hat\lambda\) |
| **\(d_{\mathrm{eff}}\)** | Participation ratio on refs + optional cache-wide estimate |
| **Chance** | `chance_R2_ols_mean ≈ n / d_eff` reported beside \(R^2_+\) |
| **Null** | Other focals’ ref sets; **matched \(n\)**; **no overlapping refs**; year/category preference; **refuse** if pool &lt; 10 |
| **Baseline script** | Correlations of scores vs \(\log n_{\mathrm{refs}}\) and year |
| **Unit tests** | Synthetic identity / blend / orthogonal / opposite / perm / dup / chance / NNLS |

### Not implemented

Streamlit UI, trained InfoNCE/Uzzi models, OT, LOF-over-arXiv, CD index, LLM facets, citation-intent weights, chunk-level “what’s new” Ask mode, promoting refs into the recommend index, ref PDFs.

---

## How to use

```bash
# 1) Algebra sanity (no network)
uv run python -m src.novelty.cli test
# or: uv run python -m src.novelty.test_metrics

# 2) Bibliography coverage only
uv run python -m src.novelty.cli coverage 1806.07572 2501.16496 2304.14997

# 3) Score (first runs: skip null until many focals exist)
uv run python -m src.novelty.cli score 2501.16496 --no-null
uv run python -m src.novelty.cli score 1806.07572 2304.14997 --no-null

# 4) After ≥10 successful scores, enable null
uv run python -m src.novelty.cli score 2501.16496

# 5) Inspect / cache size
uv run python -m src.novelty.cli show 2501.16496
uv run python -m src.novelty.cli stats

# 6) Mechanical baseline (does geometry just track ref-count?)
uv run python -m src.novelty.cli baseline
uv run python -m src.novelty.baseline --json-out results/novelty_baseline.json
```

Optional env:

- `SEMANTIC_SCHOLAR_API_KEY` — higher S2 rate limits  
- `OPENALEX_MAILTO` — polite OpenAlex identification  

---

## Reading outputs

| Field | Meaning |
|-------|---------|
| `m` | Max cosine to a single reference (high → extension-like) |
| `a` | Alignment to reference centroid |
| `D_median` / `D_q90` | How spread the references are |
| `R2_plus` | How much of the paper vector a **non-negative blend** of refs can explain |
| `Delta` | Extra explained energy beyond the best single parent (blend gain) |
| `chance_R2_ols_mean` | Rough isotropic baseline \(n/d_{\mathrm{eff}}\); if \(R^2_+\) is near this for *everyone*, geometry is uninformative |
| `null.percentiles` | Fraction of matched-n shuffled ref-sets scoring **below** the observed value |
| `typology_hint` | Crude label only (`extension_like`, `blend_like`, `outlier_like`, `mixed`, `near_chance`) |
| `attribution` | Ranked parents by NNLS weight — often the most useful qualitative output |

**Do not interpret a single paper.** Score 15–30 papers; look at distributions vs chance and vs \(\log n_{\mathrm{refs}}\).

**Kill criterion (pre-register):** if profiles do not beat \(\log n_{\mathrm{refs}}\) + year on paired contrasts you care about, stop and write a negative result.

---

## Correctness notes (review feedback addressed)

1. **Centering** — same cohort mean subtracted from focal and refs when enough cache vectors exist (`--no-center` to disable).  
2. **\(d_{\mathrm{eff}}\)** — reported on refs and optionally on cache.  
3. **Null matched on \(n\)** — each null set is subsampled to the focal’s \(n\).  
4. **Overlap** — refs shared with the focal bibliography are removed from null donors.  
5. **Min pool** — percentiles refused below 10 usable nulls; CLI prints reason.  
6. **Dedup** — canonical merge + self drop.  
7. **70% coverage** — enforced in code, not only docs.  
8. **Cap** — seeded deterministic preference for abstract-bearing refs.  
9. **Encode path** — shared with RAG embed_store.  
10. **Baseline script** — exists so the study is not “by hand only.”

Still on you: network checks (S2 counts vs website), OpenAlex fallback keys, 429 behavior under load, paired sanity labels with a supervisor.

---

## Branch / ConfAI safety

- Only adds `src/novelty/`, `docs/NOVELTY.md`, `data/novelty/` at runtime.  
- No changes required to `app.py`, `rag/`, or recommend for this feature.  
- Optional dependency: `scipy` (already common via the scientific stack) for NNLS.

