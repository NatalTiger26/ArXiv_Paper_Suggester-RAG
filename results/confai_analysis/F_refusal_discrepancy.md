# F. Refusal 0.39/0.60 vs 0.45/0.45 — what the files say

## Copy the curated snapshot (on your machine)

```bash
mkdir -p results/rag_eval/curated_2026-10-08
# From your Drive / backup of the 2026-10-08 final_eval.json:
cp "/path/to/Drive/snapshot/final_eval.json" \
   results/rag_eval/curated_2026-10-08/final_eval.json
```

Also keep the human report that still matches the curated run:

- `results/rag_eval/last_eval.md` — **2026-10-08**, index **10 / 1725**, refusal **0.45 / 0.45**

## Important: `last_eval.json` in the current tree is NOT the curated run

In this project snapshot:

| File | index_papers | refusal P/R | mean MRR | Notes |
|------|-------------:|-------------|----------|-------|
| `results/rag_eval/last_eval.md` | **10** | **0.45 / 0.45** | **0.75** | Curated-era markdown report |
| `results/rag_eval/last_eval.json` | **3830** | **0.3125 / 0.50** | **0.2833** | Same numbers as full-library `final_eval.json` (overwritten later) |
| `results/rag_eval/final_eval.json` | **3830** | **0.3125 / 0.50** | **0.2833** | Full-library run 2026-10-20 |
| `results/rag_eval/advanced_analysis.json` | **10** | confusion TP9 FP11 FN11 → **0.45 / 0.45** | 0.75 | Curated-era analysis |

So **0.45/0.45** is the curated-era **stored** grader result (`last_eval.md` + `advanced_analysis.json`).

## Where did 0.39 / 0.60 come from?

That pair was **not** found as the summary line in `last_eval.md`. It matches an **alternate row-level audit** (e.g. Gemini Phase-0 inventory / manual recount) that used a **different** definition of “did refuse”, for example:

- Counting only certain substrings as refusal, or
- Reclassifying “cited answers that say not covered…” as refusals (inflating FP → lower precision ~0.39), and/or
- Counting “there is no information” as refusal when the **repo grader** in `src/final_eval.py` at the time of the curated run used a slightly different key list.

Repo grader keys now (`src/final_eval.py` `looks_like_refusal`):

`not covered`, `do not contain`, `does not provide`, `no relevant`, `not enough information`, `ingested papers do not`, `cannot answer`, `outside the scope`, `not present in`, `i don't know`, `i do not know`, `unable to answer`.

**Poster guidance:** cite **one** source file.

- Curated protocol: **P=0.45, R=0.45** from `results/rag_eval/last_eval.md` (and `advanced_analysis.json` confusion).
- Full library: **P=0.3125, R=0.50** from `results/rag_eval/final_eval.json`.
- Do **not** mix 0.39/0.60 unless you paste the exact Drive JSON and recompute TP/FP from its rows.

## After you copy curated JSON

```bash
uv run python - <<'PY'
import json
from pathlib import Path
p = Path("results/rag_eval/curated_2026-10-08/final_eval.json")
d = json.loads(p.read_text())
s = d.get("summary", d)
print({k: s.get(k) for k in [
  "index_papers","index_chunks","refusal_precision","refusal_recall",
  "refusal_confusion","mean_mrr","timestamp"] if isinstance(s, dict)})
rows = d.get("rows") or []
if rows:
  tp=sum(1 for r in rows if r.get("should_refuse") and r.get("did_refuse"))
  fp=sum(1 for r in rows if not r.get("should_refuse") and r.get("did_refuse"))
  fn=sum(1 for r in rows if r.get("should_refuse") and not r.get("did_refuse"))
  print("from rows", dict(TP=tp,FP=fp,FN=fn,P=tp/(tp+fp) if tp+fp else None,R=tp/(tp+fn) if tp+fn else None))
PY
```
