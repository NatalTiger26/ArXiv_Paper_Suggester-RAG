# Evaluation Report — ArXiv Paper Suggester

**What was measured:** retrieval quality, answer refusal behaviour, citation hygiene, latency, and embedding-dimension trade-offs on a fixed 100-question gold set.  
**Harness:** `src/final_eval.py`  
**Index:** 10 full-text papers · 1725 chunks · EmbeddingGemma 2 (768-d)  
**Runs:** (1) retrieval-only · (2) full generation with cache **off** (cold)  
**Date of measured results:** 2026-10-08

---

## 1. What the system is

A small research assistant that:

1. Builds an arXiv paper library (search or explicit IDs).
2. Ranks papers for a research interest.
3. Ingests full text (PDF → Markdown → chunks → local embeddings → Chroma).
4. Answers questions **only** from those papers, with numbered citations.

| Piece | Choice | Why |
|-------|--------|-----|
| Embeddings | Local EmbeddingGemma 2 | No per-query embedding API cost; technical-domain model |
| Generation | Gemini (+ optional failsafe) | Strong answers; failsafe if quota fails |
| Vectors | Chroma | Simple persistent store for a course project |

This report does **not** claim the system is safe for unsupervised clinical, legal, or financial use.

---

## 2. How evaluation was done

| Step | Detail |
|------|--------|
| Gold set | 100 questions: definitions, paraphrases, methods, open problems, identifiers, comparisons, multi-hop, and **20 deliberately unanswerable / edge** items |
| Labels | Answerable items carry expected arXiv IDs; unanswerable items are flagged `should_refuse` |
| Retrieval scoring | On 80 labelled items: Recall@k, MRR, Hit@1 |
| Generation scoring | On all 100, cold (no semantic cache): refusal P/R, citation index validity, latency |
| Uncertainty | Bootstrap 95% CIs on means; Wilson 95% CIs on proportions |
| Not run | Human–judge Cohen’s κ (needs hand labels); gold-context oracle ablation |

**How to re-run**

```bash
uv run python src/final_eval.py --retrieval-only
uv run python src/final_eval.py --with-generation --no-cache
```

---

## 3. Metric glossary (read once)

| Metric | Plain meaning |
|--------|----------------|
| **Recall@k** | Did *any* expected paper appear in the top-k results? |
| **MRR** | How high was the first expected paper? (1.0 = rank 1, 0.5 = rank 2, …) |
| **Hit@1** | Was the *top* paper one of the expected ones? |
| **95% CI** | Range that would often contain the true mean if we repeated the experiment on similar questions. Overlapping CIs ⇒ do not claim a clear winner. |
| **Refusal recall** | Of questions that *should* be declined, how often we declined. |
| **Refusal precision** | Of our declines, how often declining was correct. Need **both**. |
| **Citation validity** | Every `[n]` in the answer points at a real retrieved source (code check). Not the same as “factually faithful.” |
| **Latency p50 / p95** | Half of requests finish under p50; 95% finish under p95. p95 is what users feel on bad days. |

---

## 4. Headline results

### 4.1 Quality

| Metric | Observed | 95% CI | Gate | Result |
|--------|----------|--------|------|--------|
| MRR | **0.75** | 0.66 – 0.83 | ≥ 0.55 | Pass |
| Recall@k | **0.85** | 0.76 – 0.93 | ≥ 0.55 | Pass |
| Hit@1 | **0.69** | 0.59 – 0.79 | — | — |
| Citation validity | **0.95** | 0.89 – 0.98 | ≥ 0.95 | Pass |
| Refusal recall | **0.60** | 0.39 – 0.78 | — | — |
| Refusal precision | **0.39** | 0.24 – 0.56 | ≥ 0.50 | **Fail** |

**Expected vs observed (quality)**

| Expectation before measuring | What we saw |
|------------------------------|-------------|
| With ~10 in-domain papers, Recall should land in a “usable demo” band (~0.7+) | **0.85** — met |
| Semantic questions (paraphrase / open problems) stronger than ID lookups | Confirmed (see §6) |
| Refusal would be weaker than retrieval on a small library | Confirmed: precision **0.39** |
| Citation indices mostly valid if the prompt forces `[n]` | **0.95** — met |

**In one sentence:** retrieval is demo-ready; refusal is not deployment-ready.

### 4.2 Speed and cost shape

Latency is a first-class metric here, not only a pass/fail bit.

| Stage | p50 | p95 | What dominates |
|-------|----:|----:|----------------|
| **Retrieval only** (embed query + Chroma) | **~33 ms** | **~39 ms** | Local embedding on MPS; vector search is cheap |
| **Full ask** (retrieve + Gemini, cold cache) | **~9.5 s** | **~15.6 s** | LLM generation |

| Related | Value | Meaning |
|---------|------:|---------|
| Cache hit rate (cold run) | **0.0** | Numbers are not inflated by replayed answers |
| Embeddings | Local | No per-query embedding API bill |
| Generation | API | Cost and latency scale with answer length and provider |

**Expected vs observed (latency)**

| Expectation | Observed |
|-------------|----------|
| Retrieval ≪ 1 s on a laptop GPU/MPS | **~30–40 ms** — easily met |
| Interactive Q&A often 5–20 s with a remote LLM | **p50 ≈ 9.5 s, p95 ≈ 15.6 s** — normal for unoptimized API calls |
| Cached runs can look “instant” and hide true cost | Cold run avoids that distortion |

**How to talk about this in a viva**

- Users feel **p95**, not the average of happy path only.
- Speeding up the system further means **generation** (shorter context, streaming, smaller model for easy questions)—not faster Chroma.
- A fair cost narrative: embeddings are free at query time; each full answer spends one Gemini call (plus rare failsafe).

### 4.3 Overall gate

| Check | Pass? |
|-------|-------|
| Retrieval quality | Yes |
| Citation hygiene | Yes |
| Latency (p95 ≤ 20 s for generation) | Yes |
| Refusal precision | **No** |
| **Overall** | **Fail on refusal only** |

Failing overall while passing retrieval is intentional honesty: good for a **portfolio demo**, not for unsupervised production.

---

## 5. Refusal in more detail

|  | Did refuse | Did not |
|--|----------:|--------:|
| Should refuse | **12** (TP) | **8** (FN) |
| Should answer | **19** (FP) | **61** (TN) |

- **8 false answers** — model spoke when the gold item was out of scope or adversarial.
- **19 over-refusals** — model declined questions the library could support (often low coverage score or cautious wording).

**Takeaway:** improving refusal is a **prompt / threshold** problem, not an “ingest more PDFs” problem.

---

## 6. Where retrieval is strong or weak

### By question kind

| Kind | n | MRR | Recall | Hit@1 | Reading |
|------|--:|----:|-------:|------:|---------|
| open_problem | 13 | 0.95 | 1.00 | 0.92 | Strong — library is MI-heavy |
| paraphrase | 25 | 0.91 | 0.96 | 0.88 | Strong — dense match works |
| definition | 12 | 0.69 | 0.75 | 0.67 | Adequate |
| method | 14 | 0.63 | 0.71 | 0.57 | Mixed |
| comparison | 5 | 0.52 | 0.80 | 0.40 | Small n; wide uncertainty |
| identifier | 10 | 0.42 | 0.70 | **0.20** | Weak at rank-1 — IDs favor lexical match |

### By difficulty

| Difficulty | n | MRR | Recall | Hit@1 |
|------------|--:|----:|-------:|------:|
| easy | 21 | 0.80 | 0.86 | 0.76 |
| hard | 24 | 0.75 | 0.83 | 0.71 |
| medium | 35 | 0.72 | 0.86 | 0.63 |

---

## 7. Failures (what to fix next)

| Pattern | Count | Likely fix |
|---------|------:|------------|
| Right paper in top-k but not #1 (`ranking`) | 13 | Light rerank or hybrid score |
| Expected paper never in top-k (`missing_or_ranking`) | 12 | Often ranking; check if PDF is already ingested |
| Answered when should refuse (`false_answer`) | 8 | Stricter coverage gate / prompt |
| Refused when should answer (`over_refuse`) | 6 | Lower over-cautious threshold |

Papers most often missing from top-k include **2304.14997**, **1806.07572**, **2501.16496** — these are largely **already ingested**, so the issue is ranking, not a missing download.

**Earlier improvement (story for the report):** moving from ~4 to **10** papers raised MRR from ~**0.62 → 0.75** and Recall from ~**0.65 → 0.85**. That is the Lab 5 pattern: measure → find missing content → fix one cluster → re-measure.

---

## 8. Embedding dimension (128 / 256 / 512 / 768)

Sweep on **library titles+abstracts** (not the full-text RAG index):

| Dim | MRR | 95% CI | Recall@10 |
|----:|----:|--------|----------:|
| 128 | 0.59 | 0.50 – 0.68 | 0.85 |
| 256 | 0.63 | 0.54 – 0.73 | 0.86 |
| **512** | **0.64** | 0.55 – 0.73 | 0.86 |
| 768 | 0.63 | 0.53 – 0.72 | 0.86 |

**Reading:** 512 is slightly best on the point estimate; **CIs overlap** with 768, so this is an optional storage/speed choice, not a proven quality win. All **RAG** numbers in this report used **768**. Changing dim requires resetting the Chroma collection and re-ingesting.

---

## 9. What we did not claim

- **Cohen’s κ** — no human faithfulness labels in this run.
- **Oracle (gold-context) answers** — would separate pure generation error from retrieval error more cleanly.
- **$/query from production traces** — embeddings local; generation cost is “one LLM call per cold ask.”
- **Lab 6 security block rates** — no privileged tool agent in the shipped path.

---

## 10. Practical recommendation

| Use | Verdict |
|-----|---------|
| Portfolio / lab demo on this MI–NTK library | **Yes** — retrieval and citations support it |
| Unsupervised “never wrong / always refuse OOS” assistant | **No** — refusal precision 0.39 |
| Latency-sensitive UI | Show streaming; expect ~10 s class answers unless you add a small-model fast path |

**Next experiments (one at a time)**

1. Hybrid or rerank for **identifier** / residual ranking misses → re-check MRR and Hit@1.  
2. Tighter refusal threshold → re-check precision/recall **cold**.  
3. Optional: keep an older baseline JSON and re-run for paired p-values after a change.

---

## 11. Artifacts

| File | Contents |
|------|----------|
| `results/EVALUATION_REPORT.md` | This document |
| `results/rag_eval/final_eval.json` | Per-question scores + summary |
| `results/rag_eval/dim_sweep.json` | Dimension experiment |
| `results/rag_eval/gold_questions.json` | Gold set (n=100) |
| `src/final_eval.py` | Single evaluation script |

---

*Retrieval is strong enough to demo; latency is LLM-bound; refusal still needs work before any unsupervised trust story.*
