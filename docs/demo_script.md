Here’s a tight **5-minute recording plan** you can follow with your terminal + results on screen (no slides).

---

## Setup before you hit record

1. Open a terminal in the project root (venv activated).  
2. Have these ready in tabs/windows:
   - Terminal for demos  
   - `results/phase5_results_table.md` (or `head` it in terminal)  
   - `results/phase5_error_analysis.md` (optional, for “what you’d do differently”)  
3. Do one dry run of the two demo commands so model load time doesn’t eat your minute.

---

## Script (≈5:00)

### 1. What you built — **~1:00** (with demo)

**Say:**
> I built an ArXiv paper suggester for three research themes: mechanistic interpretability, statistical mechanics of learning, and random matrix theory for neural nets.  
> You type an interest in plain English; the system retrieves papers with embeddings, then ranks them with four signals—semantic similarity, recency, citations, and a light author proxy—and shows a transparent score breakdown.  
> Here’s a live run.

**Show on screen:**
```bash
python src/recommend.py "mechanistic interpretability circuits" -k 5
```
Point at 2–3 lines: final score, `signals`, `contrib`.

Optional 10s second demo if time:
```bash
python src/recommend.py "random matrix loss surfaces" --since 2020 -k 3
```

**Close with:**
> Optional filters like year and diversity are wired in the same CLI; evaluation is against a frozen personal ranking I scored myself.

---

### 2. What you found — **~2:00** (evidence on screen)

**Show:** `results/phase5_results_table.md` (or paste the metrics table).

**Say:**
> I evaluated against 48 papers I scored 1–5 for personal importance.  
> **Citation baseline** only gets Spearman about **0.20** and almost no NDCG@10—raw citations don’t match what I care about.  
> **Embedding alone** gets the highest Spearman, about **0.68**—so my preferences are largely semantic.  
> The **full multi-signal ranker** sits at Spearman about **0.30**, but **NDCG@10 about 0.50**, better than embedding-only’s **0.35**.  
> So: embeddings track my overall order best; adding recency and citations improves the **top of the list**.  
> Ablations show citation is a double-edged signal—removing it changes correlation a lot.  
> **One sentence claim:** the multi-signal system beats pure citation on my personal ranking (0.30 vs 0.20 Spearman) and is much stronger on NDCG@10.

Optional 15s: flash one underrank from error analysis (e.g. NTK scored 5 by you, ranked low by the system).

---

### 3. One thing you’d do differently — **~2:00** (protect this)

Pick **one** main point and go deep. Recommended:

**Main point — evaluation design / overlap**

> If I restarted, I’d change how I evaluate.  
> Only about a third of my personal list overlaps the system’s top-40 for the broad eval query, so Spearman is on a small intersection.  
> I’d either (1) evaluate **per seed topic** with separate queries, or (2) force the pool to include all labeled papers and rank only among them, so correlation is measured on the full personal set.  
> I’d also **hold out part of my labels** to tune weights instead of using fixed 0.5 / 0.2 / 0.25 / 0.05—embedding-only winning on Spearman suggests the citation weight is too high for *my* taste.  
> Collection-wise, I’d plan around arXiv API limits earlier and lean harder on OpenAlex filters so less noise enters the 100-paper corpus.

**Alternate angle** (if you prefer):  
> I’d ship **embedding-heavy defaults** for “match my interests” and a separate “trending / high-cite” mode, instead of one blend that compromises Spearman for NDCG.

End with:
> The project is reproducible from the public repo: recommend, evaluate, and the numbers in `results/` match this claim.

---

## Timing cheat sheet

| Segment | Time | On screen |
|--------|------|-----------|
| Intro + demo | 0:00–1:00 | `recommend.py` output |
| Findings | 1:00–3:00 | Phase 5 metrics table |
| Do differently | 3:00–5:00 | Optional error analysis + you talking |

---

## After recording

1. Upload (YouTube unlisted / Loom / Drive).  
2. Add to **README** (top or bottom):

```markdown
## Demo video

[5-minute walkthrough](YOUR_LINK_HERE)
```

3. Moodle: **GitHub repo URL** + **video URL**.  
4. Final check: clone repo fresh (or new venv), run:

```bash
python src/recommend.py "mechanistic interpretability circuits" -k 3
python src/phase5_evaluation.py run
```

You’re ready to submit.