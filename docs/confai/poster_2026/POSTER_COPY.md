# ConfAI 2026 Academic Poster Copy (Verbatim)

**Event:** ConfAI 2026 · AI in Practice (Extended Track) · Poster Exhibition  
**Official Title:** Literature Discovery with Multi-Signal Ranking, Citation-Grounded Hybrid RAG, and 2D Novelty Geometry  
**Author:** Ajay Sharma (M.S. in AI)  
**Supervisor:** Dr. Subrat Panda  
**Affiliation:** Center for Artificial Intelligence, Plaksha University, Mohali, Punjab, India  
**GitHub Repository:** `https://github.com/NatalTiger26/ArXiv_Paper_Suggester-RAG`  
**Live Demo:** `https://nataltiger26-arxiv-paper-suggester-rag-srcapp-olkakz.streamlit.app/`  

---

## Band A: Header & Identity
- **Event:** ConfAI 2026 · AI in Practice (Extended Track) · Poster Exhibition
- **Title:** Literature Discovery with Multi-Signal Ranking, Citation-Grounded Hybrid RAG, and 2D Novelty Geometry
- **Authors:** Ajay Sharma (M.S. in AI) · Supervisor: Dr. Subrat Panda
- **Affiliation:** Center for Artificial Intelligence, Plaksha University, Mohali, Punjab, India
- **Institution Badge:** PLAKSHA UNIVERSITY (Center for Artificial Intelligence)
- **Interactive QR Access:** GitHub Code and Live Demo (Streamlit Cloud)

---

## Band B: Contributions & Headline
1. **Multi-Signal Discovery:** Ranks 5,752 catalog papers and explores 101k citation works via embeddings, 4-year half-life recency, citation saturation, and author prominence.
2. **Selective Ingest & Hybrid RAG:** Scaling causes 41.9% MRR loss. Selective BM25 plus dense RRF recovers 6% to 61% of lost MRR with 98.89% citation validity.
3. **2D Reference Cone Geometry:** NNLS cone projection decomposes innovation into Closest Parent (m), Synergy Blend (Δ), and Residual Gap (Res).

---

## Band C: Hero Architecture Pipeline
- **Title:** System Architecture: Literature Discovery, Selective Grounded RAG, and Innovation Subspaces
- **Pillar 1 (Multi-Signal Discovery):** arXiv Catalog (5,752) and Graph (101k works) feed 4 Heuristics (Dense Embeddings, Recency, Saturated Citations, Author Continuity).
  - Data Volume Arrow: Candidate pool (20-60 abstracts)
- **Pillar 2 (Selective Ingest & Hybrid RAG):** User-gated PDFs parsed into 553k passages. Hybrid RRF combines Dense Chroma (768d) and BM25 (k=60) for grounded answer synthesis.
  - Data Volume Arrow: Extracted bibliographic references
- **Pillar 3 (2D Novelty Geometry):** Reference Extractor resolves bibliographic citations. NNLS solver projects vectors into the reference cone yielding Closest Parent (m), Synergy Blend (Δ), and Residual Gap (Res).

---

## Band D: Main Three Pillars

### Column 1: Multi-Signal Discovery
- **Kicker:** Heuristic Ranking & Preset Shifts
- **Scoring Formula:**
  $$\text{Score}(q, d) = w_{\text{sem}} \cdot S_{\text{sem}} + w_{\text{rec}} \cdot 0.5^{(\Delta t / 4)} + w_{\text{cit}} \cdot C_{\text{score}} + w_{\text{auth}} \cdot A_{\text{score}}$$
- **Weight Presets Breakdown:**
  - Balanced: 50% sem · 20% rec · 25% cit · 5% auth
  - Precision: 65% sem · 10% rec · 20% cit · 5% auth
  - Recent: 45% sem · 35% rec · 15% cit · 5% auth
  - Citations: 35% sem · 10% rec · 50% cit · 5% auth
- **Worked Query ("vision transformer"):**
  1. #1 2604.133 (Transcoders, Score: 0.743)
  2. #2 2604.144 (Circuits ViT, Score: 0.740)
  3. #3 2605.282 (Explaining ViT, Score: 0.727)
  4. #4 2604.180 (Attention Prop, Score: 0.722)
  5. #5 2601.141 (Token Attr, Score: 0.718)
- **Preset Dynamics:** Citation heavy elevates classic foundations like ViT (2510.06070). Recent rewards 2026 preprints (2604.13304).

### Column 2: Hybrid RRF RAG
- **Kicker:** Scaling Degradation & Attribution Integrity
- **Distractor Scaling Degradation (Pure Dense Search):**
  - N=10: MRR 0.786, Hit@1 0.725, Recall@6 0.888
  - N=100: MRR 0.753, Hit@1 0.700, Recall@6 0.833
  - N=1,000: MRR 0.638, Hit@1 0.575, Recall@6 0.746
  - N=5,663 (Full): MRR 0.457 (-41.9%), Hit@1 0.350 (-51.7%), Recall@6 0.638
- **Dense vs Hybrid RRF (Paired Evaluation on 80 Questions):**
  - MRR: Dense 0.469 vs Hybrid 0.574 (Paired Diff: +0.105, 95% bootstrap CI: [+0.018, +0.197])
  - Hit@1: Dense 0.350 vs Hybrid 0.488 (Paired Diff: +0.138, 95% bootstrap CI: [+0.038, +0.250])
  - Recall@6: Dense 0.637 vs Hybrid 0.688 (Paired Diff: +0.050, 95% bootstrap CI: [-0.062, +0.162])
- **Attribution Integrity & Calibration Facts:**
  - 98.89% Citation Validity (89 of 90 claims cite valid chunk indices, confirming passage existence).
  - 65.0% Refusal Recall (13 of 20 unanswerable queries correctly declined).
  - Gold set LLM-generated against 10-paper index. Added distractors create valid alternative answers scored as misses.

### Column 3: 2D Novelty Geometry
- **Kicker:** Subspace Innovation & Null Control
- **Non-Negative Least Squares Formulation:**
  $$\lambda^* = \arg\min_{\lambda \ge 0} \| x - Y \lambda \|^2, \quad \hat{x} = Y \lambda^*$$
  $$m = \max_j (x^T y_j), \quad \Delta = R^2_+ - \max(m, 0)^2, \quad \text{Res} = \sqrt{1 - R^2_+}$$
- **2D Reference Cone Geometry:** Visualizes Target vector $x$, Dominant Ancestor $y_{m^*}$ at $(1, 0)$, secondary ancestors ($y_2, y_3$), cone projection $\hat{x}$, and orthogonal residual gap vector Res.
- **Cohort Distribution (5,707 valid papers):**
  - High Synergy: 5,176 papers (90.7%)
  - Moderate Synthesis: 489 papers (8.6%)
  - Linear Extension: 34 papers (0.6%)
  - Paradigm Shift: 8 papers (0.1%)
  - Diffused: 0 papers (0.0%)
- **Exemplar Ancestor Decomposition:**
  - Grad-CAM++ (1710.11063): 18/18 refs, m=0.938, Δ=0.047, Res=0.270 (Grad-CAM share: 49.7%).
  - BIG-bench (2206.04615): 80/706 refs, m=0.774, Δ=0.138, Res=0.513 (BERT share: 24.7%).
- **Empirical Reference Geometry:**
  - Real Paper Synergy: $\mu_{\Delta} = 0.088$, Median $= 0.088$, 95% CI: $[0.087, 0.088]$.
  - Isotropic 768d Null Control: $\mu_{\Delta} = 0.017$ (5.13x synergy enhancement ratio).

---

## Band E: Bottom Scientific Cards
- **Limitations (AI in Practice):**
  - Refusal: Declines 23.8% of queries under weak evidence.
  - Lexical: Dense search misses exact theorem IDs without BM25.
  - MPS Memory: 1 GB embedding weights require proactive cache release.
- **Conclusions:**
  - Gating: Scaling drops Hit@1 by 51.7%, confirming gating is vital.
  - Hybrid: Reciprocal rank fusion yields paired MRR gain of +0.1052.
  - Synergy: Real paper synergy exceeds isotropic null baseline by 5.0x.
- **Future Work & Contact:**
  - Learned Priors: Query-dependent section attention weights.
  - Lineage: Multi-hop semantic traversal and co-citations.
  - Contact: Ajay Sharma, Dr. Subrat Panda (Plaksha University).

---

## Band F: Footer References & Footnote
- **References:** [1] Vaswani et al., Attention, NeurIPS 2017. [2] Cormack et al., RRF, SIGIR 2009. [3] Selvaraju et al., Grad-CAM, ICCV 2017. [4] Srivastava et al., BIG-bench, 2022.
- **Footnote:** Snapshots: N=5,657 (final_eval.json) vs N=5,663 (sweep). RRF (k=60, α=0.5) and section priors were heuristic defaults, not tuned on eval questions. Gold questions LLM-generated against 10-paper index.
