# ConfAI 2026 Poster — Print & Design Notes

**File:** `docs/confai/ConfAI2026_Poster_A1_Vertical.pdf`  
**Size:** A1 portrait — **594 × 841 mm** (vertical, as requested by Dr. Panda)  
**Novelty track:** **not** included (per ConfAI scope)

---

## What to send / print

1. Open the PDF and check all text is sharp at 100% zoom.
2. Send to **Dr. Subrat Panda** for approval before Makers Lab print.
3. Print **A1**, **portrait**, **full bleed if possible**, or with minimal white margin.
4. Prefer **matte** paper if available (less glare under conference lights).
5. Bring a laptop for the **live demo** during the poster session.

**Demo:** https://nataltiger26-arxiv-paper-suggester-rag-srcapp-olkakz.streamlit.app/  
**Code:** https://github.com/NatalTiger26/ArXiv_Paper_Suggester-RAG

---

## Design principles (NeurIPS-level bar)

| Principle | How this poster applies it |
|-----------|----------------------------|
| **Title readable from 2–3 m** | Large white title on navy header |
| **One clear story** | Problem → loop → method → results → limits → takeaways |
| **Numbers first** | KPI strip + 2×2 metric cards |
| **Honest science** | “Measured limits” box; curated-index caveat on MRR |
| **Low ink chaos** | Two columns, cards, restrained orange accent (not rainbow) |
| **Approachable** | Plain language in problem/loop; no unexplained jargon walls |
| **Footer actionable** | Demo + GitHub URLs, print size reminder |

### Visual system
- **Navy `#0F172A`** — authority, header/footer  
- **Accent orange `#C2410C`** — section labels, KPI numbers (warm, not startup-neon)  
- **Teal soft cards** — engineering (secondary, not competing with results)  
- **Slate body text** — readable at arm’s length  
- **No novelty / research-side metrics** on this print

### Typography hierarchy
1. Title (largest)  
2. Section labels (uppercase + accent underline)  
3. Metric numerals  
4. Body 8–9 pt (A1 scale — readable when standing in front of the board)

---

## Content map (reading order)

```
HEADER     Conference · Title · Author · Supervisor
KPIs       Library scale · chunks · gold set · MRR
LEFT       1 Problem → 2 System loop → 3 Method → Design choice
RIGHT      4 Evaluation → 5 Engineering → 6 Takeaways
FOOTER     Demo URL · GitHub · A1 print note
```

---

## Numbers used (honest framing)

| Claim | Framing to say aloud |
|-------|----------------------|
| ~3,830 full-text papers | Local RAG index after stable re-ingest |
| ~374k chunks | Section-aware chunks in Chroma |
| ~100 gold questions | Evaluation set size |
| 0.75 mean MRR | **On a curated full-text index** (early protocol); CI ≈ 0.66–0.83 |
| 0.85 Recall@k / 0.69 Hit@1 / 0.95 cite validity | Same evaluation context |

If Dr. Panda prefers the **nomination** figures (4,479 library / 68,027 chunks), say so and we can swap the KPI strip only — do **not** mix curated-eval metrics with “whole corpus” without saying so.

---

## 60-second pitch (poster session)

1. **Problem:** search lists + ungrounded chat.  
2. **Loop:** rank → select ingest → ask with citations.  
3. **Difference:** inspectable ranking weights; selective full text; coverage honesty.  
4. **Evidence:** gold set, bootstrap CIs, dim sweep; limits stated.  
5. **Touch the demo** on the laptop.

---

## Pre-flight checklist before Makers Lab

- [ ] PDF opens; no missing fonts (Helvetica is standard)  
- [ ] Supervisor approved wording  
- [ ] Demo loads offline/online on presentation laptop  
- [ ] QR optional: you can add a QR to the demo URL in Illustrator/Keynote if the lab supports overlay  
- [ ] Backup: USB with this PDF + `git pull` of the repo  

---

## If you need a revision

Typical faculty feedback and how to respond:
- **“Too much text”** → cut method bullets; keep problem + loop + metrics  
- **“Show a screenshot”** → add one UI still in the right column (replace part of engineering)  
- **“Bigger title”** → already prioritized; can grow another 2–4 pt if needed  
