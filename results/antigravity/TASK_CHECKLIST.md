# Autonomous Task Checklist

| Priority | Task ID | Description | Status |
| :--- | :--- | :--- | :--- |
| **P0-A** | Citation Extractors | Fix S2 `citationCount` and OpenAlex `cited_by_count` in `refs.py` | DONE |
| **P0-B** | Batch Citation Enrichment | `scripts/enrich_citations.py` batch S2, atomic corpus write, safe Chroma sync | DONE |
| **P0-C** | Streamlit Responsiveness | Lazy paper catalog, avoid full Chroma scans on rerun, input forms | DONE |
| **P0-D** | Ingest Skip-if-Done | Verify markdown + chunks present $\rightarrow$ $<5\text{ms}$ skip; ensure `--force` overrides work | DONE |
| **P0-E** | Eval Archive Safety | Confirm `archive_previous_eval()` safely copies prior `final_eval.json` before overwrite | DONE |
| **P0-F** | Protocol Deliverables | Generate `CHANGED_FILES.md`, `VERIFY.md`, `RESUME.md` | DONE |
| **P1** | Slim Telemetry Engine | `src/telemetry.py`: tokens, cost ($USD), latency, compact expander with reset | DONE |
