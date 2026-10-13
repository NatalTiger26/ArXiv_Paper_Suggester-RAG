# Deploying on Streamlit Community Cloud

## Critical limits

- Free Cloud ~1 GB RAM. **EmbeddingGemma 2 (~1 GB) will often crash the app.**
- On Cloud the code auto-selects `sentence-transformers/all-MiniLM-L6-v2` unless you override `EMBEDDING_MODEL`.
- Indexes built with Gemma are **not** compatible with MiniLM (different vector size). For Cloud demos either:
  - ship empty indexes and ingest a few papers under MiniLM, or
  - rebuild Chroma with MiniLM before deploy.

## Checklist (do every item)

### 1. GitHub repo

Push these files (already updated in the project):

- `requirements.txt` (includes **torch**)
- `pyproject.toml` (`requires-python = ">=3.10,<3.13"`)
- `packages.txt` (tesseract-ocr)
- `.streamlit/config.toml`
- `src/settings.py` (Cloud embedder + secrets + safer Gemini default)

### 2. Streamlit Cloud app settings

1. **Main file path:** `src/app.py`
2. **Python version:** **3.11** or **3.12** (Advanced settings) — not 3.13/3.14
3. **Secrets** (App settings → Secrets), example:

```toml
GEMINI_API_KEY = "your-key"
GEMINI_MODEL = "gemini-3.8-flash"
HF_TOKEN = ""
# Optional failsafe:
# FALLBACK_LLM_API_KEY = "..."
# NVIDIA_API_KEY = "..."

# Optional: force small embedder explicitly
# EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
```

4. Reboot / redeploy after changing secrets or requirements.

### 3. After deploy

1. Open **Manage app → Logs**. First error line matters.
2. Home should load without “Oh no”.
3. Library / Recommend may work with metadata.
4. Ask needs full-text index **built with the same embedder Cloud uses**.

### 4. If you still see “Oh no”

| Log symptom | Fix |
|-------------|-----|
| No module named torch | requirements include torch; Python 3.11/3.12 |
| Error installing requirements | Python version; pin torch; reboot |
| Killed / memory | Use MiniLM; do not load Gemma on Cloud |
| Model 404 | Fix `GEMINI_MODEL` in Secrets |
| Dimension mismatch | reset-index + re-ingest on Cloud with current model |

## Local (Mac) remains unchanged

```bash
uv sync
# .env: GEMINI_API_KEY=...  GEMINI_MODEL=gemini-3.8-flash
# default embedder stays google/embeddinggemma-2
uv run python -m streamlit run src/app.py
```


## OOM / connection reset after embedding (important)

Logs that look like:

```text
Loading model: google/embeddinggemma-2  (device=cpu)
…
connection reset by peer
```

mean the free Cloud process was **killed while loading a ~1GB embedder**.

After deploying the latest `settings.py` / embed guards:

1. **Do not** set `EMBEDDING_MODEL=google/embeddinggemma-2` in Cloud Secrets.
2. Leave embedder unset, or set:
   ```toml
   EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
   ```
3. Rebuild indexes **on Cloud** after switch (Gemma Chroma ≠ MiniLM dims):
   - Library → Rebuild index, or CLI rebuild
   - Re-ingest a few paper IDs for Ask
4. Avoid the Home “quick demo” with many papers on Cloud; ingest 1–2 IDs only.
5. Full Gemma + heavy eval stays **local** (`uv run` on your Mac).
