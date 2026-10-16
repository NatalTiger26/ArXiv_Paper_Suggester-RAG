from __future__ import annotations
import json, os, sys, re, gc
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.environ["PYTHONPATH"] = str(_ROOT) + (
    os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""
)
os.chdir(_ROOT)

os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
os.environ.setdefault("STREAMLIT_LOGGER_LEVEL", "error")

_NOISE = re.compile(
    r"enableCORS|Access-Control-Allow-Origin|enableXsrfProtection|"
    r"corsAllowedOrigins|allowedHosts|Uvicorn server|Watchdog module",
    re.I,
)

class _Filter:
    def __init__(self, wrapped):
        self.wrapped = wrapped
        self._buf = ""
    def write(self, s):
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if _NOISE.search(line) or not line.strip():
                continue
            self.wrapped.write(line + "\n")
    def flush(self):
        if self._buf and not _NOISE.search(self._buf):
            self.wrapped.write(self._buf)
        self._buf = ""
        self.wrapped.flush()
    def __getattr__(self, n):
        return getattr(self.wrapped, n)

sys.stdout = _Filter(sys.__stdout__)
sys.stderr = _Filter(sys.__stderr__)


def _emit(row: dict) -> None:
    print("WORKER_RESULT " + json.dumps(row, ensure_ascii=False), flush=True)


def main() -> None:
    ids = [a.strip() for a in sys.argv[1:] if a.strip()]
    if not ids:
        _emit({"ok": False, "error": "no arxiv ids"})
        sys.exit(2)
    force = os.environ.get("FORCE", "0") == "1"
    batch = os.environ.get("RAG_EMBED_BATCH", "16")
    os.environ["RAG_EMBED_BATCH"] = str(batch)
    print(f"[worker] ids={len(ids)} force={force} batch={batch}", flush=True)

    from src.rag.embed_store import load_embedding_model
    from src.rag.ingest import ingest_paper

    print("[worker] loading embedding model…", flush=True)
    model = load_embedding_model()
    print("[worker] model ready", flush=True)

    any_ok = False
    for aid in ids:
        print(f"[worker] ingest {aid}", flush=True)
        try:
            row = ingest_paper(aid, model=model, force_download=force)
            out = {
                "ok": bool(row.get("ok")),
                "arxiv_id": row.get("arxiv_id") or aid,
                "chunks": row.get("chunks"),
                "title": (row.get("title") or "")[:80],
                "error": row.get("error"),
            }
            _emit(out)
            if out["ok"]:
                any_ok = True
            else:
                print(f"[worker] fail {aid}: {out.get('error')}", flush=True)
        except Exception as e:
            _emit({"ok": False, "arxiv_id": aid, "error": str(e)[:300]})
            print(f"[worker] exception {aid}: {e}", flush=True)
        # free peak activations between papers in the same process
        gc.collect()
        try:
            import torch
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

    sys.exit(0 if any_ok else 1)


if __name__ == "__main__":
    main()
