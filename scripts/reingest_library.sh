#!/usr/bin/env bash
# Fast, resumable, memory-aware full-text re-ingest.
# - Good papers first; prior failures deferred to the end
# - Micro-batches share one model load (much faster than 1 process/paper)
# - Hard timeout kills stuck batches; heartbeats while waiting
# - Optimistic RAM defaults (still leaves OS + peak headroom)
#
# Usage (project root):
#   chmod +x scripts/reingest_library.sh
#   caffeinate -dims bash scripts/reingest_library.sh
#
# Env BEFORE the command (not after caffeinate alone):
#   PAPER_TIMEOUT_SEC=300 caffeinate -dims bash scripts/reingest_library.sh
#   LIMIT=100 bash scripts/reingest_library.sh
#
# Env:
#   FORCE=1  REDO_DONE=1  REDO_INDEXED=1  LIMIT=N
#   SYSTEM_BUFFER_GB=0.5   PEAK_BUFFER_GB=1.0   (optimistic defaults)
#   MIN_BATCH=8  MAX_BATCH=48
#   MICRO_BATCH=4          papers per worker process (model loaded once)
#   PAPER_TIMEOUT_SEC=300  hard kill per micro-batch
#   MAX_RETRIES=2
#   STALL_LOG_SEC=45
#   PROGRESS_DIR=data/rag_ingest_progress

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv not found on PATH" >&2
  exit 1
fi

FORCE="${FORCE:-0}"
REDO_DONE="${REDO_DONE:-0}"
REDO_INDEXED="${REDO_INDEXED:-0}"
LIMIT="${LIMIT:-0}"
SYSTEM_BUFFER_GB="${SYSTEM_BUFFER_GB:-0.5}"
PEAK_BUFFER_GB="${PEAK_BUFFER_GB:-1.0}"
MIN_BATCH="${MIN_BATCH:-8}"
MAX_BATCH="${MAX_BATCH:-48}"
MICRO_BATCH="${MICRO_BATCH:-4}"
MEM_WAIT_SEC="${MEM_WAIT_SEC:-20}"
MEM_WAIT_MAX="${MEM_WAIT_MAX:-6}"
PAPER_TIMEOUT_SEC="${PAPER_TIMEOUT_SEC:-300}"
MAX_RETRIES="${MAX_RETRIES:-2}"
STALL_LOG_SEC="${STALL_LOG_SEC:-45}"
PROGRESS_DIR="${PROGRESS_DIR:-data/rag_ingest_progress}"
FIXED_BATCH="${RAG_EMBED_BATCH:-}"

mkdir -p logs "$PROGRESS_DIR"
LOG="logs/reingest_$(date +%Y%m%d_%H%M%S).log"
DONE_FILE="$PROGRESS_DIR/done.txt"
FAIL_FILE="$PROGRESS_DIR/failed.txt"
STATE_FILE="$PROGRESS_DIR/session_state.json"
WORKER_PY="$PROGRESS_DIR/_ingest_one_worker.py"

touch "$DONE_FILE" "$FAIL_FILE"
echo "$LOG" > "$PROGRESS_DIR/latest_log.txt"

export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_LOGGER_LEVEL=error
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "Project      : $ROOT"
echo "Log          : $LOG"
echo "Progress     : $PROGRESS_DIR"
echo "FORCE=$FORCE REDO_DONE=$REDO_DONE REDO_INDEXED=$REDO_INDEXED LIMIT=${LIMIT:-∞}"
echo "Buffers      : system=${SYSTEM_BUFFER_GB}GiB peak=${PEAK_BUFFER_GB}GiB (optimistic)"
echo "Embed batch  : ${MIN_BATCH}–${MAX_BATCH}" \
  "$([[ -n "$FIXED_BATCH" ]] && echo "(fixed=$FIXED_BATCH)" || echo "(adaptive)")"
echo "Micro-batch  : ${MICRO_BATCH} papers/process (shared model load)"
echo "Hard timeout : ${PAPER_TIMEOUT_SEC}s/micro-batch  retries=${MAX_RETRIES}"
echo "Order        : new papers first → prior failures last"
echo "Stop: Ctrl+C | Resume: re-run this script"
echo ""
echo "Tip: PAPER_TIMEOUT_SEC=300 caffeinate -dims bash scripts/reingest_library.sh"
echo ""

uv run python -c "import dotenv" 2>/dev/null || uv add python-dotenv

# Worker: one or more ids, load model once, print WORKER_RESULT per id
cat > "$WORKER_PY" << 'WORKER'
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
WORKER

export FORCE REDO_DONE REDO_INDEXED LIMIT DONE_FILE FAIL_FILE STATE_FILE
export SYSTEM_BUFFER_GB PEAK_BUFFER_GB MIN_BATCH MAX_BATCH MICRO_BATCH
export MEM_WAIT_SEC MEM_WAIT_MAX PAPER_TIMEOUT_SEC MAX_RETRIES STALL_LOG_SEC
export FIXED_BATCH WORKER_PY

uv run python - <<'PY' 2>&1 | tee -a "$LOG"
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

root = Path.cwd()
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
os.environ["PYTHONPATH"] = str(root) + (
    os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""
)

done_file = Path(os.environ["DONE_FILE"])
fail_file = Path(os.environ["FAIL_FILE"])
state_file = Path(os.environ["STATE_FILE"])
worker_py = Path(os.environ["WORKER_PY"])
force = os.environ.get("FORCE", "0") == "1"
redo_done = os.environ.get("REDO_DONE", "0") == "1"
redo_indexed = os.environ.get("REDO_INDEXED", "0") == "1"
system_buffer_gb = float(os.environ.get("SYSTEM_BUFFER_GB") or "0.5")
peak_buffer_gb = float(os.environ.get("PEAK_BUFFER_GB") or "1.0")
min_batch = max(1, int(os.environ.get("MIN_BATCH") or "8"))
max_batch = max(min_batch, int(os.environ.get("MAX_BATCH") or "48"))
micro_batch = max(1, int(os.environ.get("MICRO_BATCH") or "4"))
mem_wait_sec = max(5, int(os.environ.get("MEM_WAIT_SEC") or "20"))
mem_wait_max = max(1, int(os.environ.get("MEM_WAIT_MAX") or "6"))
paper_timeout = max(90, int(os.environ.get("PAPER_TIMEOUT_SEC") or "300"))
max_retries = max(1, int(os.environ.get("MAX_RETRIES") or "2"))
stall_log_sec = max(15, int(os.environ.get("STALL_LOG_SEC") or "45"))
fixed_batch = (os.environ.get("FIXED_BATCH") or "").strip()
try:
    limit = int(os.environ.get("LIMIT") or "0")
except ValueError:
    limit = 0

stop_requested = False


def _on_signal(signum, frame):
    global stop_requested
    stop_requested = True
    print(
        f"\n[signal] {signum} — stop after current micro-batch "
        f"(finished ids already saved).",
        flush=True,
    )


signal.signal(signal.SIGINT, _on_signal)
signal.signal(signal.SIGTERM, _on_signal)


def mem_stats() -> dict:
    total = avail = None
    try:
        import psutil

        vm = psutil.virtual_memory()
        total = vm.total / (1024**3)
        avail = vm.available / (1024**3)
    except Exception:
        pass
    if avail is None and sys.platform == "darwin":
        try:
            total = float(
                subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            ) / (1024**3)
            vm = subprocess.check_output(["vm_stat"], text=True)
            page = 16384.0
            m = re.search(r"page size of (\d+) bytes", vm)
            if m:
                page = float(m.group(1))

            def pages(label: str) -> float:
                mm = re.search(rf"{label}:\s+(\d+)", vm)
                return float(mm.group(1)) if mm else 0.0

            avail = (
                (pages("Pages free") + pages("Pages speculative") + pages("Pages inactive"))
                * page
                / (1024**3)
            )
        except Exception:
            pass
    if avail is None:
        total = total or 16.0
        avail = 4.0
    return {"total_gb": float(total or 0), "avail_gb": float(avail)}


def choose_batch(avail_gb: float) -> tuple[int, float]:
    """Optimistic mapping: use more of free RAM for larger embed batches."""
    if fixed_batch:
        b = max(min_batch, min(max_batch, int(fixed_batch)))
        return b, avail_gb - system_buffer_gb - peak_buffer_gb
    reserved = system_buffer_gb + peak_buffer_gb
    usable = avail_gb - reserved
    if usable < 1.0:
        b = min_batch
    elif usable < 2.0:
        b = max(min_batch, 12)
    elif usable < 3.5:
        b = max(min_batch, 24)
    elif usable < 5.5:
        b = max(min_batch, 32)
    else:
        b = max_batch
    return max(min_batch, min(max_batch, b)), usable


def choose_micro(avail_gb: float) -> int:
    """How many papers share one model load."""
    reserved = system_buffer_gb + peak_buffer_gb
    usable = avail_gb - reserved
    if usable < 1.5:
        return 1
    if usable < 3.0:
        return min(2, micro_batch)
    if usable < 5.0:
        return min(3, micro_batch)
    return micro_batch


def wait_for_memory() -> tuple[bool, dict, int, int]:
    reserved = system_buffer_gb + peak_buffer_gb
    for attempt in range(1, mem_wait_max + 1):
        if stop_requested:
            st = mem_stats()
            b, _ = choose_batch(st["avail_gb"])
            return False, st, b, 1
        st = mem_stats()
        batch, usable = choose_batch(st["avail_gb"])
        mb = choose_micro(st["avail_gb"])
        print(
            f"[mem] avail={st['avail_gb']:.1f}GiB total={st['total_gb']:.1f}GiB "
            f"reserve={reserved:.1f}GiB usable={usable:.1f}GiB "
            f"→ embed_batch={batch} micro_batch={mb}",
            flush=True,
        )
        if st["avail_gb"] >= reserved and usable >= 0.2:
            return True, st, batch, mb
        print(f"[mem] low — wait {mem_wait_sec}s ({attempt}/{mem_wait_max})", flush=True)
        time.sleep(mem_wait_sec)
    st = mem_stats()
    batch, usable = choose_batch(st["avail_gb"])
    mb = choose_micro(st["avail_gb"])
    return st["avail_gb"] >= system_buffer_gb, st, batch, mb


def load_id_set(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        aid = line.strip().split()[0] if line.strip() else ""
        aid = aid.replace("arXiv:", "").split("v")[0].strip()
        # strip trailing comments from failed.txt
        if "#" in aid:
            aid = aid.split("#")[0].strip()
        if aid:
            out.add(aid)
    return out


def append_id(path: Path, aid: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(aid + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_corpus_ids() -> list[str]:
    candidates = [
        root / "data" / "corpus.json",
        root / "data" / "corpus_100.json",
        root / "data" / "corpus_all.jsonl",
    ]
    corpus_path = next((p for p in candidates if p.exists()), None)
    if corpus_path is None:
        sys.exit("No corpus file found under data/")
    print(f"[plan] corpus = {corpus_path}", flush=True)
    text = corpus_path.read_text(encoding="utf-8")
    if corpus_path.suffix == ".jsonl":
        papers = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        data = json.loads(text)
        papers = data if isinstance(data, list) else (data.get("papers") or data.get("items") or [])
    ids: list[str] = []
    for p in papers:
        aid = str(p.get("arxiv_id") or p.get("id") or p.get("paper_id") or "").strip()
        aid = aid.replace("arXiv:", "").split("v")[0].strip()
        if aid and aid not in ids:
            ids.append(aid)
    return ids


def load_indexed_ids() -> set[str]:
    try:
        from src.rag.embed_store import list_ingested_papers

        out = set()
        for p in list_ingested_papers():
            pid = str(p.get("paper_id") or "").replace("arXiv:", "").split("v")[0].strip()
            if pid:
                out.add(pid)
        return out
    except Exception as e:
        print(f"[plan] RAG index list skipped ({e})", flush=True)
        return set()


def is_transient(err: str) -> bool:
    e = (err or "").lower()
    keys = (
        "timeout",
        "timed out",
        "killed",
        "connection",
        "temporarily",
        "reset by peer",
        "503",
        "502",
        "429",
        "out of memory",
        "oom",
        "mps",
        "network",
        "ssl",
        "urlopen",
        "remote end closed",
        "broken pipe",
    )
    return any(k in e for k in keys)


def _kill_process_tree(proc: subprocess.Popen) -> None:
    try:
        if proc.poll() is not None:
            return
        if sys.platform != "win32":
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                proc.kill()
        else:
            proc.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    try:
        proc.wait(timeout=15)
    except Exception:
        pass


def run_microbatch(ids: list[str], batch: int) -> list[dict]:
    """Run 1..N papers in one worker (shared model). Hard-kill on timeout."""
    env = os.environ.copy()
    env["RAG_EMBED_BATCH"] = str(batch)
    env["FORCE"] = "1" if force else "0"
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    env["STREAMLIT_SERVER_HEADLESS"] = "true"
    env["STREAMLIT_LOGGER_LEVEL"] = "error"
    env["PYTHONPATH"] = str(root) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env["PWD"] = str(root)

    cmd = ["uv", "run", "python", str(worker_py), *ids]
    preexec = os.setsid if sys.platform != "win32" else None
    t0 = time.time()
    last_beat = t0
    # scale timeout with micro-batch size (model load once + N papers)
    timeout = paper_timeout * max(1, len(ids))

    proc = subprocess.Popen(
        cmd,
        cwd=str(root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        preexec_fn=preexec,
    )
    results_by_id: dict[str, dict] = {}
    assert proc.stdout is not None

    while True:
        if stop_requested:
            _kill_process_tree(proc)
            for aid in ids:
                results_by_id.setdefault(
                    aid,
                    {"ok": False, "arxiv_id": aid, "error": "interrupted by signal"},
                )
            break

        elapsed = time.time() - t0
        if elapsed > timeout:
            print(
                f"[timeout] micro-batch exceeded {timeout}s — killing worker+children",
                flush=True,
            )
            _kill_process_tree(proc)
            for aid in ids:
                results_by_id.setdefault(
                    aid,
                    {
                        "ok": False,
                        "arxiv_id": aid,
                        "error": f"hard timeout after {timeout}s (killed)",
                    },
                )
            break

        if time.time() - last_beat >= stall_log_sec:
            print(
                f"[heartbeat] batch {ids[0]}…(+{len(ids)-1}) "
                f"running {elapsed:.0f}s / {timeout}s",
                flush=True,
            )
            last_beat = time.time()

        try:
            import select

            ready, _, _ = select.select([proc.stdout], [], [], 1.0)
            if not ready:
                if proc.poll() is not None:
                    break
                continue
            line = proc.stdout.readline()
        except Exception:
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            if not line:
                time.sleep(0.15)
                continue

        if not line:
            if proc.poll() is not None:
                break
            continue

        line = line.rstrip("\n")
        if line.startswith("WORKER_RESULT "):
            try:
                row = json.loads(line[len("WORKER_RESULT ") :])
                aid = str(row.get("arxiv_id") or "")
                if aid:
                    results_by_id[aid] = row
            except Exception:
                pass
        elif any(
            k in line
            for k in (
                "[worker]",
                "[ingest]",
                "[embed]",
                "Batches:",
                "OCR",
                "Done.",
                "Downloading",
                "Traceback",
                "Error",
            )
        ):
            print(f"  {line}", flush=True)

    try:
        proc.wait(timeout=5)
    except Exception:
        _kill_process_tree(proc)

    out: list[dict] = []
    for aid in ids:
        row = results_by_id.get(aid) or {
            "ok": False,
            "arxiv_id": aid,
            "error": "no result from worker (killed or crash)",
        }
        row["ok"] = bool(row.get("ok"))
        out.append(row)
    return out


# --- plan ---
st0 = mem_stats()
b0, u0 = choose_batch(st0["avail_gb"])
print(
    f"[mem] startup avail={st0['avail_gb']:.1f}/{st0['total_gb']:.1f}GiB "
    f"→ embed_batch={b0} usable={u0:.1f}GiB micro≤{micro_batch}",
    flush=True,
)

all_ids = load_corpus_ids()
done = load_id_set(done_file)
failed_prior = load_id_set(fail_file)
indexed = set() if redo_indexed else load_indexed_ids()
print(
    f"[plan] library={len(all_ids)} done_file={len(done)} "
    f"prior_fails={len(failed_prior)} indexed={len(indexed)}",
    flush=True,
)

fresh: list[str] = []
deferred: list[str] = []
for aid in all_ids:
    if not redo_done and aid in done:
        continue
    if not redo_indexed and aid in indexed:
        if aid not in done:
            append_id(done_file, aid)
            done.add(aid)
        continue
    if aid in failed_prior:
        deferred.append(aid)
    else:
        fresh.append(aid)

# session failures go to the end of *this* run as well
pending = fresh + deferred
total_pending = len(pending)
if limit > 0:
    pending = pending[:limit]
n = len(pending)

print(
    f"[plan] pending={total_pending} (fresh={len(fresh)} deferred_fails={len(deferred)}) "
    f"session={n} force={force}",
    flush=True,
)
if n == 0:
    print("[plan] nothing to do.", flush=True)
    sys.exit(0)

print(
    f"[plan] micro-batch up to {micro_batch}; timeout {paper_timeout}s × batch size",
    flush=True,
)
print("", flush=True)

times: list[float] = []
ok = fail = skipped_mem = 0
t_all = time.time()
i = 0
session_deferred: list[str] = []

while i < n:
    if stop_requested:
        print("[stop] not starting further batches.", flush=True)
        break

    ready, st, batch, mb = wait_for_memory()
    if not ready and st["avail_gb"] < system_buffer_gb:
        # push rest to deferred mentally — skip one and continue
        aid = pending[i]
        skipped_mem += 1
        print(f"[skip] {aid} — memory too low", flush=True)
        append_id(fail_file, f"{aid}  # mem_skip")
        session_deferred.append(aid)
        i += 1
        continue

    chunk = pending[i : i + mb]
    i += len(chunk)

    elapsed = sum(times)
    avg = (elapsed / max(1, ok + fail)) if (ok + fail) else 35.0
    left = n - (i - len(chunk))
    # rough: papers left / micro
    eta = avg * max(1, (n - i + len(chunk)) / max(1, mb))
    print(
        f"\n===== papers {i - len(chunk) + 1}–{i}/{n}  "
        f"{chunk}  elapsed={elapsed/60:.1f}m  eta~{eta/60:.1f}m =====",
        flush=True,
    )

    t0 = time.time()
    rows: list[dict] = []
    for attempt in range(1, max_retries + 1):
        if stop_requested:
            break
        use_batch = batch if attempt == 1 else min_batch
        use_ids = chunk if attempt == 1 else chunk  # same ids
        if attempt > 1:
            # retry only failed ids from previous attempt
            use_ids = [r["arxiv_id"] for r in rows if not r.get("ok")]
            if not use_ids:
                break
            print(
                f"[retry] attempt {attempt}/{max_retries} ids={use_ids} batch={use_batch}",
                flush=True,
            )
            time.sleep(min(30, 4 * attempt))

        print(
            f"[run] micro={len(use_ids)} embed_batch={use_batch} "
            f"timeout={paper_timeout * len(use_ids)}s",
            flush=True,
        )
        new_rows = run_microbatch(use_ids, use_batch)
        if attempt == 1:
            rows = new_rows
        else:
            by = {r["arxiv_id"]: r for r in new_rows}
            rows = [by.get(r["arxiv_id"], r) if not r.get("ok") else r for r in rows]
        if all(r.get("ok") for r in rows):
            break
        # stop retrying if nothing looks transient
        errs = " ".join(str(r.get("error") or "") for r in rows if not r.get("ok"))
        if not is_transient(errs) and attempt >= 1:
            # still allow second try once
            if attempt >= max_retries:
                break

    dt = time.time() - t0
    times.append(dt)

    for row in rows:
        aid = str(row.get("arxiv_id") or "")
        if row.get("ok"):
            ok += 1
            append_id(done_file, aid)
            print(
                f"[ok] {aid} chunks={row.get('chunks')}  "
                f"{(row.get('title') or '')[:50]}",
                flush=True,
            )
        else:
            fail += 1
            err = str(row.get("error") or "unknown")
            append_id(fail_file, f"{aid}  # {err[:100].replace(chr(10), ' ')}")
            session_deferred.append(aid)
            print(f"[fail→defer] {aid} — {err[:140]}", flush=True)

    try:
        state_file.write_text(
            json.dumps(
                {
                    "status": "stopped" if stop_requested else "running",
                    "last_ids": chunk,
                    "session_ok": ok,
                    "session_fail": fail,
                    "session_skipped_mem": skipped_mem,
                    "last_embed_batch": batch,
                    "last_micro": mb,
                    "last_avail_gb": round(st["avail_gb"], 2),
                    "done_file_count": len(load_id_set(done_file)),
                    "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    except Exception:
        pass

total = time.time() - t_all
still = len([a for a in all_ids if a not in load_id_set(done_file)])
print(
    f"\n===== SESSION END  ok={ok} fail={fail} mem_skip={skipped_mem}  "
    f"wall={total/60:.1f}m =====",
    flush=True,
)
print(
    f"[progress] done={len(load_id_set(done_file))} pending≈{still} "
    f"(failures deferred to end of queue on next run)",
    flush=True,
)
if still:
    print("Re-run this script to continue.", flush=True)
PY

echo ""
echo "Progress: $PROGRESS_DIR"
uv run python -m src.rag.cli status 2>/dev/null \
  | grep -v enableCORS | grep -v Access-Control | grep -v Xsrf \
  | grep -v allowedHosts | grep -v corsAllowed | grep -v Uvicorn || true
