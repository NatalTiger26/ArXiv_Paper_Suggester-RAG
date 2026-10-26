#!/usr/bin/env python3
"""
Stable + adaptive full-text re-ingest.

Safety (no progress loss):
  • Chroma health probe with timeout before any work
  • Append-only done.txt + journal.jsonl
  • Never trust a tiny indexed_ids-style cache
  • --fresh-chroma only moves the index aside (explicit)

Performance (use the Mac without Jetsam/MPS kills):
  • Adaptive micro-batch: 1–MAX papers per worker, model loaded once
  • Sized from free RAM + recent success streak
  • Shrinks immediately after kills/timeouts; grows only after N ok
  • Embed batch also capped by free RAM

Usage:
  uv run python scripts/reingest_stable.py
  uv run python scripts/reingest_stable.py --fresh-chroma
  uv run python scripts/reingest_stable.py --max-micro 4
  uv run python scripts/reingest_stable.py --ids-file my.txt

Overnight:
  caffeinate -dims uv run python scripts/reingest_stable.py
  FRESH=1 caffeinate -dims uv run python scripts/reingest_stable.py --fresh-chroma
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PROGRESS = ROOT / "data" / "rag_ingest_progress"
DONE = PROGRESS / "done.txt"
FAIL = PROGRESS / "failed.txt"
JOURNAL = PROGRESS / "journal.jsonl"
CORPUS = ROOT / "data" / "corpus.json"
CHROMA = ROOT / "data" / "chroma_rag"
PAPERS = ROOT / "data" / "papers"


def strip_id(aid: str) -> str:
    s = (aid or "").strip().replace("arXiv:", "").replace("arxiv:", "")
    if "v" in s and s.rsplit("v", 1)[-1].isdigit():
        s = s.rsplit("v", 1)[0]
    return s


def load_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    out, seen = [], set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.split("#", 1)[0].strip()
        m = re.match(r"^(\d{4}\.\d{4,5}|[a-z][\w\-]*/\d{7})\b", line, re.I)
        if not m:
            continue
        a = strip_id(m.group(1))
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


def append_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")
        f.flush()
        os.fsync(f.fileno())


def mem_gb() -> tuple[float, float]:
    """(available_gb, total_gb)"""
    try:
        import psutil

        vm = psutil.virtual_memory()
        return vm.available / (1024**3), vm.total / (1024**3)
    except Exception:
        pass
    if sys.platform == "darwin":
        try:
            total = float(
                subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            ) / (1024**3)
            # rough free: page free + speculative + inactive is hard; use vm_stat free only
            out = subprocess.check_output(["vm_stat"], text=True)
            page = 16384.0
            m = re.search(r"page size of (\d+)", out)
            if m:
                page = float(m.group(1))
            free = 0.0
            for label in ("Pages free", "Pages speculative"):
                mm = re.search(rf"{label}:\s+(\d+)", out)
                if mm:
                    free += int(mm.group(1)) * page
            return free / (1024**3), total
        except Exception:
            pass
    return 6.0, 16.0


def choose_micro(avail: float, *, max_micro: int, streak_ok: int, streak_fail: int) -> int:
    """
    Headroom model (EmbeddingGemma ~1.5–2.5 GiB + Chroma + PDF peak):
      leave ~2.5 GiB for OS + peak spike
    """
    usable = max(0.0, avail - 2.5)
    if usable >= 8 and streak_fail == 0:
        n = 4
    elif usable >= 5 and streak_fail == 0:
        n = 3
    elif usable >= 3.5:
        n = 2
    else:
        n = 1
    # after consecutive kills, force 1 until we recover
    if streak_fail >= 1:
        n = 1
    # only grow after sustained success
    if streak_ok < 3:
        n = min(n, 2)
    if streak_ok < 1:
        n = 1
    return max(1, min(max_micro, n))


def choose_embed_batch(avail: float) -> int:
    usable = max(0.0, avail - 2.5)
    if usable >= 8:
        return 24
    if usable >= 5:
        return 16
    if usable >= 3.5:
        return 8
    return 4


def load_corpus_ids() -> list[str]:
    if not CORPUS.exists():
        sys.exit(f"Missing {CORPUS}")
    data = json.loads(CORPUS.read_text(encoding="utf-8"))
    papers = data if isinstance(data, list) else list(data.get("papers") or [])
    ids, seen = [], set()
    for p in papers:
        aid = strip_id(str(p.get("arxiv_id") or p.get("id") or p.get("paper_id") or ""))
        if aid and aid not in seen:
            seen.add(aid)
            ids.append(aid)
    return ids


def chroma_count_with_timeout(timeout_sec: int = 45) -> int | None:
    code = r"""
import sys
from pathlib import Path
import chromadb
from chromadb.config import Settings
p = Path(sys.argv[1])
c = chromadb.PersistentClient(path=str(p), settings=Settings(anonymized_telemetry=False))
n = 0
for col in c.list_collections():
    n += col.count()
print("COUNT", n)
"""
    try:
        proc = subprocess.run(
            ["uv", "run", "python", "-c", code, str(CHROMA)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
        )
    except subprocess.TimeoutExpired:
        print(f"[health] Chroma timed out after {timeout_sec}s", flush=True)
        return None
    if proc.returncode != 0:
        print(f"[health] Chroma error: {(proc.stderr or '')[:300]}", flush=True)
        return None
    for line in (proc.stdout or "").splitlines():
        if line.startswith("COUNT"):
            try:
                return int(line.split()[1])
            except Exception:
                return None
    return None


def paper_has_chunks(arxiv_id: str, timeout_sec: int = 25) -> bool | None:
    code = r"""
import sys
from pathlib import Path
import chromadb
from chromadb.config import Settings
aid, p = sys.argv[1], Path(sys.argv[2])
c = chromadb.PersistentClient(path=str(p), settings=Settings(anonymized_telemetry=False))
cols = c.list_collections()
if not cols:
    print("HAS 0"); raise SystemExit(0)
col = cols[0]
for key in ("paper_id", "arxiv_id"):
    try:
        r = col.get(where={key: aid}, limit=1, include=[])
        if r and r.get("ids"):
            print("HAS 1"); raise SystemExit(0)
    except Exception:
        pass
print("HAS 0")
"""
    try:
        proc = subprocess.run(
            ["uv", "run", "python", "-c", code, arxiv_id, str(CHROMA)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
        )
    except subprocess.TimeoutExpired:
        return None
    if "HAS 1" in (proc.stdout or ""):
        return True
    if "HAS 0" in (proc.stdout or ""):
        return False
    return None


def run_microbatch(
    ids: list[str],
    *,
    force: bool,
    force_pdf: bool = False,
    embed_batch: int,
    timeout_sec: int,
) -> list[dict]:
    """One process, model once, N papers. Hard timeout on whole batch."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("STREAMLIT_LOGGER_LEVEL", "error")
    env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    env.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    env["RAG_EMBED_BATCH"] = str(embed_batch)

    # worker writes one RESULT line per id
    code = r"""
import json, os, sys, gc
sys.path.insert(0, "src")
ids = sys.argv[1].split(",")
force = sys.argv[2] == "1"
force_pdf = sys.argv[3] == "1" if len(sys.argv) > 3 else False
os.environ.setdefault("STREAMLIT_LOGGER_LEVEL", "error")
from src.rag.ingest import ingest_paper
from src.rag.embed_store import load_embedding_model
print("[worker] loading model…", flush=True)
model = load_embedding_model()
print("[worker] model ready", flush=True)
for aid in ids:
    print(f"[worker] ingest {aid}", flush=True)
    try:
        row = ingest_paper(aid, model=model, force_download=force, force_pdf=force_pdf)
        print("RESULT " + json.dumps({
            "ok": bool(row.get("ok")),
            "arxiv_id": row.get("arxiv_id") or aid,
            "chunks": row.get("chunks"),
            "source_format": row.get("source_format"),
            "title": (row.get("title") or "")[:100],
            "error": row.get("error"),
        }, ensure_ascii=False), flush=True)
    except Exception as e:
        print("RESULT " + json.dumps({
            "ok": False, "arxiv_id": aid, "error": str(e)[:300],
        }, ensure_ascii=False), flush=True)
    gc.collect()
    try:
        import torch
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
"""
    batch_timeout = timeout_sec * max(1, len(ids))
    try:
        proc = subprocess.run(
            ["uv", "run", "python", "-c", code, ",".join(ids), "1" if force else "0", "1" if force_pdf else "0"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=batch_timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return [
            {"ok": False, "arxiv_id": a, "error": f"batch timeout after {batch_timeout}s"}
            for a in ids
        ]

    by: dict[str, dict] = {}
    for line in (proc.stdout or "").splitlines():
        if line.startswith("RESULT "):
            try:
                row = json.loads(line[len("RESULT ") :])
                aid = str(row.get("arxiv_id") or "")
                if aid:
                    by[aid] = row
            except Exception:
                pass
        elif any(
            k in line
            for k in (
                "[worker]",
                "[embed]",
                "Batches:",
                "OCR",
                "Downloading",
                "[pdf2md]",
                "[skip",
            )
        ):
            print(f"  {line}", flush=True)

    if proc.returncode is not None and proc.returncode < 0:
        sig = -proc.returncode
        print(f"[run] worker killed by signal {sig} (memory/MPS pressure)", flush=True)

    out = []
    for a in ids:
        if a in by:
            out.append(by[a])
        else:
            out.append(
                {
                    "ok": False,
                    "arxiv_id": a,
                    "error": f"no result (exit={proc.returncode})",
                }
            )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--fresh-chroma", action="store_true")
    ap.add_argument("--ids-file", type=Path)
    ap.add_argument("--force-pdf", action="store_true", help="Force PDF download and OCR extraction instead of arXiv HTML")
    ap.add_argument("--timeout", type=int, default=int(os.environ.get("PAPER_TIMEOUT_SEC", "900")))
    ap.add_argument("--health-timeout", type=int, default=45)
    ap.add_argument("--max-micro", type=int, default=int(os.environ.get("MICRO_BATCH", "4")))
    ap.add_argument(
        "--trust-done",
        action="store_true",
        help="Skip done.txt ids without Chroma check (default behavior when healthy)",
    )
    ap.add_argument(
        "--verify-done",
        action="store_true",
        help="Slow: confirm each done.txt id still has chunks in Chroma",
    )
    ap.add_argument("--ignore-done", action="store_true")
    args = ap.parse_args()
    args.max_micro = max(1, min(6, args.max_micro))

    PROGRESS.mkdir(parents=True, exist_ok=True)
    DONE.touch(exist_ok=True)
    FAIL.touch(exist_ok=True)

    if args.fresh_chroma:
        if CHROMA.exists():
            bak = CHROMA.parent / f"chroma_rag.bak_{time.strftime('%Y%m%d_%H%M%S')}"
            CHROMA.rename(bak)
            print(f"[init] moved old index → {bak}", flush=True)
        CHROMA.mkdir(parents=True, exist_ok=True)
        if DONE.exists() and DONE.stat().st_size > 0:
            bak = DONE.with_suffix(".txt.bak_before_fresh")
            bak.write_bytes(DONE.read_bytes())
            DONE.write_text("", encoding="utf-8")
            print(f"[init] cleared done.txt (backup {bak.name})", flush=True)
        print("[init] fresh chroma_rag", flush=True)

    print("[health] probing Chroma…", flush=True)
    n_chunks = chroma_count_with_timeout(args.health_timeout)
    if n_chunks is None:
        print(
            "[health] UNHEALTHY — refuse to run.\n"
            "  uv run python scripts/reingest_stable.py --fresh-chroma",
            flush=True,
        )
        sys.exit(2)
    print(f"[health] OK chunks={n_chunks}", flush=True)

    all_ids = load_lines(args.ids_file) if args.ids_file else load_corpus_ids()
    done = set(load_lines(DONE))
    print(f"[plan] targets={len(all_ids)} done.txt={len(done)}", flush=True)

    # Fast path: trust done.txt after a healthy Chroma probe.
    # Per-id Chroma checks on 1k+ papers made resume look "stuck" for hours.
    trust_done = args.trust_done or (not args.ignore_done and not args.fresh_chroma)
    if trust_done and done:
        print(
            f"[plan] trusting done.txt ({len(done)} ids) — no per-paper Chroma verify. "
            f"Use --verify-done to re-check (slow).",
            flush=True,
        )

    pending = []
    skipped = 0
    for aid in all_ids:
        if args.ignore_done or args.fresh_chroma:
            pending.append(aid)
            continue
        if aid in done and trust_done and not getattr(args, "verify_done", False):
            skipped += 1
            continue
        if aid in done and getattr(args, "verify_done", False):
            has = paper_has_chunks(aid)
            if has is True or has is None:
                skipped += 1
                continue
            # in done but missing from index → re-queue
        pending.append(aid)

    def has_pdf(a: str) -> bool:
        return (PAPERS / f"{a}.pdf").exists() or (
            PAPERS / f"{a.replace('/', '_')}.pdf"
        ).exists()

    pending.sort(key=lambda a: (0 if has_pdf(a) else 1, a))
    if args.limit > 0:
        pending = pending[: args.limit]

    print(
        f"[plan] skip={skipped} remaining={len(pending)} max_micro={args.max_micro}",
        flush=True,
    )
    if not pending:
        print("[plan] nothing to do.", flush=True)
        return

    ok = fail = 0
    streak_ok = 0
    streak_fail = 0
    t0 = time.time()
    times: list[float] = []
    i = 0
    stop = {"flag": False}

    def _request_stop(signum, frame):
        stop["flag"] = True
        print(
            f"\n[signal] {signum} — finishing current paper/batch, then exit. "
            f"Already-[ok] ids are in done.txt.",
            flush=True,
        )

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    while i < len(pending):
        if stop["flag"]:
            print("[stop] not starting further batches.", flush=True)
            break
        avail, total = mem_gb()
        micro = choose_micro(
            avail, max_micro=args.max_micro, streak_ok=streak_ok, streak_fail=streak_fail
        )
        embed_batch = choose_embed_batch(avail)
        chunk = pending[i : i + micro]

        # skip already-in-RAG inside this chunk
        todo = []
        for aid in chunk:
            if not args.force:
                has = paper_has_chunks(aid)
                if has is True:
                    append_line(DONE, aid)
                    ok += 1
                    skipped += 1
                    print(f"[skip] {aid} already in RAG", flush=True)
                    continue
            todo.append(aid)
        i += len(chunk)
        if not todo:
            continue

        print(
            f"\n===== papers {i - len(chunk) + 1}–{i}/{len(pending)}  "
            f"micro={len(todo)} embed_batch={embed_batch} "
            f"avail={avail:.1f}/{total:.1f}GiB =====",
            flush=True,
        )
        print(f"  ids={todo}", flush=True)

        t1 = time.time()
        rows = run_microbatch(
            todo,
            force=args.force,
            force_pdf=getattr(args, "force_pdf", False),
            embed_batch=embed_batch,
            timeout_sec=args.timeout,
        )
        dt = time.time() - t1
        per = dt / max(1, len(todo))
        times.extend([per] * len(todo))

        batch_all_ok = True
        for row in rows:
            aid = str(row.get("arxiv_id") or "")
            fmt = row.get("source_format") or "ingest"
            if row.get("ok"):
                ok += 1
                streak_ok += 1
                streak_fail = 0
                append_line(DONE, aid)
                append_line(
                    JOURNAL,
                    json.dumps(
                        {
                            "ts": time.time(),
                            "arxiv_id": aid,
                            "ok": True,
                            "chunks": row.get("chunks"),
                            "format": fmt,
                            "s": round(per, 1),
                        }
                    ),
                )
                print(
                    f"[ok] {aid} [{fmt}] chunks={row.get('chunks')}  "
                    f"{(row.get('title') or '')[:50]}",
                    flush=True,
                )
            else:
                fail += 1
                streak_ok = 0
                streak_fail += 1
                batch_all_ok = False
                err = str(row.get("error") or "unknown")[:200]
                append_line(FAIL, f"{aid}  # {err}")
                append_line(
                    JOURNAL,
                    json.dumps(
                        {
                            "ts": time.time(),
                            "arxiv_id": aid,
                            "ok": False,
                            "error": err,
                            "s": round(per, 1),
                        }
                    ),
                )
                print(f"[fail] {aid} — {err}", flush=True)

        if times:
            window = times[-30:]
            med = sorted(window)[len(window) // 2]
            left = len(pending) - i
            print(
                f"[eta] ok={ok} fail={fail} streak_ok={streak_ok}  "
                f"~{med:.0f}s/paper ETA~{med * left / 60:.0f}m  "
                f"wall={(time.time() - t0) / 60:.1f}m  next_micro≤"
                f"{choose_micro(mem_gb()[0], max_micro=args.max_micro, streak_ok=streak_ok, streak_fail=streak_fail)}",
                flush=True,
            )

        if stop["flag"]:
            print("[stop] exiting after this batch.", flush=True)
            break

    print(
        f"\n===== SESSION END ok={ok} fail={fail} wall={(time.time() - t0) / 60:.1f}m =====",
        flush=True,
    )
    print("Resume with the same command. done.txt is append-only.", flush=True)


if __name__ == "__main__":
    main()
