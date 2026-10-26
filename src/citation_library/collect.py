"""Collect reference lists for library papers (resumable)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Literal

from settings import PROJECT_ROOT

from .store import (
    already_ok,
    connect,
    replace_edges,
    set_fetch_status,
    stats,
    upsert_paper,
)

Source = Literal["library", "rag", "ids"]


def _strip(aid: str) -> str:
    s = (aid or "").strip().replace("arXiv:", "").replace("arxiv:", "")
    if "v" in s and s.rsplit("v", 1)[-1].isdigit():
        s = s.rsplit("v", 1)[0]
    return s.strip()


def load_library_ids(
    *,
    source: Source = "library",
    ids_file: Path | None = None,
    limit: int = 0,
) -> list[str]:
    """Return sorted unique arXiv ids to scan."""
    ids: list[str] = []

    if source == "ids" and ids_file:
        text = Path(ids_file).read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            aid = _strip(line.split()[0])
            if aid:
                ids.append(aid)
    elif source == "rag":
        try:
            from rag.embed_store import list_ingested_papers
        except ImportError:
            from src.rag.embed_store import list_ingested_papers  # type: ignore
        for p in list_ingested_papers():
            aid = _strip(str(p.get("paper_id") or p.get("arxiv_id") or ""))
            if aid:
                ids.append(aid)
    else:
        # library = corpus.json
        corpus_path = PROJECT_ROOT / "data" / "corpus.json"
        if not corpus_path.is_file():
            raise SystemExit(f"No corpus at {corpus_path}")
        raw = json.loads(corpus_path.read_text(encoding="utf-8"))
        papers = raw if isinstance(raw, list) else list(raw.get("papers") or [])
        for p in papers:
            aid = _strip(str(p.get("arxiv_id") or p.get("id") or p.get("paper_id") or ""))
            if aid:
                ids.append(aid)

    # unique, stable
    seen: set[str] = set()
    out: list[str] = []
    for a in ids:
        if a not in seen:
            seen.add(a)
            out.append(a)
    out.sort()
    if limit and limit > 0:
        out = out[:limit]
    return out


def library_id_set() -> set[str]:
    """All arXiv ids currently in corpus.json."""
    return set(load_library_ids(source="library", limit=0))


def _record_from_novelty(rec: dict[str, Any]) -> dict[str, Any]:
    return {
        "canonical_id": rec.get("canonical_id"),
        "arxiv_id": rec.get("arxiv_id"),
        "s2_id": rec.get("s2_id"),
        "doi": rec.get("doi"),
        "title": rec.get("title"),
        "year": rec.get("year"),
        "citation_count": rec.get("citation_count") or rec.get("cited_by_count"),
        "abstract": rec.get("abstract"),
        "source": rec.get("source"),
    }


def fetch_one(
    arxiv_id: str,
    *,
    ref_limit: int = 100,
    enrich_abstracts: bool = False,
) -> tuple[dict | None, list[dict], str]:
    """
    Fetch focal + refs via novelty.refs (S2 → OpenAlex).

    Default enrich_abstracts=False: citation graph only needs ids/titles.
    Abstract enrichment is for novelty scoring and can take minutes per paper
    (rate-limit sleeps + dozens of HTTP calls).
    """
    try:
        from novelty.refs import resolve_focal_and_refs
        from novelty.ids import canonical_id, strip_version
    except ImportError:
        from src.novelty.refs import resolve_focal_and_refs  # type: ignore
        from src.novelty.ids import canonical_id, strip_version  # type: ignore

    result = resolve_focal_and_refs(
        arxiv_id,
        max_refs=ref_limit,
        enrich_abstracts=enrich_abstracts,
    )
    if not isinstance(result, dict):
        return None, [], "bad_response"
    focal = result.get("focal")
    refs = result.get("refs") or []
    note = result.get("source") or ""
    if focal and not focal.get("canonical_id"):
        focal["canonical_id"] = canonical_id(
            arxiv_id=strip_version(arxiv_id),
            s2_id=focal.get("s2_id") or focal.get("paperId"),
            doi=focal.get("doi"),
        )
    return focal, refs, note


def collect(
    *,
    source: Source = "library",
    ids_file: Path | None = None,
    limit: int = 0,
    ref_limit: int = 100,
    sleep: float = 0.35,
    force: bool = False,
    max_errors: int = 50,
    db_path: Path | None = None,
    enrich_abstracts: bool = False,
) -> dict[str, Any]:
    """
    Walk library papers, fetch reference lists, store edges.
    Resumable: papers with fetch_status=ok are skipped unless force=True.

    enrich_abstracts=False (default) skips the slow novelty abstract fill —
    citation counts only need reference identities.
    """
    ids = load_library_ids(source=source, ids_file=ids_file, limit=limit)
    if not ids:
        raise SystemExit("No papers to scan. Build the library (corpus.json) first.")

    conn = connect(db_path)
    print(f"[cite] source={source} papers={len(ids)} force={force} sleep={sleep}s")
    print(f"[cite] enrich_abstracts={enrich_abstracts}  (False = fast citation-graph mode)")
    print(f"[cite] db={db_path or 'data/citation_library/graph.sqlite3'}")
    st0 = stats(conn)
    print(f"[cite] existing status: {st0}")
    print("[cite] progress prints every paper on first 5, then every 10…", flush=True)

    ok = skip = empty = errors = fetched = 0
    t0 = time.time()
    interrupted = False

    try:
        for i, aid in enumerate(ids, 1):
            if not force and already_ok(conn, aid):
                skip += 1
                if i % 200 == 0 or i == len(ids):
                    print(
                        f"[cite] {i}/{len(ids)} skip_cached={skip} ok={ok} "
                        f"empty={empty} err={errors} api={fetched}",
                        flush=True,
                    )
                continue

            # Log *before* the network call so a hang is obvious
            if i <= 5 or i % 10 == 0:
                print(f"[cite] fetching {i}/{len(ids)}  {aid} …", flush=True)

            try:
                focal, refs, note = fetch_one(
                    aid,
                    ref_limit=ref_limit,
                    enrich_abstracts=enrich_abstracts,
                )
                fetched += 1
                if i <= 5 or i % 10 == 0:
                    print(
                        f"[cite]   → {aid}  refs={len(refs)}  source={note}",
                        flush=True,
                    )

                try:
                    from novelty.ids import canonical_id, strip_version
                except ImportError:
                    from src.novelty.ids import canonical_id, strip_version  # type: ignore

                focal_cid = None
                if focal:
                    focal_cid = focal.get("canonical_id") or canonical_id(
                        arxiv_id=strip_version(aid)
                    )
                    focal["canonical_id"] = focal_cid
                    upsert_paper(conn, _record_from_novelty(focal))
                else:
                    focal_cid = canonical_id(arxiv_id=strip_version(aid))

                cited_ids: list[str] = []
                for r in refs:
                    if not r.get("canonical_id"):
                        r["canonical_id"] = canonical_id(
                            arxiv_id=r.get("arxiv_id"),
                            s2_id=r.get("s2_id") or r.get("paperId"),
                            doi=r.get("doi"),
                        )
                    if not r.get("canonical_id"):
                        continue
                    upsert_paper(conn, _record_from_novelty(r))
                    cited_ids.append(r["canonical_id"])

                if focal_cid and cited_ids:
                    n = replace_edges(conn, focal_cid, cited_ids)
                    set_fetch_status(
                        conn,
                        aid,
                        status="ok",
                        canonical_id=focal_cid,
                        n_refs=n,
                        source=note,
                    )
                    ok += 1
                else:
                    set_fetch_status(
                        conn,
                        aid,
                        status="empty",
                        canonical_id=focal_cid,
                        n_refs=0,
                        source=note,
                    )
                    empty += 1

                conn.commit()
                pause = sleep
                try:
                    from novelty.refs import suggested_inter_request_sleep
                    pause = suggested_inter_request_sleep(sleep)
                except Exception:
                    try:
                        from src.novelty.refs import suggested_inter_request_sleep  # type: ignore
                        pause = suggested_inter_request_sleep(sleep)
                    except Exception:
                        pass
                if pause > 0:
                    time.sleep(pause)

            except KeyboardInterrupt:
                raise
            except Exception as e:
                errors += 1
                set_fetch_status(conn, aid, status="error", error=str(e)[:300])
                conn.commit()
                if errors <= 8:
                    print(f"[cite] error {aid}: {e}", flush=True)
                if errors >= max_errors:
                    print(f"[cite] stopping: max_errors={max_errors} reached", flush=True)
                    break

            if i % 25 == 0 or i == len(ids):
                elapsed = time.time() - t0
                rate = i / max(elapsed, 1e-6)
                left = len(ids) - i
                eta = left / max(rate, 1e-6) / 60
                print(
                    f"[cite] {i}/{len(ids)}  ok={ok} skip={skip} empty={empty} "
                    f"err={errors} api={fetched}  {elapsed/60:.1f}m  eta~{eta:.0f}m",
                    flush=True,
                )
    except KeyboardInterrupt:
        interrupted = True
        print(f"\n[cite] Graceful stop requested (Ctrl+C). Committing checkpoint to SQLite…", flush=True)
        conn.commit()
    finally:
        st1 = stats(conn)
        conn.close()

    summary = {
        "source": source,
        "n_input": len(ids),
        "ok": ok,
        "skip_cached": skip,
        "empty": empty,
        "errors": errors,
        "api_fetches": fetched,
        "interrupted": interrupted,
        "db_stats": st1,
        "elapsed_min": round((time.time() - t0) / 60, 2),
    }
    if interrupted:
        print(f"[cite] Checkpoint saved successfully. Re-run to resume remaining papers.")
    print(f"[cite] done: {summary}")
    return summary
