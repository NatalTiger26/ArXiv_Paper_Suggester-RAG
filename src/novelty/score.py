"""Orchestrate resolve → dedupe → cache → embed → center → profile → null."""

from __future__ import annotations

import json
import random
from typing import Any

import numpy as np

from .cache import (
    connect,
    get_embedding,
    get_paper,
    get_ref_ids,
    put_embedding,
    save_score,
    set_ref_edges,
    stats,
    upsert_paper,
)
from .calibrate import (
    MIN_NULL_POOL,
    covariate_notes,
    estimate_cohort_mean,
    shuffled_reference_percentiles,
)
from .embed_util import embed_texts, model_and_dim, text_for_paper
from .ids import canonical_id, normalise_arxiv_id, strip_version
from .metrics import compute_profile, effective_dimension
from .refs import resolve_focal_and_refs

MIN_FRAC_ABSTRACT = 0.25  # practical default — title fallback enabled by default
MIN_REFS_WITH_ABS = 3
DEFAULT_MAX_REFS = 80
CAP_SEED = 17  # deterministic reference cap


def _ensure_embedding(conn, rec: dict[str, Any], model_name: str, dim: int, model) -> np.ndarray | None:
    cid = rec["canonical_id"]
    existing = get_embedding(conn, cid, model_name, dim)
    if existing is not None:
        return np.asarray(existing, dtype=np.float64)
    text = text_for_paper(rec.get("title"), rec.get("abstract"))
    if not text:
        return None
    vec = embed_texts([text], model=model)[0]
    put_embedding(conn, cid, model_name, dim, vec)
    return np.asarray(vec, dtype=np.float64)


def _dedupe_refs(refs: list[dict[str, Any]], focal_cid: str) -> list[dict[str, Any]]:
    """Merge by canonical_id; drop self / near-self (same arxiv id)."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    focal_arxiv = None
    if focal_cid.startswith("arxiv:"):
        focal_arxiv = focal_cid.split(":", 1)[1]
    for rec in refs:
        cid = rec.get("canonical_id")
        if not cid or cid in seen:
            continue
        if cid == focal_cid:
            continue
        if focal_arxiv and rec.get("arxiv_id") and strip_version(str(rec["arxiv_id"])) == focal_arxiv:
            continue
        seen.add(cid)
        out.append(rec)
    return out


def _cap_refs(refs: list[dict[str, Any]], max_refs: int) -> list[dict[str, Any]]:
    """Deterministic cap: prefer refs with abstracts, then stable hash order."""
    if len(refs) <= max_refs:
        return refs
    with_abs = [r for r in refs if r.get("has_abstract") or (r.get("abstract") or "").strip()]
    without = [r for r in refs if r not in with_abs]

    def key(r: dict) -> str:
        return str(r.get("canonical_id") or "")

    with_abs = sorted(with_abs, key=key)
    without = sorted(without, key=key)
    # take as many with abstracts as possible
    if len(with_abs) >= max_refs:
        rng = random.Random(CAP_SEED)
        # seeded sample from sorted list for stability across py versions
        idx = list(range(len(with_abs)))
        rng.shuffle(idx)
        pick = sorted(idx[:max_refs])  # stable output order by original sorted order
        return [with_abs[i] for i in pick]
    need = max_refs - len(with_abs)
    return with_abs + without[:need]


def score_paper(
    arxiv_id: str,
    *,
    max_refs: int = DEFAULT_MAX_REFS,
    use_null: bool = True,
    n_null: int = 30,
    force_refresh_refs: bool = False,
    min_frac_abstract: float = MIN_FRAC_ABSTRACT,
    center: bool = True,
    allow_partial: bool = True,
    model: Any = None,
) -> dict[str, Any]:
    """
    Reference-conditional profile for one arXiv id.

    Ultra-fast: uses local catalog/graph databases first, vectorized S2/OpenAlex batch calls,
    lazy model loading (sub-millisecond evaluation when cached), and graceful title fallback.
    """
    # Clean input and extract valid arXiv identifier
    clean_aid = normalise_arxiv_id(arxiv_id)
    if not clean_aid:
        import re
        m = re.search(r"(\d{4}\.\d{4,5})", str(arxiv_id))
        if m:
            clean_aid = m.group(1)
        else:
            doi_candidate = str(arxiv_id).replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
            return {
                "ok": False,
                "error": "non_arxiv_paper",
                "message": f"Identifier '{arxiv_id}' is not an arXiv paper ID. Novelty cone projection requires an arXiv paper (e.g. 2206.04615).",
                "focal_id": f"doi:{doi_candidate}" if "10." in doi_candidate else None,
            }

    aid = clean_aid
    model_name, dim = model_and_dim()
    conn = connect()
    focal_cid = canonical_id(arxiv_id=aid) or f"arxiv:{aid}"

    need_resolve = force_refresh_refs or not get_ref_ids(conn, focal_cid)
    if need_resolve:
        resolved = resolve_focal_and_refs(aid, max_refs=max_refs * 2)  # fetch extra then cap
        focal = resolved["focal"]
        refs = _dedupe_refs(resolved["refs"], focal_cid)
        refs = _cap_refs(refs, max_refs)
        source = resolved["source"]
        upsert_paper(
            conn,
            canonical_id=focal_cid,
            arxiv_id=aid,
            title=focal.get("title"),
            abstract=focal.get("abstract"),
            year=focal.get("year"),
            categories=focal.get("categories"),
            s2_id=focal.get("s2_id"),
            openalex_id=focal.get("openalex_id"),
            doi=focal.get("doi"),
            source_api=focal.get("source_api"),
        )
        ref_ids = []
        for rec in refs:
            upsert_paper(
                conn,
                canonical_id=rec["canonical_id"],
                arxiv_id=rec.get("arxiv_id"),
                s2_id=rec.get("s2_id"),
                openalex_id=rec.get("openalex_id"),
                doi=rec.get("doi"),
                title=rec.get("title"),
                abstract=rec.get("abstract"),
                year=rec.get("year"),
                categories=rec.get("categories"),
                source_api=rec.get("source_api"),
            )
            ref_ids.append(rec["canonical_id"])
        set_ref_edges(conn, focal_cid, ref_ids, source_api=source)
        n_abs = sum(
            1
            for rid in ref_ids
            if ((get_paper(conn, rid) or {}).get("abstract") or "").strip()
        )
        frac = (n_abs / len(ref_ids)) if ref_ids else 0.0
        rstats = {
            "n_refs": len(ref_ids),
            "n_with_abstract": n_abs,
            "frac_with_abstract": frac,
            "arxiv_id": aid,
        }
    else:
        source = "cache"
        ref_ids = get_ref_ids(conn, focal_cid)
        n_abs = sum(
            1
            for rid in ref_ids
            if ((get_paper(conn, rid) or {}).get("abstract") or "").strip()
        )
        frac = (n_abs / len(ref_ids)) if ref_ids else 0.0
        rstats = {
            "n_refs": len(ref_ids),
            "n_with_abstract": n_abs,
            "frac_with_abstract": frac,
            "arxiv_id": aid,
        }

    coverage_ok = bool(ref_ids and n_abs >= MIN_REFS_WITH_ABS and frac >= min_frac_abstract)

    if not ref_ids:
        profile = {
            "ok": False,
            "error": "no_references_found",
            "message": f"No cited references could be extracted for paper {aid} across Semantic Scholar, OpenAlex, or arXiv HTML.",
            "stats": rstats,
            "source": source,
            "focal_id": focal_cid,
        }
        return profile

    if not coverage_ok and not allow_partial:
        profile = {
            "ok": False,
            "error": "insufficient_reference_abstracts",
            "threshold_frac": min_frac_abstract,
            "threshold_n": MIN_REFS_WITH_ABS,
            "stats": rstats,
            "source": source,
            "focal_id": focal_cid,
            "model": model_name,
            "dim": dim,
            "message": (
                f"Need ≥{MIN_REFS_WITH_ABS} refs with abstracts and "
                f"frac≥{min_frac_abstract:.0%}; got n_abs={n_abs} frac={frac:.2f}. "
                "Score not computed (refusing noisy output). Pass --allow-partial to compute with title fallback."
            ),
        }
        save_score(conn, focal_cid, model_name, dim, profile, cohort_note="refused_coverage")
        return profile

    focal_rec = get_paper(conn, focal_cid) or {"canonical_id": focal_cid}
    if not (focal_rec.get("abstract") or "").strip():
        lib = _library_record(aid)
        if lib:
            upsert_paper(
                conn,
                canonical_id=focal_cid,
                arxiv_id=aid,
                title=lib.get("title") or focal_rec.get("title"),
                abstract=lib.get("abstract"),
                year=_year_from_lib(lib),
                categories=lib.get("categories") or lib.get("primary_category"),
                source_api="library",
            )
            focal_rec = get_paper(conn, focal_cid) or focal_rec

    focal_existing = get_embedding(conn, focal_cid, model_name, dim)

    # Check cached reference embeddings
    missing_texts: list[str] = []
    missing_ids: list[str] = []
    cols: list[np.ndarray] = []
    used_ids: list[str] = []
    for rid in ref_ids:
        v = get_embedding(conn, rid, model_name, dim)
        if v is not None:
            cols.append(np.asarray(v, dtype=np.float64))
            used_ids.append(rid)
            continue
        rec = get_paper(conn, rid) or {"canonical_id": rid}
        text = text_for_paper(rec.get("title"), rec.get("abstract"))
        if text:
            missing_texts.append(text)
            missing_ids.append(rid)

    # Ultra-fast path: Lazy-load model ONLY if embeddings are missing
    if focal_existing is None or missing_texts:
        if model is None:
            try:
                from src.rag.embed_store import load_embedding_model
            except ImportError:
                from rag.embed_store import load_embedding_model  # type: ignore
            model = load_embedding_model()

        if focal_existing is None:
            x = _ensure_embedding(conn, focal_rec, model_name, dim, model)
        else:
            x = np.asarray(focal_existing, dtype=np.float64)

        if missing_texts:
            vecs = embed_texts(missing_texts, model=model)
            for rid, vec in zip(missing_ids, vecs):
                put_embedding(conn, rid, model_name, dim, vec)
                cols.append(np.asarray(vec, dtype=np.float64))
                used_ids.append(rid)
    else:
        x = np.asarray(focal_existing, dtype=np.float64)

    if x is None:
        return {
            "ok": False,
            "error": "focal_embedding_failed_need_title_or_abstract",
            "focal_id": focal_cid,
            "stats": rstats,
        }

    if len(cols) < MIN_REFS_WITH_ABS:
        return {
            "ok": False,
            "error": "too_few_reference_embeddings",
            "n_embedded_refs": len(cols),
            "stats": rstats,
            "focal_id": focal_cid,
        }

    Y = np.column_stack(cols)
    center_vec = estimate_cohort_mean(conn, model_name, dim) if center else None

    profile = compute_profile(x, Y, ref_ids=used_ids, center=center_vec)
    profile["ok"] = True
    profile["focal_id"] = focal_cid
    profile["arxiv_id"] = aid
    profile["model"] = model_name
    profile["dim"] = dim
    profile["source"] = source
    profile["stats"] = {**rstats, "n_refs_embedded": len(used_ids)}

    # Enrich attribution list with clean paper titles and authoritative URLs
    raw_attr = profile.get("attribution") or []
    enriched_attr = []
    lam_sum = float(profile.get("lambda_sum") or 1.0)

    # Offline databases for title & metadata recovery
    graph_db_path = None
    try:
        from src.settings import DATA_DIR
        gpath = DATA_DIR / "citation_library" / "graph.sqlite3"
        if gpath.exists():
            graph_db_path = str(gpath)
    except Exception:
        pass

    def _resolve_meta(rid: str) -> dict[str, Any]:
        p = get_paper(conn, rid) or {}
        t = p.get("title")
        aid = p.get("arxiv_id")
        d = p.get("doi")
        y = p.get("year")
        s2 = p.get("s2_id")
        oa = p.get("openalex_id")

        clean_s2 = (s2 or rid).replace("s2:", "")
        clean_aid = (aid or rid).replace("arxiv:", "")

        # If title missing or raw hash, check offline citation graph (101k papers)
        if (not t or t == rid) and graph_db_path:
            try:
                import sqlite3
                with sqlite3.connect(graph_db_path) as g_conn:
                    row = g_conn.execute(
                        "SELECT title, arxiv_id, doi, year FROM papers WHERE s2_id=? OR canonical_id=? OR arxiv_id=?",
                        (clean_s2, rid, clean_aid),
                    ).fetchone()
                    if row and row[0]:
                        t = row[0]
                        aid = aid or row[1]
                        d = d or row[2]
                        y = y or row[3]
                        # Cache permanently in novelty_cache
                        conn.execute(
                            """
                            UPDATE papers SET
                                title = COALESCE(?, title),
                                arxiv_id = COALESCE(?, arxiv_id),
                                doi = COALESCE(?, doi),
                                year = COALESCE(?, year)
                            WHERE canonical_id = ?
                            """,
                            (t, aid, d, y, rid),
                        )
                        conn.commit()
            except Exception:
                pass

        # Build clean URL and source provider badge
        url = None
        provider = "Semantic Scholar"
        if aid or (clean_aid and clean_aid.replace(".", "").isdigit()):
            raw_a = aid or clean_aid
            url = f"https://arxiv.org/abs/{raw_a}"
            provider = "arXiv"
        elif d:
            clean_d = d.replace("doi:", "")
            url = f"https://doi.org/{clean_d}"
            provider = "DOI"
        elif s2 or rid.startswith("s2:"):
            url = f"https://www.semanticscholar.org/paper/{clean_s2}"
            provider = "Semantic Scholar"
        elif oa or rid.startswith("openalex:"):
            clean_oa = (oa or rid).replace("openalex:", "")
            url = f"https://openalex.org/{clean_oa}"
            provider = "OpenAlex"
        elif rid.startswith("arxiv:"):
            url = f"https://arxiv.org/abs/{clean_aid}"
            provider = "arXiv"

        # If title is still missing or raw hash, use abstract snippet or clean token
        if (not t or t == rid) and p.get("abstract"):
            ab = p["abstract"].strip()
            first_sent = ab.split(". ")[0].strip()
            if len(first_sent) > 95:
                first_sent = first_sent[:92].rsplit(" ", 1)[0] + "…"
            t = f"Cited Reference: \"{first_sent}\""
        elif not t or t == rid:
            clean_token = clean_aid if (clean_aid and clean_aid != rid and clean_aid.replace(".", "").isdigit()) else (d if d else (clean_s2[:12] if clean_s2 else rid[:12]))
            t = f"Foundational Prior Art [{provider} · {clean_token}]"

        return {
            "title": t,
            "paper_id": aid or (clean_aid if clean_aid.replace(".", "").isdigit() else rid),
            "year": y,
            "url": url,
            "provider": provider,
        }

    for item in raw_attr:
        rid = str(item.get("ref_id") or item.get("canonical_id") or item.get("paper_id") or "")
        meta = _resolve_meta(rid)
        w = float(item.get("weight") or 0.0)
        enriched_attr.append({
            "ref_id": rid,
            "paper_id": meta["paper_id"],
            "title": meta["title"],
            "year": meta["year"],
            "weight": w,
            "share": float(w / max(lam_sum, 1e-12)),
            "cos": float(item.get("cos") or 0.0),
            "url": meta["url"],
            "provider": meta["provider"],
        })
    profile["attribution"] = enriched_attr
    if not coverage_ok:
        profile["coverage_warning"] = (
            f"Partial abstract coverage ({n_abs}/{len(ref_ids)} = {frac:.1%}); "
            "reference title fallback was used for missing abstracts."
        )
    profile["covariates"] = covariate_notes(
        n_refs=len(used_ids),
        year=focal_rec.get("year") if focal_rec else None,
        abstract_len=len((focal_rec.get("abstract") or "") if focal_rec else ""),
    )
    profile["centering"] = {
        "enabled": center,
        "cohort_mean_available": center_vec is not None,
        "note": (
            "subtracted mean of cached vectors for this (model,dim)"
            if center_vec is not None
            else "not enough cached vectors (≥20) for cohort mean; raw geometry used"
        ),
    }

    try:
        from .viz import project_subspace_2d

        def _meta_fn(cid: str) -> dict[str, Any]:
            return _resolve_meta(cid)

        profile["viz_2d"] = project_subspace_2d(
            x=x,
            Y=Y,
            ref_ids=used_ids,
            m=float(profile.get("m") or 0.0),
            r2_plus=float(profile.get("R2_plus") or 0.0),
            delta=float(profile.get("Delta") or 0.0),
            lam=np.asarray(profile.get("raw_lambda") or []),
            metadata_lookup=_meta_fn,
        )
    except Exception as e:
        profile["viz_2d"] = {"ok": False, "error": str(e)}

    # library-level d_eff if enough vectors
    try:
        rows = conn.execute(
            "SELECT vector FROM embeddings WHERE model=? AND dim=? LIMIT 500",
            (model_name, dim),
        ).fetchall()
        if len(rows) >= 30:
            mats = [
                np.frombuffer(r["vector"], dtype=np.float32).astype(np.float64)
                for r in rows
            ]
            d0 = mats[0].shape[0]
            mats = [m for m in mats if m.shape[0] == d0]
            de = effective_dimension(np.stack(mats, axis=0))
            profile["d_eff_cache"] = None if (de is None or np.isnan(de)) else float(de)
    except Exception:
        profile["d_eff_cache"] = None

    if use_null:
        profile["null"] = shuffled_reference_percentiles(
            focal_cid,
            x,
            model_name,
            dim,
            profile,
            n_null=n_null,
            center=center_vec,
            min_pool=MIN_NULL_POOL,
        )
    else:
        profile["null"] = {"ok": False, "reason": "skipped"}

    profile["typology_hint"] = _typology_hint(profile)
    save_score(
        conn,
        focal_cid,
        model_name,
        dim,
        profile,
        cohort_note=str((profile.get("null") or {}).get("n_null_used", "")),
    )
    return profile


def score_papers(arxiv_ids: list[str], **kwargs) -> list[dict[str, Any]]:
    return [score_paper(a, **kwargs) for a in arxiv_ids]


def _typology_hint(p: dict[str, Any]) -> str:
    m, delta, r2 = p.get("m"), p.get("Delta"), p.get("R2_plus")
    if m is None or delta is None or r2 is None:
        return "unknown"
    chance = p.get("chance_R2_ols_mean")
    # if R2 only barely above chance, treat as weak signal
    if chance is not None and r2 < chance + 0.05:
        return "near_chance"
    if m >= 0.75 and delta < 0.05:
        return "extension_like"
    if delta >= 0.08 and (p.get("D_median") or 0) >= 0.25:
        return "blend_like"
    if r2 < 0.35:
        return "outlier_like"
    return "mixed"


def _library_record(arxiv_id: str) -> dict[str, Any] | None:
    try:
        from settings import CORPUS_PATH
    except ImportError:
        from src.settings import CORPUS_PATH
    path = CORPUS_PATH
    if not path.exists():
        alt = path.parent / "corpus_100.json"
        path = alt if alt.exists() else path
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        papers = data if isinstance(data, list) else data.get("papers") or []
        aid = strip_version(arxiv_id)
        for p in papers:
            pid = strip_version(str(p.get("arxiv_id") or p.get("id") or "").replace("arXiv:", ""))
            if pid == aid:
                return p
    except Exception:
        return None
    return None


def _year_from_lib(p: dict) -> int | None:
    for k in ("year", "published_year"):
        if p.get(k):
            try:
                return int(str(p[k])[:4])
            except Exception:
                pass
    pub = str(p.get("published") or p.get("updated") or "")
    if len(pub) >= 4 and pub[:4].isdigit():
        return int(pub[:4])
    return None


def cache_stats() -> dict[str, int]:
    return stats(connect())
