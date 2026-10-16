"""Compare stored profiles against log(n_refs) and year baselines.

  uv run python -m src.novelty.baseline
  uv run python -m src.novelty.baseline --json-out results/novelty_baseline.json

Kill criterion (pre-register before interpreting):
  If rank correlations of m / R2_plus / Delta with a 'step vs incremental'
  label are no better than those of log_n_refs and year alone, stop and
  document a negative result.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _load_scores() -> list[dict[str, Any]]:
    from .cache import connect
    from .embed_util import model_and_dim

    model, dim = model_and_dim()
    conn = connect()
    rows = conn.execute(
        """
        SELECT focal_id, profile_json FROM scores
        WHERE model=? AND dim=?
        """,
        (model, dim),
    ).fetchall()
    out = []
    for r in rows:
        p = json.loads(r["profile_json"])
        if not p.get("ok"):
            continue
        cov = p.get("covariates") or {}
        out.append(
            {
                "focal_id": r["focal_id"],
                "m": p.get("m"),
                "a": p.get("a"),
                "R2_plus": p.get("R2_plus"),
                "Delta": p.get("Delta"),
                "D_median": p.get("D_median"),
                "log_n_refs": cov.get("log_n_refs"),
                "year": cov.get("year"),
                "abstract_len": cov.get("abstract_len"),
                "typology_hint": p.get("typology_hint"),
                "chance_R2": p.get("chance_R2_ols_mean"),
                "d_eff_refs": p.get("d_eff_refs"),
            }
        )
    return out


def _corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 5:
        return None
    a = np.asarray(xs, dtype=np.float64)
    b = np.asarray(ys, dtype=np.float64)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "message": "no successful scores in cache"}

    def col(k: str) -> list[float]:
        return [float(r[k]) for r in rows if r.get(k) is not None and r[k] == r[k]]

    report: dict[str, Any] = {
        "n_papers": len(rows),
        "means": {},
        "corr_with_log_n_refs": {},
        "corr_with_year": {},
        "corr_Delta_with_D_median": None,
        "kill_criterion_note": (
            "If metric correlations with external labels do not beat "
            "log_n_refs and year, treat geometry scores as non-informative."
        ),
    }
    for k in ("m", "a", "R2_plus", "Delta", "D_median", "log_n_refs"):
        v = col(k)
        if v:
            report["means"][k] = {
                "mean": float(np.mean(v)),
                "std": float(np.std(v)),
                "min": float(np.min(v)),
                "max": float(np.max(v)),
            }

    ln = col("log_n_refs")
    year = [float(r["year"]) for r in rows if r.get("year")]
    for k in ("m", "R2_plus", "Delta", "D_median"):
        kv = col(k)
        # align lengths carefully
        pairs_ln = [
            (float(r["log_n_refs"]), float(r[k]))
            for r in rows
            if r.get("log_n_refs") is not None and r.get(k) is not None
        ]
        if pairs_ln:
            report["corr_with_log_n_refs"][k] = _corr(
                [a for a, _ in pairs_ln], [b for _, b in pairs_ln]
            )
        pairs_y = [
            (float(r["year"]), float(r[k]))
            for r in rows
            if r.get("year") is not None and r.get(k) is not None
        ]
        if pairs_y:
            report["corr_with_year"][k] = _corr(
                [a for a, _ in pairs_y], [b for _, b in pairs_y]
            )

    pairs_dd = [
        (float(r["Delta"]), float(r["D_median"]))
        for r in rows
        if r.get("Delta") is not None
        and r.get("D_median") is not None
        and r["D_median"] == r["D_median"]
    ]
    if pairs_dd:
        report["corr_Delta_with_D_median"] = _corr(
            [a for a, _ in pairs_dd], [b for _, b in pairs_dd]
        )
        if report["corr_Delta_with_D_median"] is not None and abs(
            report["corr_Delta_with_D_median"]
        ) > 0.85:
            report["warning"] = (
                "Delta is nearly collinear with D_median — blend gain may add "
                "little beyond dispersion."
            )

    # chance comparison
    high_r2 = sum(
        1
        for r in rows
        if r.get("R2_plus") is not None
        and r.get("chance_R2") is not None
        and r["R2_plus"] > (r["chance_R2"] or 0) + 0.1
    )
    report["frac_R2_clearly_above_chance"] = high_r2 / len(rows)

    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Baseline report for novelty scores")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()
    rows = _load_scores()
    rep = summarize(rows)
    print(json.dumps(rep, indent=2))
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
