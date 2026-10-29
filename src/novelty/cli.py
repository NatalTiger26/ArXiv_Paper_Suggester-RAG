"""CLI for reference-conditional profiles.

  uv run python -m src.novelty.test_metrics
  uv run python -m src.novelty.cli coverage 1806.07572 2501.16496
  uv run python -m src.novelty.cli score 2501.16496 --no-null
  uv run python -m src.novelty.cli score 2501.16496
  uv run python -m src.novelty.cli show 2501.16496
  uv run python -m src.novelty.cli stats
  uv run python -m src.novelty.baseline
"""

from __future__ import annotations

import argparse
import json


def cmd_coverage(args: argparse.Namespace) -> None:
    from .refs import resolve_focal_and_refs

    rows = []
    for aid in args.arxiv_ids:
        r = resolve_focal_and_refs(aid, max_refs=args.max_refs)
        st = r["stats"]
        rows.append(
            {
                "arxiv_id": aid,
                "source": r["source"],
                "n_refs": st["n_refs"],
                "n_with_abstract": st["n_with_abstract"],
                "frac_with_abstract": round(st["frac_with_abstract"], 3),
                "focal_title": (r["focal"].get("title") or "")[:60],
            }
        )
        print(
            f"{aid}: refs={st['n_refs']} abs={st['n_with_abstract']} "
            f"frac={st['frac_with_abstract']:.2f} source={r['source']}",
            flush=True,
        )
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        print(f"Wrote {args.json_out}", flush=True)


def cmd_score(args: argparse.Namespace) -> None:
    from .score import score_paper

    last = None
    for aid in args.arxiv_ids:
        print(f"\n=== scoring {aid} ===", flush=True)
        last = score_paper(
            aid,
            max_refs=args.max_refs,
            use_null=not args.no_null,
            n_null=args.n_null,
            force_refresh_refs=args.refresh,
            center=not args.no_center,
            allow_partial=getattr(args, "allow_partial", False),
        )
        _print_profile(last)
    if args.json_out and last is not None:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(last, f, indent=2, default=str)
        print(f"Wrote {args.json_out}", flush=True)


def cmd_show(args: argparse.Namespace) -> None:
    from .cache import connect, get_score
    from .embed_util import model_and_dim
    from .ids import canonical_id, strip_version

    model, dim = model_and_dim()
    conn = connect()
    for aid in args.arxiv_ids:
        cid = canonical_id(arxiv_id=strip_version(aid))
        row = get_score(conn, cid, model, dim) if cid else None
        if not row:
            print(f"{aid}: no stored score for model={model} dim={dim}", flush=True)
            continue
        _print_profile(row["profile"])


def cmd_stats(args: argparse.Namespace) -> None:
    from .score import cache_stats

    print(json.dumps(cache_stats(), indent=2))


def cmd_test(args: argparse.Namespace) -> None:
    from .test_metrics import main as test_main

    test_main()


def cmd_baseline(args: argparse.Namespace) -> None:
    from .baseline import main as baseline_main
    import sys

    argv = []
    if args.json_out:
        argv = ["--json-out", args.json_out]
    sys.argv = ["baseline"] + argv
    baseline_main()


def _print_profile(p: dict) -> None:
    if not p.get("ok"):
        print(f"  ok=False error={p.get('error')}", flush=True)
        if p.get("message"):
            print(f"  {p['message']}", flush=True)
        print(f"  stats={p.get('stats')}", flush=True)
        return
    print(
        f"  focal={p.get('focal_id')} model={p.get('model')} dim={p.get('dim')} "
        f"centered={p.get('centered')}",
        flush=True,
    )
    print(
        f"  m={_fmt(p.get('m'))}  a={_fmt(p.get('a'))}  "
        f"R2+={_fmt(p.get('R2_plus'))}  Δ={_fmt(p.get('Delta'))}",
        flush=True,
    )
    print(
        f"  D_median={_fmt(p.get('D_median'))}  D_q90={_fmt(p.get('D_q90'))}  "
        f"hint={p.get('typology_hint')}",
        flush=True,
    )
    print(
        f"  d_eff_refs={_fmt(p.get('d_eff_refs'))}  "
        f"chance_R2≈{_fmt(p.get('chance_R2_ols_mean'))}",
        flush=True,
    )
    null = p.get("null") or {}
    if null.get("ok"):
        pct = null.get("percentiles") or {}
        print(
            f"  null: used={null.get('n_null_used')} pool={null.get('pool_candidates')} "
            f"matched_n={null.get('matched_n')}",
            flush=True,
        )
        print(f"  null percentiles: {{{', '.join(f'{k}={_fmt(v)}' for k,v in pct.items())}}}", flush=True)
    else:
        print(
            f"  null: {null.get('reason')} "
            f"(pool={null.get('pool')}, msg={null.get('message', '')})",
            flush=True,
        )
    attr = p.get("attribution") or []
    if attr:
        print("  top parents:", flush=True)
        for a in attr[:5]:
            print(
                f"    {a.get('weight', 0):.3f}  cos={a.get('cos', 0):.3f}  {a.get('ref_id')}",
                flush=True,
            )


def _fmt(v) -> str:
    if v is None:
        return "None"
    try:
        return f"{float(v):.4f}"
    except Exception:
        return str(v)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="src.novelty.cli",
        description="Reference-conditional embedding profiles (offline, separate from RAG)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_cov = sub.add_parser("coverage", help="Ref/abstract coverage (S2/OpenAlex)")
    p_cov.add_argument("arxiv_ids", nargs="+")
    p_cov.add_argument("--max-refs", type=int, default=80)
    p_cov.add_argument("--json-out", default="")
    p_cov.set_defaults(func=cmd_coverage)

    p_sc = sub.add_parser("score", help="Compute profile")
    p_sc.add_argument("arxiv_ids", nargs="+")
    p_sc.add_argument("--max-refs", type=int, default=80)
    p_sc.add_argument("--no-null", action="store_true")
    p_sc.add_argument("--no-center", action="store_true", help="skip cohort mean centering")
    p_sc.add_argument("--n-null", type=int, default=30)
    p_sc.add_argument("--refresh", action="store_true")
    p_sc.add_argument("--allow-partial", action="store_true", help="Fallback to reference titles if abstract coverage < 70%%")
    p_sc.add_argument("--json-out", default="")
    p_sc.set_defaults(func=cmd_score)

    p_sh = sub.add_parser("show", help="Show stored profile")
    p_sh.add_argument("arxiv_ids", nargs="+")
    p_sh.set_defaults(func=cmd_show)

    p_st = sub.add_parser("stats", help="Cache counts")
    p_st.set_defaults(func=cmd_stats)

    p_t = sub.add_parser("test", help="Run synthetic metric unit tests")
    p_t.set_defaults(func=cmd_test)

    p_b = sub.add_parser("baseline", help="Correlate scores vs log n_refs / year")
    p_b.add_argument("--json-out", default="")
    p_b.set_defaults(func=cmd_baseline)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
