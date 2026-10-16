"""Synthetic unit tests with known answers (no network).

  uv run python -m src.novelty.test_metrics
"""

from __future__ import annotations

import sys

import numpy as np

from .metrics import compute_profile, nnls_cone_r2, prepare_matrix


def _ok(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        raise AssertionError(name)


def test_identity() -> None:
    d = 64
    y1 = np.zeros(d)
    y1[0] = 1.0
    Y = np.column_stack([y1, np.eye(d)[:, 1:4]])
    x = y1.copy()
    p = compute_profile(x, Y)
    _ok("identity m≈1", abs(p["m"] - 1.0) < 1e-6, str(p["m"]))
    _ok("identity R2≈1", abs(p["R2_plus"] - 1.0) < 1e-5, str(p["R2_plus"]))
    _ok("identity Delta≈0", abs(p["Delta"]) < 1e-5, str(p["Delta"]))


def test_exact_blend() -> None:
    d = 64
    y1 = np.zeros(d)
    y1[0] = 1.0
    y2 = np.zeros(d)
    y2[1] = 1.0
    Y = np.column_stack([y1, y2])
    x = (y1 + y2) / np.sqrt(2.0)
    p = compute_profile(x, Y, ref_ids=["a", "b"])
    _ok("blend m≈1/sqrt2", abs(p["m"] - 1 / np.sqrt(2)) < 1e-5, str(p["m"]))
    _ok("blend R2≈1", abs(p["R2_plus"] - 1.0) < 1e-4, str(p["R2_plus"]))
    _ok("blend Delta≈0.5", abs(p["Delta"] - 0.5) < 1e-4, str(p["Delta"]))
    w = {a["ref_id"]: a["weight"] for a in p["attribution"]}
    _ok("blend weights ~0.5", abs(w.get("a", 0) - 0.5) < 0.05 and abs(w.get("b", 0) - 0.5) < 0.05, str(w))


def test_orthogonal() -> None:
    d = 64
    Y = np.eye(d)[:, :5]
    x = np.zeros(d)
    x[10] = 1.0
    p = compute_profile(x, Y)
    _ok("orthogonal R2≈0", p["R2_plus"] < 1e-6, str(p["R2_plus"]))
    _ok("orthogonal Delta≈0", abs(p["Delta"]) < 1e-6, str(p["Delta"]))


def test_opposite() -> None:
    d = 32
    y1 = np.zeros(d)
    y1[0] = 1.0
    x = -y1
    p = compute_profile(x, y1.reshape(-1, 1))
    _ok("opposite m_plus_sq=0", abs(p["m_plus_sq"]) < 1e-12, str(p["m_plus_sq"]))
    _ok("opposite R2≈0", p["R2_plus"] < 1e-6, str(p["R2_plus"]))


def test_permutation() -> None:
    rng = np.random.default_rng(0)
    d, n = 40, 6
    Y = rng.normal(size=(d, n))
    Y = Y / np.linalg.norm(Y, axis=0, keepdims=True)
    x = _unit(Y[:, 0] * 0.7 + Y[:, 1] * 0.3)
    p1 = compute_profile(x, Y, ref_ids=[str(i) for i in range(n)])
    perm = rng.permutation(n)
    Y2 = Y[:, perm]
    ids2 = [str(i) for i in perm]
    p2 = compute_profile(x, Y2, ref_ids=ids2)
    for k in ("m", "a", "R2_plus", "Delta", "D_median"):
        _ok(f"perm {k}", abs((p1[k] or 0) - (p2[k] or 0)) < 1e-8, f"{p1[k]} vs {p2[k]}")


def test_duplication() -> None:
    d = 32
    y1 = np.zeros(d)
    y1[0] = 1.0
    y2 = np.zeros(d)
    y2[1] = 1.0
    Y = np.column_stack([y1, y2])
    x = y1.copy()
    p1 = compute_profile(x, Y)
    Ydup = np.column_stack([y1, y1, y2])
    p2 = compute_profile(x, Ydup)
    _ok("dup R2 unchanged", abs(p1["R2_plus"] - p2["R2_plus"]) < 1e-5)
    # D may change — just record
    print(f"    (info) D_median {p1['D_median']:.4f} → {p2['D_median']:.4f} under dup")


def test_chance_level() -> None:
    rng = np.random.default_rng(1)
    d, n, trials = 64, 8, 40
    r2s = []
    for _ in range(trials):
        Y = rng.normal(size=(d, n))
        Y = Y / np.linalg.norm(Y, axis=0, keepdims=True)
        x = rng.normal(size=d)
        x = x / np.linalg.norm(x)
        p = compute_profile(x, Y)
        r2s.append(p["R2_plus"])
    mean_r2 = float(np.mean(r2s))
    bound = n / d
    _ok(
        "chance R2 mean ≤ n/d + margin",
        mean_r2 <= bound + 0.15,
        f"mean={mean_r2:.3f} bound={bound:.3f}",
    )


def test_nnls_crosscheck() -> None:
    rng = np.random.default_rng(2)
    d, n = 50, 7
    Y = rng.normal(size=(d, n))
    x = np.abs(rng.normal(size=d))  # positive-ish combination
    x = x / np.linalg.norm(x)
    Y = Y / np.linalg.norm(Y, axis=0, keepdims=True)
    r2_a, lam_a = nnls_cone_r2(x, Y)
    # independent: scipy nnls residual
    from scipy.optimize import nnls

    lam_b, _ = nnls(Y, x)
    proj = Y @ lam_b
    r2_b = float(np.dot(proj, proj))
    _ok("nnls cross residual", abs(r2_a - r2_b) < 1e-6, f"{r2_a} vs {r2_b}")


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / max(np.linalg.norm(v), 1e-12)


def main() -> None:
    print("novelty metric unit tests")
    tests = [
        test_identity,
        test_exact_blend,
        test_orthogonal,
        test_opposite,
        test_permutation,
        test_duplication,
        test_chance_level,
        test_nnls_crosscheck,
    ]
    failed = 0
    for t in tests:
        print(f"\n{t.__name__}")
        try:
            t()
        except Exception as e:
            failed += 1
            print(f"  ERROR: {e}")
    print(f"\n{'ALL PASS' if failed == 0 else f'{failed} TEST(S) FAILED'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
