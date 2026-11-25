"""
General-n Exact Formula Autoresearch Harness (Target: Base < 2.0 for Dense Perm_n).

Evaluates any candidate general-n formula `candidate_perm(A_wire, p)` across n = 3..8
using strict TrackedFieldElement wire accounting over dual 61-bit primes.

Scoring:
  - Hard Gate: 100% exact match against glynn_perm_mod on dense random matrices over
    PRIME_61_A (2^61 - 1) and PRIME_61_B (2^61 - 31) for every n in ns.
  - Primary Fitness: Effective asymptotic base b_eff = (L_mul(n_max) / L_mul(n_min))^(1/(n_max - n_min))
    and terminal doubling ratio L_mul(n_max) / L_mul(n_max - 1).
  - Target: Beat Gen 10 Mechanism F2 (asymptotic base 2.0) with b_eff < 2.000.

Usage:
  python3 general_n_formula_autoresearch.py --candidate candidate_formula.py --max-n 8
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import random
import sys
import time

from dag_verification import (
    PRIME_61_A,
    PRIME_61_B,
    OpCounter,
    TrackedFieldElement,
    glynn_perm_mod,
)

# Reference non-scalar multiplication counts for Glynn and Gen 10 Mechanism F2
GLYNN_MULS = {n: (n - 1) * (1 << (n - 1)) for n in range(2, 17)}
GEN10_F2_MULS = {
    2: 2, 3: 7, 4: 18, 5: 41, 6: 83, 7: 169, 8: 318,
    9: 619, 10: 1152, 11: 2135, 12: 3941,
}


def load_candidate(path: str):
    spec = importlib.util.spec_from_file_location("candidate_mod", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load candidate module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "candidate_perm"):
        raise AttributeError(f"{path} must define candidate_perm(A_wire, p) -> TrackedFieldElement")
    return mod.candidate_perm


def evaluate_general_n_formula(
    candidate_fn,
    ns: tuple[int, ...] = (3, 4, 5, 6, 7, 8),
    trials_per_prime: int = 8,
    seed: int = 2026,
) -> dict:
    rng = random.Random(seed)
    results = {}
    t0 = time.time()

    for n in ns:
        last_counter = None
        last_depth = None
        for p in (PRIME_61_A, PRIME_61_B):
            for trial in range(trials_per_prime):
                A_int = [[rng.randrange(p) for _ in range(n)] for _ in range(n)]
                counter = OpCounter()
                A_wire = [
                    [TrackedFieldElement(A_int[r][c], p, counter) for c in range(n)]
                    for r in range(n)
                ]
                out = candidate_fn(A_wire, p)
                if not isinstance(out, TrackedFieldElement):
                    return {
                        "passed": False,
                        "reason": f"Output at n={n} is not a TrackedFieldElement (got {type(out)})",
                    }
                ref = glynn_perm_mod(A_int, p)
                if out.val != ref:
                    return {
                        "passed": False,
                        "reason": f"Exactness failure at n={n}, p={p}, trial={trial}: got {out.val}, expected {ref}",
                    }
                last_counter = counter
                last_depth = out.depth

        results[n] = {
            **last_counter.as_dict(),
            "depth": last_depth,
            "glynn_muls": GLYNN_MULS[n],
            "gen10_f2_muls": GEN10_F2_MULS.get(n),
        }

    # Compute incremental doubling ratios L_mul(n) / L_mul(n-1)
    ratios = {}
    for i in range(1, len(ns)):
        n_prev, n_cur = ns[i - 1], ns[i]
        m_prev = results[n_prev]["non_scalar_muls"]
        m_cur = results[n_cur]["non_scalar_muls"]
        ratios[f"{n_prev}->{n_cur}"] = m_cur / max(m_prev, 1)

    # Fit log2(L_mul(n)) = a + n * log2(b_eff) on the upper half of ns (n >= 5)
    fit_ns = [n for n in ns if n >= 5]
    if len(fit_ns) >= 2:
        xs = [float(n) for n in fit_ns]
        ys = [math.log(results[n]["non_scalar_muls"]) for n in fit_ns]
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
        b_eff = math.exp(slope)
    else:
        b_eff = float("nan")

    beats_gen10 = all(
        results[n]["non_scalar_muls"] < GEN10_F2_MULS[n]
        for n in ns if n in GEN10_F2_MULS and n >= 5
    )

    return {
        "passed": True,
        "elapsed_s": time.time() - t0,
        "per_n": results,
        "doubling_ratios": ratios,
        "fitted_base_n_ge_5": b_eff,
        "beats_gen10_f2_all_n_ge_5": beats_gen10,
    }


def _baseline_glynn_tracked(A_wire, p: int) -> TrackedFieldElement:
    """Reference Glynn formula implemented strictly with TrackedFieldElement."""
    n = len(A_wire)
    N = 1 << (n - 1)
    total = None
    for mask in range(N):
        sign = -1 if (bin(mask).count("1") & 1) else 1
        prod = None
        for r in range(n):
            row = A_wire[r]
            s = row[0]
            for c in range(1, n):
                if (mask >> (c - 1)) & 1:
                    s = s - row[c]
                else:
                    s = s + row[c]
            prod = s if prod is None else (prod * s)
        if total is None:
            total = prod if sign == 1 else ((-1 % p) * prod)
        else:
            total = (total + prod) if sign == 1 else (total - prod)
    return pow(N, -1, p) * total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", type=str, default=None, help="Path to candidate_formula.py")
    ap.add_argument("--max-n", type=int, default=8)
    ap.add_argument("--trials", type=int, default=5)
    args = ap.parse_args()

    ns = tuple(range(3, args.max_n + 1))
    if args.candidate:
        fn = load_candidate(args.candidate)
        label = args.candidate
    else:
        fn = _baseline_glynn_tracked
        label = "Built-in Glynn Baseline"

    print(f"=== General-n Formula Autoresearch Verifier | Candidate: {label} ===")
    rep = evaluate_general_n_formula(fn, ns=ns, trials_per_prime=args.trials)
    if not rep["passed"]:
        print(f"[FAIL] {rep['reason']}")
        sys.exit(1)

    print(f"{'n':>3} | {'Non-Scalar Muls':>15} | {'Gen10 F2':>10} | {'Total Gates':>11} | {'Depth':>5}")
    print("-" * 58)
    for n in ns:
        r = rep["per_n"][n]
        print(
            f"{n:3d} | {r['non_scalar_muls']:15,d} | {str(r['gen10_f2_muls']):>10} | "
            f"{r['total_gates']:11,d} | {r['depth']:5d}"
        )
    print("\nDoubling ratios L_mul(n)/L_mul(n-1):")
    for k, v in rep["doubling_ratios"].items():
        print(f"  {k}: {v:.4f}")
    print(f"Fitted exponential base (n >= 5): {rep['fitted_base_n_ge_5']:.4f}  (target: < 2.0000)")
    print(f"Beats Gen 10 F2 for all n >= 5:   {rep['beats_gen10_f2_all_n_ge_5']}")
    print(f"Elapsed: {rep['elapsed_s']:.2f}s")


if __name__ == "__main__":
    main()