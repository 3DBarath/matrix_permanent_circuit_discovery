"""
Generation 10 Champion (Mode A, Minimum Non-Scalar Multiplications):
Mechanism F2 (Simultaneous Upper-Degree Interpolation + Dual-Ball Annihilation
via Low-Rank Schur-Complement Butterfly) for the n x n Matrix Permanent.

Achieves the exact theoretical non-scalar multiplication floor of linear
degree-parity compression on the Boolean hypercube:
    T(1) = 0
    T(k) = min_{1 <= a <= k//2} [ T(a) + T(k - a) + K(n, k) - K(n, a - 2) - K(n, k - a - 2) ]
    Opt(n) = min_{1 <= a <= n//2} [ T(a) + T(n - a) + C(n, a) ]

How the Schur-Complement Butterfly works (no N x N matrix inversions):
For an interp_ann2(a, b, k = a + b) node with 2 <= a <= b and k < n - 1:
1. Annihilate V (deg b) on the low-weight ball Z_b = {|y| <= b - 2} (fused).
2. Annihilate U (deg a) on the high-weight ball Z_a = {|y| >= m - (a - 2)} (fused).
3. Drop r = K(n, a - 2) weight-k pivot masks D subset P_k \ Z_b and multiply ONLY
   on the remaining K(n, k) - K(n, a - 2) - K(n, b - 2) masks.
4. Interpolate W^(0) via O(m * 2^m) Mobius-Zeta, solve the tiny r x r incidence
   system M * lambda = -W^(0)[Z_a] (for a = 2, r = 1 and M = [[1]], 0 scalar muls!),
   and add the sparse superset correction sum_t lambda_t * 1[y_t subset z].

Verified exact modulo both 61-bit primes (n = 3..12):
- n = 6:  83 muls (-48.1% vs Glynn 160, 1 scalar mul)
- n = 7:  169 muls (-56.0% vs Glynn 384, 1 scalar mul)
- n = 8:  318 muls (-64.5% vs Glynn 896, 1 scalar mul)
- n = 9:  619 muls (-69.8% vs Glynn 2,048, 1 scalar mul)
- n = 10: 1,152 muls (-75.0% vs Glynn 4,608)
- n = 11: 2,135 muls (-79.2% vs Glynn 10,240)
- n = 12: 3,941 muls (-82.5% vs Glynn 22,528)
- n = 14: 14,485 muls (-86.4% vs Glynn 106,496)
- n = 16: 52,488 muls (-89.3% vs Glynn 491,520, 1.602 * 2^15)
"""

from __future__ import annotations

from math import comb
import sympy as sp

from src.general_n_butterfly_opttree import (
    K,
    _s_add,
    _s_mul,
    _s_sub,
    eval_row_range,
    fwht_pruned,
    hamming_weight,
    mobius_annihilate,
    mobius_interpolate,
)


def solve_f2_plan(n: int) -> tuple[int, tuple, dict]:
    """Exact Mechanism F2 DP: cost(a, b) = K(n, a + b) - K(n, a - 2) - K(n, b - 2)."""
    T: dict[int, tuple[int, tuple]] = {1: (0, ("row",))}
    for k in range(2, n):
        best = (10**18, None)
        for a in range(1, k // 2 + 1):
            b = k - a  # a <= b
            c = K(n, k) - K(n, a - 2) - K(n, b - 2)
            tot = T[a][0] + T[b][0] + c
            if tot < best[0]:
                if k == n - 1:
                    best = (tot, ("dual_ann", a, b, a - 2, b - 2, c))
                elif a < 2:
                    best = (tot, ("interp_ann1", b, a, k, b - 2, c))
                else:
                    best = (tot, ("interp_ann2", a, b, k, a - 2, b - 2, c))
        T[k] = best

    best_root = (10**18, None)
    for a in range(1, n // 2 + 1):
        b = n - a
        pm = comb(n, a)
        tot = T[a][0] + T[b][0] + pm
        if tot < best_root[0]:
            best_root = (tot, ("parseval", a, b, pm))
    return best_root[0], best_root[1], T


def _find_schur_pivots(m: int, k: int, d_a: int, d_b: int, weights: list[int]):
    """
    Find r = K(m+1, d_a) dropped pivot masks D subset P_k \ Z_b (preferring weight-k
    masks where e_y(z) = 1[y subset z]) such that M[z, t] = e_{y_t}(z) on Z_a is
    invertible over Q. For d_a = 0 (r = 1), M = [[1]] with zero matrix inversion.
    """
    N = 1 << m
    Z_a = [z for z in range(N) if weights[z] >= m - d_a]
    r = len(Z_a)

    if r == 1:
        y0 = (1 << k) - 1
        return [y0], Z_a, sp.Matrix([[1]]), True

    wt_k = [y for y in range(N) if weights[y] == k]
    B = sp.Matrix([[1 if (y & z) == y else 0 for y in wt_k] for z in Z_a])
    _, piv_cols = B.rref()
    if len(piv_cols) == r:
        D = [wt_k[c] for c in piv_cols]
        M = B[:, list(piv_cols)]
        return D, Z_a, M.inv(), True

    cand = [y for y in range(N) if d_b < weights[y] <= k]

    def e_val(y: int, z: int) -> int:
        if (y & z) != y:
            return 0
        diff = weights[z] - weights[y]
        lim = k - weights[y]
        return sum(((-1) ** t) * comb(diff, t) for t in range(min(diff, lim) + 1))

    B_full = sp.Matrix([[e_val(y, z) for y in cand] for z in Z_a])
    _, piv_cols = B_full.rref()
    if len(piv_cols) != r:
        raise RuntimeError(f"Schur system rank-deficient for m={m}, k={k}, d_a={d_a}, d_b={d_b}")
    D = [cand[c] for c in piv_cols]
    M = B_full[:, list(piv_cols)]
    return D, Z_a, M.inv(), False


def _s_scale_exact(sw, rat: sp.Rational, p: int):
    if sw is None or rat == 0:
        return None
    w, s = sw
    num, den = int(rat.p), int(rat.q)
    if num == 1 and den == 1:
        return (w, s)
    if num == -1 and den == 1:
        return (w, -s)
    coeff = (num * pow(den, -1, p)) % p
    return (coeff * w, s)


def build_schur_f2_circuit(n: int):
    """
    Build the Generation 10 Mechanism F2 Schur-Complement Butterfly circuit for Perm_n(A).
    Returns (plan, evaluator_fn) where evaluator_fn(A, p) computes Perm(A) mod p.
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    if n == 2:
        return {"n": 2, "N": 2, "non_scalar_muls": 2}, lambda A, p: A[0][0] * A[1][1] + A[0][1] * A[1][0]

    min_muls, root_spec, T = solve_f2_plan(n)
    m = n - 1
    N = 1 << m
    full_mask = N - 1
    weights = [hamming_weight(y) for y in range(N)]
    schur_cache: dict[tuple[int, int, int], tuple] = {}

    def get_schur(k: int, d_a: int, d_b: int):
        key = (k, d_a, d_b)
        if key not in schur_cache:
            schur_cache[key] = _find_schur_pivots(m, k, d_a, d_b, weights)
        return schur_cache[key]

    def eval_node(A_rows, k: int, offset: int, p: int, flipped=False, min_w=0, max_w=m, drop_le=-1):
        if k == 1:
            if not flipped:
                return eval_row_range(A_rows[offset], min_w, max_w, weights=weights)
            raw = eval_row_range(A_rows[offset], m - max_w, m - min_w, weights=weights)
            return [raw[full_mask ^ z] for z in range(N)]

        _, spec = T[k]
        kind = spec[0]

        if kind == "interp_ann1":
            _, b, a, deg, d_b, _ = spec
            min_active = max(0, d_b + 1)
            L = eval_node(A_rows, b, offset, p, flipped=flipped, min_w=min_active, max_w=deg, drop_le=d_b)
            R = eval_node(A_rows, a, offset + b, p, flipped=flipped, min_w=min_active, max_w=deg, drop_le=-1)
            piv = [_s_mul(L[z], R[z]) if (min_active <= weights[z] <= deg) else None for z in range(N)]
            return mobius_interpolate(
                piv, m, deg, max_out_w=max_w, min_out_w=min_w, drop_deg_le=drop_le, weights=weights
            )

        if kind == "dual_ann":
            _, a, b, d_left, d_right, _ = spec
            max_z_L = m - 1 - d_right if d_right >= 0 else m
            L_sh = eval_node(
                A_rows, a, offset, p, flipped=flipped,
                min_w=max(0, d_left + 1), max_w=max_z_L, drop_le=d_left,
            )
            if d_right >= 0:
                max_z_R = m - 1 - d_left if d_left >= 0 else m
                R_opp = eval_node(
                    A_rows, b, offset + a, p, flipped=(not flipped),
                    min_w=d_right + 1, max_w=max_z_R, drop_le=d_right,
                )
                R_sh = [R_opp[full_mask ^ z] for z in range(N)]
            else:
                R_sh = eval_node(
                    A_rows, b, offset + a, p, flipped=flipped,
                    min_w=max(0, d_left + 1), max_w=m, drop_le=-1,
                )
            prod = [
                _s_mul(L_sh[z], R_sh[z]) if (L_sh[z] is not None and R_sh[z] is not None) else None
                for z in range(N)
            ]
            if drop_le >= 0:
                return mobius_annihilate(prod, m, drop_le, max_out_w=max_w, weights=weights)
            return prod

        if kind == "interp_ann2":
            _, a, b, deg, d_a, d_b, _ = spec
            D, Z_a, M_inv, all_wt_k = get_schur(deg, d_a, d_b)
            D_set = set(D)
            min_active = d_b + 1
            max_active = min(deg, m - 1 - d_a)

            V_sh = eval_node(
                A_rows, b, offset, p, flipped=flipped,
                min_w=min_active, max_w=max_active, drop_le=d_b,
            )
            U_opp = eval_node(
                A_rows, a, offset + b, p, flipped=(not flipped),
                min_w=m - max_active, max_w=m - min_active, drop_le=d_a,
            )
            U_sh = [U_opp[full_mask ^ z] for z in range(N)]

            piv = [None] * N
            for z in range(N):
                if min_active <= weights[z] <= deg and z not in D_set:
                    if U_sh[z] is not None and V_sh[z] is not None:
                        piv[z] = _s_mul(U_sh[z], V_sh[z])

            W0 = mobius_interpolate(piv, m, deg, max_out_w=m, min_out_w=0, drop_deg_le=-1, weights=weights)

            r = len(D)
            lambdas = []
            for t in range(r):
                acc = None
                for j, z in enumerate(Z_a):
                    val = -M_inv[t, j]
                    if val != 0 and W0[z] is not None:
                        acc = _s_add(acc, _s_scale_exact(W0[z], val, p))
                lambdas.append(acc)

            if all_wt_k and drop_le < 0:
                out = list(W0)
                for y_t, lam in zip(D, lambdas):
                    if lam is None:
                        continue
                    for z in range(N):
                        if min_w <= weights[z] <= max_w and (z & y_t) == y_t:
                            out[z] = _s_add(out[z], lam)
                return out

            for y_t, lam in zip(D, lambdas):
                piv[y_t] = lam
            return mobius_interpolate(
                piv, m, deg, max_out_w=max_w, min_out_w=min_w, drop_deg_le=drop_le, weights=weights
            )

        raise RuntimeError(f"Unknown node kind: {kind}")

    def evaluate(A, p: int):
        _, a, b, _ = root_spec
        if a == 1:
            cur = eval_node(A, b, 1, p, flipped=False, min_w=0, max_w=m, drop_le=-1)
            hat = fwht_pruned(cur, m, m - 1, m)
            row0 = A[0]
            total = None
            for col in range(n):
                S = full_mask if col == 0 else (full_mask ^ (1 << (col - 1)))
                cw = hat.get(S)
                if cw is not None:
                    total = _s_add(total, _s_mul(cw, (row0[col], 1)))
            w, s = total
            coeff = pow(N, -1, p) if s == 1 else (-pow(N, -1, p)) % p
            return coeff * w
        raise RuntimeError(f"Unexpected root split {root_spec}")

    plan = {"n": n, "N": N, "non_scalar_muls": min_muls, "root": root_spec, "trees": T}
    return plan, evaluate
