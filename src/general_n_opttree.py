"""
Generation 3 Breakthrough (Mode A, General n >= 3):
Optimal Binary Evaluation Tree + Dual Degree-Parity Compression for Perm_n(A).

Asymptotic improvement:
- Glynn + Gray code: (n - 1) * 2^(n-1) = Theta(n * 2^n) non-scalar multiplications.
- Sequential Dual Degree-Parity: Theta(sqrt(n) * 2^n) non-scalar multiplications.
- Optimal Binary Evaluation Tree (OptTree): O(2^n) non-scalar multiplications
  (e.g., n=4: 18, n=5: 42, n=6: 85, n=7: 172, n=8: 322, n=10: 1176, n=12: 4081 vs 22528).

Includes Fast Walsh-Hadamard Transform (FWHT) for O(n * 2^n) row linear forms
and exact rational interpolation/annihilation matrices over Q (exact mod any odd prime p).
"""

from __future__ import annotations

from math import comb
import sympy as sp


def hamming_weight(mask: int) -> int:
    return bin(mask).count("1")


def K(n: int, d: int) -> int:
    """Dimension of degree <= d multilinear polynomials on {+1, -1}^(n-1)."""
    if d < 0:
        return 0
    if d >= n - 1:
        return 1 << (n - 1)
    return sum(comb(n - 1, w) for w in range(d + 1))


def fwht_row(row) -> list:
    """
    Fast Walsh-Hadamard Transform evaluating (v_mask . row) for all
    N = 2^(n-1) sign vectors v_mask in {+1} x {+1, -1}^(n-1).
    Uses exactly 2*(2^(n-1) - 1) additions/subtractions.
    """
    n = len(row)
    cur = [row[0]]
    for c in range(1, n):
        rc = row[c]
        nxt = []
        for x in cur:
            nxt.append(x + rc)
        for x in cur:
            nxt.append(x - rc)
        cur = nxt
    return cur


def fwht_inverse(vec) -> list:
    """Unnormalized Fast Walsh-Hadamard Transform (H_{n-1} @ vec) in (n-1)*2^(n-1) ops."""
    N = len(vec)
    arr = list(vec)
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            for j in range(step):
                u = arr[i + j]
                v = arr[i + j + step]
                arr[i + j] = u + v
                arr[i + j + step] = u - v
        step <<= 1
    return arr


def walsh_character_matrix(n: int, max_deg: int) -> sp.Matrix:
    """
    N x K(n, max_deg) matrix whose (mask, S) entry is (-1)^popcount(mask & S),
    matching the ordering of fwht_row (bit c-1 corresponds to column c).
    """
    N = 1 << (n - 1)
    chars = [S for S in range(N) if hamming_weight(S) <= max_deg]
    return sp.Matrix([
        [-1 if (hamming_weight(mask & S) & 1) else 1 for S in chars]
        for mask in range(N)
    ])


def solve_optimal_tree_plan(n: int) -> dict:
    """
    Dynamic program finding the minimum non-scalar multiplication binary evaluation
    tree combining Prefix Degree Interpolation, Suffix Degree-Parity Annihilation,
    and Parseval Walsh-Hadamard Inner Products.
    """
    N = 1 << (n - 1)

    # 1. Optimal tree to build full N-vector of degree m (1 <= m <= n-1)
    full: dict[int, tuple[int, tuple]] = {1: (0, ("row",))}
    for m in range(2, n):
        best = (10**18, None)
        for a in range(1, m // 2 + 1):
            b = m - a
            cost = full[a][0] + full[b][0] + K(n, m)
            if cost < best[0]:
                best = (cost, ("mul_interp", a, b, m))
        full[m] = best

    # 2. Optimal suffix plan starting from a current vector of degree a
    memo: dict[tuple[int, bool], tuple[int, list]] = {}

    def finish(a: int, shifted: bool) -> tuple[int, list]:
        rem = n - a
        if (a, shifted) in memo:
            return memo[(a, shifted)]

        # Option 1: Direct Parseval inner product with full[rem]
        # Only characters S with |S| >= a - 1 (and <= a if unshifted) survive.
        max_w = min(a, n - 1) if not shifted else (n - 1)
        parseval_muls = sum(comb(n - 1, w) for w in range(a - 1, max_w + 1))
        best = (
            full[rem][0] + parseval_muls,
            [("parseval", a, rem, shifted, parseval_muls)],
        )

        # Option 2: Multiply by full[b] (1 <= b < rem)
        for b in range(1, rem):
            ann_cost = N - K(n, a - 2)
            if not shifted and K(n, a + b) < ann_cost:
                step_cost = full[b][0] + K(n, a + b)
                sub_cost, sub_ops = finish(a + b, False)
                if step_cost + sub_cost < best[0]:
                    best = (
                        step_cost + sub_cost,
                        [("interp_step", b, a + b, K(n, a + b))] + sub_ops,
                    )
            else:
                step_cost = full[b][0] + ann_cost
                sub_cost, sub_ops = finish(a + b, True)
                if step_cost + sub_cost < best[0]:
                    best = (
                        step_cost + sub_cost,
                        [("ann_step", b, a - 2, ann_cost)] + sub_ops,
                    )

        memo[(a, shifted)] = best
        return best

    overall = (10**18, None, None)
    for a in range(1, n):
        c, ops = finish(a, False)
        tot = full[a][0] + c
        if tot < overall[0]:
            overall = (tot, a, ops)

    return {
        "n": n,
        "N": N,
        "non_scalar_muls": overall[0],
        "start_deg": overall[1],
        "ops": overall[2],
        "full_trees": full,
    }


def _compile_matrices(n: int, plan: dict) -> dict:
    """Precompute exact rational interpolation and annihilation matrices for a plan."""
    N = 1 << (n - 1)
    interp_cache: dict[int, tuple[list[int], sp.Matrix]] = {}
    ann_cache: dict[int, tuple[list[int], list[int], sp.Matrix]] = {}

    def get_interp(deg: int):
        if deg not in interp_cache:
            W = walsh_character_matrix(n, deg)
            _, pivots = W.T.rref()
            pivots = list(pivots)
            R = W * W[pivots, :].inv()
            interp_cache[deg] = (pivots, R)
        return interp_cache[deg]

    def get_ann(deg_ann: int):
        if deg_ann not in ann_cache:
            W = walsh_character_matrix(n, deg_ann)
            _, Z = W.T.rref()
            Z = list(Z)
            active = [i for i in range(N) if i not in Z]
            S_mat = W[active, :] * W[Z, :].inv()
            ann_cache[deg_ann] = (Z, active, S_mat)
        return ann_cache[deg_ann]

    def ensure_full(m: int):
        if m == 1:
            return
        _, spec = plan["full_trees"][m]
        _, a, b, deg = spec
        ensure_full(a)
        ensure_full(b)
        get_interp(deg)

    ensure_full(plan["start_deg"])
    for op in plan["ops"]:
        kind = op[0]
        if kind == "interp_step":
            _, b, deg, _ = op
            ensure_full(b)
            get_interp(deg)
        elif kind == "ann_step":
            _, b, deg_ann, _ = op
            ensure_full(b)
            if deg_ann >= 0:
                get_ann(deg_ann)
        elif kind == "parseval":
            _, _, rem, _, _ = op
            ensure_full(rem)

    return {"interp": interp_cache, "ann": ann_cache}


def _apply_row_exact(mat_row: sp.Matrix, vec_wires: list, p: int):
    acc = None
    for c in range(mat_row.cols):
        w = vec_wires[c]
        val = mat_row[0, c]
        if w is None or val == 0:
            continue
        num, den = int(val.p), int(val.q)
        if num == 1 and den == 1:
            acc = w if acc is None else (acc + w)
        elif num == -1 and den == 1:
            acc = ((p - 1) * w) if acc is None else (acc - w)
        else:
            coeff = (num * pow(den, -1, p)) % p
            term = coeff * w
            acc = term if acc is None else (acc + term)
    return acc


def build_opttree_circuit(n: int):
    """
    Build a reusable exact OptTree circuit evaluator for n x n matrices.
    Returns (plan, evaluator_fn) where evaluator_fn(A, p) computes Perm(A) mod p.
    """
    plan = solve_optimal_tree_plan(n)
    mats = _compile_matrices(n, plan)
    N = plan["N"]
    full_trees = plan["full_trees"]
    parities = [-1 if (hamming_weight(mask) & 1) else 1 for mask in range(N)]

    def eval_full(A_rows, m: int, offset: int, p: int) -> list:
        if m == 1:
            return fwht_row(A_rows[offset])
        _, spec = full_trees[m]
        _, a, b, deg = spec
        left = eval_full(A_rows, a, offset, p)
        right = eval_full(A_rows, b, offset + a, p)
        pivots, R = mats["interp"][deg]
        piv_prods = [left[i] * right[i] for i in pivots]
        piv_map = {idx: pos for pos, idx in enumerate(pivots)}
        out = [None] * N
        for i in range(N):
            if i in piv_map:
                out[i] = piv_prods[piv_map[i]]
            else:
                out[i] = _apply_row_exact(R[i : i + 1, :], piv_prods, p)
        return out

    def evaluate(A, p: int):
        row_ptr = 0
        cur = eval_full(A, plan["start_deg"], row_ptr, p)
        row_ptr += plan["start_deg"]

        for op in plan["ops"]:
            kind = op[0]
            if kind == "interp_step":
                _, b, deg, _ = op
                rhs = eval_full(A, b, row_ptr, p)
                row_ptr += b
                pivots, R = mats["interp"][deg]
                piv_prods = [cur[i] * rhs[i] for i in pivots]
                piv_map = {idx: pos for pos, idx in enumerate(pivots)}
                nxt = [None] * N
                for i in range(N):
                    if i in piv_map:
                        nxt[i] = piv_prods[piv_map[i]]
                    else:
                        nxt[i] = _apply_row_exact(R[i : i + 1, :], piv_prods, p)
                cur = nxt

            elif kind == "ann_step":
                _, b, deg_ann, _ = op
                rhs = eval_full(A, b, row_ptr, p)
                row_ptr += b
                if deg_ann < 0:
                    cur = [cur[i] * rhs[i] for i in range(N)]
                else:
                    Z, active, S_mat = mats["ann"][deg_ann]
                    z_vals = [cur[i] for i in Z]
                    nxt = [None] * N
                    for pos, i in enumerate(active):
                        shift = _apply_row_exact(S_mat[pos : pos + 1, :], z_vals, p)
                        if shift is None:
                            shifted_val = cur[i]
                        elif cur[i] is None:
                            shifted_val = (p - 1) * shift
                        else:
                            shifted_val = cur[i] - shift
                        nxt[i] = shifted_val * rhs[i]
                    cur = nxt

            elif kind == "parseval":
                _, a, rem, shifted, _ = op
                if rem == 1:
                    # Direct Final-Layer Collapse: pull last row outside the sum (n muls)
                    last_row = A[row_ptr]
                    row_ptr += 1
                    total = None
                    for col in range(n):
                        coeff_wire = None
                        for i in range(N):
                            if cur[i] is None:
                                continue
                            v_col = 1 if col == 0 else (-1 if ((i >> (col - 1)) & 1) else 1)
                            sgn = parities[i] * v_col
                            if coeff_wire is None:
                                coeff_wire = cur[i] if sgn == 1 else (p - 1) * cur[i]
                            else:
                                coeff_wire = (
                                    (coeff_wire + cur[i]) if sgn == 1 else (coeff_wire - cur[i])
                                )
                        term = coeff_wire * last_row[col]
                        total = term if total is None else (total + term)
                    return pow(N, -1, p) * total
                else:
                    # General Parseval inner product in Walsh-Hadamard basis
                    rhs = eval_full(A, rem, row_ptr, p)
                    row_ptr += rem
                    zero = cur[0] - cur[0]
                    cur_filled = [zero if x is None else x for x in cur]
                    hat_a = fwht_inverse(cur_filled)
                    hat_b = fwht_inverse(rhs)
                    max_w = min(a, n - 1) if not shifted else (n - 1)
                    full_mask = N - 1
                    total = None
                    for S in range(N):
                        w = hamming_weight(S)
                        if a - 1 <= w <= max_w:
                            term = hat_a[S] * hat_b[full_mask ^ S]
                            total = term if total is None else (total + term)
                    # Parseval identity: sum_j sigma_j a_j b_j = (1/N) sum_S hat_a[S] hat_b[~S]
                    return pow(N * N, -1, p) * total

        raise RuntimeError("Plan terminated without a final Parseval/collapse step.")

    return plan, evaluate
