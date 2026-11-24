"""
Generation 7 Champion (Mode A, General n >= 3):
Mechanism D (Double-Sided Dual-Ball Annihilation) + Hasse-Row +
Lexicographically Tie-Broken Fused Mobius-Zeta Butterfly OptTree.

NEW IN GENERATION 7 — Mechanism D (Double-Sided Dual-Ball Annihilation):
When combining two unshifted subtrees U (degree a >= 2) and V (degree b >= 2)
with r = n - a - b remaining rows, we simultaneously subtract a degree <= a-2
polynomial c1 from U (vanishing on the low-weight ball Z1 = {y : |y| <= a-2})
AND a degree <= b-2 polynomial c2 from V (vanishing on the disjoint high-weight
ball Z2 = {y : |y| >= (n-1) - (b-2)}). Every cross error term c1*V*rem,
U*c2*rem, and c1*c2*rem has degree <= n-2 < n-1 and vanishes identically against
the parity character sigma over every field F (char(F) != 2).
This reduces the product layer from N - K(n, a-2) to N - K(n, a-2) - K(n, b-2)
non-scalar multiplications, strictly breaking the Gen 3-6 OptTree record at
EVERY size n >= 5 (e.g. n=5: 41 vs 42, n=6: 84 vs 85, n=11: 2151 vs 2207,
n=12: 4014 vs 4081, n=14: 14611 vs 14703), while keeping scalar_muls = 1 and
fusing both balls directly into their O(n * 2^n) Mobius-Zeta butterflies.
"""

from __future__ import annotations

from math import comb


def hamming_weight(mask: int) -> int:
    return bin(mask).count("1")


def K(n: int, d: int) -> int:
    """Dimension of degree <= d multilinear polynomials on {+1, -1}^(n-1)."""
    if d < 0:
        return 0
    if d >= n - 1:
        return 1 << (n - 1)
    return sum(comb(n - 1, w) for w in range(d + 1))


def list_optimal_tree_plans(n: int) -> tuple[int, dict, list[tuple[int, list]]]:
    """
    Dynamic program returning (min_non_scalar_muls, full_trees, all_tied_plans)
    incorporating Prefix Interpolation, Single-Sided Annihilation, Mechanism D
    (Double-Sided Dual-Ball Annihilation), and Parseval Collapse.
    """
    N = 1 << (n - 1)

    full: dict[int, tuple[int, tuple]] = {1: (0, ("row",))}
    for m_deg in range(2, n):
        best = (10**18, None)
        for a in range(1, m_deg // 2 + 1):
            b = m_deg - a
            cost = full[a][0] + full[b][0] + K(n, m_deg)
            if cost < best[0]:
                best = (cost, ("mul_interp", a, b, m_deg))
        full[m_deg] = best

    memo: dict[tuple[int, bool], tuple[int, list[list]]] = {}

    def finish(a: int, shifted: bool) -> tuple[int, list[list]]:
        rem = n - a
        if (a, shifted) in memo:
            return memo[(a, shifted)]

        max_w = min(a, n - 1) if not shifted else (n - 1)
        parseval_muls = sum(comb(n - 1, w) for w in range(a - 1, max_w + 1))
        best_cost = full[rem][0] + parseval_muls
        best_plans: list[list] = [[("parseval", a, rem, shifted, parseval_muls)]]

        def consider(tot: int, op: tuple, sub_plans: list[list]) -> None:
            nonlocal best_cost, best_plans
            if tot < best_cost:
                best_cost = tot
                best_plans = [[op] + sp for sp in sub_plans]
            elif tot == best_cost:
                best_plans.extend([[op] + sp for sp in sub_plans])

        for b in range(1, rem):
            # Option 1: Prefix interpolation step (unshifted only)
            if not shifted:
                sc = full[b][0] + K(n, a + b)
                sub_c, sub_ps = finish(a + b, False)
                consider(sc + sub_c, ("interp_step", b, a + b, K(n, a + b)), sub_ps)

            # Option 2: Mechanism D — Double-Sided Dual-Ball Annihilation (unshifted, a>=2, b>=2)
            if not shifted and a >= 2 and b >= 2:
                d_left = a - 2
                d_right = b - 2
                dual_muls = N - K(n, d_left) - K(n, d_right)
                sc = full[b][0] + dual_muls
                sub_c, sub_ps = finish(a + b, True)
                consider(sc + sub_c, ("dual_ann_step", b, d_left, d_right, dual_muls), sub_ps)

            # Option 3: Single-sided annihilation step
            ann_cost = N - K(n, a - 2)
            sc = full[b][0] + ann_cost
            sub_c, sub_ps = finish(a + b, True)
            consider(sc + sub_c, ("ann_step", b, a - 2, ann_cost), sub_ps)

        memo[(a, shifted)] = (best_cost, best_plans)
        return best_cost, best_plans

    overall_cost = 10**18
    all_plans: list[tuple[int, list]] = []
    for a in range(1, n):
        c, ps = finish(a, False)
        tot = full[a][0] + c
        if tot < overall_cost:
            overall_cost = tot
            all_plans = [(a, p) for p in ps]
        elif tot == overall_cost:
            all_plans.extend([(a, p) for p in ps])

    return overall_cost, full, all_plans


def solve_optimal_tree_plan(n: int) -> dict:
    """Return the first minimum-multiplication Gen 7 plan."""
    min_muls, full, all_plans = list_optimal_tree_plans(n)
    start_deg, ops = all_plans[0]
    return {
        "n": n,
        "N": 1 << (n - 1),
        "non_scalar_muls": min_muls,
        "start_deg": start_deg,
        "ops": ops,
        "full_trees": full,
    }


# ---------------------------------------------------------------------------
# Zero-cost signed-wire arithmetic: None (zero) or (wire, sign in {+1, -1})
# ---------------------------------------------------------------------------


def _s_neg(sw):
    return None if sw is None else (sw[0], -sw[1])


def _s_add(a, b):
    if a is None:
        return b
    if b is None:
        return a
    wa, sa = a
    wb, sb = b
    if sa == sb:
        return (wa + wb, sa)
    return (wa - wb, 1) if sa == 1 else (wb - wa, 1)


def _s_sub(a, b):
    return _s_add(a, _s_neg(b))


def _s_mul(a, b):
    if a is None or b is None:
        return None
    return (a[0] * b[0], a[1] * b[1])


# ---------------------------------------------------------------------------
# Structured O(n * 2^n) Butterfly & Hasse-Diagram Operators (Zero Scalar Muls)
# ---------------------------------------------------------------------------


def eval_row_range(row, min_w: int, max_w: int, weights: list[int] | None = None) -> list:
    """
    Evaluate (v_mask . row) for all sign masks with min_w <= |mask| <= max_w.
    Uses the Hasse-diagram 1-add-per-mask evaluator (2m + K - 1 ops) for n >= 4
    prefix/suffix balls, and pruned butterfly otherwise.
    """
    n = len(row)
    m = n - 1
    N = 1 << m
    full_mask = N - 1
    if weights is None:
        weights = [hamming_weight(y) for y in range(N)]

    if n >= 4 and min_w == 0 and max_w <= m:
        two_r = [None] + [(row[c] + row[c], 1) for c in range(1, n)]
        s0 = (row[0], 1)
        for c in range(1, n):
            s0 = _s_add(s0, (row[c], 1))
        out = [None] * N
        out[0] = s0
        for mask in range(1, N):
            if weights[mask] <= max_w:
                lsb = mask & -mask
                out[mask] = _s_sub(out[mask ^ lsb], two_r[lsb.bit_length()])
        return out

    if n >= 4 and max_w >= m and min_w > 0:
        limit = m - min_w
        two_r = [None] + [(row[c] + row[c], 1) for c in range(1, n)]
        sf = (row[0], 1)
        for c in range(1, n):
            sf = _s_sub(sf, (row[c], 1))
        out = [None] * N
        out[full_mask] = sf
        for cmask in range(1, N):
            if weights[cmask] <= limit:
                lsb = cmask & -cmask
                out[full_mask ^ cmask] = _s_add(
                    out[full_mask ^ (cmask ^ lsb)], two_r[lsb.bit_length()]
                )
        return out

    cur = {0: ((row[0], 1), 0)}
    for c in range(1, n):
        rc = (row[c], 1)
        bit = 1 << (c - 1)
        rem_after = m - c
        nxt = {}
        for mask, (sw, w) in cur.items():
            if w + rem_after >= min_w:
                nxt[mask] = (_s_add(sw, rc), w)
            if w + 1 <= max_w:
                nxt[mask | bit] = (_s_sub(sw, rc), w + 1)
        cur = nxt
    out = [None] * N
    for mask, (sw, _) in cur.items():
        out[mask] = sw
    return out


def mobius_interpolate(
    piv_vals: list,
    m: int,
    d: int,
    max_out_w: int | None = None,
    min_out_w: int = 0,
    drop_deg_le: int = -1,
    weights: list[int] | None = None,
) -> list:
    """
    Exact O(m * 2^m) Mobius-Zeta interpolation of a degree <= d multilinear
    polynomial on {0, 1}^m from its values on P_d = {y : |y| <= d}.
    Supports fused annihilation (drop_deg_le >= 0) with zero extra gates.
    """
    N = 1 << m
    if max_out_w is None:
        max_out_w = m
    if weights is None:
        weights = [hamming_weight(y) for y in range(N)]

    if drop_deg_le < 0 and d >= max_out_w:
        return list(piv_vals)

    if drop_deg_le < 0 and d == m - 1 and max_out_w >= m:
        out = list(piv_vals)
        full_mask = N - 1
        acc = None
        for y in range(full_mask):
            sw = piv_vals[y]
            if sw is None:
                continue
            acc = _s_sub(acc, sw) if ((m - 1 - weights[y]) & 1) else _s_add(acc, sw)
        out[full_mask] = acc
        return out

    if drop_deg_le == 0 and d == m - 1 and max_out_w >= m:
        full_mask = N - 1
        acc = None
        for y in range(full_mask):
            sw = piv_vals[y]
            if sw is None:
                continue
            acc = _s_sub(acc, sw) if ((m - 1 - weights[y]) & 1) else _s_add(acc, sw)
        u0 = piv_vals[0]
        out = [None] * N
        for y in range(1, full_mask):
            out[y] = _s_sub(piv_vals[y], u0)
        out[full_mask] = _s_sub(acc, u0)
        return out

    # Stage 1: Inverse Mobius on |y| <= d
    c = [piv_vals[y] if weights[y] <= d else None for y in range(N)]
    for bit in range(m):
        step = 1 << bit
        for y in range(N):
            if (y & step) and weights[y] <= d:
                c[y] = _s_sub(c[y], c[y ^ step])

    full_mask = N - 1

    if drop_deg_le >= 0:
        for y in range(N):
            if weights[y] <= drop_deg_le:
                c[y] = None
        target_min = max(drop_deg_le + 1, min_out_w)
        for bit in range(m):
            step = 1 << bit
            high_mask = full_mask ^ ((1 << (bit + 1)) - 1)
            rem_bits = m - 1 - bit
            for y in range(N):
                if not (y & step):
                    continue
                w = weights[y]
                if w > max_out_w or w <= drop_deg_le:
                    continue
                rem_zeros = rem_bits - weights[y & high_mask]
                if w + rem_zeros < target_min:
                    continue
                prev = c[y ^ step]
                if prev is not None:
                    c[y] = prev if c[y] is None else _s_add(c[y], prev)
        return c

    target_min = max(d + 1, min_out_w)
    for bit in range(m):
        step = 1 << bit
        high_mask = full_mask ^ ((1 << (bit + 1)) - 1)
        rem_bits = m - 1 - bit
        for y in range(N):
            if not (y & step):
                continue
            w = weights[y]
            if w > max_out_w:
                continue
            rem_zeros = rem_bits - weights[y & high_mask]
            if w + rem_zeros < target_min:
                continue
            prev = c[y ^ step]
            if prev is not None:
                c[y] = prev if c[y] is None else _s_add(c[y], prev)

    if min_out_w <= d:
        for y in range(N):
            if min_out_w <= weights[y] <= d:
                c[y] = piv_vals[y]
    return c


def _choose_annihilate_method(present: list[bool], m: int, d_ann: int, weights: list[int]) -> str:
    """Data-independent exact add-count selection between Method A and Method B."""
    N = 1 << m
    full_mask = N - 1

    pA = [present[y] and (weights[y] <= d_ann) for y in range(N)]
    costA = 0
    for bit in range(m):
        step = 1 << bit
        for y in range(N):
            if (y & step) and weights[y] <= d_ann:
                if pA[y ^ step]:
                    if pA[y]:
                        costA += 1
                    else:
                        pA[y] = True
    for bit in range(m):
        step = 1 << bit
        high_mask = full_mask ^ ((1 << (bit + 1)) - 1)
        rem_bits = m - 1 - bit
        for y in range(N):
            if not (y & step):
                continue
            rem_zeros = rem_bits - weights[y & high_mask]
            if weights[y] + rem_zeros <= d_ann:
                continue
            if pA[y ^ step]:
                if pA[y]:
                    costA += 1
                else:
                    pA[y] = True
    for y in range(N):
        if weights[y] > d_ann and present[y] and pA[y]:
            costA += 1

    pB = list(present)
    costB = 0
    for bit in range(m):
        step = 1 << bit
        for y in range(N):
            if y & step:
                if pB[y ^ step]:
                    if pB[y]:
                        costB += 1
                    else:
                        pB[y] = True
    for y in range(N):
        if weights[y] <= d_ann:
            pB[y] = False
    for bit in range(m):
        step = 1 << bit
        for y in range(N):
            if (y & step) and weights[y] > d_ann:
                if pB[y ^ step]:
                    if pB[y]:
                        costB += 1
                    else:
                        pB[y] = True

    return "A" if costA <= costB else "B"


def mobius_annihilate(U: list, m: int, d_ann: int, weights: list[int] | None = None) -> list:
    """Subtract the unique degree <= d_ann polynomial matching U on |y| <= d_ann."""
    N = 1 << m
    if weights is None:
        weights = [hamming_weight(y) for y in range(N)]
    if d_ann < 0:
        return list(U)
    if d_ann == 0:
        u0 = U[0]
        out = [None] * N
        for y in range(1, N):
            out[y] = _s_sub(U[y], u0)
        return out

    present = [x is not None for x in U]
    if _choose_annihilate_method(present, m, d_ann, weights) == "A":
        c = mobius_interpolate(U, m, d_ann, max_out_w=m, min_out_w=d_ann + 1, weights=weights)
        out = [None] * N
        for y in range(N):
            if weights[y] > d_ann:
                out[y] = _s_sub(U[y], c[y])
        return out

    arr = list(U)
    for bit in range(m):
        step = 1 << bit
        for y in range(N):
            if y & step:
                arr[y] = _s_sub(arr[y], arr[y ^ step])
    for y in range(N):
        if weights[y] <= d_ann:
            arr[y] = None
    for bit in range(m):
        step = 1 << bit
        for y in range(N):
            if (y & step) and weights[y] > d_ann:
                prev = arr[y ^ step]
                if prev is not None:
                    arr[y] = prev if arr[y] is None else _s_add(arr[y], prev)
    return arr


def fwht_pruned(vec: list, m: int, w_min: int, w_max: int) -> dict:
    """Pruned signed-wire Fast Walsh-Hadamard Transform for w_min <= |S| <= w_max."""
    def rec(arr: list, k: int, lo: int, hi: int) -> dict:
        if lo > hi or lo > k or hi < 0:
            return {}
        if k == 0:
            return {0: arr[0]} if (lo <= 0 <= hi and arr[0] is not None) else {}
        half = 1 << (k - 1)
        res = {}
        if lo <= k - 1:
            s_arr = [_s_add(arr[i], arr[i + half]) for i in range(half)]
            res.update(rec(s_arr, k - 1, lo, min(hi, k - 1)))
        if hi >= 1:
            d_arr = [_s_sub(arr[i], arr[i + half]) for i in range(half)]
            sub1 = rec(d_arr, k - 1, max(0, lo - 1), hi - 1)
            for S, val in sub1.items():
                res[S | half] = val
        return res

    return rec(vec, m, w_min, w_max)


class _DryWire:
    __slots__ = ("depth", "stats")

    def __init__(self, depth: int, stats: list[int]) -> None:
        self.depth = depth
        self.stats = stats

    def __add__(self, other: _DryWire) -> _DryWire:
        self.stats[0] += 1
        return _DryWire(max(self.depth, other.depth) + 1, self.stats)

    def __sub__(self, other: _DryWire) -> _DryWire:
        self.stats[0] += 1
        return _DryWire(max(self.depth, other.depth) + 1, self.stats)

    def __mul__(self, other) -> _DryWire:
        self.stats[0] += 1
        d = max(self.depth, other.depth) + 1 if isinstance(other, _DryWire) else self.depth + 1
        return _DryWire(d, self.stats)

    def __rmul__(self, other) -> _DryWire:
        self.stats[0] += 1
        return _DryWire(self.depth + 1, self.stats)


def _make_evaluator(n: int, full_trees: dict, start_deg: int, ops: list):
    m = n - 1
    N = 1 << m
    full_mask = N - 1
    weights = [hamming_weight(y) for y in range(N)]

    def eval_full(
        A_rows, deg_m: int, offset: int, min_out_w: int = 0, max_out_w: int = m, drop_deg_le: int = -1
    ) -> list:
        if deg_m == 1:
            return eval_row_range(A_rows[offset], min_out_w, max_out_w, weights=weights)
        _, spec = full_trees[deg_m]
        _, a, b, deg = spec
        left = eval_full(A_rows, a, offset, 0, deg)
        right = eval_full(A_rows, b, offset + a, 0, deg)
        piv = [_s_mul(left[i], right[i]) if weights[i] <= deg else None for i in range(N)]
        return mobius_interpolate(
            piv, m, deg, max_out_w=max_out_w, min_out_w=min_out_w,
            drop_deg_le=drop_deg_le, weights=weights,
        )

    def eval_full_flipped_ann(A_rows, deg_m: int, offset: int, d_right: int, min_y_w: int) -> list:
        """
        Build full[deg_m] on the bit-flipped hypercube z = full_mask ^ y,
        fusing high-weight ball annihilation |y| >= m - d_right (|z| <= d_right)
        directly into the Mobius-Zeta interpolation with zero extra gates.
        Only computes target y with min_y_w <= |y| < m - d_right, i.e. d_right < |z| <= m - min_y_w.
        """
        max_z_out = m - min_y_w

        def eval_rev(dm: int, off: int, max_z_w: int, drop_z_le: int = -1) -> list:
            if dm == 1:
                raw = eval_row_range(A_rows[off], m - max_z_w, m, weights=weights)
                return [raw[full_mask ^ z] for z in range(N)]
            _, spec = full_trees[dm]
            _, a, b, deg = spec
            L = eval_rev(a, off, deg)
            R = eval_rev(b, off + a, deg)
            piv = [_s_mul(L[z], R[z]) if weights[z] <= deg else None for z in range(N)]
            return mobius_interpolate(
                piv, m, deg, max_out_w=max_z_w,
                min_out_w=max(0, drop_z_le + 1), drop_deg_le=drop_z_le, weights=weights,
            )

        v_flip = eval_rev(deg_m, offset, max_z_out, drop_z_le=d_right)
        return [v_flip[full_mask ^ y] for y in range(N)]

    def next_max_w(op_idx: int) -> int:
        if op_idx < len(ops) and ops[op_idx][0] == "interp_step":
            return ops[op_idx][2]
        return m

    def evaluate(A, p: int | None = None, inv_N=None, inv_N2=None):
        row_ptr = 0
        if start_deg >= 2 and ops[0][0] in ("ann_step", "dual_ann_step") and ops[0][2] >= 0:
            d_ann = ops[0][2]
            max_y = (m - 1 - ops[0][3]) if ops[0][0] == "dual_ann_step" else m
            cur = eval_full(
                A, start_deg, row_ptr, min_out_w=d_ann + 1, max_out_w=max_y, drop_deg_le=d_ann
            )
            already_ann_deg = d_ann
        else:
            cur = eval_full(A, start_deg, row_ptr, 0, next_max_w(0))
            already_ann_deg = -1
        row_ptr += start_deg

        for idx, op in enumerate(ops):
            kind = op[0]
            if kind == "interp_step":
                _, b, deg, _ = op
                rhs = eval_full(A, b, row_ptr, 0, deg)
                row_ptr += b
                piv = [_s_mul(cur[i], rhs[i]) if weights[i] <= deg else None for i in range(N)]
                if idx + 1 < len(ops) and ops[idx + 1][0] in ("ann_step", "dual_ann_step") and ops[idx + 1][2] >= 0:
                    nxt_op = ops[idx + 1]
                    d_ann = nxt_op[2]
                    max_y = (m - 1 - nxt_op[3]) if nxt_op[0] == "dual_ann_step" else m
                    cur = mobius_interpolate(
                        piv, m, deg, max_out_w=max_y, min_out_w=d_ann + 1,
                        drop_deg_le=d_ann, weights=weights,
                    )
                    already_ann_deg = d_ann
                else:
                    cur = mobius_interpolate(
                        piv, m, deg, max_out_w=next_max_w(idx + 1), weights=weights
                    )
                    already_ann_deg = -1

            elif kind == "dual_ann_step":
                _, b, d_left, d_right, _ = op
                shifted_cur = (
                    cur if already_ann_deg == d_left else mobius_annihilate(cur, m, d_left, weights=weights)
                )
                shifted_rhs = eval_full_flipped_ann(A, b, row_ptr, d_right, min_y_w=d_left + 1)
                row_ptr += b
                cur = [
                    _s_mul(shifted_cur[i], shifted_rhs[i])
                    if (shifted_cur[i] is not None and shifted_rhs[i] is not None)
                    else None
                    for i in range(N)
                ]
                already_ann_deg = -1

            elif kind == "ann_step":
                _, b, deg_ann, _ = op
                rhs = eval_full(A, b, row_ptr, deg_ann + 1, m)
                row_ptr += b
                shifted_cur = (
                    cur if already_ann_deg == deg_ann else mobius_annihilate(cur, m, deg_ann, weights=weights)
                )
                cur = [
                    _s_mul(shifted_cur[i], rhs[i]) if shifted_cur[i] is not None else None
                    for i in range(N)
                ]
                already_ann_deg = -1

            elif kind == "parseval":
                _, a, rem, shifted, _ = op
                if rem == 1:
                    last_row = A[row_ptr]
                    row_ptr += 1
                    hat = fwht_pruned(cur, m, m - 1, m)
                    total = None
                    for col in range(n):
                        S = full_mask if col == 0 else (full_mask ^ (1 << (col - 1)))
                        cw = hat.get(S)
                        if cw is not None:
                            total = _s_add(total, _s_mul(cw, (last_row[col], 1)))
                    w, s = total
                    if p is not None:
                        coeff = pow(N, -1, p) if s == 1 else (-pow(N, -1, p)) % p
                        return coeff * w
                    scale = inv_N if s == 1 else -inv_N
                    return scale * w
                else:
                    rhs = eval_full(A, rem, row_ptr, 0, m)
                    row_ptr += rem
                    max_w = min(a, m) if not shifted else m
                    hat_a = fwht_pruned(cur, m, a - 1, max_w)
                    hat_b = fwht_pruned(rhs, m, m - max_w, m - (a - 1))
                    total = None
                    for S, va in hat_a.items():
                        vb = hat_b.get(full_mask ^ S)
                        if vb is not None:
                            total = _s_add(total, _s_mul(va, vb))
                    w, s = total
                    if p is not None:
                        coeff = pow(N * N, -1, p) if s == 1 else (-pow(N * N, -1, p)) % p
                        return coeff * w
                    scale = inv_N2 if s == 1 else -inv_N2
                    return scale * w

        raise RuntimeError("Plan terminated without a final Parseval/collapse step.")

    return evaluate


def build_butterfly_opttree_circuit(n: int):
    """
    Build the lexicographically optimal Gen 7 (Mechanism D + Hasse-Row + Fused Butterfly)
    circuit for Perm_n(A): minimizes (non_scalar_muls, total_gates, depth).
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    if n == 2:
        plan = {"n": 2, "N": 2, "non_scalar_muls": 2, "start_deg": 1, "ops": []}

        def eval2(A, p: int | None = None, inv_N=None, inv_N2=None):
            return A[0][0] * A[1][1] + A[0][1] * A[1][0]

        return plan, eval2

    min_muls, full_trees, all_plans = list_optimal_tree_plans(n)
    best_key = (10**18, 10**18)
    best_choice = None
    best_eval = None

    for start_deg, ops in all_plans:
        ev = _make_evaluator(n, full_trees, start_deg, ops)
        if len(all_plans) == 1:
            best_choice = (start_deg, ops)
            best_eval = ev
            break
        stats = [0]
        dummy_A = [[_DryWire(0, stats) for _ in range(n)] for _ in range(n)]
        out = ev(dummy_A, p=3)
        key = (stats[0], out.depth)
        if key < best_key:
            best_key = key
            best_choice = (start_deg, ops)
            best_eval = ev

    plan = {
        "n": n,
        "N": 1 << (n - 1),
        "non_scalar_muls": min_muls,
        "start_deg": best_choice[0],
        "ops": best_choice[1],
        "full_trees": full_trees,
    }
    return plan, best_eval
