"""
Generation 8 Champion (Mode A, General n >= 3):
Mechanism E (Degree-Preserving Recursive Dual-Ball Annihilation) +
Hasse-Diagram Row Evaluator + Lexicographically Tie-Broken Fused Mobius-Zeta
Butterfly OptTree for the n x n Matrix Permanent.

CORE THEOREM (Generation 8 — Degree Preservation Under Annihilation):
Subtracting a degree <= (a - 2) polynomial c1 from a degree <= a polynomial U
never increases degree: deg(U - c1) <= a. Consequently, the product of two
annihilated subtrees (U - c1)(V - c2) retains degree <= a + b and remains exact
modulo degree <= (a + b - 2). This collapses the entire OptTree search into a
single symmetric binary-tree recurrence:

    T(1) = 0
    T(k) = min_{1 <= a <= k//2} [
        T(a) + T(k - a) + min(K(n, k), N - K(n, a - 2) - K(n, k - a - 2))
    ]                                                       (2 <= k <= n - 1)
    Opt(n) = min_{1 <= a <= n//2} [ T(a) + T(n - a) + C(n, a) ]

Every dual-ball annihilation step fuses into BOTH its left child (low-weight
ball |y| <= a - 2) and its bit-flipped right child (high-weight ball
|y| >= (n - 1) - (b - 2)) with zero extra passes and exactly 1 scalar
multiplication total across the entire circuit.

Record non-scalar multiplications (vs. Glynn (n-1)*2^(n-1)):
- n = 3:  7 (vs 8, -12.5%)        - n = 10: 1,165 (vs 4,608, -74.7%)
- n = 4:  18 (vs 24, -25.0%)      - n = 11: 2,150 (vs 10,240, -79.0%)
- n = 5:  41 (vs 64, -35.9%)      - n = 12: 4,013 (vs 22,528, -82.2%)
- n = 6:  84 (vs 160, -47.5%)     - n = 14: 14,610 (vs 106,496, -86.3%)
- n = 7:  171 (vs 384, -55.5%)    - n = 15: 27,909 (vs 229,376, -87.8%)
- n = 8:  321 (vs 896, -64.2%)    - n = 16: 53,118 (vs 491,520, -89.2%)
- n = 9:  623 (vs 2,048, -69.6%)
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


def list_optimal_tree_plans(n: int) -> tuple[int, dict, list[tuple]]:
    """
    Unified Mechanism E dynamic program returning:
      (min_non_scalar_muls, T_options_by_deg, root_options)
    where T_options_by_deg[k] is the list of all optimal split specs for degree k
    and root_options is the list of all optimal root Parseval splits.
    """
    N = 1 << (n - 1)
    T_cost: dict[int, int] = {1: 0}
    T_opts: dict[int, list[tuple]] = {1: [("row",)]}

    for k in range(2, n):
        best_c = 10**18
        opts: list[tuple] = []
        for a in range(1, k // 2 + 1):
            b = k - a
            c_interp = K(n, k)
            c_dual = N - K(n, a - 2) - K(n, b - 2)
            base = T_cost[a] + T_cost[b]

            # Consider interpolation
            if base + c_interp < best_c:
                best_c = base + c_interp
                opts = [("interp", a, b, k, c_interp)]
            elif base + c_interp == best_c:
                opts.append(("interp", a, b, k, c_interp))

            # Consider dual-ball annihilation (strictly valid for all a, b >= 1; K(n, <0) = 0)
            if base + c_dual < best_c:
                best_c = base + c_dual
                opts = [("dual_ann", a, b, a - 2, b - 2, c_dual)]
            elif base + c_dual == best_c:
                # Avoid duplicate if c_dual == c_interp and already added? Keep both for tie-breaking!
                opts.append(("dual_ann", a, b, a - 2, b - 2, c_dual))

        T_cost[k] = best_c
        T_opts[k] = opts

    best_root = 10**18
    root_opts: list[tuple] = []
    for a in range(1, n // 2 + 1):
        b = n - a
        pm = comb(n, a)
        tot = T_cost[a] + T_cost[b] + pm
        if tot < best_root:
            best_root = tot
            root_opts = [("parseval", a, b, pm)]
        elif tot == best_root:
            root_opts.append(("parseval", a, b, pm))

    return best_root, T_opts, root_opts


def solve_optimal_tree_plan(n: int) -> dict:
    """Return the minimum-multiplication Mechanism E plan summary for size n."""
    min_muls, T_opts, root_opts = list_optimal_tree_plans(n)
    return {
        "n": n,
        "N": 1 << (n - 1),
        "non_scalar_muls": min_muls,
        "root": root_opts[0],
        "trees": {k: opts[0] for k, opts in T_opts.items()},
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
    Uses Hasse-diagram 1-add-per-mask evaluation (2m + K - 1 ops) for n >= 4
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
    polynomial on {0, 1}^m from P_d = {y : |y| <= d}, with optional fused
    annihilation (drop_deg_le >= 0).
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


def _choose_annihilate_method(present: list[bool], m: int, d_ann: int, max_out_w: int, weights: list[int]) -> str:
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
            w = weights[y]
            if w > max_out_w:
                continue
            rem_zeros = rem_bits - weights[y & high_mask]
            if w + rem_zeros <= d_ann:
                continue
            if pA[y ^ step]:
                if pA[y]:
                    costA += 1
                else:
                    pA[y] = True
    for y in range(N):
        if d_ann < weights[y] <= max_out_w and present[y] and pA[y]:
            costA += 1

    pB = list(present)
    costB = 0
    for bit in range(m):
        step = 1 << bit
        for y in range(N):
            if (y & step) and weights[y] <= max_out_w:
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
            if (y & step) and d_ann < weights[y] <= max_out_w:
                if pB[y ^ step]:
                    if pB[y]:
                        costB += 1
                    else:
                        pB[y] = True

    return "A" if costA <= costB else "B"


def mobius_annihilate(
    U: list, m: int, d_ann: int, max_out_w: int | None = None, weights: list[int] | None = None
) -> list:
    """Subtract the unique degree <= d_ann polynomial matching U on |y| <= d_ann."""
    N = 1 << m
    if max_out_w is None:
        max_out_w = m
    if weights is None:
        weights = [hamming_weight(y) for y in range(N)]
    if d_ann < 0:
        return list(U)
    if d_ann == 0:
        u0 = U[0]
        out = [None] * N
        for y in range(1, N):
            if weights[y] <= max_out_w:
                out[y] = _s_sub(U[y], u0)
        return out

    present = [x is not None for x in U]
    if _choose_annihilate_method(present, m, d_ann, max_out_w, weights) == "A":
        c = mobius_interpolate(
            U, m, d_ann, max_out_w=max_out_w, min_out_w=d_ann + 1, weights=weights
        )
        out = [None] * N
        for y in range(N):
            if d_ann < weights[y] <= max_out_w:
                out[y] = _s_sub(U[y], c[y])
        return out

    arr = [U[y] if weights[y] <= max_out_w else None for y in range(N)]
    for bit in range(m):
        step = 1 << bit
        for y in range(N):
            if (y & step) and weights[y] <= max_out_w:
                arr[y] = _s_sub(arr[y], arr[y ^ step])
    for y in range(N):
        if weights[y] <= d_ann:
            arr[y] = None
    for bit in range(m):
        step = 1 << bit
        for y in range(N):
            if (y & step) and d_ann < weights[y] <= max_out_w:
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


def _make_mech_e_evaluator(n: int, tree_specs: dict[int, tuple], root_spec: tuple):
    m = n - 1
    N = 1 << m
    full_mask = N - 1
    weights = [hamming_weight(y) for y in range(N)]

    def eval_node(
        A_rows,
        k: int,
        offset: int,
        flipped: bool = False,
        min_w: int = 0,
        max_w: int = m,
        drop_le: int = -1,
    ) -> list:
        """
        Evaluate degree-k subtree in coordinate frame `flipped` (z = y if not flipped,
        z = full_mask ^ y if flipped), returning values for min_w <= |z| <= max_w,
        with optional fused annihilation of |z| <= drop_le.
        """
        if k == 1:
            if not flipped:
                return eval_row_range(A_rows[offset], min_w, max_w, weights=weights)
            raw = eval_row_range(A_rows[offset], m - max_w, m - min_w, weights=weights)
            return [raw[full_mask ^ z] for z in range(N)]

        spec = tree_specs[k]
        kind = spec[0]

        if kind == "interp":
            _, a, b, deg, _ = spec
            L = eval_node(A_rows, a, offset, flipped=flipped, min_w=0, max_w=deg, drop_le=-1)
            R = eval_node(A_rows, b, offset + a, flipped=flipped, min_w=0, max_w=deg, drop_le=-1)
            piv = [_s_mul(L[z], R[z]) if weights[z] <= deg else None for z in range(N)]
            return mobius_interpolate(
                piv, m, deg, max_out_w=max_w, min_out_w=min_w, drop_deg_le=drop_le, weights=weights
            )

        if kind == "dual_ann":
            _, a, b, d_left, d_right, _ = spec
            # In frame `flipped`, annihilate L on |z| <= d_left and R on |z| >= m - d_right!
            # So active z range for L * R is d_left < |z| < m - d_right.
            max_z_L = m - 1 - d_right if d_right >= 0 else m
            L_sh = eval_node(
                A_rows, a, offset, flipped=flipped,
                min_w=max(0, d_left + 1), max_w=max_z_L, drop_le=d_left,
            )
            if d_right >= 0:
                # Build R in opposite frame `not flipped` so |z| >= m - d_right becomes |z'| <= d_right!
                max_z_R = m - 1 - d_left if d_left >= 0 else m
                R_opp = eval_node(
                    A_rows, b, offset + a, flipped=(not flipped),
                    min_w=d_right + 1, max_w=max_z_R, drop_le=d_right,
                )
                R_sh = [R_opp[full_mask ^ z] for z in range(N)]
            else:
                R_sh = eval_node(
                    A_rows, b, offset + a, flipped=flipped,
                    min_w=max(0, d_left + 1), max_w=m, drop_le=-1,
                )

            prod = [
                _s_mul(L_sh[z], R_sh[z])
                if (L_sh[z] is not None and R_sh[z] is not None)
                else None
                for z in range(N)
            ]
            if drop_le >= 0:
                return mobius_annihilate(prod, m, drop_le, max_out_w=max_w, weights=weights)
            return prod

        raise RuntimeError(f"Unknown spec kind: {kind}")

    def evaluate(A, p: int | None = None, inv_N=None, inv_N2=None):
        _, a, b, _ = root_spec
        if a == 1:
            # Final-Layer Collapse on row 0 (n non-scalar muls)
            cur = eval_node(A, b, 1, flipped=False, min_w=0, max_w=m, drop_le=-1)
            hat = fwht_pruned(cur, m, m - 1, m)
            row0 = A[0]
            total = None
            for col in range(n):
                S = full_mask if col == 0 else (full_mask ^ (1 << (col - 1)))
                cw = hat.get(S)
                if cw is not None:
                    total = _s_add(total, _s_mul(cw, (row0[col], 1)))
            w, s = total
            if p is not None:
                coeff = pow(N, -1, p) if s == 1 else (-pow(N, -1, p)) % p
                return coeff * w
            scale = inv_N if s == 1 else -inv_N
            return scale * w

        U = eval_node(A, a, 0, flipped=False, min_w=0, max_w=m, drop_le=-1)
        V = eval_node(A, b, a, flipped=False, min_w=0, max_w=m, drop_le=-1)
        hat_u = fwht_pruned(U, m, a - 1, a)
        hat_v = fwht_pruned(V, m, b - 1, b)
        total = None
        for S, vu in hat_u.items():
            vv = hat_v.get(full_mask ^ S)
            if vv is not None:
                total = _s_add(total, _s_mul(vu, vv))
        w, s = total
        if p is not None:
            coeff = pow(N * N, -1, p) if s == 1 else (-pow(N * N, -1, p)) % p
            return coeff * w
        scale = inv_N2 if s == 1 else -inv_N2
        return scale * w

    return evaluate


def build_butterfly_opttree_circuit(n: int):
    """
    Build the lexicographically optimal Gen 8 (Mechanism E + Hasse-Row + Fused Butterfly)
    circuit for Perm_n(A): minimizes (non_scalar_muls, total_gates, depth).
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    if n == 2:
        plan = {"n": 2, "N": 2, "non_scalar_muls": 2, "root": ("direct2",), "trees": {}}

        def eval2(A, p: int | None = None, inv_N=None, inv_N2=None):
            return A[0][0] * A[1][1] + A[0][1] * A[1][0]

        return plan, eval2

    min_muls, T_opts, root_opts = list_optimal_tree_plans(n)

    # Bottom-up greedy lexicographic tie-breaking per degree k (fast even for large n!)
    chosen: dict[int, tuple] = {1: ("row",)}
    for k in range(2, n):
        opts = T_opts[k]
        if len(opts) == 1:
            chosen[k] = opts[0]
            continue
        # Score each tied option for degree k by building a dummy evaluator for k (with root (1, k) if k==n-1 else direct)
        best_key = (10**18, 10**18)
        best_opt = opts[0]
        for opt in opts:
            # Prefer larger a (more balanced) / fewer dry-run gates
            cand_trees = dict(chosen)
            cand_trees[k] = opt
            # Measure dry-run gates of building node k
            ev_test = _make_mech_e_evaluator(n, cand_trees, ("parseval", 1, n - 1, n))
            # Only evaluate full circuit if k == n - 1; for smaller k, measure eval_node directly via a quick proxy
            if k == n - 1:
                stats = [0]
                dummy_A = [[_DryWire(0, stats) for _ in range(n)] for _ in range(n)]
                out = ev_test(dummy_A, p=3)
                key = (stats[0], out.depth)
            else:
                # Prefer dual_ann only if strictly cheaper, otherwise prefer interp (fusable!) and larger a
                key = (0 if opt[0] == "interp" else 1, -opt[1])
            if key < best_key:
                best_key = key
                best_opt = opt
        chosen[k] = best_opt

    # Tie-break across root_opts
    best_root_key = (10**18, 10**18)
    best_root = root_opts[0]
    best_eval = None
    for r_opt in root_opts:
        ev = _make_mech_e_evaluator(n, chosen, r_opt)
        if len(root_opts) == 1:
            best_root = r_opt
            best_eval = ev
            break
        stats = [0]
        dummy_A = [[_DryWire(0, stats) for _ in range(n)] for _ in range(n)]
        out = ev(dummy_A, p=3)
        key = (stats[0], out.depth)
        if key < best_root_key:
            best_root_key = key
            best_root = r_opt
            best_eval = ev

    plan = {
        "n": n,
        "N": 1 << (n - 1),
        "non_scalar_muls": min_muls,
        "root": best_root,
        "trees": chosen,
    }
    return plan, best_eval
