"""
Exact Polynomial-Time O(n^r) Permanent for Rank-r Matrices (Barvinok / MacMahon).

For any n x n matrix A = U @ V^T (U, V of shape n x r) over F_p (p > n):
  Perm(U V^T) = sum_{k_1 + ... + k_r = n} (prod_j k_j!) * c_k(U) * c_k(V)
where c_k(M) = [x_1^{k_1} ... x_r^{k_r}] prod_{i=1}^n (sum_{j=1}^r M_{ij} x_j).

Collapses #P-hard exponential 2^n complexity to polynomial O(n^r) whenever rank r is fixed.
"""

from __future__ import annotations


def perm_low_rank_mod(U: list[list[int]], V: list[list[int]], p: int) -> int:
    """Compute exact Perm(U @ V^T) mod p in O(n^r) time."""
    n = len(U)
    r = len(U[0])

    def expand_poly(M: list[list[int]]) -> dict[tuple[int, ...], int]:
        dp: dict[tuple[int, ...], int] = {(0,) * r: 1}
        for i in range(n):
            row = M[i]
            nxt: dict[tuple[int, ...], int] = {}
            for exp, val in dp.items():
                for j in range(r):
                    mj = row[j] % p
                    if mj == 0:
                        continue
                    ne = list(exp)
                    ne[j] += 1
                    t = tuple(ne)
                    nxt[t] = (nxt.get(t, 0) + val * mj) % p
            dp = nxt
        return dp

    cu = expand_poly(U)
    cv = expand_poly(V)
    fact = [1] * (n + 1)
    for k in range(1, n + 1):
        fact[k] = (fact[k - 1] * k) % p

    total = 0
    for exp, vu in cu.items():
        vv = cv.get(exp, 0)
        if vv:
            w = 1
            for kj in exp:
                w = (w * fact[kj]) % p
            total = (total + w * vu % p * vv) % p
    return total
