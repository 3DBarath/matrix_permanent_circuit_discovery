"""
Deep Non-Hypercube Algebraic Probes for Perm_n (n = 3, 4):
1. Quadratic-Splitting Flattening Rank of Perm_4 over Sym^2(F^16) (136 deg-2 monomials):
   Bounds the minimum number of deg2*deg2 products d4 in any (d2, 0, d4) circuit.
2. Determinant-Hadamard Linear Span Probe:
   Tests whether Perm_n(A) = sum_{k=1}^K c_k * det(M_k * A) can achieve K < 2^{n-1}
   using arbitrary rank-1 or general masks M_k (n = 3, 4).
3. Perm_4 Essential-Variable & Singular-Locus / Jacobian Probe:
   Checks whether the Appendix A lower-bound invariants for Perm_3 lift to Perm_4.
4. Perm_4 Pure-Quad Bilinear Tensor Rank Probe (136x136 symmetric flattening).
"""

from __future__ import annotations

import itertools
import numpy as np
import sympy as sp


def perm_monomials(n: int) -> list[tuple[int, ...]]:
    """Return list of permutations pi in S_n representing monomials prod_i A[i, pi[i]]."""
    return list(itertools.permutations(range(n)))


def probe_quadratic_splitting_rank_perm4() -> dict:
    """
    Any pure deg2*deg2 circuit sum_{r=1}^{d4} Q_r(A) * R_r(A) = Perm_4(A)
    represents Perm_4 as a bilinear form on the 136-dimensional space of
    homogeneous degree-2 polynomials in 16 variables x_0..x_15.

    More specifically, in the monomial basis of deg-2 polynomials (136 monomials),
    Perm_4(A) = m2(A)^T * H * m2(A) is NOT unique because x_a*x_b * x_c*x_d = x_a*x_c * x_b*x_d.
    For each of the 24 degree-4 monomials of Perm_4 (e.g. a00*a11*a22*a33), there are
    3 ways to split the 4 distinct variables into two unordered pairs:
      {v0,v1}|{v2,v3}, {v0,v2}|{v1,v3}, {v0,v3}|{v1,v2}.
    A symmetric 136x136 matrix H satisfies m2^T H m2 = Perm_4 iff:
      - For each of the 24 perm monomials, the sum of the 3 split entries (with multiplicity 2) equals 1.
      - For every other degree-4 monomial, the sum of its split entries equals 0.

    Notice: for row-separated splits (rows {0,1} vs {2,3}), H has rank 12 (6 symmetric
    2x2 blocks [[0,1],[1,0]], i.e., 6 products P_ij * Q_kl).
    Wait: for general bilinear Q_r * R_r (not necessarily Q_r = R_r), sum_{r=1}^d4 Q_r R_r
    corresponds to a matrix M (not necessarily symmetric) with (M + M^T)/2 = H,
    and d4 >= rank(M) >= rank(H)/2!
    Let's compute the exact structure of the canonical symmetric flattening H_sym
    (equal weight 1/6 on all 6 ordered splits of each perm monomial) AND check the
    row-partitioned flattening (rows {0,1} vs {2,3}) and mixed-row flattenings!
    """
    # Map 16 variables (r, c) -> idx 0..15
    vars16 = [(r, c) for r in range(4) for c in range(4)]
    # All 136 degree-2 monomials (i <= j)
    deg2_mons = [(i, j) for i in range(16) for j in range(i, 16)]
    mon2_idx = {m: idx for idx, m in enumerate(deg2_mons)}

    # Canonical symmetric flattening H_sym (every ordered split gets 1/6)
    H_sym = np.zeros((len(deg2_mons), len(deg2_mons)), dtype=np.float64)
    # Row-partitioned flattening H_row ({0,1} vs {2,3})
    H_row = np.zeros((len(deg2_mons), len(deg2_mons)), dtype=np.float64)

    for pi in itertools.permutations(range(4)):
        vs = sorted(4 * r + pi[r] for r in range(4))
        v0, v1, v2, v3 = vs
        splits = [
            ((v0, v1), (v2, v3)),
            ((v0, v2), (v1, v3)),
            ((v0, v3), (v1, v2)),
        ]
        for p1, p2 in splits:
            i1, i2 = mon2_idx[p1], mon2_idx[p2]
            H_sym[i1, i2] += 1.0 / 6.0
            H_sym[i2, i1] += 1.0 / 6.0
        # Row-partitioned split: rows {0,1} vs {2,3}
        p_top = tuple(sorted([0 * 4 + pi[0], 1 * 4 + pi[1]]))
        p_bot = tuple(sorted([2 * 4 + pi[2], 3 * 4 + pi[3]]))
        H_row[mon2_idx[p_top], mon2_idx[p_bot]] += 1.0

    rank_sym = int(np.linalg.matrix_rank(H_sym, tol=1e-9))
    rank_row = int(np.linalg.matrix_rank(H_row, tol=1e-9))

    # Also compute rank of the 3-way mixed flattening:
    # For any partition of the 16 variables into two disjoint sets S and T,
    # the rank of the submatrix of Perm_4(x_S, x_T) gives a lower bound on
    # communication/bilinear rank across that cut!
    # Check 8-vs-8 cuts: (rows 0,1 vs 2,3), (cols 0,1 vs 2,3), (checkerboard)
    def cut_rank(S_set: set[int]) -> int:
        # Collect monomials supported on S and T
        S_mons = {}
        T_mons = {}
        entries = []
        for pi in itertools.permutations(range(4)):
            vs = [4 * r + pi[r] for r in range(4)]
            ms = tuple(sorted(v for v in vs if v in S_set))
            mt = tuple(sorted(v for v in vs if v not in S_set))
            if ms not in S_mons:
                S_mons[ms] = len(S_mons)
            if mt not in T_mons:
                T_mons[mt] = len(T_mons)
            entries.append((S_mons[ms], T_mons[mt]))
        M = np.zeros((len(S_mons), len(T_mons)), dtype=np.float64)
        for i, j in entries:
            M[i, j] += 1.0
        return int(np.linalg.matrix_rank(M, tol=1e-9))

    cut_rows = cut_rank({r * 4 + c for r in (0, 1) for c in range(4)})
    cut_diag = cut_rank({r * 4 + c for r in range(4) for c in range(4) if (r + c) % 2 == 0})
    cut_block = cut_rank({r * 4 + c for r in (0, 1) for c in (0, 1)} | {r * 4 + c for r in (2, 3) for c in (2, 3)})

    return {
        "canonical_H_sym_rank": rank_sym,
        "row_partitioned_H_row_rank": rank_row,
        "cut_rank_rows01_vs_23": cut_rows,
        "cut_rank_checkerboard": cut_diag,
        "cut_rank_2x2_diag_blocks": cut_block,
    }


def probe_determinant_hadamard_rank(n: int) -> dict:
    """
    Test: can Perm_n(A) = sum_{k=1}^K alpha_k * det(M_k * A) with K < 2^{n-1}?
    Notice: det(M * A) = sum_{pi in S_n} sgn(pi) * (prod_i M[i, pi[i]]) * (prod_i A[i, pi[i]]).
    So in the n!-dimensional vector space indexed by pi in S_n,
    we want to write the vector v_perm = [1, 1, ..., 1] as a linear combination of K
    vectors of the form w(M)_pi = sgn(pi) * prod_i M[i, pi[i]],
    equivalently write v_sgn = [sgn(pi)] as a linear combination of K rank-1-like
    permutation-product vectors u(M)_pi = prod_i M[i, pi[i]]!

    Wait: for n = 3 (n! = 6), let's check the EXACT minimum K over R/C by testing
    whether K = 2 or K = 3 works via nonlinear least squares on u(M)!
    For n = 4 (n! = 24), test K = 4, 5, 6, 7!
    """
    from scipy.optimize import least_squares

    perms = perm_monomials(n)
    n_fact = len(perms)
    # Compute signs of permutations
    sgns = np.array([
        1.0 if sum(1 for i in range(n) for j in range(i + 1, n) if pi[i] > pi[j]) % 2 == 0 else -1.0
        for pi in perms
    ])
    # Target: sum_k alpha_k * det(M_k * A) = Perm_n(A)
    # Equivalently, sum_k (prod_i M_k[i, pi[i]]) = sgn(pi) for all pi in S_n
    # (absorbing alpha_k into row 0 of M_k).
    perm_idx = np.array(perms, dtype=np.int32)  # (n!, n)
    rows = np.arange(n)[None, :]  # (1, n)

    results = {}
    max_k = (1 << (n - 1))
    test_ks = range(2, max_k + 1) if n == 3 else [3, 4, 5, 6, 7, 8]

    rng = np.random.default_rng(12345)
    for K in test_ks:
        best_res = float("inf")
        n_restarts = 12 if n == 3 else 6
        for _ in range(n_restarts):
            x0 = rng.standard_normal(K * n * n)

            def resid(params):
                Ms = params.reshape(K, n, n)
                # vals[k, p] = prod_{r=0}^{n-1} Ms[k, r, perm_idx[p, r]]
                # Shape: (K, n!, n) -> prod over axis 2 -> (K, n!)
                gathered = Ms[:, rows, perm_idx]  # (K, n!, n)
                prods = np.prod(gathered, axis=2)  # (K, n!)
                return np.sum(prods, axis=0) - sgns

            sol = least_squares(resid, x0, method="trf", max_nfev=3000)
            err = float(np.linalg.norm(sol.fun))
            if err < best_res:
                best_res = err
            if best_res < 1e-10:
                break
        results[f"K={K}"] = best_res
    return results


def probe_perm4_jacobian_and_lie() -> dict:
    """
    Compute exact algebraic invariants of Perm_4:
    - Rank of the 16-variable gradient nabla Perm_4 (essential variables)
    - Dimension of the symmetry Lie algebra g(Perm_4) (analogous to Appendix A Lemma 5)
    """
    # Symmetry Lie algebra of Perm_n: A -> X A + A Y where X, Y diagonal with tr(X)+tr(Y)=0
    # plus for n>=3, is dim g(Perm_n) = 2n - 1?
    # Let's verify numerically by evaluating d/dt Perm_4(A + t * E_{pq} A) etc.
    # Actually, let's check the dimension of the space of 16x16 matrices L such that
    # sum_{i,j} (L x)_{ij} * d(Perm_4)/d(x_{ij}) = 0 identically!
    rng = np.random.default_rng(7)
    n_samples = 300
    perms = perm_monomials(4)
    # Build matrix of evaluations of x_b * dPerm4/dx_a for all 16*16 = 256 pairs (a, b)
    M = np.zeros((n_samples, 256), dtype=np.float64)
    for s in range(n_samples):
        A = rng.standard_normal((4, 4))
        x = A.reshape(16)
        grad = np.zeros(16, dtype=np.float64)
        for pi in perms:
            vals = [A[r, pi[r]] for r in range(4)]
            for r in range(4):
                p = 1.0
                for r2 in range(4):
                    if r2 != r:
                        p *= vals[r2]
                grad[4 * r + pi[r]] += p
        M[s] = np.outer(grad, x).reshape(256)
    svals = np.linalg.svd(M, compute_uv=False)
    rank = int(np.sum(svals > 1e-8))
    lie_dim = 256 - rank
    # Also check gradient rank (essential variables)
    G = np.zeros((n_samples, 16), dtype=np.float64)
    for s in range(n_samples):
        A = rng.standard_normal((4, 4))
        for pi in perms:
            vals = [A[r, pi[r]] for r in range(4)]
            for r in range(4):
                p = 1.0
                for r2 in range(4):
                    if r2 != r:
                        p *= vals[r2]
                G[s, 4 * r + pi[r]] += p
    ess_vars = int(np.linalg.matrix_rank(G, tol=1e-8))
    return {"essential_vars": ess_vars, "lie_algebra_dim": lie_dim, "expected_2n_minus_1": 2 * 4 - 1}


if __name__ == "__main__":
    print("=== Probe 1: Perm_4 Quadratic-Splitting & Cut Ranks ===")
    q_res = probe_quadratic_splitting_rank_perm4()
    for k, v in q_res.items():
        print(f"  {k}: {v}")

    print("\n=== Probe 2: Determinant-Hadamard Linear Span Rank (Perm_n = sum_k det(M_k * A)) ===")
    for n in (3, 4):
        d_res = probe_determinant_hadamard_rank(n)
        print(f"  n={n} (2^(n-1)={1<<(n-1)}):")
        for k, v in d_res.items():
            print(f"    {k}: residual = {v:.3e}")

    print("\n=== Probe 3: Perm_4 Essential Variables & Symmetry Lie Algebra ===")
    l_res = probe_perm4_jacobian_and_lie()
    for k, v in l_res.items():
        print(f"  {k}: {v}")
