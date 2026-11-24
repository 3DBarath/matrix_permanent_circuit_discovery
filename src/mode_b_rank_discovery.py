"""
Mode B Continuous-to-Exact Discovery Pipeline:
1. Alternating Least Squares (ALS) tensor rank probe for the 4x4x6 (2x4 permanents)
   and 6x4x4 (2->3 permanent lift) bilinear tensors.
2. Exact rational lifting via quotient-space Walsh-Hadamard projection and SymPy
   Gauss-Jordan solving over Q.
"""

from __future__ import annotations

import numpy as np
import sympy as sp


def build_2x4_perm_tensor() -> np.ndarray:
    """4x4x6 tensor T[i, j, p] = 1 iff {i, j} == pair p."""
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    T = np.zeros((4, 4, 6), dtype=np.float64)
    for idx, (i, j) in enumerate(pairs):
        T[i, j, idx] = 1.0
        T[j, i, idx] = 1.0
    return T


def build_2to3_lift_tensor() -> np.ndarray:
    """6x4x4 tensor mapping six 2x2 permanents and row 2 to four 3x3 permanents."""
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    pair_idx = {p: idx for idx, p in enumerate(pairs)}
    triples = [(i, j, k) for i in range(4) for j in range(i + 1, 4) for k in range(j + 1, 4)]
    T = np.zeros((6, 4, 4), dtype=np.float64)
    for t_idx, (i, j, k) in enumerate(triples):
        T[pair_idx[(i, j)], k, t_idx] = 1.0
        T[pair_idx[(i, k)], j, t_idx] = 1.0
        T[pair_idx[(j, k)], i, t_idx] = 1.0
    return T


def als_cp_rank_probe(
    T: np.ndarray, rank: int, n_restarts: int = 5, n_iters: int = 2000, seed: int = 42
) -> tuple[float, float]:
    """Run ALS to test whether tensor T admits an exact (bounded-norm) CP decomposition."""
    rng = np.random.default_rng(seed)
    d0, d1, d2 = T.shape
    T0 = T.reshape(d0, -1)
    T1 = np.transpose(T, (1, 0, 2)).reshape(d1, -1)
    T2 = np.transpose(T, (2, 0, 1)).reshape(d2, -1)
    best_err = float("inf")
    best_norm = float("inf")
    reg = 1e-12

    for _ in range(n_restarts):
        A = rng.standard_normal((d0, rank))
        B = rng.standard_normal((d1, rank))
        C = rng.standard_normal((d2, rank))
        for _ in range(n_iters):
            KR = np.einsum("jr,kr->jkr", B, C).reshape(d1 * d2, rank)
            A = np.linalg.solve(KR.T @ KR + reg * np.eye(rank), KR.T @ T0.T).T
            KR = np.einsum("ir,kr->ikr", A, C).reshape(d0 * d2, rank)
            B = np.linalg.solve(KR.T @ KR + reg * np.eye(rank), KR.T @ T1.T).T
            KR = np.einsum("ir,jr->ijr", A, B).reshape(d0 * d1, rank)
            C = np.linalg.solve(KR.T @ KR + reg * np.eye(rank), KR.T @ T2.T).T
        recon = np.einsum("ir,jr,kr->ijk", A, B, C)
        err = float(np.linalg.norm(T - recon))
        nA = np.linalg.norm(A, axis=0)
        nB = np.linalg.norm(B, axis=0)
        max_c = float(np.max(np.abs(C * (nA * nB)[None, :])))
        if err < best_err:
            best_err = err
            best_norm = max_c
    return best_err, best_norm


def lift_exact_rank7_pair_and_chain() -> tuple[int, int, sp.Matrix]:
    """
    Exact rational lifting over Q for:
    1) 4x4x6 2x2-permanent tensor (C_exact, 7x6)
    2) 6x4x4 2->3 permanent lift tensor (A_exact, 6x7)
    3) Fused 7x7 projection L = A_exact.T * C_exact.T (rank-1 update of sign-diagonal).
    """
    vecs = [
        [1, 1, 1, 1],
        [1, 1, 1, -1],
        [1, 1, -1, 1],
        [1, 1, -1, -1],
        [1, -1, 1, 1],
        [1, -1, 1, -1],
        [1, -1, -1, 1],
    ]
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    triples = [(i, j, k) for i in range(4) for j in range(i + 1, 4) for k in range(j + 1, 4)]

    M1 = sp.Matrix([[v[i] * v[j] for v in vecs] for i in range(4) for j in range(4)])
    T1 = sp.Matrix(
        [[1 if (i, j) == p or (j, i) == p else 0 for p in pairs] for i in range(4) for j in range(4)]
    )
    C_exact, _ = M1.gauss_jordan_solve(T1)
    res1 = int((M1 * C_exact - T1).norm())

    KR = sp.Matrix([[v[k] * v[3 - t] for v in vecs] for k in range(4) for t in range(4)])
    T2 = sp.Matrix([
        [
            1 if (k in triples[t] and pairs[p] == tuple(sorted(set(triples[t]) - {k}))) else 0
            for p in range(6)
        ]
        for k in range(4)
        for t in range(4)
    ])
    A_T, _ = KR.gauss_jordan_solve(T2)
    res2 = int((KR * A_T - T2).norm())
    L_fused = A_T * C_exact.T  # 7x7 exact fused projection
    return res1, res2, 8 * L_fused
