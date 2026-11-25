"""
Deep Continuous-Relaxation + Exact Analytic-Jacobian Levenberg-Marquardt Solver
for Unconstrained M-Multiplication Perm_4 Arithmetic DAGs.

Supports four circuit families that the bit-sliced CUDA kernel excluded:
  1. stratified_dense: arbitrary (d2, d3, d4) splits with dense linear forms (no MAX_TERMS=6 cap)
  2. pure_quad: (d2, 0, d4) pure deg2*deg2 mixed-row bilinear-quadratic circuits
  3. general_inhomog: full unconstrained lower-triangular DAG allowing inhomogeneous cancellation
  4. homotopy_18_to_17: warm-start from the exact 18-mul Gen 2 champion, pruning each gate
     and re-optimizing the remaining 17 gates in the unconstrained DAG manifold.
"""

from __future__ import annotations

import itertools
import time
import numpy as np
from scipy.optimize import least_squares

from dag_verification import lift_float_dag_to_rationals, verify_rational_dag

_PERMS_4 = np.array(list(itertools.permutations(range(4))), dtype=np.int32)
_ROWS_4 = np.arange(4)[None, :]


def eval_perm4_batch(X: np.ndarray) -> np.ndarray:
    """Exact Perm_4(A) for a batch X of shape (B, 16)."""
    A = X.reshape(-1, 4, 4)
    gathered = A[:, _ROWS_4, _PERMS_4]  # (B, 24, 4)
    return np.sum(np.prod(gathered, axis=2), axis=1)


def sample_matrices(rng: np.random.Generator, B: int, radius: float = 2.0) -> np.ndarray:
    """Sample B random 4x4 matrices normalized to Frobenius norm = radius (prevents overflow)."""
    X = rng.standard_normal((B, 16))
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X * (radius / norms)


def make_topology_mask(family: str, M: int, split: tuple[int, int, int] | None = None):
    """Build boolean parameter masks (mLx, mLg, mRx, mRg, mw) for a given DAG family."""
    n2 = 16
    mLx = np.zeros((M, n2), dtype=bool)
    mLg = np.zeros((M, M), dtype=bool)
    mRx = np.zeros((M, n2), dtype=bool)
    mRg = np.zeros((M, M), dtype=bool)
    mw = np.zeros(M, dtype=bool)

    if family == "general_inhomog":
        mLx[:, :] = True
        mRx[:, :] = True
        for k in range(M):
            mLg[k, :k] = True
            mRg[k, :k] = True
        mw[:] = True
    elif family == "stratified_dense":
        assert split is not None and sum(split) == M
        d2, d3, d4 = split
        mLx[:d2, :] = True
        mRx[:d2, :] = True
        mLx[d2:d2 + d3, :] = True
        mRg[d2:d2 + d3, :d2] = True
        if d3 > 0:
            mLx[d2 + d3:, :] = True
            mRg[d2 + d3:, d2:d2 + d3] = True
        else:
            mLg[d2 + d3:, :d2] = True
            mRg[d2 + d3:, :d2] = True
        mw[d2 + d3:] = True
    elif family == "pure_quad":
        assert split is not None and split[1] == 0 and sum(split) == M
        d2, _, d4 = split
        mLx[:d2, :] = True
        mRx[:d2, :] = True
        mLg[d2:, :d2] = True
        mRg[d2:, :d2] = True
        mw[d2:] = True
    elif family == "mixed_deg4":
        assert split is not None and sum(split) == M
        d2, d3, d4 = split
        mLx[:d2, :] = True
        mRx[:d2, :] = True
        mLx[d2:d2 + d3, :] = True
        mRg[d2:d2 + d3, :d2] = True
        mLx[d2 + d3:, :] = True
        mLg[d2 + d3:, :d2] = True
        mRg[d2 + d3:, :d2 + d3] = True
        mw[d2 + d3:] = True
    else:
        raise ValueError(f"Unknown family: {family}")

    return mLx, mLg, mRx, mRg, mw


class MaskedDAGModel:
    """Vectorized general arithmetic DAG with exact reverse-mode analytic Jacobian."""

    def __init__(self, M: int, masks: tuple[np.ndarray, ...]):
        self.M = M
        self.n2 = 16
        self.mLx, self.mLg, self.mRx, self.mRg, self.mw = masks
        self.iLx = np.flatnonzero(self.mLx)
        self.iLg = np.flatnonzero(self.mLg)
        self.iRx = np.flatnonzero(self.mRx)
        self.iRg = np.flatnonzero(self.mRg)
        self.iw = np.flatnonzero(self.mw)
        self.sizes = [len(self.iLx), len(self.iLg), len(self.iRx), len(self.iRg), len(self.iw)]
        self.offsets = np.cumsum([0] + self.sizes)
        self.n_params = int(self.offsets[-1])

    def unpack(self, theta: np.ndarray):
        M, n2 = self.M, self.n2
        Lx = np.zeros((M, n2), dtype=np.float64)
        Lg = np.zeros((M, M), dtype=np.float64)
        Rx = np.zeros((M, n2), dtype=np.float64)
        Rg = np.zeros((M, M), dtype=np.float64)
        w = np.zeros(M, dtype=np.float64)
        o = self.offsets
        Lx.ravel()[self.iLx] = theta[o[0]:o[1]]
        Lg.ravel()[self.iLg] = theta[o[1]:o[2]]
        Rx.ravel()[self.iRx] = theta[o[2]:o[3]]
        Rg.ravel()[self.iRg] = theta[o[3]:o[4]]
        w.ravel()[self.iw] = theta[o[4]:o[5]]
        return Lx, Lg, Rx, Rg, w

    def pack(self, Lx, Lg, Rx, Rg, w) -> np.ndarray:
        return np.concatenate([
            Lx.ravel()[self.iLx],
            Lg.ravel()[self.iLg],
            Rx.ravel()[self.iRx],
            Rg.ravel()[self.iRg],
            w.ravel()[self.iw],
        ])

    def init_safe_theta(self, rng: np.random.Generator, scale_x: float = 0.35, scale_g: float = 0.08) -> np.ndarray:
        """Depth-scaled initialization that prevents exponential blowup in inhomogeneous DAGs."""
        M = self.M
        Lx = rng.standard_normal((M, 16)) * scale_x
        Rx = rng.standard_normal((M, 16)) * scale_x
        Lg = np.zeros((M, M), dtype=np.float64)
        Rg = np.zeros((M, M), dtype=np.float64)
        for k in range(1, M):
            s = scale_g / np.sqrt(k)
            Lg[k, :k] = rng.standard_normal(k) * s
            Rg[k, :k] = rng.standard_normal(k) * s
        w = rng.standard_normal(M) * 0.25
        return self.pack(Lx, Lg, Rx, Rg, w)

    def forward_and_jac(self, theta: np.ndarray, X: np.ndarray):
        B = X.shape[0]
        M = self.M
        Lx, Lg, Rx, Rg, w = self.unpack(theta)

        G = np.empty((B, M), dtype=np.float64)
        L_vals = np.empty((B, M), dtype=np.float64)
        R_vals = np.empty((B, M), dtype=np.float64)

        for k in range(M):
            lk = X @ Lx[k]
            rk = X @ Rx[k]
            if k > 0:
                if np.any(Lg[k, :k]):
                    lk = lk + G[:, :k] @ Lg[k, :k]
                if np.any(Rg[k, :k]):
                    rk = rk + G[:, :k] @ Rg[k, :k]
            L_vals[:, k] = lk
            R_vals[:, k] = rk
            G[:, k] = lk * rk

        y_hat = G @ w

        dG = np.broadcast_to(w[None, :], (B, M)).copy()
        dLx = np.zeros((B, M, self.n2), dtype=np.float64)
        dLg = np.zeros((B, M, M), dtype=np.float64)
        dRx = np.zeros((B, M, self.n2), dtype=np.float64)
        dRg = np.zeros((B, M, M), dtype=np.float64)

        for k in range(M - 1, -1, -1):
            dg_k = dG[:, k]
            dL_k = dg_k * R_vals[:, k]
            dR_k = dg_k * L_vals[:, k]
            if np.any(self.mLx[k]):
                dLx[:, k, :] = dL_k[:, None] * X
            if np.any(self.mRx[k]):
                dRx[:, k, :] = dR_k[:, None] * X
            if k > 0:
                if np.any(self.mLg[k, :k]):
                    dLg[:, k, :k] = dL_k[:, None] * G[:, :k]
                    dG[:, :k] += dL_k[:, None] * Lg[k, :k][None, :]
                if np.any(self.mRg[k, :k]):
                    dRg[:, k, :k] = dR_k[:, None] * G[:, :k]
                    dG[:, :k] += dR_k[:, None] * Rg[k, :k][None, :]

        J = np.empty((B, self.n_params), dtype=np.float64)
        o = self.offsets
        J[:, o[0]:o[1]] = dLx.reshape(B, -1)[:, self.iLx]
        J[:, o[1]:o[2]] = dLg.reshape(B, -1)[:, self.iLg]
        J[:, o[2]:o[3]] = dRx.reshape(B, -1)[:, self.iRx]
        J[:, o[3]:o[4]] = dRg.reshape(B, -1)[:, self.iRg]
        J[:, o[4]:o[5]] = G[:, self.iw]
        return y_hat, J

    def forward(self, theta: np.ndarray, X: np.ndarray) -> np.ndarray:
        B = X.shape[0]
        M = self.M
        Lx, Lg, Rx, Rg, w = self.unpack(theta)
        G = np.empty((B, M), dtype=np.float64)
        for k in range(M):
            lk = X @ Lx[k]
            rk = X @ Rx[k]
            if k > 0:
                if np.any(Lg[k, :k]):
                    lk = lk + G[:, :k] @ Lg[k, :k]
                if np.any(Rg[k, :k]):
                    rk = rk + G[:, :k] @ Rg[k, :k]
            G[:, k] = lk * rk
        return G @ w


def encode_gen2_18mul_dag():
    """Encode the exact Generation 2 champion (perm4_gen2_18muls.py, 7+7+4 = 18 muls)."""
    M = 18
    Lx = np.zeros((M, 16), dtype=np.float64)
    Lg = np.zeros((M, M), dtype=np.float64)
    Rx = np.zeros((M, 16), dtype=np.float64)
    Rg = np.zeros((M, M), dtype=np.float64)
    w = np.zeros(M, dtype=np.float64)

    walsh7 = np.array([
        [1,  1,  1,  1],
        [1,  1,  1, -1],
        [1,  1, -1,  1],
        [1,  1, -1, -1],
        [1, -1,  1,  1],
        [1, -1,  1, -1],
        [1, -1, -1,  1],
    ], dtype=np.float64)
    parities = np.array([1, -1, -1, 1, -1, 1, 1], dtype=np.float64)

    for k in range(7):
        Lx[k, 0:4] = walsh7[k]
        Rx[k, 4:8] = walsh7[k]

    for k in range(7):
        Lx[7 + k, 8:12] = walsh7[k]
        for j in range(7):
            coeff = (1.0 if j == k else 0.0) - parities[j]
            Rg[7 + k, j] = parities[k] * coeff

    inv_w = np.array([
        [1,  1,  1,  1,  1,  1,  1],
        [1,  1,  1,  1, -1, -1, -1],
        [1,  1, -1, -1,  1,  1, -1],
        [1, -1,  1, -1,  1, -1,  1],
    ], dtype=np.float64)
    for c in range(4):
        Lx[14 + c, 12 + c] = 1.0
        Rg[14 + c, 7:14] = inv_w[c]
        w[14 + c] = 1.0 / 8.0

    return Lx, Lg, Rx, Rg, w


def run_lm_solve(
    model: MaskedDAGModel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    theta0: np.ndarray,
    max_nfev: int = 80,
    reg_weight: float = 0.0,
) -> dict:
    """Run exact-Jacobian Trust-Region Reflective optimization on an overdetermined batch."""
    B = X_train.shape[0]
    scale = 1.0 / np.sqrt(B)

    def fun(th):
        r = (model.forward(th, X_train) - y_train) * scale
        if reg_weight > 0.0:
            return np.concatenate([r, reg_weight * th])
        return r

    def jac(th):
        _, J = model.forward_and_jac(th, X_train)
        J *= scale
        if reg_weight > 0.0:
            return np.vstack([J, reg_weight * np.eye(model.n_params)])
        return J

    sol = least_squares(
        fun, theta0, jac=jac, method="trf", ftol=1e-13, xtol=1e-13, gtol=1e-13, max_nfev=max_nfev
    )
    train_mse = float(np.mean((model.forward(sol.x, X_train) - y_train) ** 2))
    val_mse = float(np.mean((model.forward(sol.x, X_val) - y_val) ** 2))
    # Relative RMSE vs target std
    val_rel_rmse = float(np.sqrt(val_mse) / (np.std(y_val) + 1e-12))
    return {
        "theta": sol.x,
        "train_mse": train_mse,
        "val_mse": val_mse,
        "val_rel_rmse": val_rel_rmse,
        "nfev": sol.nfev,
        "param_norm": float(np.linalg.norm(sol.x)),
    }


def main():
    t0 = time.time()
    rng = np.random.default_rng(2026)

    # Strictly overdetermined batches: B_train = 1500 > max n_params (833)
    X_train = sample_matrices(rng, 1500, radius=2.0)
    y_train = eval_perm4_batch(X_train)
    X_val = sample_matrices(rng, 500, radius=2.0)
    y_val = eval_perm4_batch(X_val)

    Lx18, Lg18, Rx18, Rg18, w18 = encode_gen2_18mul_dag()
    m18 = MaskedDAGModel(18, make_topology_mask("stratified_dense", 18, (7, 7, 4)))
    th18 = m18.pack(Lx18, Lg18, Rx18, Rg18, w18)

    # Sanity check: 18-mul exact recovery from sigma=0.25 perturbation on overdetermined B=1500!
    th_pert = th18 + rng.standard_normal(m18.n_params) * 0.25
    res_rec = run_lm_solve(m18, X_train, y_train, X_val, y_val, th_pert, max_nfev=60)
    print(
        f"[OVERDETERMINED SANITY CHECK] 18-mul recovery from sigma=0.25 noise (B=1500):\n"
        f"  train_mse={res_rec['train_mse']:.3e} | val_mse={res_rec['val_mse']:.3e} | "
        f"val_rel_rmse={res_rec['val_rel_rmse']:.3e}"
    )

    print("\n=== Deep Sweep A: Warm-Start Homotopy 18 -> 17 (Unconstrained General DAG, B=1500) ===")
    m17_full = MaskedDAGModel(17, make_topology_mask("general_inhomog", 17))
    for drop_k, label in [(6, "Layer1(deg2)"), (13, "Layer2(deg3)"), (17, "Layer3(deg4)")]:
        keep = [i for i in range(18) if i != drop_k]
        th0 = m17_full.pack(
            Lx18[keep], Lg18[np.ix_(keep, keep)], Rx18[keep], Rg18[np.ix_(keep, keep)], w18[keep]
        ) + rng.standard_normal(m17_full.n_params) * 0.02
        res = run_lm_solve(m17_full, X_train, y_train, X_val, y_val, th0, max_nfev=50, reg_weight=1e-5)
        print(
            f"  Drop {label} (gate {drop_k:2d}): train_mse={res['train_mse']:.3e} | "
            f"val_mse={res['val_mse']:.3e} | rel_rmse={res['val_rel_rmse']:.3f}"
        )

    print("\n=== Deep Sweep B: Multi-Topology 17-Mul Cold-Start Sweep (Overdetermined B=1500) ===")
    configs = [
        ("stratified_dense (7,6,4)", "stratified_dense", 17, (7, 6, 4)),
        ("stratified_dense (6,7,4)", "stratified_dense", 17, (6, 7, 4)),
        ("stratified_dense (8,5,4)", "stratified_dense", 17, (8, 5, 4)),
        ("pure_quad (11,0,6)",       "pure_quad",        17, (11, 0, 6)),
        ("pure_quad (12,0,5)",       "pure_quad",        17, (12, 0, 5)),
        ("general_inhomog (17)",     "general_inhomog",  17, None),
    ]
    for name, fam, M, split in configs:
        model = MaskedDAGModel(M, make_topology_mask(fam, M, split))
        th0 = model.init_safe_theta(rng, scale_x=0.30, scale_g=0.06)
        res = run_lm_solve(model, X_train, y_train, X_val, y_val, th0, max_nfev=45, reg_weight=1e-5)
        print(
            f"  {name:26s} (params={model.n_params:3d}): "
            f"train_mse={res['train_mse']:.3e} | val_mse={res['val_mse']:.3e} | rel_rmse={res['val_rel_rmse']:.3f}"
        )

    print(f"\nTotal wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
