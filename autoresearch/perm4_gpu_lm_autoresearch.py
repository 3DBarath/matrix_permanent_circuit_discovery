"""
Evolutionary Autoresearch Engine for Unconstrained 17-Mul Perm_4 Arithmetic DAGs
(Dual-Backend: GPU-Batched PyTorch Gauss-Newton/LM on CUDA, or Analytic-Jacobian SciPy TRF on CPU).

Key architectural fixes over bit-sliced F_3/F_5 CUDA SA:
  1. Unconstrained topologies: arbitrary (d2, d3, d4) splits (including (7,6,4), (6,7,4), (11,0,6), (12,0,5))
     and full lower-triangular general_inhomog DAGs with NO MAX_TERMS=6 cap.
  2. Multi-radius homogeneity sampling (r in {1.0, 1.5, 2.0}) to prevent high-degree blowup
     in inhomogeneous cancellation DAGs.
  3. Dense continuous fitness (log10 val_mse on held-out matrices) + automatic exact rational
     lifting and 3-gate verification via dag_verification.py whenever val_mse < 1e-10.

Usage on Google Colab (Tesla T4 / A100):
  python3 perm4_gpu_lm_autoresearch.py --generations 20 --pop-size 8 --restarts 64
Usage locally (CPU fallback):
  python3 perm4_gpu_lm_autoresearch.py --generations 3 --pop-size 4 --restarts 1 --max-nfev 25
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import time
import numpy as np

from dag_verification import lift_float_dag_to_rationals, verify_rational_dag
from perm4_deep_dag_solver import (
    MaskedDAGModel,
    encode_gen2_18mul_dag,
    eval_perm4_batch,
    make_topology_mask,
    run_lm_solve,
)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def sample_multiradius_matrices(rng: np.random.Generator, B: int, radii=(1.0, 1.5, 2.0)) -> np.ndarray:
    """
    Sample B random 4x4 matrices across multiple Frobenius radii.
    Crucial for general_inhomog DAGs: forces non-degree-4 terms to cancel across scales.
    """
    X = rng.standard_normal((B, 16))
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    r_choices = rng.choice(radii, size=(B, 1))
    return X * (r_choices / norms)


@dataclass
class SolverGenome:
    family: str                   # "stratified_dense", "pure_quad", "mixed_deg4", "general_inhomog"
    split: tuple[int, int, int]   # (d2, d3, d4) summing to M=17
    scale_x: float                # init std for input linear forms
    scale_g: float                # init std for gate feedback linear forms
    reg_weight: float             # L2 damping weight
    warm_drop_gate: int           # -1 for cold start, 0..17 to warm-start by dropping gate k from Gen 2
    warm_noise: float             # perturbation std around warm-start


_VALID_SPLITS = {
    "stratified_dense": [(7, 6, 4), (6, 7, 4), (8, 5, 4), (9, 4, 4), (7, 5, 5), (6, 6, 5)],
    "pure_quad": [(11, 0, 6), (12, 0, 5), (10, 0, 7), (13, 0, 4)],
    "mixed_deg4": [(7, 5, 5), (8, 4, 5), (9, 3, 5), (8, 5, 4), (7, 6, 4)],
    "general_inhomog": [(17, 0, 0)],
}


def random_genome(rng: np.random.Generator) -> SolverGenome:
    fam = str(rng.choice(list(_VALID_SPLITS.keys())))
    split = _VALID_SPLITS[fam][int(rng.integers(len(_VALID_SPLITS[fam])))]
    warm_drop = int(rng.choice([-1, -1, 6, 13, 17])) if fam == "general_inhomog" else -1
    return SolverGenome(
        family=fam,
        split=split,
        scale_x=float(10 ** rng.uniform(-0.8, -0.3)),
        scale_g=float(10 ** rng.uniform(-1.6, -0.9)),
        reg_weight=float(10 ** rng.uniform(-6.0, -3.5)),
        warm_drop_gate=warm_drop,
        warm_noise=float(10 ** rng.uniform(-2.0, -0.8)),
    )


def mutate_genome(g: SolverGenome, rng: np.random.Generator) -> SolverGenome:
    d = asdict(g)
    d["split"] = tuple(d["split"])
    move = int(rng.integers(5))
    if move == 0:
        fam = str(rng.choice(list(_VALID_SPLITS.keys())))
        d["family"] = fam
        d["split"] = _VALID_SPLITS[fam][int(rng.integers(len(_VALID_SPLITS[fam])))]
        if fam != "general_inhomog":
            d["warm_drop_gate"] = -1
    elif move == 1:
        splits = _VALID_SPLITS[d["family"]]
        d["split"] = splits[int(rng.integers(len(splits)))]
    elif move == 2:
        d["scale_x"] = float(np.clip(d["scale_x"] * (10 ** rng.uniform(-0.25, 0.25)), 0.08, 0.7))
        d["scale_g"] = float(np.clip(d["scale_g"] * (10 ** rng.uniform(-0.25, 0.25)), 0.01, 0.2))
    elif move == 3:
        d["reg_weight"] = float(np.clip(d["reg_weight"] * (10 ** rng.uniform(-0.6, 0.6)), 1e-7, 1e-2))
    else:
        if d["family"] == "general_inhomog":
            d["warm_drop_gate"] = int(rng.choice([-1, 0, 6, 7, 13, 14, 17]))
            d["warm_noise"] = float(np.clip(d["warm_noise"] * (10 ** rng.uniform(-0.3, 0.3)), 0.005, 0.3))
    return SolverGenome(**d)


def evaluate_genome_torch_cuda(
    genome: SolverGenome,
    X_train_np: np.ndarray,
    y_train_np: np.ndarray,
    X_val_np: np.ndarray,
    y_val_np: np.ndarray,
    restarts: int,
    steps: int,
    seed: int,
) -> dict:
    """
    GPU-batched PyTorch AdamW -> L-BFGS / damped Gauss-Newton solver on CUDA.
    Runs `restarts` parallel DAGs simultaneously on GPU.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    M = 17
    mLx, mLg, mRx, mRg, mw = make_topology_mask(
        genome.family, M, genome.split if genome.family != "general_inhomog" else None
    )
    t_mLx = torch.tensor(mLx, device=device, dtype=torch.float64)
    t_mLg = torch.tensor(mLg, device=device, dtype=torch.float64)
    t_mRx = torch.tensor(mRx, device=device, dtype=torch.float64)
    t_mRg = torch.tensor(mRg, device=device, dtype=torch.float64)
    t_mw = torch.tensor(mw, device=device, dtype=torch.float64)

    X_tr = torch.tensor(X_train_np, device=device, dtype=torch.float64)
    y_tr = torch.tensor(y_train_np, device=device, dtype=torch.float64)
    X_va = torch.tensor(X_val_np, device=device, dtype=torch.float64)
    y_va = torch.tensor(y_val_np, device=device, dtype=torch.float64)

    Lx = torch.randn((restarts, M, 16), device=device, dtype=torch.float64) * genome.scale_x
    Rx = torch.randn((restarts, M, 16), device=device, dtype=torch.float64) * genome.scale_x
    Lg = torch.randn((restarts, M, M), device=device, dtype=torch.float64) * genome.scale_g
    Rg = torch.randn((restarts, M, M), device=device, dtype=torch.float64) * genome.scale_g
    w = torch.randn((restarts, M), device=device, dtype=torch.float64) * 0.25

    if genome.family == "general_inhomog" and genome.warm_drop_gate >= 0:
        Lx18, Lg18, Rx18, Rg18, w18 = encode_gen2_18mul_dag()
        keep = [i for i in range(18) if i != genome.warm_drop_gate]
        Lx = torch.tensor(Lx18[keep], device=device)[None] + torch.randn_like(Lx) * genome.warm_noise
        Rx = torch.tensor(Rx18[keep], device=device)[None] + torch.randn_like(Rx) * genome.warm_noise
        Lg = torch.tensor(Lg18[np.ix_(keep, keep)], device=device)[None] + torch.randn_like(Lg) * genome.warm_noise
        Rg = torch.tensor(Rg18[np.ix_(keep, keep)], device=device)[None] + torch.randn_like(Rg) * genome.warm_noise
        w = torch.tensor(w18[keep], device=device)[None] + torch.randn_like(w) * genome.warm_noise

    for p in (Lx, Lg, Rx, Rg, w):
        p.requires_grad_(True)

    opt = torch.optim.AdamW([Lx, Lg, Rx, Rg, w], lr=2e-2, weight_decay=genome.reg_weight)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-4)

    def forward_batch(X_in):
        # X_in: (B, 16), output: (R, B)
        B_cur = X_in.shape[0]
        G_list = []
        eLx = Lx * t_mLx
        eLg = Lg * t_mLg
        eRx = Rx * t_mRx
        eRg = Rg * t_mRg
        ew = w * t_mw
        for k in range(M):
            lk = torch.einsum("ri,bi->rb", eLx[:, k, :], X_in)
            rk = torch.einsum("ri,bi->rb", eRx[:, k, :], X_in)
            if k > 0:
                G_prev = torch.stack(G_list, dim=2)  # (R, B, k)
                lk = lk + torch.einsum("rbj,rj->rb", G_prev, eLg[:, k, :k])
                rk = rk + torch.einsum("rbj,rj->rb", G_prev, eRg[:, k, :k])
            G_list.append(lk * rk)
        G_all = torch.stack(G_list, dim=2)  # (R, B, M)
        return torch.einsum("rbm,rm->rb", G_all, ew)

    for _ in range(steps):
        opt.zero_grad()
        pred = forward_batch(X_tr)
        loss = torch.mean((pred - y_tr[None, :]) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([Lx, Lg, Rx, Rg, w], 10.0)
        opt.step()
        sched.step()

    # Phase 2: Batched GPU Levenberg-Marquardt (torch.func.vmap + jacrev + linalg.solve)
    # Packs only active masked parameters to keep the P x P normal equations compact.
    lm_steps = max(8, steps // 20)
    with torch.no_grad():
        idx_Lx = torch.nonzero(t_mLx, as_tuple=False)
        idx_Lg = torch.nonzero(t_mLg, as_tuple=False)
        idx_Rx = torch.nonzero(t_mRx, as_tuple=False)
        idx_Rg = torch.nonzero(t_mRg, as_tuple=False)
        idx_w = torch.nonzero(t_mw, as_tuple=False)
        nLx, nLg, nRx, nRg, nw = len(idx_Lx), len(idx_Lg), len(idx_Rx), len(idx_Rg), len(idx_w)
        P = nLx + nLg + nRx + nRg + nw
        theta_batch = torch.cat(
            [
                Lx[:, idx_Lx[:, 0], idx_Lx[:, 1]],
                Lg[:, idx_Lg[:, 0], idx_Lg[:, 1]],
                Rx[:, idx_Rx[:, 0], idx_Rx[:, 1]],
                Rg[:, idx_Rg[:, 0], idx_Rg[:, 1]],
                w[:, idx_w[:, 0]],
            ],
            dim=1,
        ).detach()

    def _unpack_single(th):
        o0 = nLx
        o1 = o0 + nLg
        o2 = o1 + nRx
        o3 = o2 + nRg
        pLx = torch.zeros((M, 16), device=device, dtype=torch.float64).index_put(
            (idx_Lx[:, 0], idx_Lx[:, 1]), th[:o0]
        )
        pLg = torch.zeros((M, M), device=device, dtype=torch.float64).index_put(
            (idx_Lg[:, 0], idx_Lg[:, 1]), th[o0:o1]
        )
        pRx = torch.zeros((M, 16), device=device, dtype=torch.float64).index_put(
            (idx_Rx[:, 0], idx_Rx[:, 1]), th[o1:o2]
        )
        pRg = torch.zeros((M, M), device=device, dtype=torch.float64).index_put(
            (idx_Rg[:, 0], idx_Rg[:, 1]), th[o2:o3]
        )
        pw = torch.zeros((M,), device=device, dtype=torch.float64).index_put((idx_w[:, 0],), th[o3:])
        return pLx, pLg, pRx, pRg, pw

    def _residual_single(th):
        pLx, pLg, pRx, pRg, pw = _unpack_single(th)
        G_list = []
        for k in range(M):
            lk = X_tr @ pLx[k]
            rk = X_tr @ pRx[k]
            if k > 0:
                G_prev = torch.stack(G_list, dim=1)
                lk = lk + G_prev @ pLg[k, :k]
                rk = rk + G_prev @ pRg[k, :k]
            G_list.append(lk * rk)
        G_all = torch.stack(G_list, dim=1)
        return (G_all @ pw) - y_tr

    # Chunk restarts so Jacobian (chunk, B, P) fits comfortably in T4 VRAM (15.6 GB)
    chunk_sz = max(1, min(restarts, 8))
    eye_P = torch.eye(P, device=device, dtype=torch.float64)
    vmap_res = torch.func.vmap(_residual_single)
    vmap_jac = torch.func.vmap(torch.func.jacrev(_residual_single))

    with torch.no_grad():
        for start in range(0, restarts, chunk_sz):
            th_c = theta_batch[start : start + chunk_sz].clone()
            lam = torch.full((th_c.shape[0],), max(genome.reg_weight, 1e-3), device=device, dtype=torch.float64)
            r_c = vmap_res(th_c)
            loss_c = torch.mean(r_c * r_c, dim=1)
            for _ in range(lm_steps):
                J_c = vmap_jac(th_c)  # (C, B, P)
                JTJ = torch.einsum("cbp,cbq->cpq", J_c, J_c)
                JTr = torch.einsum("cbp,cb->cp", J_c, r_c)
                A_sys = JTJ + lam[:, None, None] * eye_P[None]
                delta = torch.linalg.solve(A_sys, -JTr.unsqueeze(-1)).squeeze(-1)
                th_try = th_c + delta
                r_try = vmap_res(th_try)
                loss_try = torch.mean(r_try * r_try, dim=1)
                accept = loss_try < loss_c
                th_c = torch.where(accept[:, None], th_try, th_c)
                r_c = torch.where(accept[:, None], r_try, r_c)
                loss_c = torch.where(accept, loss_try, loss_c)
                lam = torch.where(accept, torch.clamp(lam * 0.3, min=1e-12), torch.clamp(lam * 5.0, max=1e6))
            theta_batch[start : start + chunk_sz] = th_c

        for r_idx in range(restarts):
            pLx, pLg, pRx, pRg, pw = _unpack_single(theta_batch[r_idx])
            Lx[r_idx], Lg[r_idx], Rx[r_idx], Rg[r_idx], w[r_idx] = pLx, pLg, pRx, pRg, pw

    with torch.no_grad():
        val_pred = forward_batch(X_va)
        val_mses = torch.mean((val_pred - y_va[None, :]) ** 2, dim=1)
        best_idx = int(torch.argmin(val_mses).item())
        best_val_mse = float(val_mses[best_idx].item())
        val_rms_y = float(torch.sqrt(torch.mean(y_va * y_va)).item())
        best_rel_rmse = math.sqrt(best_val_mse) / max(val_rms_y, 1e-12)
        tr_pred = forward_batch(X_tr)
        best_tr_mse = float(torch.mean((tr_pred[best_idx] - y_tr) ** 2).item())

    out = {
        "train_mse": best_tr_mse,
        "val_mse": best_val_mse,
        "val_rel_rmse": best_rel_rmse,
        "log10_val_mse": math.log10(max(best_val_mse, 1e-30)),
        "verified_exact": False,
    }
    if best_val_mse < 1e-10:
        rat = lift_float_dag_to_rationals(
            (Lx[best_idx] * t_mLx).detach().cpu().numpy(),
            (Lg[best_idx] * t_mLg).detach().cpu().numpy(),
            (Rx[best_idx] * t_mRx).detach().cpu().numpy(),
            (Rg[best_idx] * t_mRg).detach().cpu().numpy(),
            (w[best_idx] * t_mw).detach().cpu().numpy(),
        )
        v = verify_rational_dag(4, *rat, trials_per_prime=50, run_symbolic=True)
        out["verified_exact"] = bool(v["passed"])
        out["verifier_report"] = v
    return out


def evaluate_genome_cpu(
    genome: SolverGenome,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    restarts: int,
    max_nfev: int,
    rng: np.random.Generator,
) -> dict:
    M = 17
    masks = make_topology_mask(
        genome.family, M, genome.split if genome.family != "general_inhomog" else None
    )
    model = MaskedDAGModel(M, masks)
    best = None

    for _ in range(restarts):
        if genome.family == "general_inhomog" and genome.warm_drop_gate >= 0:
            Lx18, Lg18, Rx18, Rg18, w18 = encode_gen2_18mul_dag()
            keep = [i for i in range(18) if i != genome.warm_drop_gate]
            th0 = model.pack(
                Lx18[keep], Lg18[np.ix_(keep, keep)], Rx18[keep], Rg18[np.ix_(keep, keep)], w18[keep]
            ) + rng.standard_normal(model.n_params) * genome.warm_noise
        else:
            th0 = model.init_safe_theta(rng, scale_x=genome.scale_x, scale_g=genome.scale_g)
        res = run_lm_solve(
            model, X_train, y_train, X_val, y_val, th0, max_nfev=max_nfev, reg_weight=genome.reg_weight
        )
        if best is None or res["val_mse"] < best["val_mse"]:
            best = res

    out = {
        "train_mse": best["train_mse"],
        "val_mse": best["val_mse"],
        "val_rel_rmse": best["val_rel_rmse"],
        "log10_val_mse": math.log10(max(best["val_mse"], 1e-30)),
        "verified_exact": False,
    }
    if best["val_mse"] < 1e-10:
        Lx, Lg, Rx, Rg, w = model.unpack(best["theta"])
        rat = lift_float_dag_to_rationals(Lx, Lg, Rx, Rg, w)
        v = verify_rational_dag(4, *rat, trials_per_prime=50, run_symbolic=True)
        out["verified_exact"] = bool(v["passed"])
        out["verifier_report"] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--pop-size", type=int, default=4)
    ap.add_argument("--restarts", type=int, default=1)
    ap.add_argument("--max-nfev", type=int, default=25)
    ap.add_argument("--torch-steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--log", type=str, default="evolution_log.jsonl")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    # Multi-radius overdetermined batch (prevents high-degree blowup in general_inhomog!)
    X_train = sample_multiradius_matrices(rng, 1600, radii=(1.0, 1.5, 2.0))
    y_train = eval_perm4_batch(X_train)
    X_val = sample_multiradius_matrices(rng, 500, radii=(1.0, 1.5, 2.0))
    y_val = eval_perm4_batch(X_val)

    use_gpu = HAS_TORCH and torch.cuda.is_available()
    backend = "PyTorch CUDA" if use_gpu else "SciPy Analytic-Jacobian TRF (CPU)"
    print(f"=== Evolutionary Autoresearch Engine | Backend: {backend} ===")

    pop = [random_genome(rng) for _ in range(args.pop_size)]
    t0 = time.time()

    with open(args.log, "w", encoding="utf-8") as logf:
        for gen in range(1, args.generations + 1):
            scored = []
            for idx, g in enumerate(pop):
                if use_gpu:
                    res = evaluate_genome_torch_cuda(
                        g, X_train, y_train, X_val, y_val, args.restarts, args.torch_steps, args.seed + gen * 100 + idx
                    )
                else:
                    res = evaluate_genome_cpu(
                        g, X_train, y_train, X_val, y_val, args.restarts, args.max_nfev, rng
                    )
                scored.append((res["val_mse"], g, res))
                rec = {"gen": gen, "idx": idx, "genome": asdict(g), "metrics": res}
                logf.write(json.dumps(rec) + "\n")
                logf.flush()
                if res["verified_exact"]:
                    print(f"!!! BREAKTHROUGH: VERIFIED EXACT 17-MUL PERM_4 CIRCUIT AT GEN {gen} !!!")
                    return

            scored.sort(key=lambda x: x[0])
            best_mse, best_g, best_r = scored[0]
            print(
                f"[Gen {gen:2d}] Best val_mse={best_mse:.4e} (log10={best_r['log10_val_mse']:.2f}) | "
                f"family={best_g.family} split={best_g.split} warm_drop={best_g.warm_drop_gate} "
                f"elapsed={time.time() - t0:.1f}s"
            )

            # Elitism (top 2) + mutated offspring
            elites = [g for _, g, _ in scored[: max(1, args.pop_size // 2)]]
            next_pop = list(elites)
            while len(next_pop) < args.pop_size:
                parent = elites[int(rng.integers(len(elites)))]
                next_pop.append(mutate_genome(parent, rng))
            pop = next_pop


if __name__ == "__main__":
    main()
