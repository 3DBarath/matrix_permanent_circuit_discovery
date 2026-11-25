# Evolutionary Code-Mutation Autoresearch Specification (Gen 21+)
**Target:** Breaking $L_{\mathrm{mul}}(\mathrm{Perm}_4) < 18$ and Exploring Non-Hypercube $(2-\varepsilon)^n$ Algebraic Circuits

---

## 1. Ground Truth: What Is Proved / Ruled Out (Do Not Waste Compute Here)

1. **$L_{\mathrm{mul}}(\mathrm{Perm}_3) = 7$ (Unconditional, $\mathrm{char}(\mathbb{F}) \neq 2$):** Closed for all general arithmetic DAGs (including arbitrary degree-$\ge 3$ feedback) via 5 machine-certified lemmas over $\mathbb{Z}[1/2]$ (Section 5.3 of `paper/main.tex`).
2. **Depth-3 $\Sigma\Pi\Sigma$ Product Rank of $\mathrm{Perm}_3 = 4$:** Closed even with mixed-row linear forms.
3. **Row-Separated $\Sigma\Pi\Sigma$ Over ANY Field (Including Roots of Unity $\zeta_k$):** By Ranestad–Schreyer (2011) / Carlini–Catalisano–Geramita (2012), the symmetric Waring rank of $x_1 x_2 \cdots x_n$ over $\mathbb{C}$ is strictly $2^{n-1}$. Any row-separated polarization over $\{\pm 1\}^{n-1}$ OR $\zeta_k$ OR arbitrary complex grids requires $\ge 2^{n-1}$ products.
4. **Hypercube Linear Floor (Gen 10 `general_n_schur_f2.py`):** Hits the exact DP floor $O(2^n)$ ($18$ at $n=4$, $52{,}488$ at $n=16$). No butterfly, Schur-complement, or CSE optimization inside $\{\pm 1\}^{n-1}$ can drop below base $2$.
5. **Determinant-Hadamard Sums $\sum_{k=1}^K \det(M_k \odot A)$:** Exact numerical rank is $K = 2$ ($n=3$), $K = 3$ ($n=4$), $K = 7$ ($n=5$)—below $2^{n-1}$ for small $n$, but dimension counting ($K n^2 \ge n!$) forces $K = \Omega(n!/n^2) \gg 2^n$ asymptotically. Do not pursue for large-$n$ $(2-\varepsilon)^n$.
6. **Row-Layered $n=4$ First Layer:** $\mathrm{rank}_{\mathbb{F}}(T_1) \ge 7$ proved (Prop. 5.1, Lean 4).

---

## 2. Genuinely Open Frontiers (Where Mutations Should Focus)

### Track 1: Unconstrained 17-Mul General DAGs for $\mathrm{Perm}_4$
The 6.55B CUDA search (`perm4_17mul_search.cu`) left four blind spots open:
- Hardcoded `(8, 4, 5)` split (excluded `(7, 6, 4)`, `(6, 7, 4)`, pure quadratic `(11, 0, 6)`, `(12, 0, 5)`).
- Hardcoded `MAX_TERMS = 6` (excluded even Gen 2's 7-term shift `S`).
- Strictly homogeneous layers (excluded inhomogeneous cancellation DAGs where $Y^{(2)} = Y^{(3)} = 0$ cancels across gates).
- Flat $\mathbb{F}_3/\mathbb{F}_5$ Hamming loss (near-random walk).

**Mutation Protocol (Track 1):**
- Mutate **continuous-relaxation solvers** (`perm4_gpu_lm_autoresearch.py` / `perm4_deep_dag_solver.py`) using **overdetermined multi-radius batches** ($B \ge 2 \cdot n_{\text{params}}$, radii $r \in \{1.0, 1.5, 2.0\}$ so inhomogeneous high-degree terms cannot overfit a single sphere).
- Prioritize second-order methods (exact analytic-Jacobian Levenberg-Marquardt / Gauss-Newton / L-BFGS) over first-order Adam alone.
- Mutate: `(d2, d3, d4)` splits, $S_4 \times S_4$ symmetry-tied weight subspaces, homotopy pruning schedules from the exact 18-mul Gen 2 manifold, and LLL / rational reconstruction thresholds.
- **Calibrated sanity gate:** Any mutated solver MUST recover the exact 18-mul Gen 2 circuit from $\sigma = 0.25$ noise to `val_mse < 1e-20` before its 17-mul runs are trusted.

### Track 2: Non-Hypercube General-$n$ Circuit Compilers
- **Forbidden:** Any $\{\pm 1\}^{n-1}$ row-separated polarization, low-rank assumptions, sparse assumptions, $\mathrm{char}=2$ shortcuts, or floating-point approximations.
- **Permitted non-hypercube search spaces:**
  1. Mixed-row bilinear-quadratic tensor contractions (sharing degree-2 gates across row partitions).
  2. Recursive block-permanent composition (composing optimal $3\times 3$ / $4\times 4$ blocks with shared minors).
  3. Algebraic border-rank / $\varepsilon$-perturbation identities (if a border-rank circuit has small error degree $e$, interpolation costs $O(e)$).
- **Fitness:** Incremental doubling ratio $L_{\mathrm{mul}}(n) / L_{\mathrm{mul}}(n-1)$ across $n \in \{4, 5, 6, 7, 8\}$. Reject any mutation that only shaves a constant factor without lowering the doubling ratio below $2.0$.

---

## 3. Immutable Anti-Cheat Verification Contract (`dag_verification.py` & `tracked_field.py`)

Every candidate with `val_mse < 1e-10` (Track 1) or claiming sub-$2^n$ scaling (Track 2) MUST pass all three gates unmodified:
1. **Gate 1 (Dual 61-Bit Prime Schwartz-Zippel):** 100/100 exact matches against `glynn_perm_mod` over `PRIME_61_A = 2^61 - 1` and `PRIME_61_B = 2^61 - 31` on dense random matrices.
2. **Gate 2 (Strict Wire Accounting):** Executed exclusively via `TrackedFieldElement` + `OpCounter` (no divisions, no data-dependent branching, no hidden scalar/wire multiplications).
3. **Gate 3 (Exact Symbolic Proof):** `sp.expand(sym_out - sym_ref) == 0` in $\mathbb{Q}[A]$ for $n \le 4$.

---

## 4. Fallback Value Under `#ETH`
If `#ETH` holds and no $(2-\varepsilon)^n$ dense permanent circuit exists, every Track 1 generation still yields publishable value:
- Either a **17-mul $\mathrm{Perm}_4$ breakthrough** (breaking the 18-mul record), or
- **Hessian / Jacobian rank & basin-of-attraction data** guiding an extension of Section 5.3's 5-lemma $\mathbb{Z}[1/2]$ lower-bound proof from $L_{\mathrm{mul}}(\mathrm{Perm}_3) = 7$ ($\dim \mathfrak{g} = 4$) to $L_{\mathrm{mul}}(\mathrm{Perm}_4) = 18$ ($\dim \mathfrak{g} = 6$, essential variables $= 16$).
