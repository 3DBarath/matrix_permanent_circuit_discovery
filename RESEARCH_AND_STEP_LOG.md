# Research & Evolutionary Step Log

This log records every mathematical hypothesis, computational experiment, failure/correction, and verified milestone across Generations 0–3.

---

## Step 0 — Baseline Analysis & Problem Framing ($n = 4$)
- **Target:** Beat Glynn + Gray code at $n = 4$ (24 non-scalar multiplications) using an exact, branch-free arithmetic straight-line program verified over two 61-bit primes ($p_1 = 2^{61}-1$, $p_2 = 2^{61}-31$) and `TrackedFieldElement` op accounting.
- **Initial algebraic probes (discarded):**
  - Naive Block Laplace (splitting rows $\{0,1\}$ and $\{2,3\}$): $12 + 12 + 6 = 30$ muls.
  - Diagonal-reuse trick $p_{ij} = (x_i+x_j)(y_i+y_j) - x_i y_i - x_j y_j$: $10 + 10 + 6 = 26$ muls (still $> 24$).
  - Pairing $\delta_4 = \pm 1$ in Glynn's formula: eliminates $t^3, t^4$ terms in $Q(t)$ but reduces back to column-4 Laplace or Glynn (24–26 muls).

---

## Step 1 — Generation 1: Continuous ALS Rank-7 Discovery & Exact Lifting (20 Muls at $n = 4$)
- **Hypothesis:** Compress the $4 \times 4 \times 6$ bilinear tensor $T_{ij, p} = \mathbf{1}[\{i,j\}=p]$ (all six $2 \times 2$ permanents of a $2 \times 4$ matrix) below rank 10.
- **Experiment:** Ran ALS CP-decomposition across ranks $r \in \{6, 7, 8, 9\}$:
  - Rank 6: residual `1.255e+00` (infeasible).
  - **Rank 7:** residual `8.862e-09` (refined to `4.932e-11`), max scaled factor norm `1.208` — confirming **exact rank 7** (not border rank).
- **Algebraic Lift:** Observed that for any $v \in \{+1\} \times \{\pm 1\}^3$, $\mathrm{diag}(v v^T) = (1,1,1,1)$ and $\mathrm{antisym}(v v^T) = 0$, so all 8 rank-1 outer products span a 7-dimensional space containing all 6 off-diagonal permanent slices. Dropping $v_7 = (1,-1,-1,-1)$ and solving via `sympy.Matrix.gauss_jordan_solve` yielded exact rational coefficients in $\{-1/4, 0, 1/4\}$ (residual `0`).
- **CSE & Verification (`src/perm4_gen1_20muls.py`):**
  - Factored the 7 Walsh-Hadamard forms in 11 adds/subs per row and the $6 \times 7$ reconstruction in 14 adds/subs.
  - Verified symbolically in $\mathbb{Q}[A]$ (residual `0`) and 100/100 modular trials over $p_1, p_2$.
  - **Fitness:** `(True, -20, -98, -10)` — 20 non-scalar muls, 1 scalar mul (`inv16`), 77 adds/subs, 98 total gates, depth 10.

---

## Step 2 — Generation 2: Layered Bilinear Chain & Rank-1 Fusion (18 Muls at $n = 4$)
- **Hypothesis:** Split $4 \times 4$ as $3 \times 4$ plus row 3 (4 final muls). If the $6 \times 4 \times 4$ bilinear lift tensor $(P_{01}, A_2) \mapsto Q_{012}$ has rank $r \le 8$, total multiplications drop to $7 + r + 4 \le 19 < 20$.
- **Experiment:** Ran ALS CP-decomposition on the $6 \times 4 \times 4$ tensor:
  - Rank 6: residual `1.255e+00`.
  - **Rank 7:** residual `1.084e-11`, max scaled factor norm `1.078` — **exact rank 7**, yielding $7 + 7 + 4 = \mathbf{18}$ non-scalar multiplications.
- **Algebraic Lift & Fusion:**
  - Inspecting the numerical ALS factors revealed that both $B$ and $C$ were diagonally rescaled Walsh-Hadamard sign vectors ($C$ being the reversed sign vector of $B$).
  - Solving over $\mathbb{Q}$ via SymPy yielded exact residual `0` and revealed that composing Layer 1's reconstruction $C^T$ with Layer 2's input projection $A^T$ collapses to a single $7 \times 7$ rank-1 update:
    $$8 L = \mathrm{diag}(+1,-1,-1,+1,-1,+1,+1)\,(I_7 - \mathbf{1}\sigma^T)$$
    costing only 13 adds/subs and reducing the final scalar divisor from 16 to 8.
- **Debugging Note:** The first test run of `perm4_circuit_18` failed the symbolic check due to reversed indexing between the 4 reconstructed $3 \times 3$ permanents `q_012..q_123` and row 3 entries `r3[0..3]`. Reversing the pairing immediately brought the symbolic residual in $\mathbb{Q}[A]$ to `0`.
- **Verification (`src/perm4_gen2_18muls.py`):**
  - Symbolic $\mathbb{Q}[A]$ residual `0`; 100/100 modular checks over $p_1, p_2$.
  - **Fitness:** `(True, -18, -84, -15)` — 18 non-scalar muls (−25.0% vs. Glynn), 1 scalar mul (`inv8`), 65 adds/subs, 84 total gates, depth 15.

---

## Step 3 — Generalization: Sequential Dual Degree-Parity Compression (All $n \ge 3$)
- **Insight:** The fused rank-1 shift at $n = 4$ is a special case of a general Boolean hypercube theorem:
  - Prefix products of degree $d$ lie in a $K(n, d) = \sum_{w=0}^d \binom{n-1}{w}$-dimensional Walsh subspace (Prefix Degree Interpolation).
  - Remaining suffix products of degree $n - k$ are annihilated by any degree $\le k - 2$ shift (Suffix Degree-Parity Annihilation), zeroing $K(n, k-2)$ indices.
  - Pulling row $n - 1$ outside collapses the final layer to $n$ muls.
- **Debugging Note:** Initial prototype raised `TypeError` when `apply_linear_exact` encountered `None` (already-annihilated zero wires) in a subsequent annihilation layer; skipping `None` entries resolved it cleanly.
- **Verified Sequential Counts (50/50 over $p_1, p_2$):**
  - $n = 3$: **7** (Glynn 8, −12.5%)
  - $n = 4$: **18** (Glynn 24, −25.0%)
  - $n = 5$: **42** (Glynn 64, −34.4%)
  - $n = 6$: **90** (Glynn 160, −43.8%)
  - Asymptotic scaling: $\Theta(\sqrt{n} \cdot 2^n)$ non-scalar multiplications.

---

## Step 4 — Generation 3: Optimal Binary Evaluation Tree (`OptTree`, $O(2^n)$ Non-Scalar Muls)
- **Hypothesis:** Instead of multiplying one row at a time, build low-degree subtrees (`full[2]`, `full[3]`, `full[4]`) via Prefix Degree Interpolation and combine them in chunks via Annihilation and Parseval Walsh-Hadamard Inner Products, skipping dense middle layers.
- **DP Optimization & Exact Implementation (`src/general_n_opttree.py`):**
  - Added Fast Walsh-Hadamard Transform (`fwht_row`, `fwht_inverse`) for $O(n 2^n)$ row linear forms.
  - Discovered strictly lower counts starting at $n = 6$: **85** at $n = 6$ (vs. 90 sequential, 160 Glynn), **172** at $n = 7$ (vs. 192 sequential, 384 Glynn), **322** at $n = 8$ (vs. 896 Glynn), **1,176** at $n = 10$ (vs. 4,608 Glynn), and **4,081** at $n = 12$ (vs. 22,528 Glynn, −81.9%).
- **Full Suite Verification (`tests/run_all_verifications.py`):**
  - All Gen 1, Gen 2, Mode B lifting, and Gen 3 `OptTree` ($n = 3, 4, 5, 6, 7$) checks passed with 0 errors and exact op counts matching the DP forecasts.
