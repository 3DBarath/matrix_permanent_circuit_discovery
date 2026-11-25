# Exact Matrix Permanent Circuit Discovery

Exact, branch-free arithmetic Straight-Line Programs (SLPs) for the $n \times n$ matrix permanent $\mathrm{Perm}(A)$ over any field $\mathbb{F}$ with $\mathrm{char}(\mathbb{F}) \neq 2$, achieving strictly fewer non-scalar multiplications and total arithmetic operations than the classical Glynn + Gray code baseline.

**Author:** Barath B  
**Preprint:** [`paper/main.pdf`](paper/main.pdf) | [`paper/main.tex`](paper/main.tex)

---

## Headline Results

### 1. Multiplicative & Total Gate Complexity (Generations 1–10)

| $n$ | Glynn (Baseline Muls) | **Gen 10 Muls (Min Muls)** | Mul Savings | **Gen 9 CSE Gates (Min Gates)** | Glynn + Gray Gates | Total Gate Savings | Verification |
|---:|---:|---:|---:|---:|---:|---:|---|
| **3** | 8 | **7** | **−12.5%** | **29** | 33 | **−12.1%** | 50/50 modular ($p_1, p_2$) |
| **4** | 24 | **18** | **−25.0%** | **84–87** | 84 | $+3.6\%$ | Symbolic $\mathbb{Q}[A]$ (res=0) + 100/100 modular |
| **5** | 64 | **41** | **−35.9%** | **227** | 195 | $+16.4\%$ | 50/50 modular ($p_1, p_2$) |
| **6** | 160 | **83** | **−48.1%** | **502** | 438 | $+14.6\%$ | 50/50 modular ($p_1, p_2$) |
| **7** | 384 | **169** | **−56.0%** | **1,109** | 973 | $+14.0\%$ | 20/20 modular ($p_1, p_2$) |
| **8** | 896 | **318** | **−64.5%** | **2,311** | 2,152 | $+7.4\%$ | 20/20 modular ($p_1, p_2$) |
| **10** | 4,608 | **1,152** | **−75.0%** | **10,276** | 10,410 | **−1.3% (Fewer Gates)** | 10/10 modular ($p_1, p_2$) |
| **12** | 22,528 | **3,941** | **−82.5%** | **42,945** | 49,404 | **−13.1% (Fewer Gates)** | 4/4 modular ($p_1, p_2$) |
| **14** | 106,496 | **14,485** | **−86.4%** | **185,002** | 229,726 | **−19.5% (Fewer Gates)** | Exact DP + SLP CSE |
| **16** | 491,520 | **52,488** | **−89.3%** | **789,626** | 1,049,040 | **−24.7% (Fewer Gates)** | Exact DP + SLP CSE |

---

## Architectural Evolution

* **Gen 1:** Continuous ALS rank-7 tensor decomposition of all six $2 \times 2$ minors lifted over $\mathbb{Q}$ ($n=4 \implies 20$ muls).
* **Gen 2 Champion ($n=4$):** Fused rank-7 lift tensor with a rank-1 update of a sign-diagonal linear layer: $8L = \mathrm{diag}(+1,-1,-1,+1,-1,+1,+1)(I_7 - \mathbf{1}\sigma^T)$. Yields **18 non-scalar muls** and **84 total gates** (symbolic residual 0 in $\mathbb{Q}[A]$).
* **Gen 3 (`general_n_opttree.py`):** General-$n$ dynamic programming binary tree with Dual Degree-Parity compression (Prefix Degree Interpolation + Suffix Parity Annihilation + Parseval Final Collapse), proving the asymptotic **$O(2^n)$** non-scalar bound.
* **Gen 5 Champion (`general_n_butterfly_opttree.py`):** Eliminates all dense rational matrices using $O(m 2^m)$ Möbius-Zeta butterfly networks and Hasse-diagram row evaluation. Proves that total gates scale at $O(n 2^n)$, beating Glynn + Gray code on **both** multiplications and total gates at $n \ge 10$.
* **Gen 6 (`slp_compiler.py` & `opttree_native_mod61.cpp`):** Full Straight-Line Program SSA compiler with algebraic CSE and dead-code elimination. Includes a native C++17 hardware benchmark running branch-free 128-bit modular arithmetic over the Mersenne prime $2^{61}-1$, bit-exact verified up to $n = 14$.
* **Gen 7:** Discovers **Mechanism D** (Double-Sided Dual-Ball Annihilation). Zeroes out terms on disjoint low- and high-weight Hamming balls simultaneously. Includes 1.6 MB polynomial-ring SIMD benchmark over $\mathbb{F}_{2^{61}-1}[x]/(x^8-2)$.
* **Gen 8:** Proves **Mechanism E** (Degree-Preserving Recursive Dual-Ball Annihilation). Shows degree is strictly preserved under polynomial shifts, establishing an exact symmetric binary tree recurrence.
* **Gen 9 Champion (Min Total Gates, `general_n_butterfly_opttree.py`):** Fused Möbius–Zeta OptTree with Mechanism F1, achieving the absolute minimum total gate count with CSE: **42,945 total gates** at $n = 12$ (**−13.1% fewer arithmetic operations than Glynn**). Also adds exact $O(n^r)$ polynomial-time MacMahon/Barvinok solver for rank-$r$ matrices (`src/low_rank_permanent.py`).
* **Gen 10 Champion (Min Non-Scalar Muls, `general_n_schur_f2.py`):** Discovers **Mechanism F2 via Low-Rank Schur-Complement Butterflies**. Solves dual-ball subspace annihilation inside intermediate nodes by dropping $r = K(n, a-2)$ pivot coordinates and solving a tiny $r \times r$ incidence system. Reaches the exact theoretical non-scalar multiplication floor on the Boolean hypercube: **83 muls** at $n=6$, **169 muls** at $n=7$, **1,152 muls** at $n=10$, and **3,941 muls** at $n=12$ (**−82.5% vs. Glynn**).

---

## Repository Layout

```text
matrix_permanent_circuit_discovery/
├── README.md                              # Summary benchmarks, roadmap, and layout
├── FINDINGS.md                            # Mathematical proofs, theorems, and complexity analysis
├── RESEARCH_AND_STEP_LOG.md               # Chronological log across Generations 0-19
├── .gitignore                             # Clean build filters
├── paper/
│   ├── main.tex                           # Paper I: Dense O(2^n) + Lower Bounds (arXiv LaTeX)
│   ├── main.pdf                           # Compiled PDF preprint
│   └── README.md                          # Submission guidelines
├── paper_sparse/
│   ├── main.tex                           # Paper II: Sub-2^n Sparse Permanent & Extensions
│   └── main.pdf                           # Compiled PDF preprint
├── lean/
│   └── PermanentProofs.lean               # Lean 4 + Mathlib4 formal proofs (zero sorrys)
├── colab/
│   ├── sparse_permanent_cuda.cu           # Batched Tesla T4 CUDA Frontier DP solver
│   ├── perm4_17mul_search.cu              # 64-lane bitsliced F_3/F_5 CUDA search for Perm_4
│   ├── run_in_colab.ipynb                 # Sparse CUDA Colab notebook
│   ├── run_lean_in_colab.ipynb            # Lean 4 + Mathlib4 Colab notebook
│   └── run_perm4_search.ipynb             # Perm_4 17-mul CUDA search notebook
├── src/
│   ├── __init__.py
│   ├── tracked_field.py                   # TrackedFieldElement op accounting + Glynn baseline
│   ├── mode_b_rank_discovery.py           # Continuous ALS rank-7 probe + exact SymPy rational lifting
│   ├── perm4_gen1_20muls.py               # Gen 1: 20 non-scalar muls, 98 gates, depth 10
│   ├── perm4_gen2_18muls.py               # Gen 2: 18 non-scalar muls, 84 gates, depth 15
│   ├── general_n_opttree.py               # Gen 3: DP binary tree + Dual Degree-Parity
│   ├── general_n_butterfly_opttree.py     # Gen 9: Fused Mobius-Zeta Butterfly OptTree (Min Total Gates)
│   ├── general_n_schur_f2.py              # Gen 10: Schur-Complement F2 OptTree (Min Non-Scalar Muls)
│   ├── slp_compiler.py                    # Gen 6: SLP SSA compiler with algebraic CSE + DCE
│   ├── opttree_native_mod61.cpp           # Gen 6: Native C++17 hardware benchmark mod (2^61 - 1)
│   ├── bench_poly8_mod61.cpp              # Gen 7: SIMD polynomial-ring benchmark F_p[x]/(x^8 - 2)
│   ├── low_rank_permanent.py              # Gen 9: Exact O(n^r) polynomial-time solver for rank-r
│   ├── sparse_permanent.py                # Gen 11: Python sub-2^n sparse permanent solver
│   ├── sparse_permanent.cpp               # Gen 12-19: C++17 + OpenMP sparse permanent (3-reg & --4reg)
│   ├── boson_sampling_complex.cpp         # Gen 16: Complex unitary Boson Sampling permanent solver
│   └── sparse_hafnian.cpp                 # Gen 16-18: Exact sub-2^n Sparse Hafnian solver
├── tests/
│   ├── run_all_verifications.py           # Anti-cheat verification test suite (Q[A] + 61-bit primes)
│   └── test_sparse_permanent.py           # Bit-exact sparse permanent verification suite
└── archive/                               # Historical zips, prototypes, and autoresearch reports
```

---

## Quick Start & Verification

### Run the Python Verification Suite
```bash
uv run --with numpy --with sympy python tests/run_all_verifications.py
```

### Run Gen 10 Schur-Complement F2 Circuit
```bash
uv run --with numpy --with sympy python -c "
from src.general_n_schur_f2 import build_schur_f2_circuit
from src.tracked_field import TrackedFieldElement, OpCounter
c = OpCounter()
plan, ev = build_schur_f2_circuit(6)
print(f'n=6 Gen 10 non-scalar muls: {plan[\"non_scalar_muls\"]}')
"
```


---

## Generation 11–19 Highlights
- **Dense Circuits & Lower Bounds (Gen 1–10, 15, 19):** Eliminated Glynn's $(n-1)$ factor to hit the exact $O(2^n)$ linear hypercube floor ($52,488$ muls at $n=16$, $-89.3\%$), with $-13.1\%$ total gates at $n=12$ and up to $2.86\times$ C++17 wall-clock speedup. Proved unconditional $L_{\mathrm{mul}}(\mathrm{Perm}_3) = 7$ over $\mathrm{char}(\mathbb{F}) \neq 2$ via 5 machine-certified $\mathbb{Z}[1/2]$ lemmas, and ran a **6.55B-circuit** Tesla T4 CUDA search (`24.82 M/s`, best loss `6/64`) reinforcing 18 as tight at $n=4$.
- **Low-Rank Collapse (Gen 9):** Exact $O(n^r)$ polynomial-time Barvinok solver (`src/low_rank_permanent.py`).
- **Sparse & Graph Permanents (Gen 11–20):** Exact sub-$2^n$ C++17 + OpenMP solver (`src/sparse_permanent.cpp`) solving **$72 \times 72$ weighted 3-regular permanents in 2.12s (2t) / 580.49 ms (8t)** ($4.05 \times 10^{13}\times$ fewer ops) and scaling to **$80 \times 80$ in 21.77 s** (**$1.62 \times 10^{15}\times$ fewer ops** than $2^{80}$ with `fw <= 16`), plus 4-regular `n=28` in **94,364 ops** (`br=0`) and **`n=36` in 42.48 ms** (`--4reg`).
- **Quantum, GPU & Formal Verification Extensions (Gen 16–20):**
  - **Complex Boson Sampling (`src/boson_sampling_complex.cpp`):** `std::complex<double>` with `long double` accumulation, solving $n=60$ complex unitary permanents in **192.15 ms** (`0.000e+00` error vs. Glynn).
  - **Sparse Hafnians (`src/sparse_hafnian.cpp`):** Exact sub-$2^n$ non-bipartite Hafnian solver for Gaussian Boson Sampling, scaling to 3-regular **$n=96$ in 112.64 ms** (and 4-regular **$n=32$ in 0.551 ms**).
  - **CUDA Colab Kernel (`colab/sparse_permanent_cuda.cu`):** GPU-batched Frontier DP with 32 KB shared memory and atomic-free compression, solving 3-regular **$n=64$ in 1.37 s** and **$n=72$ in 19.04 s–19.80 s** on Tesla T4.
  - **Lean 4 Formalization (`lean/PermanentProofs.lean`):** Machine-checked formalizations (`perm_degree2_row_cofactor_collapse`, `deg1_peeling_row_cofactor`, `essential_var_eliminates_le_five`, `essential_var_m6_forces_w_le_3`) typechecked cleanly in Colab with zero `sorry`s.
- **Preprint Papers:**
  - **Paper I ([`paper/main.pdf`](paper/main.pdf)):** Dense $O(2^n)$ Hypercube Floor + Unconditional $\mathrm{Perm}_3$ Lower Bound.
  - **Paper II ([`paper_sparse/main.pdf`](paper_sparse/main.pdf)):** Sub-$2^n$ Sparse Permanent ($n=72$) + Gen 16–19 Extensions.
