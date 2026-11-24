# Exact Matrix Permanent Circuit Discovery

Exact, branch-free arithmetic Straight-Line Programs (SLPs) for the $n \times n$ matrix permanent $\mathrm{Perm}(A)$ over any field $\mathbb{F}$ with $\mathrm{char}(\mathbb{F}) \neq 2$, achieving strictly fewer non-scalar multiplications and total arithmetic operations than the classical Glynn + Gray code baseline.

**Author:** Barath B  
**Preprint:** [`paper/main.pdf`](paper/main.pdf) | [`paper/main.tex`](paper/main.tex)

---

## Headline Results

### 1. Multiplicative & Total Gate Complexity (Generations 1–6)

| $n$ | Glynn (Baseline Muls) | **Our Multiplications (Gen 7)** | Mul Savings | **Our Total Gates (Gen 7)** | Glynn + Gray Gates | Gate Advantage | Verification |
|---:|---:|---:|---:|---:|---:|---:|---|
| **3** | 8 | **7** | **−12.5%** | **29** | 33 | **−12.1%** | 50/50 modular ($p_1, p_2$) |
| **4** | 24 | **18** | **−25.0%** | **84–87** | 84 | $+3.6\%$ | Symbolic $\mathbb{Q}[A]$ (res=0) + 100/100 modular |
| **5** | 64 | **41** | **−35.9%** | **227** | 195 | $+16.4\%$ | 50/50 modular ($p_1, p_2$) |
| **6** | 160 | **84** | **−47.5%** | **497** | 438 | $+13.5\%$ | 50/50 modular ($p_1, p_2$) |
| **7** | 384 | **171** | **−55.5%** | **1,120** | 973 | $+15.1\%$ | 20/20 modular ($p_1, p_2$) |
| **8** | 896 | **321** | **−64.2%** | **2,312** | 2,152 | $+7.4\%$ | 20/20 modular ($p_1, p_2$) |
| **10** | 4,608 | **1,166** | **−74.7%** | **10,115** | **10,410** | **Strictly Fewer Total Gates (−2.8%)** | 10/10 modular ($p_1, p_2$) |
| **12** | 22,528 | **4,014** | **−82.2%** | **45,265** | **49,404** | **Strictly Fewer Total Gates (−8.4%)** | 4/4 modular ($p_1, p_2$) |

---

## Generations & Architectural Evolution

* **Gen 1:** Continuous ALS rank-7 tensor decomposition of all six $2 \times 2$ minors lifted over $\mathbb{Q}$ ($n=4 \implies 20$ muls).
* **Gen 2 Champion ($n=4$):** Fused rank-7 lift tensor with a rank-1 update of a sign-diagonal linear layer: $8L = \mathrm{diag}(+1,-1,-1,+1,-1,+1,+1)(I_7 - \mathbf{1}\sigma^T)$. Yields **18 non-scalar muls** and **84 total gates** (symbolic residual 0 in $\mathbb{Q}[A]$).
* **Gen 3 (`general_n_opttree.py`):** General-$n$ dynamic programming binary tree with Dual Degree-Parity compression (Prefix Degree Interpolation + Suffix Parity Annihilation + Parseval Final Collapse), proving the asymptotic **$O(2^n)$** non-scalar bound.
* **Gen 5 Champion (`general_n_butterfly_opttree.py`):** Eliminates all dense rational matrices using $O(m 2^m)$ Möbius-Zeta butterfly networks and Hasse-diagram row evaluation. Proves that total gates scale at $O(n 2^n)$, beating Glynn + Gray code on **both** multiplications and total gates at $n \ge 10$.
* **Gen 6 (`slp_compiler.py` & `opttree_native_mod61.cpp`):** Full Straight-Line Program SSA compiler with algebraic CSE and dead-code elimination. Includes a native C++17 hardware benchmark running branch-free 128-bit modular arithmetic over the Mersenne prime $2^{61}-1$, bit-exact verified up to $n = 14$.
* **Gen 7 Champion (`general_n_butterfly_opttree.py` & `bench_poly8_mod61.cpp`):** Discovers **Mechanism D** (Double-Sided Dual-Ball Annihilation). By subtracting degree-bounded polynomials from *both* subtrees simultaneously on disjoint low- and high-weight Hamming balls, all cross-terms vanish against the parity character. Strictly breaks prior records across dimensions (e.g. $n=5 \implies 41$ muls; $n=10 \implies 1{,}166$ muls and $-2.8\%$ gates; $n=12 \implies 4{,}014$ muls and $-8.4\%$ gates). Includes 1.6 MB polynomial-ring SIMD benchmark over $\mathbb{F}_{2^{61}-1}[x]/(x^8-2)$.

---

## Repository Layout

```text
matrix_permanent_circuit_discovery/
├── README.md                              # Summary benchmarks, roadmap, and layout
├── FINDINGS.md                            # Mathematical proofs, theorems, and complexity analysis
├── RESEARCH_AND_STEP_LOG.md               # Chronological log across Generations 0-6
├── .gitignore                             # Clean build filters
├── paper/
│   ├── main.tex                           # Full arXiv / conference LaTeX manuscript
│   ├── main.pdf                           # Compiled PDF preprint (Tectonic XeTeX)
│   └── README.md                          # Submission guidelines
├── src/
│   ├── __init__.py
│   ├── tracked_field.py                   # TrackedFieldElement op accounting + Glynn baseline
│   ├── mode_b_rank_discovery.py           # Continuous ALS rank-7 probe + exact SymPy rational lifting
│   ├── perm4_gen1_20muls.py               # Gen 1: 20 non-scalar muls, 98 gates, depth 10
│   ├── perm4_gen2_18muls.py               # Gen 2: 18 non-scalar muls, 84 gates, depth 15
│   ├── general_n_opttree.py               # Gen 3: DP binary tree + Dual Degree-Parity
│   ├── general_n_butterfly_opttree.py     # Gen 5: Fused Mobius-Zeta Butterfly OptTree (O(n 2^n) gates)
│   ├── slp_compiler.py                    # Gen 6: SLP SSA compiler with algebraic CSE + DCE
│   └── opttree_native_mod61.cpp           # Gen 6: Native C++17 hardware benchmark mod (2^61 - 1)
└── tests/
    └── run_all_verifications.py           # Anti-cheat verification test suite (Q[A] + 61-bit primes)
```

---

## Quick Start & Verification

### Run the Python Verification Suite
```bash
uv run --with numpy --with sympy python tests/run_all_verifications.py
```

### Run the Native C++ Hardware Benchmark
```bash
g++ -O3 -march=native src/opttree_native_mod61.cpp -o run_native_bench
./run_native_bench
```
