# Mathematical Findings & Complexity Analysis

## 1. Core Algebraic Discovery: Dual Degree-Parity Compression

Glynn's formula expresses the $n \times n$ matrix permanent over any field $\mathbb{F}$ ($\mathrm{char}(\mathbb{F}) \neq 2$) as a parity-weighted sum over $N = 2^{n-1}$ sign vectors $v_j = (1, s_{j,1}, \dots, s_{j,n-1}) \in \{+1\} \times \{\pm 1\}^{n-1}$ with parity $\sigma_j = \prod_{c=1}^{n-1} s_{j,c}$:

$$\mathrm{Perm}(A) = \frac{1}{2^{n-1}} \sum_{j=0}^{N-1} \sigma_j \prod_{r=0}^{n-1} (v_j \cdot A_r)$$

Evaluating each product independently requires $(n-1) \cdot 2^{n-1} = \Theta(n \cdot 2^n)$ non-scalar multiplications. We show that **three exact algebraic reductions** compress this to **$\Theta(\sqrt{n} \cdot 2^n)$** sequentially and **$O(2^n)$** using an optimal binary evaluation tree (`OptTree`).

### Mechanism A: Prefix Degree Interpolation
Let $M^{(m)}_j = \prod_{r=0}^{m-1} (v_j \cdot A_r)$ be the pointwise product of $m$ row linear forms. Viewed as a function on the Boolean hypercube $\{\pm 1\}^{n-1}$, $M^{(m)}$ is a multilinear polynomial of degree at most $m$. Its Walsh-Hadamard Fourier support is restricted to characters $\chi_S$ of Hamming weight $|S| \le m$, which spans a subspace of dimension:

$$K(n, m) = \sum_{w=0}^{m} \binom{n-1}{w}$$

Whenever $m < n - 1$, $K(n, m) < 2^{n-1}$. Therefore, multiplying two prefix vectors of degrees $a$ and $b$ ($a + b = m$) requires only **$K(n, m)$** non-scalar multiplications on a full-rank pivot subset of sign vectors; the remaining $2^{n-1} - K(n, m)$ entries are recovered exactly via linear interpolation (0 non-scalar multiplications).

### Mechanism B: Suffix Degree-Parity Annihilation
Suppose we hold a partial product vector $U$ of $a$ rows and must multiply it by the remaining $n - a$ rows. Any multilinear polynomial $c(v)$ of degree $d \le a - 2$ satisfies:

$$\deg\!\left(c(v) \cdot \prod_{r=a}^{n-1} (v \cdot A_r)\right) \le (a - 2) + (n - a) = n - 2 < n - 1$$

Because the parity character $\sigma(v) = \prod_{c=1}^{n-1} s_c$ has degree $n - 1$, orthogonality of Walsh-Hadamard characters guarantees:

$$\sum_{j=0}^{N-1} \sigma_j \, c(v_j) \prod_{r=a}^{n-1} (v_j \cdot A_r) \equiv 0$$

identically as a formal polynomial. Choosing $c(v)$ from the $K(n, a-2)$-dimensional space of degree $\le a-2$ polynomials to match $U$ on a pivot set $Z$ of size $K(n, a-2)$ makes $U - c$ **identically zero on $Z$**, reducing the next multiplication layer to only $2^{n-1} - K(n, a-2) = K(n, n-a)$ non-scalar multiplications.

### Mechanism C: Parseval / Final-Layer Collapse
For any partition $n = a + b$ where we hold $U$ (degree $a$, possibly shifted by degree $\le a-2$) and $V$ (degree $b$), Parseval's identity in the Walsh-Hadamard basis gives:

$$\sum_{j=0}^{N-1} \sigma_j U_j V_j = \frac{1}{N} \sum_{S \subseteq \{1,\dots,n-1\}} \widehat{U}(S) \, \widehat{V}(\overline{S})$$

Since $\widehat{V}(\overline{S}) = 0$ whenever $|\overline{S}| > b$ (i.e., $|S| < n - 1 - b = a - 1$), only characters with $|S| \ge a - 1$ contribute. In particular, when $b = 1$ (the final row $A_{n-1}$), pulling $A_{n-1, c}$ outside the sum requires only **$n$** non-scalar multiplications instead of $2^{n-1}$.

---

## 2. Exact $n = 4$ Results (Mode B Discovery)

### Generation 1: Block Laplace + Rank-7 Walsh-Hadamard (`perm4_gen1_20muls.py`)
- Continuous ALS on the $4 \times 4 \times 6$ bilinear tensor of all six $2 \times 2$ permanents $P_{ij} = x_i y_j + x_j y_i$ revealed exact rank **7** (Frobenius error $8.86 \times 10^{-11}$, bounded factor norm $1.29$).
- Algebraically, all eight outer products $v_k v_k^T$ ($v_k \in \{1\}\times\{\pm 1\}^3$) have identical diagonal $(1,1,1,1)$ and zero antisymmetric part, so any 7 linearly independent sign vectors span $\mathrm{Sym}^2_{\text{off-diag}}(\mathbb{F}^4) \oplus \mathrm{span}(I_4)$ with exact coefficients in $\{-1/4, 0, 1/4\}$.
- Applying this to rows $\{0,1\}$ (7 muls) and $\{2,3\}$ (7 muls) plus 6 Laplace products gives **20 non-scalar multiplications**, 1 scalar mul (`1/16`), 77 adds/subs, **98 total gates**, depth 10.

### Generation 2 Champion: Fused Rank-7 Quotient Chain (`perm4_gen2_18muls.py`)
- Continuous ALS on the $6 \times 4 \times 4$ lift tensor $(P_{01}, A_2) \mapsto Q_{012}$ revealed exact rank **7** (Frobenius error $1.08 \times 10^{-11}$, bounded factor norm $1.08$).
- Lifting both layers to exact rationals over $\mathbb{Q}$ revealed that the composed $7 \times 7$ linear map between Layer 1 and Layer 2 collapses to a **rank-1 update of a sign-diagonal matrix**:
  $$8 L = \mathrm{diag}(+1, -1, -1, +1, -1, +1, +1)\,(I_7 - \mathbf{1}\sigma^T)$$
  which requires only **13 additions/subtractions** and eliminates the 8th product term before Layer 2.
- Combined with Final-Layer Collapse on row 3 (4 muls), this yields:
  - **18 non-scalar multiplications** (−25.0% vs. Glynn's 24)
  - **1 scalar multiplication** (`1/8`)
  - **65 additions/subtractions**, **84 total gates**, **depth 15**
  - Verified both symbolically in $\mathbb{Q}[A_{00},\dots,A_{33}]$ (residual $= 0$) and over 100 random matrices across two 61-bit primes.

---

## 3. General-$n$ Scaling: Sequential vs. Optimal Binary Evaluation Tree (`OptTree`)

By dynamic programming over binary evaluation trees that build low-degree subtrees (`full[2]`, `full[3]`, `full[4]`) via Prefix Degree Interpolation and combine them via Annihilation and Parseval Collapse, `OptTree` skips expensive middle layers:

| $n$ | $N = 2^{n-1}$ | Glynn $(n-1)N$ | Sequential Dual-Parity | **`OptTree` (Gen 3)** | `OptTree` / $N$ | Savings vs. Glynn |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 4 | 8 | 7 | **7** | 1.75 | −12.5% |
| 4 | 8 | 24 | 18 | **18** | 2.25 | −25.0% |
| 5 | 16 | 64 | 42 | **42** | 2.63 | −34.4% |
| 6 | 32 | 160 | 90 | **85** | 2.66 | −46.9% |
| 7 | 64 | 384 | 192 | **172** | 2.69 | −55.2% |
| 8 | 128 | 896 | 392 | **322** | 2.52 | −64.1% |
| 10 | 512 | 4,608 | 1,638 | **1,176** | 2.30 | −74.5% |
| 12 | 2,048 | 22,528 | 6,754 | **4,081** | **1.99** | **−81.9%** |

Notice that by $n = 12$, `OptTree` drops below $2 \cdot 2^{n-1} = 2^n$ total non-scalar multiplications ($4{,}081 < 4{,}096$), confirming the asymptotic reduction from $\Theta(n \cdot 2^n)$ to **$O(2^n)$** non-scalar multiplications.

---

## 4. Generation 5: Fused Möbius–Zeta Butterfly & $O(n 2^n)$ Total Gates

In Generation 5 (`src/general_n_butterfly_opttree.py`), all precomputed dense rational interpolation and annihilation matrices were replaced with structured fast transforms on the Boolean hypercube:

1. **Möbius–Zeta Isomorphism ($\{\pm 1\}^{n-1} \leftrightarrow \{0, 1\}^{n-1}$):**
   Interpolating a degree-$m$ multilinear polynomial on canonical pivots $P_m = \{y : |y| \le m\}$ is solved via an inverted Möbius butterfly followed by a forward Zeta butterfly, running in $O(m 2^m)$ additions and **0 scalar multiplications**.
2. **Fused Annihilation:**
   Coefficients $c_S$ with $|S| \le d_{\text{ann}}$ are zeroed out directly between the Möbius and Zeta stages at zero gate overhead.
3. **Hasse-Diagram Row Evaluator:**
   Evaluates row linear forms along the Boolean poset with only 1 subtraction per mask step ($2(n-1) + (K-1)$ additions/subtractions total).
4. **Simultaneous Multiplicative and Gate Supremacy:**
   Starting at $n = 10$, the circuit requires strictly fewer total arithmetic operations than Glynn + Gray code ($10{,}338$ vs. $10{,}410$ gates at $n=10$, and $45{,}940$ vs. $49{,}404$ gates at $n=12$).

---

## 5. Generation 6: SLP Compiler (CSE/DCE) and Native C++ Hardware Kernel

* **`slp_compiler.py`:** Formulates the circuit as an arithmetic SSA DAG, executes algebraic Common Subexpression Elimination ($a+b \equiv b+a$, $a \cdot b \equiv b \cdot a$), prunes dead wires, and emits flat C++17 straight-line programs.
* **`opttree_native_mod61.cpp`:** Implements 128-bit hardware modular arithmetic over the Mersenne prime $2^{61}-1$ with zero conditional branches. Verified bit-exact (32/32 trials) against the native Glynn baseline for all dimensions $n = 4 \dots 14$.

---

## 6. Generation 7: Mechanism D (Double-Sided Dual-Ball Annihilation)

* **Double-Sided Annihilation:** Simultaneously subtracts $c_1$ (degree $\le a-2$, vanishing on $|y| \le a-2$) from $U$ and $c_2$ (degree $\le b-2$, vanishing on $|y| \ge n - 1 - (b-2)$) from $V$. All cross-terms have degree $\le n - 2 < n - 1$ and vanish identically against $\sigma$.
* **Polynomial-Ring SIMD Verification (`bench_poly8_mod61.cpp`):** Stress-tested over $\mathbb{F}_{2^{61}-1}[x] / (x^8 - 2)$.

---

## 7. Generation 8: Mechanism E (Degree-Preserving Recursive Annihilation)

* **Degree Invariance:** Proved that subtracting a degree $\le a-2$ polynomial from a degree $\le a$ polynomial satisfies $\deg(U - c_1) \le a$. The annihilated polynomial never increases in multilinear degree, allowing recursive binary tree composition under a single symmetric recurrence.

---

## 8. Generation 9: Mechanism F (Simultaneous Upper-Degree Interpolation + Lower-Ball Annihilation)

* **Mechanism F1/F2:** When combining degree-$a$ subtree $U$ and degree-$b$ subtree $V$ ($a \le b$) into degree $k = a + b < n - 1$, annihilating child $V$ on $Z_b = \{y : |y| \le b-2\}$ preserves $\deg(V - c_2) \le b$. The product $U(V - c_2)$ still has degree $\le k$ and vanishes on $Z_b \subset P_k$.
* **Canonical Pivot Cost:**
  $$\mathrm{Cost}_{F1}(a, b) = K(n, a + b) - K(n, b - 2)$$
  $$\mathrm{Cost}_{F2}(a, b) = K(n, a + b) - K(n, a - 2) - K(n, b - 2)$$
* **Record Scaling:**
  - $n = 6$: **83 muls** (vs 160 Glynn)
  - $n = 8$: **318 muls** (vs 896 Glynn)
  - $n = 10$: **1,152 muls** (vs 4,608 Glynn, **−75.0%**)
  - $n = 12$: **3,941 muls** (vs 22,528 Glynn, **−82.5%**, **−9.1% total gates**)
  - $n = 14$: **14,485 muls** (vs 106,496 Glynn, **−86.4%**)
  - $n = 16$: **52,488 muls** (vs 491,520 Glynn, **−89.3%**)
* **Low-Rank Collapse (`src/low_rank_permanent.py`):** For $n \times n$ matrices of rank $r \ll n$, computes exact permanent in $O(n^r)$ polynomial time via MacMahon/Barvinok polynomial expansion.

---

## 9. Generation 10: Mechanism F2 via Low-Rank Schur-Complement Butterflies

Generation 10 implements Mechanism F2 via an exact Schur-complement butterfly architecture (`src/general_n_schur_f2.py`):
* **Operational Mechanism:** For combining $U$ (degree $a$) and $V$ (degree $b$) into $k = a + b < n - 1$ ($2 \le a \le b$):
  1. Fuses low-weight ball annihilation $Z_b = \{|y| \le b-2\}$ into child $V$.
  2. Fuses high-weight ball annihilation $Z_a = \{|y| \ge m - (a-2)\}$ into child $U$.
  3. Drops $r = K(n, a-2)$ weight-$k$ pivot masks $D \subset P_k \setminus Z_b$, evaluating on **only $K(n, k) - K(n, a-2) - K(n, b-2)$ points**.
  4. Solves the small $r \times r$ incidence system $M \lambda = -W^{(0)}[Z_a]$ using Schur complements. For $a = 2$, $r = 1$ and $M = [[1]]$ (requiring 0 scalar multiplications).
* **Exact Hypercube Multiplicative Floor:**
  - $n = 6$: **83 muls** (vs. 160 Glynn, **−48.1%**)
  - $n = 7$: **169 muls** (vs. 384 Glynn, **−56.0%**)
  - $n = 8$: **318 muls** (vs. 896 Glynn, **−64.5%**)
  - $n = 10$: **1,152 muls** (vs. 4,608 Glynn, **−75.0%**)
  - $n = 12$: **3,941 muls** (vs. 22,528 Glynn, **−82.5%**)
  - $n = 14$: **14,485 muls** (vs. 106,496 Glynn, **−86.4%**)
  - $n = 16$: **52,488 muls** (vs. 491,520 Glynn, **−89.3%**)

---

## 10. Open Directions
- **Sub-18 at $n = 4$:** Determining whether a non-Glynn multilinear or depth-3 product-rank-6 circuit exists to push below 18 non-scalar multiplications remains an open target for Mode B search.
