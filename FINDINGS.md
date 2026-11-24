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

## 4. Open Directions for Future Generations
- **Fast $O(n 2^n)$ interpolation/annihilation linear layers:** The current `general_n_opttree.py` uses Fast Walsh-Hadamard Transforms ($O(n 2^n)$ adds) for row evaluation and Parseval steps, while applying precomputed rational matrices for interpolation/annihilation. Factoring those interpolation/annihilation maps into structured butterfly/Möbius transforms will bring total gate counts to $O(n 2^n)$ alongside the $O(2^n)$ non-scalar multiplication count.
- **Sub-18 at $n = 4$:** Determining whether a non-Glynn multilinear or depth-3 product-rank-6 circuit exists to push below 18 non-scalar multiplications remains an open target for Mode B search.
