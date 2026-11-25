# Breaking Glynn's Barrier: Exact $O(2^n)$-Multiplication Circuits for the Matrix Permanent via Degree-Preserving Dual-Ball Annihilation and Möbius–Zeta Butterflies

**Author:** Barath B (Independent Researcher, ORCID: [`0009-0008-8195-539X`](https://orcid.org/0009-0008-8195-539X))  
**Target Submission:** arXiv (`cs.CC` / `math.CO`) / STOC / FOCS / CCC  
**Source File:** [`paper/main.tex`](./main.tex)  
**Compiled PDF:** [`paper/main.pdf`](./main.pdf)

---

## Abstract

We present an algebraic complexity framework spanning ten generations that strictly breaks the 15-year-old Glynn barrier for the exact $n \times n$ matrix permanent over any field $\mathbb{F}$ with $\mathrm{char}(\mathbb{F}) \neq 2$. We reduce the non-scalar (wire $\times$ wire) multiplication complexity from Glynn's $\Theta(n \cdot 2^n)$ baseline to $O(2^n)$, while bounding total arithmetic gate complexity to $O(n \cdot 2^n)$ and beating Glynn + Gray code on **both** non-scalar multiplications and total arithmetic operations for all $n \ge 10$.

The construction is founded on six exact algebraic mechanisms (Mechanisms A–F2) on the Boolean hypercube $\{+1,-1\}^{n-1} \cong \{0,1\}^{n-1}$, replacing precomputed dense rational interpolation matrices with matrix-free Möbius–Zeta butterflies, Hasse-diagram 1-add-per-mask row evaluators, signed-wire tracking (1 scalar multiplication across the entire circuit), and SSA Common Subexpression Elimination (CSE):
- **Generation 9 (Mechanism F1):** 4,011 non-scalar multiplications ($-82.2\%$) and 42,945 total gates ($-13.1\%$ vs. Glynn + Gray's 49,404) at $n = 12$.
- **Generation 10 (Mechanism F2, Schur-Complement Butterfly):** Reaches the exact information-theoretic hypercube floor of 3,941 multiplications ($-82.5\%$) at $n = 12$ and 52,488 ($-89.3\%$, $1.602 \cdot 2^{15}$) at $n = 16$.
- **Hardware & Lower Bounds:** Compiled C++17 kernels over $\mathbb{F}_{2^{61}-1}[x]/(x^8 - 2)$ achieve up to $2.86\times$ wall-clock throughput speedups over Glynn + Gray code. We also establish the row-layered rank lower bound at $n=4$, prove Depth-3 $\Sigma\Pi\Sigma$ product rank $4$ and unconditional general-DAG $L_{\mathrm{mul}}(\mathrm{Perm}_3) = 7$, and provide an $O(n^r)$ solver for rank-$r$ matrices.

---

## Instructions for Uploading to arXiv / Overleaf

1. **Overleaf:**
   - Create a new blank project.
   - Upload `main.tex`.
   - Compile using `pdfLaTeX`, `XeLaTeX`, or `tectonic`.
2. **arXiv Submission:**
   - Upload `main.tex` directly to arXiv under subject categories:
     - Primary: **cs.CC** (Computational Complexity)
     - Secondary: **math.CO** (Combinatorics) / **cs.DS** (Data Structures and Algorithms)
