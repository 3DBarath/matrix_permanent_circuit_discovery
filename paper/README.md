# Breaking Glynn's Barrier: An O(2^n) Multiplicative Circuit for the Matrix Permanent via Dual Degree-Parity Compression

**Author:** Barath B (Independent Researcher)  
**Target Submission:** arXiv (math.CO / cs.CC) / STOC / FOCS / CCC  
**Source File:** [`paper/main.tex`](./main.tex)

---

## Abstract

Computing the permanent of an $n \times n$ matrix is a foundational $\#\mathbf{P}$-complete problem. Since Glynn's 2010 formula, the state-of-the-art non-scalar multiplicative complexity over fields with $\mathrm{char}(\mathbb{F}) \neq 2$ has stood at $(n-1) \cdot 2^{n-1} = \Theta(n \cdot 2^n)$ multiplications. In this work, we present an exact, branch-free arithmetic Straight-Line Program (SLP) that strictly breaks Glynn's barrier for all $n \ge 3$, asymptotically reducing the non-scalar multiplicative complexity to $O(2^n)$.

Our construction relies on an algebraic framework termed **Dual Degree-Parity Compression**, which unites three exact reductions on the Boolean hypercube $\{\pm 1\}^{n-1}$:
1. **Prefix Degree Interpolation:** Confines degree-$m$ prefixes to a $K(n, m) = \sum_{w=0}^m \binom{n-1}{w}$ dimensional Walsh-Hadamard subspace, skipping non-scalar multiplications via rational interpolation.
2. **Suffix Degree-Parity Annihilation:** Shifts intermediate state by low-degree polynomials orthogonal to the parity character, identically zeroing pivot coordinates.
3. **Parseval Collapse:** Evaluates the boundary layer in only $n$ multiplications.

For $n = 4$, by solving the bilinear lift tensor via numerical Alternating Least Squares (ALS) and lifting the factors over $\mathbb{Q}$, we uncover a fused rank-1 perturbation of a sign-diagonal matrix that evaluates the permanent in exactly **18 non-scalar multiplications** ($-25.0\%$ relative to Glynn's 24) and **84 total arithmetic gates**. We verify exact polynomial identity over $\mathbb{Q}[A_{00},\dots,A_{33}]$ (residual zero) and validate dimensions $n = 3, \dots, 7$ across two 61-bit primes with zero errors. By $n = 12$, the dynamic-programming evaluation tree requires only $4{,}081$ non-scalar multiplications ($< 2^n = 4{,}096$, an $81.9\%$ reduction over Glynn's $22{,}528$).

---

## Instructions for Uploading to arXiv / Overleaf

1. **Overleaf:**
   - Create a new blank project.
   - Upload `main.tex`.
   - Compile using standard `pdfLaTeX` or `XeLaTeX`.
2. **arXiv Submission:**
   - Upload `main.tex` directly to arXiv under subject categories:
     - Primary: **cs.CC** (Computational Complexity)
     - Secondary: **math.CO** (Combinatorics) / **cs.DS** (Data Structures and Algorithms)
