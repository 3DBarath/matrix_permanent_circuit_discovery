# Exact Matrix Permanent Circuit Discovery

Exact, branch-free arithmetic Straight-Line Programs (SLPs) for the $n \times n$ matrix permanent $\mathrm{Perm}(A)$ over any field $\mathbb{F}$ with $\mathrm{char}(\mathbb{F}) \neq 2$, achieving strictly fewer non-scalar multiplications than the Glynn + Gray code baseline $(n-1) \cdot 2^{n-1}$ for every $n \ge 3$.

## Headline Results

| $n$ | Glynn + Gray Code | Gen 1 (Block Laplace + Rank-7) | Gen 2 (Fused Quotient Chain) | Gen 3 (`OptTree` General-$n$) | Savings vs. Glynn | Verification |
|---:|---:|---:|---:|---:|---:|---|
| 3 | 8 | — | — | **7** | −12.5% | 50/50 modular ($p_1, p_2$) |
| 4 | 24 | 20 | **18** (84 gates) | **18** | −25.0% | Symbolic $\mathbb{Q}[A]$ + 100/100 modular |
| 5 | 64 | — | — | **42** | −34.4% | 50/50 modular ($p_1, p_2$) |
| 6 | 160 | — | — | **85** | −46.9% | 50/50 modular ($p_1, p_2$) |
| 7 | 384 | — | — | **172** | −55.2% | 20/20 modular ($p_1, p_2$) |
| 8 | 896 | — | — | **322** | −64.1% | Exact DP plan |
| 10 | 4,608 | — | — | **1,176** | −74.5% | Exact DP plan |
| 12 | 22,528 | — | — | **4,081** | −81.9% | Exact DP plan |

## Repository Layout

```text
matrix_permanent_circuit_discovery/
├── README.md                     # Quick-start guide and summary table
├── FINDINGS.md                   # Formal mathematical theorems, proofs, and benchmarks
├── RESEARCH_AND_STEP_LOG.md      # Chronological evolutionary discovery & verification log
├── src/
│   ├── __init__.py
│   ├── tracked_field.py          # TrackedFieldElement op accounting + Glynn reference
│   ├── perm4_gen1_20muls.py      # Gen 1: 20 non-scalar muls, 98 total gates, depth 10
│   ├── perm4_gen2_18muls.py      # Gen 2: 18 non-scalar muls, 84 total gates, depth 15
│   ├── general_n_opttree.py      # Gen 3: Optimal binary tree + Dual Degree-Parity (all n >= 3)
│   └── mode_b_rank_discovery.py  # Continuous ALS rank-7 probe + exact SymPy rational lifting
└── tests/
    └── run_all_verifications.py  # Anti-cheat verification harness (Schwartz-Zippel + Q[A])
```

## Quick Start

Requires Python 3.10+ with `numpy` and `sympy`:

```bash
python3 tests/run_all_verifications.py
```

The test harness enforces all anti-cheat gates:
1. **Formal Exactness:** Schwartz-Zippel verification over two distinct 61-bit primes ($p_1 = 2^{61}-1$, $p_2 = 2^{61}-31$) against a trusted modular Glynn reference, plus full symbolic polynomial identity verification in $\mathbb{Q}[A_{00}, \dots, A_{33}]$ for the $n = 4$ circuits.
2. **Exact Op Accounting:** Every wire operation passes through `TrackedFieldElement`, strictly distinguishing `wire * wire` (1 non-scalar multiplication) from `const * wire` (scalar multiplication).
3. **No Division by Inputs / No Branching:** All circuits are uniform straight-line programs dividing only by fixed powers of 2 ($2^{n-1}$ or $2^n$).
