"""
Exact Verification Harness for Matrix Permanent Arithmetic DAGs.
Uses TrackedFieldElement / OpCounter accounting over dual 61-bit primes
(PRIME_61_A = 2^61 - 1, PRIME_61_B = 2^61 - 31) and SymPy Q[A] symbolic checks.
"""

from __future__ import annotations

from fractions import Fraction
import itertools
import random
import numpy as np
import sympy as sp

PRIME_61_A = (1 << 61) - 1
PRIME_61_B = (1 << 61) - 31


class OpCounter:
    def __init__(self) -> None:
        self.non_scalar_muls = 0
        self.scalar_muls = 0
        self.adds_subs = 0

    @property
    def total_gates(self) -> int:
        return self.non_scalar_muls + self.scalar_muls + self.adds_subs

    def as_dict(self) -> dict[str, int]:
        return {
            "non_scalar_muls": self.non_scalar_muls,
            "scalar_muls": self.scalar_muls,
            "adds_subs": self.adds_subs,
            "total_gates": self.total_gates,
        }


class TrackedFieldElement:
    __slots__ = ("val", "p", "counter", "depth")

    def __init__(self, val: int, p: int, counter: OpCounter, depth: int = 0) -> None:
        self.val = int(val) % p
        self.p = p
        self.counter = counter
        self.depth = depth

    def __add__(self, other: TrackedFieldElement) -> TrackedFieldElement:
        if isinstance(other, TrackedFieldElement):
            self.counter.adds_subs += 1
            return TrackedFieldElement(
                self.val + other.val, self.p, self.counter, max(self.depth, other.depth) + 1
            )
        raise TypeError("Wire addition requires another TrackedFieldElement.")

    def __sub__(self, other: TrackedFieldElement) -> TrackedFieldElement:
        if isinstance(other, TrackedFieldElement):
            self.counter.adds_subs += 1
            return TrackedFieldElement(
                self.val - other.val, self.p, self.counter, max(self.depth, other.depth) + 1
            )
        raise TypeError("Wire subtraction requires another TrackedFieldElement.")

    def __mul__(self, other: TrackedFieldElement | int) -> TrackedFieldElement:
        if isinstance(other, TrackedFieldElement):
            self.counter.non_scalar_muls += 1
            return TrackedFieldElement(
                self.val * other.val, self.p, self.counter, max(self.depth, other.depth) + 1
            )
        if isinstance(other, int):
            self.counter.scalar_muls += 1
            return TrackedFieldElement(self.val * other, self.p, self.counter, self.depth + 1)
        raise TypeError(f"Unsupported operand type: {type(other)}")

    def __rmul__(self, other: int) -> TrackedFieldElement:
        if isinstance(other, int):
            self.counter.scalar_muls += 1
            return TrackedFieldElement(other * self.val, self.p, self.counter, self.depth + 1)
        raise TypeError(f"Unsupported operand type: {type(other)}")


def glynn_perm_mod(A_int: list[list[int]], p: int) -> int:
    n = len(A_int)
    N = 1 << (n - 1)
    total = 0
    for mask in range(N):
        sign = -1 if (bin(mask).count("1") & 1) else 1
        prod = 1
        for r in range(n):
            row = A_int[r]
            s = row[0]
            for c in range(1, n):
                if (mask >> (c - 1)) & 1:
                    s -= row[c]
                else:
                    s += row[c]
            prod = (prod * (s % p)) % p
        total = (total + sign * prod) % p
    return (total * pow(N, -1, p)) % p


def _scale_tracked(wire: TrackedFieldElement, rat: Fraction, p: int) -> TrackedFieldElement | None:
    if rat == 0:
        return None
    if rat == 1:
        return wire
    coeff = (rat.numerator * pow(rat.denominator, -1, p)) % p
    return coeff * wire


def eval_rational_dag_tracked(
    A_wire: list[list[TrackedFieldElement]],
    L_x: list[list[Fraction]],
    L_g: list[list[Fraction]],
    R_x: list[list[Fraction]],
    R_g: list[list[Fraction]],
    w_out: list[Fraction],
    p: int,
) -> TrackedFieldElement:
    n = len(A_wire)
    x = [A_wire[r][c] for r in range(n) for c in range(n)]
    M = len(w_out)
    g: list[TrackedFieldElement] = []

    def lincomb(coeffs_x, coeffs_g):
        acc: TrackedFieldElement | None = None
        for i, c in enumerate(coeffs_x):
            term = _scale_tracked(x[i], c, p)
            if term is not None:
                acc = term if acc is None else (acc + term)
        for j, c in enumerate(coeffs_g):
            term = _scale_tracked(g[j], c, p)
            if term is not None:
                acc = term if acc is None else (acc + term)
        return acc

    for k in range(M):
        L = lincomb(L_x[k], L_g[k][:k])
        R = lincomb(R_x[k], R_g[k][:k])
        if L is None or R is None:
            g.append(TrackedFieldElement(0, p, x[0].counter, depth=0))
        else:
            g.append(L * R)

    out: TrackedFieldElement | None = None
    for k, c in enumerate(w_out):
        term = _scale_tracked(g[k], c, p)
        if term is not None:
            out = term if out is None else (out + term)
    if out is None:
        return TrackedFieldElement(0, p, x[0].counter, depth=0)
    return out


def lift_float_dag_to_rationals(
    L_x: np.ndarray,
    L_g: np.ndarray,
    R_x: np.ndarray,
    R_g: np.ndarray,
    w_out: np.ndarray,
    max_denom: int = 128,
    zero_tol: float = 1e-6,
):
    def snap(val: float) -> Fraction:
        if abs(val) < zero_tol:
            return Fraction(0)
        return Fraction(float(val)).limit_denominator(max_denom)

    M, n2 = L_x.shape
    return (
        [[snap(L_x[k, i]) for i in range(n2)] for k in range(M)],
        [[snap(L_g[k, j]) if j < k else Fraction(0) for j in range(M)] for k in range(M)],
        [[snap(R_x[k, i]) for i in range(n2)] for k in range(M)],
        [[snap(R_g[k, j]) if j < k else Fraction(0) for j in range(M)] for k in range(M)],
        [snap(w_out[k]) for k in range(M)],
    )


def verify_rational_dag(
    n: int,
    L_x: list[list[Fraction]],
    L_g: list[list[Fraction]],
    R_x: list[list[Fraction]],
    R_g: list[list[Fraction]],
    w_out: list[Fraction],
    trials_per_prime: int = 50,
    run_symbolic: bool = True,
    seed: int = 2026,
) -> dict:
    rng = random.Random(seed)
    last_counter = None
    last_depth = None

    for p in (PRIME_61_A, PRIME_61_B):
        for trial in range(trials_per_prime):
            A_int = [[rng.randrange(p) for _ in range(n)] for _ in range(n)]
            counter = OpCounter()
            A_wire = [
                [TrackedFieldElement(A_int[r][c], p, counter) for c in range(n)]
                for r in range(n)
            ]
            out = eval_rational_dag_tracked(A_wire, L_x, L_g, R_x, R_g, w_out, p)
            ref = glynn_perm_mod(A_int, p)
            if out.val != ref:
                return {
                    "passed": False,
                    "failed_gate": f"Gate 1 modular mismatch at p={p}, trial={trial}",
                    "non_scalar_muls": counter.non_scalar_muls,
                }
            last_counter = counter
            last_depth = out.depth

    symbolic_ok = None
    if run_symbolic and n <= 4:
        sym_A = [[sp.Symbol(f"a{r}{c}") for c in range(n)] for r in range(n)]
        x = [sym_A[r][c] for r in range(n) for c in range(n)]
        M = len(w_out)
        g_sym = []
        for k in range(M):
            L = sum(sp.Rational(L_x[k][i].numerator, L_x[k][i].denominator) * x[i] for i in range(n * n))
            L += sum(sp.Rational(L_g[k][j].numerator, L_g[k][j].denominator) * g_sym[j] for j in range(k))
            R = sum(sp.Rational(R_x[k][i].numerator, R_x[k][i].denominator) * x[i] for i in range(n * n))
            R += sum(sp.Rational(R_g[k][j].numerator, R_g[k][j].denominator) * g_sym[j] for j in range(k))
            g_sym.append(sp.expand(L * R))
        sym_out = sp.expand(
            sum(sp.Rational(w_out[k].numerator, w_out[k].denominator) * g_sym[k] for k in range(M))
        )
        sym_ref = sp.expand(
            sum(sp.prod(sym_A[r][pi[r]] for r in range(n)) for pi in itertools.permutations(range(n)))
        )
        symbolic_ok = (sp.expand(sym_out - sym_ref) == 0)
        if not symbolic_ok:
            return {
                "passed": False,
                "failed_gate": "Gate 3 symbolic Q[A] residual non-zero",
                "non_scalar_muls": last_counter.non_scalar_muls,
            }

    return {
        "passed": True,
        "non_scalar_muls": last_counter.non_scalar_muls,
        "scalar_muls": last_counter.scalar_muls,
        "adds_subs": last_counter.adds_subs,
        "total_gates": last_counter.total_gates,
        "depth": last_depth,
        "symbolic_ok": symbolic_ok,
    }
