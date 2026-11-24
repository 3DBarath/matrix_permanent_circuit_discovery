"""
Anti-cheat exact operation accounting over finite fields F_p (char(F) != 2)
and trusted Glynn reference implementation.
"""

from __future__ import annotations

PRIME_61_A = (1 << 61) - 1   # 2305843009213693951 (Mersenne prime)
PRIME_61_B = (1 << 61) - 31  # 2305843009213693921 (61-bit prime)


class OpCounter:
    """Tracks exact arithmetic gate counts for a straight-line program."""

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
    """
    Modular field element over F_p that distinguishes wire*wire (non-scalar mul)
    from const*wire (scalar mul) and tracks exact arithmetic DAG depth.
    """

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
                self.val + other.val,
                self.p,
                self.counter,
                max(self.depth, other.depth) + 1,
            )
        raise TypeError("Wire addition requires another TrackedFieldElement.")

    def __sub__(self, other: TrackedFieldElement) -> TrackedFieldElement:
        if isinstance(other, TrackedFieldElement):
            self.counter.adds_subs += 1
            return TrackedFieldElement(
                self.val - other.val,
                self.p,
                self.counter,
                max(self.depth, other.depth) + 1,
            )
        raise TypeError("Wire subtraction requires another TrackedFieldElement.")

    def __mul__(self, other: TrackedFieldElement | int) -> TrackedFieldElement:
        if isinstance(other, TrackedFieldElement):
            self.counter.non_scalar_muls += 1
            return TrackedFieldElement(
                self.val * other.val,
                self.p,
                self.counter,
                max(self.depth, other.depth) + 1,
            )
        if isinstance(other, int):
            self.counter.scalar_muls += 1
            return TrackedFieldElement(
                self.val * other,
                self.p,
                self.counter,
                self.depth + 1,
            )
        raise TypeError(f"Unsupported operand type: {type(other)}")

    def __rmul__(self, other: int) -> TrackedFieldElement:
        if isinstance(other, int):
            self.counter.scalar_muls += 1
            return TrackedFieldElement(
                other * self.val,
                self.p,
                self.counter,
                self.depth + 1,
            )
        raise TypeError(f"Unsupported operand type: {type(other)}")


def glynn_perm_mod(A_int: list[list[int]], p: int) -> int:
    """Trusted Glynn formula reference for n x n permanent modulo prime p."""
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
