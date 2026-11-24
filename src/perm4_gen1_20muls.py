"""
Generation 1 Candidate: Exact 20-multiplication circuit for Perm_4(A).
Mechanism: Block Laplace expansion (rows {0,1} x {2,3}) combined with exact
rank-7 Walsh-Hadamard bilinear decomposition of the 4x4x6 2x2-permanent tensor.
Op counts: 20 non-scalar muls, 1 scalar mul (inv16), 77 adds/subs, 98 total gates, depth 10.
"""


def _walsh7(row):
    """Evaluate 7 of 8 Walsh-Hadamard forms on a 4-vector (11 adds/subs, depth 2)."""
    x0, x1, x2, x3 = row
    s01 = x0 + x1
    d01 = x0 - x1
    s23 = x2 + x3
    d23 = x2 - x3
    return (
        s01 + s23,  # (1,  1,  1,  1)
        s01 + d23,  # (1,  1,  1, -1)
        s01 - d23,  # (1,  1, -1,  1)
        s01 - s23,  # (1,  1, -1, -1)
        d01 + s23,  # (1, -1,  1,  1)
        d01 + d23,  # (1, -1,  1, -1)
        d01 - d23,  # (1, -1, -1,  1)
    )


def _pair_perms_scaled4(row_a, row_b):
    """Compute 4 * P_{a,b}(i, j) for all 6 pairs (i < j) in 7 non-scalar muls."""
    u = _walsh7(row_a)
    v = _walsh7(row_b)

    m0 = u[0] * v[0]
    m1 = u[1] * v[1]
    m2 = u[2] * v[2]
    m3 = u[3] * v[3]
    m4 = u[4] * v[4]
    m5 = u[5] * v[5]
    m6 = u[6] * v[6]

    a01 = m0 - m1
    a02 = m0 - m2
    a13 = m1 - m3
    a15 = m1 - m5
    a23 = m2 - m3
    a26 = m2 - m6
    a45 = m4 - m5
    a46 = m4 - m6

    p01 = a15 + a26
    p02 = a13 + a46
    p03 = a23 + a45
    p12 = a02 - a46
    p13 = a01 - a45
    p23 = a01 - a23
    return p01, p02, p03, p12, p13, p23


def perm4_circuit_20(A, inv16):
    """Exact straight-line program for Perm_4(A) using 20 non-scalar multiplications."""
    p01, p02, p03, p12, p13, p23 = _pair_perms_scaled4(A[0], A[1])
    q01, q02, q03, q12, q13, q23 = _pair_perms_scaled4(A[2], A[3])

    t0 = p01 * q23
    t1 = p02 * q13
    t2 = p03 * q12
    t3 = p12 * q03
    t4 = p13 * q02
    t5 = p23 * q01

    s0 = t0 + t1
    s1 = t2 + t3
    s2 = t4 + t5
    return inv16 * ((s0 + s1) + s2)
