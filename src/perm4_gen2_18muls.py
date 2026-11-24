"""
Generation 2 Champion: Exact 18-multiplication circuit for Perm_4(A).
Mechanism: Fused rank-7 Walsh-Hadamard quotient chain (7 + 7 + 4 = 18 non-scalar muls).
- Layer 1 (rows 0, 1): 7 non-scalar muls via prefix degree-2 interpolation.
- Fused rank-1 shift: S = (m0+m3+m5+m6) - (m1+m2+m4) recovers m7 and annihilates
  index 7 before Layer 2 in only 13 adds/subs.
- Layer 2 (row 2): 7 non-scalar muls.
- Layer 3 (row 3): inverse Walsh-Hadamard + 4 non-scalar muls.
Op counts: 18 non-scalar muls, 1 scalar mul (inv8), 65 adds/subs, 84 total gates, depth 15.
"""


def _walsh7(row):
    """Evaluate 7 of 8 Walsh-Hadamard forms on a 4-vector (11 adds/subs, depth 2)."""
    x0, x1, x2, x3 = row
    s01 = x0 + x1
    d01 = x0 - x1
    s23 = x2 + x3
    d23 = x2 - x3
    return (
        s01 + s23,  # 0: (1,  1,  1,  1)  parity +
        s01 + d23,  # 1: (1,  1,  1, -1)  parity -
        s01 - d23,  # 2: (1,  1, -1,  1)  parity -
        s01 - s23,  # 3: (1,  1, -1, -1)  parity +
        d01 + s23,  # 4: (1, -1,  1,  1)  parity -
        d01 + d23,  # 5: (1, -1,  1, -1)  parity +
        d01 - d23,  # 6: (1, -1, -1,  1)  parity +
    )


def perm4_circuit_18(A, inv8):
    """Exact straight-line program for Perm_4(A) using 18 non-scalar multiplications."""
    # Layer 1: 7 non-scalar muls
    u = _walsh7(A[0])
    v = _walsh7(A[1])
    m0 = u[0] * v[0]
    m1 = u[1] * v[1]
    m2 = u[2] * v[2]
    m3 = u[3] * v[3]
    m4 = u[4] * v[4]
    m5 = u[5] * v[5]
    m6 = u[6] * v[6]

    # Fused rank-1 parity-annihilation shift (13 adds/subs)
    pos = (m0 + m3) + (m5 + m6)
    neg = (m1 + m2) + m4
    S = pos - neg

    y0 = m0 - S
    y1 = S - m1
    y2 = S - m2
    y3 = m3 - S
    y4 = S - m4
    y5 = m5 - S
    y6 = m6 - S

    # Layer 2: 7 non-scalar muls
    z = _walsh7(A[2])
    w0 = y0 * z[0]
    w1 = y1 * z[1]
    w2 = y2 * z[2]
    w3 = y3 * z[3]
    w4 = y4 * z[4]
    w5 = y5 * z[5]
    w6 = y6 * z[6]

    # Inverse Walsh-Hadamard for the four 3x3 permanents (16 adds/subs)
    a0 = w0 + w1
    b0 = w0 - w1
    a1 = w2 + w3
    b1 = w2 - w3
    a2 = w4 + w5
    b2 = w4 - w5

    top_s = a0 + a1
    top_d = a0 - a1
    bot_s = a2 + w6
    bot_d = a2 - w6

    q_123 = top_s + bot_s
    q_023 = top_s - bot_s
    q_013 = top_d + bot_d
    q_012 = (b0 + b1) + (b2 + w6)

    # Layer 3: 4 non-scalar muls
    r3 = A[3]
    t0 = q_123 * r3[0]
    t1 = q_023 * r3[1]
    t2 = q_013 * r3[2]
    t3 = q_012 * r3[3]

    return inv8 * ((t0 + t1) + (t2 + t3))
