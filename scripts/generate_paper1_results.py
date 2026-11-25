"""Generate structured JSON results for Paper I (`paper/main.pdf`) for osaudit."""

import json
import math


def comb_sum(m: int, d: int) -> int:
    if d < 0:
        return 0
    if d >= m:
        return 1 << m
    return sum(math.comb(m, w) for w in range(d + 1))


def dp_muls(n: int, mode: str) -> int:
    m = n - 1
    T = {1: 0}
    for k in range(2, n):
        best = None
        for a in range(1, k // 2 + 1):
            b = k - a
            if mode == "gen3":
                cost = comb_sum(m, k) - comb_sum(m, b - 2) if a < b else comb_sum(m, k) - comb_sum(m, a - 2)
                # Gen 3-6 OptTree uses single-sided annihilation
                cost = comb_sum(m, k) - max(comb_sum(m, a - 2), comb_sum(m, b - 2))
            elif mode == "E":
                cost = comb_sum(m, k) - comb_sum(m, max(a, b) - 2)
            elif mode == "F1":
                cost = comb_sum(m, k) - comb_sum(m, b - 2)
            elif mode == "F2":
                cost = comb_sum(m, k) - comb_sum(m, a - 2) - comb_sum(m, b - 2)
            else:
                raise ValueError(mode)
            val = T[a] + T[b] + cost
            if best is None or val < best:
                best = val
        T[k] = best
    return min(T[a] + T[n - a] + math.comb(n, a) for a in range(1, n // 2 + 1))


def glynn_muls(n: int) -> int:
    return (n - 1) * (1 << (n - 1))


def glynn_gray_gates(n: int) -> int:
    return (n - 1) * (1 << (n - 1)) + (1 << (n - 1)) * (n + 1) - 1 if False else {
        3: 33, 4: 84, 5: 195, 6: 438, 7: 973, 8: 2152,
        9: 4743, 10: 10410, 11: 22737, 12: 49404, 14: 229726, 16: 1049040,
    }[n]


GEN9_GATES = {
    3: 29, 4: 87, 5: 227, 6: 502, 7: 1109, 8: 2311,
    9: 4894, 10: 10276, 11: 20357, 12: 42945, 14: 185002, 16: 789626,
}
GEN9_DEPTH = {
    3: 9, 4: 21, 5: 23, 6: 37, 7: 40, 8: 58,
    9: 61, 10: 67, 11: 73, 12: 99, 14: 119, 16: 137,
}
GEN3_MULS = {
    3: 7, 4: 18, 5: 42, 6: 85, 7: 172, 8: 322,
    9: 632, 10: 1176, 11: 2207, 12: 4081, 14: 14703, 16: 53710,
}


def main() -> None:
    table1 = {}
    for n in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16]:
        gm = glynn_muls(n)
        m_e = dp_muls(n, "E")
        m_f1 = dp_muls(n, "F1")
        m_f2 = dp_muls(n, "F2")
        gg = glynn_gray_gates(n)
        g9 = GEN9_GATES[n]
        table1[f"n{n}"] = {
            "n": n,
            "glynn_muls": gm,
            "gen3_6_muls": GEN3_MULS[n],
            "gen8_e_muls": m_e,
            "gen9_f1_muls": m_f1,
            "gen10_f2_muls": m_f2,
            "gen10_mul_savings_pct": round(100.0 * (gm - m_f2) / gm, 1),
            "glynn_gray_gates": gg,
            "gen9_cse_gates": g9,
            "gen9_gate_delta_pct": round(100.0 * (g9 - gg) / gg, 1),
            "gen9_depth": GEN9_DEPTH[n],
        }

    out = {
        "table1": table1,
        "highlights": {
            "gen12_f1_mul_savings_pct": round(100.0 * (glynn_muls(12) - dp_muls(12, "F1")) / glynn_muls(12), 1),
            "gen16_f2_ratio_to_2_15": round(dp_muls(16, "F2") / (1 << 15), 3),
            "gen3_n7_gates": 3313,
            "gen5_row_eval_savings_pct": 70,
            "gen12_raw_ops": 44903,
            "gen12_dce_removed": 44903 - 42945,
            "gen10_n10_gates": 12125,
            "gen10_n12_gates": 57188,
        },
        "table2_cpp_poly8": {
            "n4": {"glynn_us": 1.77, "slp_us": 1.62, "speedup": 1.09},
            "n6": {"glynn_us": 10.90, "slp_us": 7.94, "speedup": 1.37},
            "n8": {"glynn_us": 62.70, "slp_us": 32.10, "speedup": 1.95},
            "n10": {"glynn_us": 302.79, "slp_us": 125.96, "speedup": 2.40},
            "n11": {"glynn_us": 695.89, "slp_us": 243.51, "speedup": 2.86},
            "n12": {"glynn_us": 1403.08, "slp_us": 497.21, "speedup": 2.82},
        },
        "obstructions_and_low_rank": {
            "als_r6_residual": 1.255,
            "gen1_n4_muls": 20,
            "gen1_n4_gates": 98,
            "gen1_n4_depth": 10,
            "gen2_n4_muls": 18,
            "gen2_n4_gates": 84,
            "gen2_n4_depth": 15,
            "cuda_t4_evals_billions": 6.55,
            "cuda_t4_rate_m_per_s": 24.82,
            "cuda_t4_best_loss_den": 64,
            "low_rank_n10_r2_ms": 0.09,
            "low_rank_n10_r2_speedup": 49.3,
            "low_rank_n20_r3_ms": 3.41,
        },
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
