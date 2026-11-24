#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <random>
#include <vector>
#include <algorithm>

static constexpr uint64_t P61 = (1ULL << 61) - 1;

static inline uint64_t add61(uint64_t a, uint64_t b) {
    uint64_t s = a + b;
    return s >= P61 ? s - P61 : s;
}
static inline uint64_t sub61(uint64_t a, uint64_t b) {
    return a >= b ? (a - b) : (a + P61 - b);
}
static inline uint64_t mul61(uint64_t a, uint64_t b) {
    __uint128_t prod = (__uint128_t)a * b;
    uint64_t lo = (uint64_t)prod & P61;
    uint64_t hi = (uint64_t)(prod >> 61);
    return add61(lo, hi);
}
static inline uint64_t pow61(uint64_t base, uint64_t exp) {
    uint64_t res = 1;
    while (exp) {
        if (exp & 1) res = mul61(res, base);
        base = mul61(base, base);
        exp >>= 1;
    }
    return res;
}

// Optimized Glynn + Gray code baseline mod (2^61 - 1)
template<int N_DIM>
uint64_t glynn_gray_mod61(const uint64_t* A) {
    constexpr int M = 1 << (N_DIM - 1);
    uint64_t row_sum[N_DIM];
    uint64_t twoA[N_DIM][N_DIM];
    for (int r = 0; r < N_DIM; ++r) {
        uint64_t s = A[r * N_DIM];
        for (int c = 1; c < N_DIM; ++c) {
            uint64_t v = A[r * N_DIM + c];
            s = add61(s, v);
            twoA[r][c] = add61(v, v);
        }
        row_sum[r] = s;
    }
    uint64_t total = 0;
    {
        uint64_t prod = row_sum[0];
        for (int r = 1; r < N_DIM; ++r) prod = mul61(prod, row_sum[r]);
        total = prod;
    }
    int prev_g = 0;
    for (int step = 1; step < M; ++step) {
        int g = step ^ (step >> 1);
        int diff = g ^ prev_g;
        int col = __builtin_ctz(diff) + 1;
        if (g & diff) {
            for (int r = 0; r < N_DIM; ++r) row_sum[r] = sub61(row_sum[r], twoA[r][col]);
        } else {
            for (int r = 0; r < N_DIM; ++r) row_sum[r] = add61(row_sum[r], twoA[r][col]);
        }
        uint64_t prod = row_sum[0];
        for (int r = 1; r < N_DIM; ++r) prod = mul61(prod, row_sum[r]);
        if (__builtin_parity(g)) total = sub61(total, prod);
        else total = add61(total, prod);
        prev_g = g;
    }
    static const uint64_t invM = pow61(M, P61 - 2);
    return mul61(total, invM);
}

// =========================================================================
// Native C++17 Generation 6 Fused Mobius-Zeta Butterfly OptTree
// =========================================================================
struct OptTreeNative {
    int n, m, N;
    uint64_t invN;
    std::vector<uint8_t> wt;
    std::vector<std::vector<int>> masks_le; // masks_le[d] = masks with |y| <= d (sorted)
    std::vector<std::vector<int>> masks_gt; // masks_gt[d] = masks with |y| > d

    explicit OptTreeNative(int n_) : n(n_), m(n_ - 1), N(1 << (n_ - 1)) {
        invN = pow61(N, P61 - 2);
        wt.resize(N);
        for (int y = 0; y < N; ++y) wt[y] = (uint8_t)__builtin_popcount(y);
        masks_le.resize(m + 1);
        masks_gt.resize(m + 1);
        for (int d = 0; d <= m; ++d) {
            for (int y = 0; y < N; ++y) {
                if (wt[y] <= d) masks_le[d].push_back(y);
                else masks_gt[d].push_back(y);
            }
        }
    }

    // Hasse-diagram 1-add-per-mask row evaluation on |y| <= max_w
    inline void eval_row_le(const uint64_t* row, int max_w, uint64_t* out) const {
        uint64_t two_r[24];
        uint64_t s0 = row[0];
        for (int c = 1; c < n; ++c) {
            s0 = add61(s0, row[c]);
            two_r[c] = add61(row[c], row[c]);
        }
        out[0] = s0;
        const auto& ms = masks_le[max_w];
        for (size_t idx = 1; idx < ms.size(); ++idx) {
            int mask = ms[idx];
            int lsb = mask & -mask;
            int col = __builtin_ctz(lsb) + 1;
            out[mask] = sub61(out[mask ^ lsb], two_r[col]);
        }
    }

    // Hasse-diagram 1-add-per-mask row evaluation on |y| >= min_w
    inline void eval_row_ge(const uint64_t* row, int min_w, uint64_t* out) const {
        int full = N - 1;
        int limit = m - min_w;
        uint64_t two_r[24];
        uint64_t sf = row[0];
        for (int c = 1; c < n; ++c) {
            sf = sub61(sf, row[c]);
            two_r[c] = add61(row[c], row[c]);
        }
        out[full] = sf;
        const auto& cms = masks_le[limit];
        for (size_t idx = 1; idx < cms.size(); ++idx) {
            int cmask = cms[idx];
            int lsb = cmask & -cmask;
            int col = __builtin_ctz(lsb) + 1;
            out[full ^ cmask] = add61(out[full ^ (cmask ^ lsb)], two_r[col]);
        }
    }

    // Mobius-Zeta interpolation from |y| <= d to |y| <= max_w (with optional fused drop <= drop_d)
    inline void mobius_interp(uint64_t* arr, int d, int max_w, int drop_d = -1) const {
        if (drop_d < 0 && d >= max_w) return;
        // Stage 1: Inverse Mobius on |y| <= d
        const auto& le_d = masks_le[d];
        for (int bit = 0; bit < m; ++bit) {
            int step = 1 << bit;
            for (int y : le_d) {
                if (y & step) arr[y] = sub61(arr[y], arr[y ^ step]);
            }
        }
        // Zero entries outside active Mobius support
        if (drop_d >= 0) {
            for (int y : masks_le[drop_d]) arr[y] = 0;
        }
        for (int y : masks_gt[d]) arr[y] = 0;

        // Stage 2: Forward Zeta up to max_w
        const auto& le_max = masks_le[max_w];
        for (int bit = 0; bit < m; ++bit) {
            int step = 1 << bit;
            for (int y : le_max) {
                if (y & step) arr[y] = add61(arr[y], arr[y ^ step]);
            }
        }
    }

    // Mobius annihilation: U -> U - c_{<= d_ann} (zero on |y| <= d_ann)
    inline void mobius_ann(uint64_t* U, int d_ann) const {
        if (d_ann == 0) {
            uint64_t u0 = U[0];
            U[0] = 0;
            for (int y = 1; y < N; ++y) U[y] = sub61(U[y], u0);
            return;
        }
        // Full inverse Mobius -> zero |S| <= d_ann -> forward Zeta on |S| > d_ann
        for (int bit = 0; bit < m; ++bit) {
            int step = 1 << bit;
            for (int y = 0; y < N; ++y) {
                if (y & step) U[y] = sub61(U[y], U[y ^ step]);
            }
        }
        for (int y : masks_le[d_ann]) U[y] = 0;
        const auto& gt = masks_gt[d_ann];
        for (int bit = 0; bit < m; ++bit) {
            int step = 1 << bit;
            for (int y : gt) {
                if (y & step) U[y] = add61(U[y], U[y ^ step]);
            }
        }
    }

    // Final Parseval Collapse with last row (rem == 1): computes hat(U)[full] and hat(U)[full ^ (1<<c)]
    inline uint64_t final_collapse(const uint64_t* U, const uint64_t* last_row, int d_ann_prev) const {
        // Only masks with |y| > d_ann_prev are nonzero in U!
        // Compute sum_y (-1)^|y| U[y] * (v_y . last_row) by accumulating the n coefficients directly on sparse gt!
        // Even faster: for |gt| = K(n, m - 1 - d_ann_prev) (e.g., 67 at n=12!), direct loop over gt is only 67*12 ops!
        uint64_t coeff[24] = {0};
        const auto& gt = (d_ann_prev >= 0) ? masks_gt[d_ann_prev] : masks_le[m];
        for (int y : gt) {
            uint64_t u = U[y];
            int par = wt[y] & 1;
            if (par) coeff[0] = sub61(coeff[0], u);
            else coeff[0] = add61(coeff[0], u);
            for (int c = 1; c < n; ++c) {
                int s = par ^ ((y >> (c - 1)) & 1);
                if (s) coeff[c] = sub61(coeff[c], u);
                else coeff[c] = add61(coeff[c], u);
            }
        }
        uint64_t total = 0;
        for (int c = 0; c < n; ++c) total = add61(total, mul61(coeff[c], last_row[c]));
        return mul61(total, invN);
    }

    // Helper: build deg-2 product of 2 rows on |y| <= max_w
    inline void build_deg2(const uint64_t* r0, const uint64_t* r1, uint64_t* out, uint64_t* tmp, int max_w, int drop_d = -1) const {
        eval_row_le(r0, 2, out);
        eval_row_le(r1, 2, tmp);
        for (int y : masks_le[2]) out[y] = mul61(out[y], tmp[y]);
        mobius_interp(out, 2, max_w, drop_d);
    }

    // Helper: build deg-4 product of 4 rows on |y| <= max_w
    inline void build_deg4(const uint64_t* A_off, uint64_t* out, uint64_t* tmp1, uint64_t* tmp2, int max_w, int drop_d = -1) const {
        build_deg2(A_off, A_off + n, out, tmp1, 4, -1);
        build_deg2(A_off + 2 * n, A_off + 3 * n, tmp1, tmp2, 4, -1);
        for (int y : masks_le[4]) out[y] = mul61(out[y], tmp1[y]);
        mobius_interp(out, 4, max_w, drop_d);
    }

    // Exact Gen 5/6 optimal plans for n = 4..14
    uint64_t eval_perm(const uint64_t* A) const {
        static thread_local uint64_t buf0[1 << 14], buf1[1 << 14], buf2[1 << 14], buf3[1 << 14];
        if (n == 4) {
            // start=1, interp(1,2) fused with ann(d_ann=0), ann_mul(row2), collapse(row3)
            build_deg2(A, A + n, buf0, buf1, m, 0);
            eval_row_ge(A + 2 * n, 1, buf1);
            for (int y : masks_gt[0]) buf0[y] = mul61(buf0[y], buf1[y]);
            return final_collapse(buf0, A + 3 * n, 0);
        } else if (n == 5) {
            build_deg2(A, A + n, buf0, buf1, m, 0);
            eval_row_ge(A + 2 * n, 1, buf1);
            for (int y : masks_gt[0]) buf0[y] = mul61(buf0[y], buf1[y]);
            mobius_ann(buf0, 1);
            eval_row_ge(A + 3 * n, 2, buf1);
            for (int y : masks_gt[1]) buf0[y] = mul61(buf0[y], buf1[y]);
            return final_collapse(buf0, A + 4 * n, 1);
        } else if (n == 6) {
            // deg2 fused drop 0 -> mul deg2(rows 2,3) -> ann(2) -> mul row4 -> collapse(row5)
            build_deg2(A, A + n, buf0, buf1, m, 0);
            build_deg2(A + 2 * n, A + 3 * n, buf1, buf2, m, -1);
            for (int y : masks_gt[0]) buf0[y] = mul61(buf0[y], buf1[y]);
            mobius_ann(buf0, 2);
            eval_row_ge(A + 4 * n, 3, buf1);
            for (int y : masks_gt[2]) buf0[y] = mul61(buf0[y], buf1[y]);
            return final_collapse(buf0, A + 5 * n, 2);
        } else if (n == 7) {
            // deg4 fused drop 2 -> mul row4 -> ann(3) -> mul row5 -> collapse(row6)
            build_deg4(A, buf0, buf1, buf2, m, 2);
            eval_row_ge(A + 4 * n, 3, buf1);
            for (int y : masks_gt[2]) buf0[y] = mul61(buf0[y], buf1[y]);
            mobius_ann(buf0, 3);
            eval_row_ge(A + 5 * n, 4, buf1);
            for (int y : masks_gt[3]) buf0[y] = mul61(buf0[y], buf1[y]);
            return final_collapse(buf0, A + 6 * n, 3);
        } else if (n == 8) {
            // deg4 fused drop 2 -> mul deg2(4,5) -> ann(4) -> mul row6 -> collapse(row7)
            build_deg4(A, buf0, buf1, buf2, m, 2);
            build_deg2(A + 4 * n, A + 5 * n, buf1, buf2, m, -1);
            for (int y : masks_gt[2]) buf0[y] = mul61(buf0[y], buf1[y]);
            mobius_ann(buf0, 4);
            eval_row_ge(A + 6 * n, 5, buf1);
            for (int y : masks_gt[4]) buf0[y] = mul61(buf0[y], buf1[y]);
            return final_collapse(buf0, A + 7 * n, 4);
        } else if (n == 9) {
            // deg4 fused drop 2 -> mul deg2(4,5) -> ann(4) -> mul row6 -> ann(5) -> mul row7 -> collapse(row8)
            build_deg4(A, buf0, buf1, buf2, m, 2);
            build_deg2(A + 4 * n, A + 5 * n, buf1, buf2, m, -1);
            for (int y : masks_gt[2]) buf0[y] = mul61(buf0[y], buf1[y]);
            mobius_ann(buf0, 4);
            eval_row_ge(A + 6 * n, 5, buf1);
            for (int y : masks_gt[4]) buf0[y] = mul61(buf0[y], buf1[y]);
            mobius_ann(buf0, 5);
            eval_row_ge(A + 7 * n, 6, buf1);
            for (int y : masks_gt[5]) buf0[y] = mul61(buf0[y], buf1[y]);
            return final_collapse(buf0, A + 8 * n, 5);
        } else if (n == 10) {
            // deg4 fused drop 2 -> mul deg3(4,5,6) -> ann(5) -> mul row7 -> ann(6) -> mul row8 -> collapse(row9)
            build_deg4(A, buf0, buf1, buf2, m, 2);
            build_deg2(A + 4 * n, A + 5 * n, buf1, buf2, 3, -1);
            eval_row_le(A + 6 * n, 3, buf2);
            for (int y : masks_le[3]) buf1[y] = mul61(buf1[y], buf2[y]);
            mobius_interp(buf1, 3, m, -1);
            for (int y : masks_gt[2]) buf0[y] = mul61(buf0[y], buf1[y]);
            mobius_ann(buf0, 5);
            eval_row_ge(A + 7 * n, 6, buf1);
            for (int y : masks_gt[5]) buf0[y] = mul61(buf0[y], buf1[y]);
            mobius_ann(buf0, 6);
            eval_row_ge(A + 8 * n, 7, buf1);
            for (int y : masks_gt[6]) buf0[y] = mul61(buf0[y], buf1[y]);
            return final_collapse(buf0, A + 9 * n, 6);
        } else if (n == 11) {
            // deg4 fused drop 2 -> mul deg4(4..7) -> ann(6) -> mul row8 -> ann(7) -> mul row9 -> collapse(row10)
            build_deg4(A, buf0, buf1, buf2, m, 2);
            build_deg4(A + 4 * n, buf1, buf2, buf3, m, -1);
            for (int y : masks_gt[2]) buf0[y] = mul61(buf0[y], buf1[y]);
            mobius_ann(buf0, 6);
            eval_row_ge(A + 8 * n, 7, buf1);
            for (int y : masks_gt[6]) buf0[y] = mul61(buf0[y], buf1[y]);
            mobius_ann(buf0, 7);
            eval_row_ge(A + 9 * n, 8, buf1);
            for (int y : masks_gt[7]) buf0[y] = mul61(buf0[y], buf1[y]);
            return final_collapse(buf0, A + 10 * n, 7);
        } else if (n == 12) {
            // start=8 (deg4 * deg4 -> interp 8 fused drop 6!) -> mul deg2(8,9) -> ann(8) -> mul row10 -> collapse(row11)
            build_deg4(A, buf0, buf1, buf2, 8, -1);
            build_deg4(A + 4 * n, buf1, buf2, buf3, 8, -1);
            for (int y : masks_le[8]) buf0[y] = mul61(buf0[y], buf1[y]);
            mobius_interp(buf0, 8, m, 6);
            build_deg2(A + 8 * n, A + 9 * n, buf1, buf2, m, -1);
            for (int y : masks_gt[6]) buf0[y] = mul61(buf0[y], buf1[y]);
            mobius_ann(buf0, 8);
            eval_row_ge(A + 10 * n, 9, buf1);
            for (int y : masks_gt[8]) buf0[y] = mul61(buf0[y], buf1[y]);
            return final_collapse(buf0, A + 11 * n, 8);
        } else if (n == 14) {
            // start=8 (deg4 * deg4 -> interp 8 fused drop 6) -> mul deg4(8..11) -> ann(10) -> mul row12 -> collapse(row13)
            build_deg4(A, buf0, buf1, buf2, 8, -1);
            build_deg4(A + 4 * n, buf1, buf2, buf3, 8, -1);
            for (int y : masks_le[8]) buf0[y] = mul61(buf0[y], buf1[y]);
            mobius_interp(buf0, 8, m, 6);
            build_deg4(A + 8 * n, buf1, buf2, buf3, m, -1);
            for (int y : masks_gt[6]) buf0[y] = mul61(buf0[y], buf1[y]);
            mobius_ann(buf0, 10);
            eval_row_ge(A + 12 * n, 11, buf1);
            for (int y : masks_gt[10]) buf0[y] = mul61(buf0[y], buf1[y]);
            return final_collapse(buf0, A + 13 * n, 10);
        }
        return 0;
    }
};

template<int N_DIM>
void bench(int iters) {
    OptTreeNative engine(N_DIM);
    std::mt19937_64 rng(2026 + N_DIM);
    constexpr int NUM_MATS = 32;
    std::vector<uint64_t> mats(NUM_MATS * N_DIM * N_DIM);
    for (auto& x : mats) x = rng() % P61;

    for (int m = 0; m < NUM_MATS; ++m) {
        const uint64_t* A = &mats[m * N_DIM * N_DIM];
        uint64_t r1 = glynn_gray_mod61<N_DIM>(A);
        uint64_t r2 = engine.eval_perm(A);
        if (r1 != r2) {
            std::printf("MISMATCH n=%d mat=%d: glynn=%llu opt=%llu\n", N_DIM, m, (unsigned long long)r1, (unsigned long long)r2);
            std::exit(1);
        }
    }

    volatile uint64_t s1 = 0, s2 = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i)
        s1 ^= glynn_gray_mod61<N_DIM>(&mats[(i & (NUM_MATS - 1)) * N_DIM * N_DIM]);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ns_g = std::chrono::duration<double, std::nano>(t1 - t0).count() / iters;

    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i)
        s2 ^= engine.eval_perm(&mats[(i & (NUM_MATS - 1)) * N_DIM * N_DIM]);
    auto t3 = std::chrono::high_resolution_clock::now();
    double ns_o = std::chrono::duration<double, std::nano>(t3 - t2).count() / iters;

    if (s1 != s2) { std::printf("SINK MISMATCH\n"); std::exit(1); }
    std::printf("n=%2d | Glynn+Gray: %9.1f ns | Native Gen6 C++: %9.1f ns | Speedup: %5.2fx | BIT-EXACT (32/32)\n",
                N_DIM, ns_g, ns_o, ns_g / ns_o);
}

int main() {
    bench<4>(200000);
    bench<5>(100000);
    bench<6>(50000);
    bench<7>(20000);
    bench<8>(10000);
    bench<9>(5000);
    bench<10>(2000);
    bench<11>(1000);
    bench<12>(500);
    bench<14>(100);
    return 0;
}
