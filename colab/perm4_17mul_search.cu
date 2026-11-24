// colab/perm4_17mul_search.cu
// Self-contained Google Colab CUDA kernel for bit-sliced F_3 / F_5
// evolutionary + simulated-annealing search for a 17-multiplication
// non-row-separated DAG for Perm_4 (mixing deg2*deg2 and deg1*deg3 gates).
//
// Compile on Colab (Tesla T4, sm_75):
//   nvcc -O3 -arch=sm_75 colab/perm4_17mul_search.cu -o perm4_17mul_search
// Run:
//   ./perm4_17mul_search --mod 3 --iters 200000 --blocks 256 --threads 256
//   ./perm4_17mul_search --mod 5 --iters 200000 --blocks 256 --threads 256

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <ctime>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err__));                               \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Circuit architecture:
//   Inputs: x[0..15] (4x4 matrix, row-major), degree 1
//   17 multiplication gates g[0..16]:
//     - Layer A (deg 2): g[0..7]   compute (lin of x) * (lin of x)
//     - Layer B (deg 3): g[8..11]  compute (lin of x) * (lin of g[0..7])
//     - Layer C (deg 4): g[12..16] mix deg2*deg2 and deg1*deg3:
//         type 0 (deg2*deg2): (lin of g[0..7]) * (lin of g[0..7])
//         type 1 (deg1*deg3): (lin of x)       * (lin of g[8..11])
//   Output: linear combination of g[12..16] (all degree 4).
//   Every gate's left/right linear form has up to MAX_TERMS sparse (idx, coeff) pairs.

constexpr int NUM_INPUTS = 16;
constexpr int NUM_GATES  = 17;
constexpr int NUM_DEG2   = 8;   // g[0..7]
constexpr int NUM_DEG3   = 4;   // g[8..11]
constexpr int NUM_DEG4   = 5;   // g[12..16]
constexpr int MAX_TERMS  = 6;

struct Gate {
    uint8_t mode; // For deg4 gates: 0 = deg2*deg2, 1 = deg1*deg3
    uint8_t l_idx[MAX_TERMS];
    uint8_t l_coef[MAX_TERMS];
    uint8_t r_idx[MAX_TERMS];
    uint8_t r_coef[MAX_TERMS];
};

struct Circuit {
    Gate gates[NUM_GATES];
    uint8_t out_coef[NUM_DEG4];
};

// -----------------------------------------------------------------------------
// Fast per-thread PRNG (xorshift64*)
// -----------------------------------------------------------------------------
__device__ __forceinline__ uint64_t xorshift64(uint64_t &state) {
    uint64_t x = state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

__device__ __forceinline__ uint32_t rand_u32(uint64_t &state, uint32_t bound) {
    return (uint32_t)(xorshift64(state) % bound);
}

__device__ __forceinline__ float rand_f32(uint64_t &state) {
    return (float)(xorshift64(state) >> 40) * (1.0f / 16777216.0f);
}

// -----------------------------------------------------------------------------
// Bit-sliced F_3 arithmetic (64 lanes per uint64_t pair: 0=(0,0), 1=(1,0), 2=(0,1))
// -----------------------------------------------------------------------------
struct F3Vec {
    uint64_t b0;
    uint64_t b1;
};

__device__ __forceinline__ F3Vec f3_add(F3Vec a, F3Vec b) {
    F3Vec r;
    r.b0 = (a.b0 & ~b.b0 & ~b.b1) | (~a.b0 & ~a.b1 & b.b0) | (a.b1 & b.b1);
    r.b1 = (a.b1 & ~b.b0 & ~b.b1) | (~a.b0 & ~a.b1 & b.b1) | (a.b0 & b.b0);
    return r;
}

__device__ __forceinline__ F3Vec f3_sub(F3Vec a, F3Vec b) {
    F3Vec neg_b = {b.b1, b.b0};
    return f3_add(a, neg_b);
}

__device__ __forceinline__ F3Vec f3_mul(F3Vec a, F3Vec b) {
    F3Vec r;
    r.b0 = (a.b0 & b.b0) | (a.b1 & b.b1);
    r.b1 = (a.b0 & b.b1) | (a.b1 & b.b0);
    return r;
}

__device__ __forceinline__ F3Vec f3_scale(F3Vec a, uint8_t c) {
    if (c == 0) return {0ULL, 0ULL};
    if (c == 1) return a;
    return {a.b1, a.b0}; // c == 2
}

// -----------------------------------------------------------------------------
// Bit-sliced F_5 arithmetic (64 lanes per 3-word tuple: binary 0..4)
// -----------------------------------------------------------------------------
struct F5Vec {
    uint64_t b0;
    uint64_t b1;
    uint64_t b2;
};

__device__ __forceinline__ F5Vec f5_add(F5Vec a, F5Vec b) {
    // 3-bit addition: s = a + b in [0..8]
    uint64_t c0 = a.b0 & b.b0;
    uint64_t s0 = a.b0 ^ b.b0;
    uint64_t c1 = (a.b1 & b.b1) | (c0 & (a.b1 ^ b.b1));
    uint64_t s1 = a.b1 ^ b.b1 ^ c0;
    uint64_t c2 = (a.b2 & b.b2) | (c1 & (a.b2 ^ b.b2));
    uint64_t s2 = a.b2 ^ b.b2 ^ c1;
    // Subtract 5 if s >= 5 (i.e. c2 == 1 or (s2 & (s1 | s0)))
    uint64_t ge5 = c2 | (s2 & (s1 | s0));
    // s - 5 = s + 3 (mod 8) when ge5 is set
    uint64_t k0 = ge5;
    uint64_t k1 = ge5;
    uint64_t rc0 = s0 & k0;
    uint64_t r0 = s0 ^ k0;
    uint64_t rc1 = (s1 & k1) | (rc0 & (s1 ^ k1));
    uint64_t r1 = s1 ^ k1 ^ rc0;
    uint64_t r2 = s2 ^ rc1;
    return {r0, r1, r2};
}

__device__ __forceinline__ F5Vec f5_mul(F5Vec a, F5Vec b) {
    // Lane-wise 3-bit multiply mod 5 using 2-bit partial products
    F5Vec acc = {0ULL, 0ULL, 0ULL};
    // b0 * a
    F5Vec t0 = {a.b0 & b.b0, a.b1 & b.b0, a.b2 & b.b0};
    acc = f5_add(acc, t0);
    // 2 * b1 * a = (a + a) masked by b1
    F5Vec a2 = f5_add(a, a);
    F5Vec t1 = {a2.b0 & b.b1, a2.b1 & b.b1, a2.b2 & b.b1};
    acc = f5_add(acc, t1);
    // 4 * b2 * a = (a2 + a2) masked by b2
    F5Vec a4 = f5_add(a2, a2);
    F5Vec t2 = {a4.b0 & b.b2, a4.b1 & b.b2, a4.b2 & b.b2};
    acc = f5_add(acc, t2);
    return acc;
}

__device__ __forceinline__ F5Vec f5_scale(F5Vec a, uint8_t c) {
    if (c == 0) return {0ULL, 0ULL, 0ULL};
    if (c == 1) return a;
    F5Vec a2 = f5_add(a, a);
    if (c == 2) return a2;
    if (c == 3) return f5_add(a2, a);
    return f5_add(a2, a2); // c == 4
}

// -----------------------------------------------------------------------------
// Circuit evaluation (templated on MOD = 3 or 5)
// -----------------------------------------------------------------------------
template <int MOD>
__device__ int eval_circuit(const Circuit &c,
                            const uint64_t * __restrict__ x_words,
                            const uint64_t * __restrict__ target_words);

template <>
__device__ int eval_circuit<3>(const Circuit &c,
                               const uint64_t * __restrict__ x_words,
                               const uint64_t * __restrict__ target_words) {
    F3Vec x[NUM_INPUTS];
    for (int i = 0; i < NUM_INPUTS; ++i) {
        x[i] = {x_words[2 * i], x_words[2 * i + 1]};
    }
    F3Vec g[NUM_GATES];

    // Layer A: deg 2 (g[0..7])
    for (int k = 0; k < NUM_DEG2; ++k) {
        F3Vec L = {0ULL, 0ULL}, R = {0ULL, 0ULL};
        for (int t = 0; t < MAX_TERMS; ++t) {
            if (c.gates[k].l_coef[t]) L = f3_add(L, f3_scale(x[c.gates[k].l_idx[t]], c.gates[k].l_coef[t]));
            if (c.gates[k].r_coef[t]) R = f3_add(R, f3_scale(x[c.gates[k].r_idx[t]], c.gates[k].r_coef[t]));
        }
        g[k] = f3_mul(L, R);
    }

    // Layer B: deg 3 (g[8..11]) = (lin of x) * (lin of g[0..7])
    for (int k = NUM_DEG2; k < NUM_DEG2 + NUM_DEG3; ++k) {
        F3Vec L = {0ULL, 0ULL}, R = {0ULL, 0ULL};
        for (int t = 0; t < MAX_TERMS; ++t) {
            if (c.gates[k].l_coef[t]) L = f3_add(L, f3_scale(x[c.gates[k].l_idx[t]], c.gates[k].l_coef[t]));
            if (c.gates[k].r_coef[t]) R = f3_add(R, f3_scale(g[c.gates[k].r_idx[t]], c.gates[k].r_coef[t]));
        }
        g[k] = f3_mul(L, R);
    }

    // Layer C: deg 4 (g[12..16]) mixing deg2*deg2 (mode 0) and deg1*deg3 (mode 1)
    F3Vec out = {0ULL, 0ULL};
    for (int k = NUM_DEG2 + NUM_DEG3; k < NUM_GATES; ++k) {
        F3Vec L = {0ULL, 0ULL}, R = {0ULL, 0ULL};
        if (c.gates[k].mode == 0) {
            for (int t = 0; t < MAX_TERMS; ++t) {
                if (c.gates[k].l_coef[t]) L = f3_add(L, f3_scale(g[c.gates[k].l_idx[t]], c.gates[k].l_coef[t]));
                if (c.gates[k].r_coef[t]) R = f3_add(R, f3_scale(g[c.gates[k].r_idx[t]], c.gates[k].r_coef[t]));
            }
        } else {
            for (int t = 0; t < MAX_TERMS; ++t) {
                if (c.gates[k].l_coef[t]) L = f3_add(L, f3_scale(x[c.gates[k].l_idx[t]], c.gates[k].l_coef[t]));
                if (c.gates[k].r_coef[t]) R = f3_add(R, f3_scale(g[NUM_DEG2 + c.gates[k].r_idx[t]], c.gates[k].r_coef[t]));
            }
        }
        g[k] = f3_mul(L, R);
        if (c.out_coef[k - (NUM_DEG2 + NUM_DEG3)]) {
            out = f3_add(out, f3_scale(g[k], c.out_coef[k - (NUM_DEG2 + NUM_DEG3)]));
        }
    }

    uint64_t diff = (out.b0 ^ target_words[0]) | (out.b1 ^ target_words[1]);
    return __popcll(diff);
}

template <>
__device__ int eval_circuit<5>(const Circuit &c,
                               const uint64_t * __restrict__ x_words,
                               const uint64_t * __restrict__ target_words) {
    F5Vec x[NUM_INPUTS];
    for (int i = 0; i < NUM_INPUTS; ++i) {
        x[i] = {x_words[3 * i], x_words[3 * i + 1], x_words[3 * i + 2]};
    }
    F5Vec g[NUM_GATES];

    for (int k = 0; k < NUM_DEG2; ++k) {
        F5Vec L = {0ULL, 0ULL, 0ULL}, R = {0ULL, 0ULL, 0ULL};
        for (int t = 0; t < MAX_TERMS; ++t) {
            if (c.gates[k].l_coef[t]) L = f5_add(L, f5_scale(x[c.gates[k].l_idx[t]], c.gates[k].l_coef[t]));
            if (c.gates[k].r_coef[t]) R = f5_add(R, f5_scale(x[c.gates[k].r_idx[t]], c.gates[k].r_coef[t]));
        }
        g[k] = f5_mul(L, R);
    }

    for (int k = NUM_DEG2; k < NUM_DEG2 + NUM_DEG3; ++k) {
        F5Vec L = {0ULL, 0ULL, 0ULL}, R = {0ULL, 0ULL, 0ULL};
        for (int t = 0; t < MAX_TERMS; ++t) {
            if (c.gates[k].l_coef[t]) L = f5_add(L, f5_scale(x[c.gates[k].l_idx[t]], c.gates[k].l_coef[t]));
            if (c.gates[k].r_coef[t]) R = f5_add(R, f5_scale(g[c.gates[k].r_idx[t]], c.gates[k].r_coef[t]));
        }
        g[k] = f5_mul(L, R);
    }

    F5Vec out = {0ULL, 0ULL, 0ULL};
    for (int k = NUM_DEG2 + NUM_DEG3; k < NUM_GATES; ++k) {
        F5Vec L = {0ULL, 0ULL, 0ULL}, R = {0ULL, 0ULL, 0ULL};
        if (c.gates[k].mode == 0) {
            for (int t = 0; t < MAX_TERMS; ++t) {
                if (c.gates[k].l_coef[t]) L = f5_add(L, f5_scale(g[c.gates[k].l_idx[t]], c.gates[k].l_coef[t]));
                if (c.gates[k].r_coef[t]) R = f5_add(R, f5_scale(g[c.gates[k].r_idx[t]], c.gates[k].r_coef[t]));
            }
        } else {
            for (int t = 0; t < MAX_TERMS; ++t) {
                if (c.gates[k].l_coef[t]) L = f5_add(L, f5_scale(x[c.gates[k].l_idx[t]], c.gates[k].l_coef[t]));
                if (c.gates[k].r_coef[t]) R = f5_add(R, f5_scale(g[NUM_DEG2 + c.gates[k].r_idx[t]], c.gates[k].r_coef[t]));
            }
        }
        g[k] = f5_mul(L, R);
        if (c.out_coef[k - (NUM_DEG2 + NUM_DEG3)]) {
            out = f5_add(out, f5_scale(g[k], c.out_coef[k - (NUM_DEG2 + NUM_DEG3)]));
        }
    }

    uint64_t diff = (out.b0 ^ target_words[0]) | (out.b1 ^ target_words[1]) | (out.b2 ^ target_words[2]);
    return __popcll(diff);
}

// -----------------------------------------------------------------------------
// Circuit initialization & mutation
// -----------------------------------------------------------------------------
__device__ void init_circuit(Circuit &c, uint64_t &rng, int mod) {
    for (int k = 0; k < NUM_GATES; ++k) {
        c.gates[k].mode = (k >= NUM_DEG2 + NUM_DEG3) ? (uint8_t)rand_u32(rng, 2) : 0;
        int l_bound = (k < NUM_DEG2 + NUM_DEG3) ? NUM_INPUTS
                      : (c.gates[k].mode == 0 ? NUM_DEG2 : NUM_INPUTS);
        int r_bound = (k < NUM_DEG2) ? NUM_INPUTS
                      : (k < NUM_DEG2 + NUM_DEG3) ? NUM_DEG2
                      : (c.gates[k].mode == 0 ? NUM_DEG2 : NUM_DEG3);
        for (int t = 0; t < MAX_TERMS; ++t) {
            c.gates[k].l_idx[t]  = (uint8_t)rand_u32(rng, l_bound);
            c.gates[k].l_coef[t] = (uint8_t)(t < 2 ? 1 + rand_u32(rng, mod - 1) : 0);
            c.gates[k].r_idx[t]  = (uint8_t)rand_u32(rng, r_bound);
            c.gates[k].r_coef[t] = (uint8_t)(t < 2 ? 1 + rand_u32(rng, mod - 1) : 0);
        }
    }
    for (int i = 0; i < NUM_DEG4; ++i) {
        c.out_coef[i] = (uint8_t)(1 + rand_u32(rng, mod - 1));
    }
}

__device__ void mutate_circuit(Circuit &c, uint64_t &rng, int mod) {
    uint32_t kind = rand_u32(rng, 100);
    if (kind < 10) {
        // Mutate output coefficient
        int idx = rand_u32(rng, NUM_DEG4);
        c.out_coef[idx] = (uint8_t)rand_u32(rng, mod);
        return;
    }
    int k = rand_u32(rng, NUM_GATES);
    if (kind < 15 && k >= NUM_DEG2 + NUM_DEG3) {
        // Flip deg4 gate between deg2*deg2 and deg1*deg3
        c.gates[k].mode ^= 1;
        int l_bound = (c.gates[k].mode == 0) ? NUM_DEG2 : NUM_INPUTS;
        int r_bound = (c.gates[k].mode == 0) ? NUM_DEG2 : NUM_DEG3;
        for (int t = 0; t < MAX_TERMS; ++t) {
            c.gates[k].l_idx[t] %= l_bound;
            c.gates[k].r_idx[t] %= r_bound;
        }
        return;
    }
    int t = rand_u32(rng, MAX_TERMS);
    bool left = (rand_u32(rng, 2) == 0);
    int bound = left
        ? ((k < NUM_DEG2 + NUM_DEG3) ? NUM_INPUTS : (c.gates[k].mode == 0 ? NUM_DEG2 : NUM_INPUTS))
        : ((k < NUM_DEG2) ? NUM_INPUTS : (k < NUM_DEG2 + NUM_DEG3) ? NUM_DEG2 : (c.gates[k].mode == 0 ? NUM_DEG2 : NUM_DEG3));

    if (left) {
        c.gates[k].l_idx[t]  = (uint8_t)rand_u32(rng, bound);
        c.gates[k].l_coef[t] = (uint8_t)rand_u32(rng, mod);
    } else {
        c.gates[k].r_idx[t]  = (uint8_t)rand_u32(rng, bound);
        c.gates[k].r_coef[t] = (uint8_t)rand_u32(rng, mod);
    }
}

// -----------------------------------------------------------------------------
// Search kernel
// -----------------------------------------------------------------------------
template <int MOD>
__global__ void perm4_sa_kernel(const uint64_t * __restrict__ x_words,
                                const uint64_t * __restrict__ target_words,
                                int iters,
                                uint64_t seed_base,
                                int * __restrict__ global_best_loss,
                                Circuit * __restrict__ global_best_circuit) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t rng = seed_base ^ ((uint64_t)(tid + 1) * 0x9E3779B97F4A7C15ULL);

    Circuit cur, best;
    init_circuit(cur, rng, MOD);
    int cur_loss = eval_circuit<MOD>(cur, x_words, target_words);
    best = cur;
    int best_loss = cur_loss;

    const float t_start = 4.0f;
    const float t_end   = 0.05f;
    const float inv_iters = 1.0f / (float)iters;

    for (int it = 0; it < iters; ++it) {
        Circuit cand = cur;
        mutate_circuit(cand, rng, MOD);
        int cand_loss = eval_circuit<MOD>(cand, x_words, target_words);
        int delta = cand_loss - cur_loss;

        float temp = t_start * powf(t_end / t_start, (float)it * inv_iters);
        if (delta <= 0 || rand_f32(rng) < expf(-(float)delta / temp)) {
            cur = cand;
            cur_loss = cand_loss;
            if (cur_loss < best_loss) {
                best = cur;
                best_loss = cur_loss;
                if (best_loss == 0) break;
            }
        }
    }

    int old = atomicMin(global_best_loss, best_loss);
    if (best_loss < old) {
        *global_best_circuit = best;
    }
}

// -----------------------------------------------------------------------------
// Host-side exact Perm_4 mod p & bit-sliced test-matrix generator
// -----------------------------------------------------------------------------
static int perm4_mod(const int m[16], int mod) {
    static const int perms[24][4] = {
        {0,1,2,3},{0,1,3,2},{0,2,1,3},{0,2,3,1},{0,3,1,2},{0,3,2,1},
        {1,0,2,3},{1,0,3,2},{1,2,0,3},{1,2,3,0},{1,3,0,2},{1,3,2,0},
        {2,0,1,3},{2,0,3,1},{2,1,0,3},{2,1,3,0},{2,3,0,1},{2,3,1,0},
        {3,0,1,2},{3,0,2,1},{3,1,0,2},{3,1,2,0},{3,2,0,1},{3,2,1,0}
    };
    int sum = 0;
    for (int p = 0; p < 24; ++p) {
        int prod = 1;
        for (int r = 0; r < 4; ++r) {
            prod = (prod * m[4 * r + perms[p][r]]) % mod;
        }
        sum = (sum + prod) % mod;
    }
    return sum;
}

static void generate_test_batch(int mod, uint64_t *x_words, uint64_t *target_words, uint64_t seed) {
    int words_per_val = (mod == 3) ? 2 : 3;
    memset(x_words, 0, NUM_INPUTS * words_per_val * sizeof(uint64_t));
    memset(target_words, 0, words_per_val * sizeof(uint64_t));

    uint64_t s = seed;
    for (int lane = 0; lane < 64; ++lane) {
        int mat[16];
        for (int i = 0; i < 16; ++i) {
            s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
            mat[i] = (int)((s * 0x2545F4914F6CDD1DULL) % mod);
            for (int b = 0; b < words_per_val; ++b) {
                if ((mat[i] >> b) & 1) {
                    x_words[words_per_val * i + b] |= (1ULL << lane);
                }
            }
        }
        int p4 = perm4_mod(mat, mod);
        for (int b = 0; b < words_per_val; ++b) {
            if ((p4 >> b) & 1) {
                target_words[b] |= (1ULL << lane);
            }
        }
    }
}

int main(int argc, char **argv) {
    int mod     = 3;
    int iters   = 100000;
    int blocks  = 256;
    int threads = 256;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--mod") == 0 && i + 1 < argc) mod = atoi(argv[++i]);
        else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) iters = atoi(argv[++i]);
        else if (strcmp(argv[i], "--blocks") == 0 && i + 1 < argc) blocks = atoi(argv[++i]);
        else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) threads = atoi(argv[++i]);
    }

    if (mod != 3 && mod != 5) {
        fprintf(stderr, "Error: --mod must be 3 or 5\n");
        return EXIT_FAILURE;
    }

    int words_per_val = (mod == 3) ? 2 : 3;
    uint64_t h_x[NUM_INPUTS * 3];
    uint64_t h_target[3];
    generate_test_batch(mod, h_x, h_target, (uint64_t)time(nullptr) ^ 0xCAFEBABE12345678ULL);

    uint64_t *d_x = nullptr, *d_target = nullptr;
    int *d_best_loss = nullptr;
    Circuit *d_best_circuit = nullptr;

    CUDA_CHECK(cudaMalloc(&d_x, NUM_INPUTS * words_per_val * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_target, words_per_val * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_best_loss, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_best_circuit, sizeof(Circuit)));

    CUDA_CHECK(cudaMemcpy(d_x, h_x, NUM_INPUTS * words_per_val * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_target, h_target, words_per_val * sizeof(uint64_t), cudaMemcpyHostToDevice));

    int init_loss = 65;
    CUDA_CHECK(cudaMemcpy(d_best_loss, &init_loss, sizeof(int), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    printf("Starting Perm_4 17-mul bit-sliced search (F_%d): %d blocks x %d threads x %d iters\n",
           mod, blocks, threads, iters);

    CUDA_CHECK(cudaEventRecord(start));
    if (mod == 3) {
        perm4_sa_kernel<3><<<blocks, threads>>>(d_x, d_target, iters, (uint64_t)time(nullptr),
                                                d_best_loss, d_best_circuit);
    } else {
        perm4_sa_kernel<5><<<blocks, threads>>>(d_x, d_target, iters, (uint64_t)time(nullptr),
                                                d_best_loss, d_best_circuit);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    int best_loss = 65;
    Circuit best_circuit;
    CUDA_CHECK(cudaMemcpy(&best_loss, d_best_loss, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&best_circuit, d_best_circuit, sizeof(Circuit), cudaMemcpyDeviceToHost));

    double total_evals = (double)blocks * (double)threads * (double)iters;
    double evals_per_sec = total_evals / (ms * 1e-3);

    printf("Done in %.2f ms | %.2f M circuits/sec | Best loss: %d / 64\n",
           ms, evals_per_sec / 1e6, best_loss);

    if (best_loss == 0) {
        printf("FOUND 0-loss candidate on 64-matrix F_%d batch!\n", mod);
        printf("Output coeffs: [%d, %d, %d, %d, %d]\n",
               best_circuit.out_coef[0], best_circuit.out_coef[1],
               best_circuit.out_coef[2], best_circuit.out_coef[3],
               best_circuit.out_coef[4]);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_target));
    CUDA_CHECK(cudaFree(d_best_loss));
    CUDA_CHECK(cudaFree(d_best_circuit));
    return 0;
}
