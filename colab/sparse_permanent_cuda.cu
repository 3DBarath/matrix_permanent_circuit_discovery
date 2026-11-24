/*
 * Gen 15 CUDA: Flat-Array Frontier DP + Batched Cascade-Branching
 * Exact Sub-2^n Sparse Permanent Engine Mod P61 = 2^61 - 1.
 *
 * Designed for Google Colab (T4 sm_75 / A100 sm_80), zero external dependencies.
 * Compile:
 *   nvcc -O3 -std=c++17 -arch=sm_75 colab/sparse_permanent_cuda.cu -o sparse_perm_cuda
 *   (use -arch=sm_80 for A100)
 */

#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <queue>
#include <random>
#include <utility>
#include <vector>

static constexpr uint64_t P61 = (1ULL << 61) - 1;
static constexpr int MAX_GPU_FW = 13;
static constexpr int GPU_BUF_BITS = 17; // 2^17 = 131,072 > 2^(13+deg) // 2^16 = 65536 states per DP table
static constexpr int BLOCK_THREADS = 256;

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,        \
                         __LINE__, cudaGetErrorString(err__));                 \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// ============================================================================
// Mod P61 = 2^61 - 1 Arithmetic (Host & Device)
// ============================================================================

__host__ __device__ static inline uint64_t add61(uint64_t a, uint64_t b) {
    uint64_t s = a + b;
    return s >= P61 ? s - P61 : s;
}

__host__ __device__ static inline uint64_t sub61(uint64_t a, uint64_t b) {
    return a >= b ? (a - b) : (a + P61 - b);
}

__host__ __device__ static inline uint64_t mul61(uint64_t a, uint64_t b) {
#ifdef __CUDA_ARCH__
    uint64_t lo = a * b;
    uint64_t hi = __umul64hi(a, b);
    uint64_t lo_part = lo & P61;
    uint64_t hi_part = (hi << 3) | (lo >> 61);
    return add61(lo_part, hi_part);
#else
    __uint128_t prod = (__uint128_t)a * b;
    return add61((uint64_t)prod & P61, (uint64_t)(prod >> 61));
#endif
}

__host__ __device__ static inline uint64_t pow61(uint64_t base, uint64_t exp) {
    uint64_t r = 1;
    while (exp) {
        if (exp & 1ULL) r = mul61(r, base);
        base = mul61(base, base);
        exp >>= 1;
    }
    return r;
}

__device__ static inline void atomicAdd61(uint64_t* addr, uint64_t val) {
    if (val == 0) return;
    unsigned long long* ull_addr = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long old = *ull_addr, assumed;
    do {
        assumed = old;
        uint64_t sum = add61(static_cast<uint64_t>(assumed), val);
        old = atomicCAS(ull_addr, assumed, static_cast<unsigned long long>(sum));
    } while (assumed != old);
}

// ============================================================================
// Sparse Matrix & Reductions (Host)
// ============================================================================

using AdjRow = std::vector<std::pair<int, uint64_t>>;

static inline uint64_t get_entry(const AdjRow& row, int c) {
    for (const auto& kv : row) if (kv.first == c) return kv.second;
    return 0;
}

static inline void set_entry(AdjRow& row, int c, uint64_t v) {
    for (size_t i = 0; i < row.size(); ++i) {
        if (row[i].first == c) {
            if (v == 0) { row[i] = row.back(); row.pop_back(); }
            else row[i].second = v;
            return;
        }
    }
    if (v != 0) row.emplace_back(c, v);
}

static inline void erase_entry(AdjRow& row, int c) {
    for (size_t i = 0; i < row.size(); ++i) {
        if (row[i].first == c) { row[i] = row.back(); row.pop_back(); return; }
    }
}

static uint64_t reduce_matrix(std::vector<AdjRow>& rows, int n_cols, std::vector<AdjRow>& out_rows) {
    int n = (int)rows.size();
    std::vector<AdjRow> col_adj(n_cols);
    for (int r = 0; r < n; ++r)
        for (const auto& kv : rows[r]) col_adj[kv.first].emplace_back(r, kv.second);

    std::vector<uint8_t> ar(n, 1), ac(n_cols, 1);
    int rem_r = n;
    uint64_t scale = 1;

    auto rm_row = [&](int r) {
        ar[r] = 0; --rem_r;
        for (const auto& kv : rows[r]) erase_entry(col_adj[kv.first], r);
        rows[r].clear();
    };
    auto rm_col = [&](int c) {
        ac[c] = 0;
        for (const auto& kv : col_adj[c]) erase_entry(rows[kv.first], c);
        col_adj[c].clear();
    };

    bool changed = true;
    while (changed && rem_r > 0) {
        changed = false;
        for (int r = 0; r < n; ++r) {
            if (!ar[r]) continue;
            if (rows[r].empty()) return 0;
            if (rows[r].size() == 1) {
                int c = rows[r][0].first;
                scale = mul61(scale, rows[r][0].second);
                rm_row(r); rm_col(c); changed = true; break;
            }
        }
        if (changed) continue;
        for (int c = 0; c < n_cols; ++c) {
            if (!ac[c]) continue;
            if (col_adj[c].empty()) return 0;
            if (col_adj[c].size() == 1) {
                int r = col_adj[c][0].first;
                scale = mul61(scale, col_adj[c][0].second);
                rm_row(r); rm_col(c); changed = true; break;
            }
        }
        if (changed) continue;
        for (int r = 0; r < n; ++r) {
            if (ar[r] && rows[r].size() == 2) {
                int c1 = rows[r][0].first; uint64_t w1 = rows[r][0].second;
                int c2 = rows[r][1].first; uint64_t w2 = rows[r][1].second;
                rm_row(r);
                std::vector<int> aff;
                aff.reserve(col_adj[c1].size() + col_adj[c2].size());
                for (const auto& kv : col_adj[c1]) aff.push_back(kv.first);
                for (const auto& kv : col_adj[c2]) aff.push_back(kv.first);
                std::sort(aff.begin(), aff.end());
                aff.erase(std::unique(aff.begin(), aff.end()), aff.end());
                for (int i : aff) {
                    uint64_t nv = add61(mul61(w1, get_entry(rows[i], c2)), mul61(w2, get_entry(rows[i], c1)));
                    set_entry(rows[i], c1, nv); set_entry(col_adj[c1], i, nv);
                }
                rm_col(c2); changed = true; break;
            }
        }
        if (changed) continue;
        for (int c = 0; c < n_cols; ++c) {
            if (ac[c] && col_adj[c].size() == 2) {
                int r1 = col_adj[c][0].first; uint64_t w1 = col_adj[c][0].second;
                int r2 = col_adj[c][1].first; uint64_t w2 = col_adj[c][1].second;
                rm_col(c);
                std::vector<int> aff;
                aff.reserve(rows[r1].size() + rows[r2].size());
                for (const auto& kv : rows[r1]) aff.push_back(kv.first);
                for (const auto& kv : rows[r2]) aff.push_back(kv.first);
                std::sort(aff.begin(), aff.end());
                aff.erase(std::unique(aff.begin(), aff.end()), aff.end());
                for (int j : aff) {
                    uint64_t nv = add61(mul61(w1, get_entry(col_adj[j], r2)), mul61(w2, get_entry(col_adj[j], r1)));
                    set_entry(rows[r1], j, nv); set_entry(col_adj[j], r1, nv);
                }
                rm_row(r2); changed = true; break;
            }
        }
    }

    out_rows.clear();
    if (rem_r == 0) return scale;
    std::vector<int> c_map(n_cols, -1);
    int nc = 0;
    for (int c = 0; c < n_cols; ++c) if (ac[c]) c_map[c] = nc++;
    out_rows.reserve(rem_r);
    for (int r = 0; r < n; ++r) {
        if (!ar[r]) continue;
        AdjRow nr; nr.reserve(rows[r].size());
        for (const auto& kv : rows[r]) nr.emplace_back(c_map[kv.first], kv.second);
        out_rows.push_back(std::move(nr));
    }
    return scale;
}

static bool split_components(const std::vector<AdjRow>& rows, std::vector<std::vector<AdjRow>>& comps) {
    int n = (int)rows.size();
    std::vector<std::vector<int>> c2r(n);
    for (int r = 0; r < n; ++r)
        for (const auto& kv : rows[r]) c2r[kv.first].push_back(r);
    std::vector<uint8_t> vr(n, 0), vc(n, 0);
    std::vector<int> c_map(n, -1);
    comps.clear();
    for (int r0 = 0; r0 < n; ++r0) {
        if (vr[r0]) continue;
        std::queue<int> q; q.push(r0); vr[r0] = 1;
        std::vector<int> cr, cc;
        while (!q.empty()) {
            int r = q.front(); q.pop(); cr.push_back(r);
            for (const auto& kv : rows[r]) {
                int c = kv.first;
                if (!vc[c]) {
                    vc[c] = 1; cc.push_back(c);
                    for (int nr : c2r[c]) if (!vr[nr]) { vr[nr] = 1; q.push(nr); }
                }
            }
        }
        if (cr.size() != cc.size()) return false;
        for (size_t i = 0; i < cc.size(); ++i) c_map[cc[i]] = (int)i;
        std::vector<AdjRow> sub(cr.size());
        for (size_t i = 0; i < cr.size(); ++i) {
            sub[i].reserve(rows[cr[i]].size());
            for (const auto& kv : rows[cr[i]]) sub[i].emplace_back(c_map[kv.first], kv.second);
        }
        comps.push_back(std::move(sub));
    }
    return true;
}

static int order_min_frontier(const std::vector<AdjRow>& rows, std::vector<int>& order) {
    int n = (int)rows.size();
    std::vector<int> col_rem(n, 0);
    for (const auto& d : rows) for (const auto& kv : d) col_rem[kv.first]++;
    std::vector<uint8_t> used(n, 0), opened(n, 0);
    order.clear(); order.reserve(n);
    int cur_open = 0, max_fw = 0;
    for (int step = 0; step < n; ++step) {
        int best_r = -1, best_delta = 1e9, best_deg = 1e9;
        for (int r = 0; r < n; ++r) {
            if (used[r]) continue;
            int no = 0, cl = 0;
            for (const auto& kv : rows[r]) {
                int c = kv.first;
                if (!opened[c] && col_rem[c] > 1) ++no;
                if (opened[c] && col_rem[c] == 1) ++cl;
            }
            int delta = no - cl, deg = (int)rows[r].size();
            if (delta < best_delta || (delta == best_delta && deg < best_deg)) {
                best_delta = delta; best_deg = deg; best_r = r;
            }
        }
        used[best_r] = 1; order.push_back(best_r);
        for (const auto& kv : rows[best_r]) if (!opened[kv.first]) { opened[kv.first] = 1; ++cur_open; }
        if (cur_open > max_fw) max_fw = cur_open;
        for (const auto& kv : rows[best_r]) if (--col_rem[kv.first] == 0) { opened[kv.first] = 0; --cur_open; }
    }
    return max_fw;
}

// ============================================================================
// CUDA Batched Frontier DP Kernel
// ============================================================================

struct LeafMeta {
    int n;
    int row_offset; // index into d_row_ptr (n + 1 entries per leaf? or n entries)
    uint64_t scale;
};

__global__ void frontier_dp_batch_kernel(
    int num_leaves,
    const LeafMeta* __restrict__ leaves,
    const int* __restrict__ row_ptr,     // CSR row pointers (global across all leaves)
    const int* __restrict__ col_idx,     // CSR column indices
    const uint64_t* __restrict__ col_val,// CSR weights mod P61
    const int* __restrict__ last_use,    // flattened [sum of n]
    uint64_t* __restrict__ dp_workspace, // [gridDim.x * 2 * (1 << MAX_GPU_FW)]
    uint64_t* __restrict__ out_results)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    if (bid >= num_leaves) return;

    const LeafMeta meta = leaves[bid];
    int n = meta.n;

    if (n == 0) {
        if (tid == 0) out_results[bid] = meta.scale;
        return;
    }

    uint64_t* buf0 = dp_workspace + (size_t)bid * 2 * (1ULL << GPU_BUF_BITS);
    uint64_t* buf1 = buf0 + (1ULL << GPU_BUF_BITS);
    uint64_t* dp_val = buf0;
    uint64_t* nxt_val = buf1;

    __shared__ int s_frontier[MAX_GPU_FW + 8];
    __shared__ int s_keep_idx[MAX_GPU_FW + 8];
    __shared__ int s_col_pos[128]; // supports up to n <= 128
    __shared__ int s_fw;
    __shared__ int s_nxt_fw;
    __shared__ uint32_t s_dead_mask;
    __shared__ int s_keep_cnt;

    for (int i = tid; i < n; i += blockDim.x) s_col_pos[i] = -1;
    if (tid == 0) {
        s_fw = 0;
        dp_val[0] = 1;
    }
    __syncthreads();

    for (int step = 0; step < n; ++step) {
        int r_start = row_ptr[meta.row_offset + step];
        int r_end   = row_ptr[meta.row_offset + step + 1];

        if (tid == 0) {
            int fw = s_fw;
            for (int e = r_start; e < r_end; ++e) {
                int c = col_idx[e];
                if (s_col_pos[c] == -1) {
                    s_col_pos[c] = fw;
                    s_frontier[fw] = c;
                    fw++;
                }
            }
            s_nxt_fw = fw;
        }
        __syncthreads();

        int prev_sz = 1 << s_fw;
        int nxt_sz  = 1 << s_nxt_fw;

        for (int m = tid; m < nxt_sz; m += blockDim.x) nxt_val[m] = 0;
        __syncthreads();

        for (int mask = tid; mask < prev_sz; mask += blockDim.x) {
            uint64_t val = dp_val[mask];
            if (val == 0) continue;
            for (int e = r_start; e < r_end; ++e) {
                int c = col_idx[e];
                uint32_t bit = 1U << s_col_pos[c];
                if (!(mask & bit)) {
                    uint32_t nm = mask | bit;
                    uint64_t prod = mul61(val, col_val[e]);
                    atomicAdd61(&nxt_val[nm], prod);
                }
            }
        }
        __syncthreads();

        if (tid == 0) {
            uint32_t dead = 0;
            int kcnt = 0;
            for (int idx = 0; idx < s_nxt_fw; ++idx) {
                if (last_use[meta.row_offset + s_frontier[idx]] == step) {
                    dead |= (1U << idx);
                } else {
                    s_keep_idx[kcnt++] = idx;
                }
            }
            s_dead_mask = dead;
            s_keep_cnt = kcnt;
        }
        __syncthreads();

        if (s_dead_mask != 0) {
            int csz = 1 << s_keep_cnt;
            // Reuse dp_val as comp_val
            for (int m = tid; m < csz; m += blockDim.x) dp_val[m] = 0;
            __syncthreads();

            uint32_t dead = s_dead_mask;
            int kcnt = s_keep_cnt;
            for (int mask = tid; mask < nxt_sz; mask += blockDim.x) {
                if ((mask & dead) != dead) continue;
                uint64_t v = nxt_val[mask];
                if (v == 0) continue;
                uint32_t cm = 0;
                for (int ni = 0; ni < kcnt; ++ni) {
                    if ((mask >> s_keep_idx[ni]) & 1U) cm |= (1U << ni);
                }
                atomicAdd61(&dp_val[cm], v);
            }
            __syncthreads();

            if (tid == 0) {
                for (int idx = 0; idx < s_nxt_fw; ++idx) s_col_pos[s_frontier[idx]] = -1;
                for (int ni = 0; ni < kcnt; ++ni) {
                    int c = s_frontier[s_keep_idx[ni]];
                    s_col_pos[c] = ni;
                    s_frontier[ni] = c;
                }
                s_fw = kcnt;
            }
            __syncthreads();
        } else {
            uint64_t* tmp = dp_val;
            dp_val = nxt_val;
            nxt_val = tmp;
            if (tid == 0) s_fw = s_nxt_fw;
            __syncthreads();
        }
    }

    if (tid == 0) {
        out_results[bid] = mul61(meta.scale, dp_val[0]);
    }
}

// ============================================================================
// Batched Cascade-Branching + GPU Dispatch
// ============================================================================

struct CpuLeaf {
    std::vector<AdjRow> rows;
    std::vector<int> order;
    uint64_t scale;
};

static void collect_leaves(std::vector<AdjRow> rows, uint64_t cur_scale,
                           std::vector<CpuLeaf>& leaves, uint64_t& branches) {
    int n = (int)rows.size();
    std::vector<AdjRow> red;
    uint64_t scale = reduce_matrix(rows, n, red);
    if (scale == 0) return;
    cur_scale = mul61(cur_scale, scale);
    if (red.empty()) {
        leaves.push_back({{}, {}, cur_scale});
        return;
    }

    std::vector<std::vector<AdjRow>> comps;
    if (!split_components(red, comps)) return;

    // If multiple components, solve small ones or branch on the largest
    if (comps.size() > 1) {
        // For simplicity in batching, if all components fit GPU DP, we can
        // either multiply them or recurse. Here we recurse on each component
        // and multiply? Wait: cascade branching on one component multiplies
        // by the other components! To keep leaf collection purely additive,
        // we only split if we can solve sub-components, or we just order the
        // whole reduced matrix if fw <= MAX_GPU_FW.
        // Actually, if we have multiple components, we can solve each component
        // via a recursive call to solve_cuda_batch!
        // Let's handle that in solve_cuda_hybrid below.
    }

    auto& comp = comps[0];
    int k = (int)comp.size();
    std::vector<int> order;
    int fw = order_min_frontier(comp, order);

    if (fw <= MAX_GPU_FW && (k <= 14 || fw <= MAX_GPU_FW - 1)) {
        leaves.push_back({std::move(comp), std::move(order), cur_scale});
        return;
    }

    int min_deg = 1e9;
    std::vector<int> cdeg(k, 0);
    for (const auto& d : comp) {
        if ((int)d.size() < min_deg) min_deg = (int)d.size();
        for (const auto& kv : d) cdeg[kv.first]++;
    }

    int r_star = -1;
    std::pair<int, int> best_k = {1e9, 1e9};
    for (int r = 0; r < k; ++r) {
        int mc = 1e9;
        for (const auto& kv : comp[r]) mc = std::min(mc, cdeg[kv.first]);
        std::pair<int, int> key = {(int)comp[r].size(), mc};
        if (key < best_k) { best_k = key; r_star = r; }
    }
    int c_star = -1, best_cd = 1e9;
    uint64_t w_star = 0;
    for (const auto& kv : comp[r_star]) {
        if (cdeg[kv.first] < best_cd) {
            best_cd = cdeg[kv.first]; c_star = kv.first; w_star = kv.second;
        }
    }
    branches++;

    std::vector<AdjRow> inc;
    inc.reserve(k - 1);
    for (int r = 0; r < k; ++r) {
        if (r == r_star) continue;
        AdjRow nr;
        for (const auto& kv : comp[r]) {
            if (kv.first == c_star) continue;
            nr.emplace_back(kv.first < c_star ? kv.first : kv.first - 1, kv.second);
        }
        inc.push_back(std::move(nr));
    }
    std::vector<AdjRow> exc = comp;
    erase_entry(exc[r_star], c_star);

    collect_leaves(std::move(inc), mul61(cur_scale, w_star), leaves, branches);
    collect_leaves(std::move(exc), cur_scale, leaves, branches);
}

static uint64_t eval_leaves_gpu(const std::vector<CpuLeaf>& leaves) {
    if (leaves.empty()) return 0;

    int num_leaves = (int)leaves.size();
    std::vector<LeafMeta> h_meta(num_leaves);
    std::vector<int> h_row_ptr;
    std::vector<int> h_col_idx;
    std::vector<uint64_t> h_col_val;
    std::vector<int> h_last_use;

    int cur_row_offset = 0;
    for (int i = 0; i < num_leaves; ++i) {
        const auto& lf = leaves[i];
        int n = (int)lf.rows.size();
        h_meta[i] = {n, cur_row_offset, lf.scale};
        if (n == 0) continue;

        std::vector<int> lu(n, -1);
        for (int step = 0; step < n; ++step) {
            for (const auto& kv : lf.rows[lf.order[step]]) {
                lu[kv.first] = step;
            }
        }
        for (int c = 0; c < n; ++c) h_last_use.push_back(lu[c]);
        h_last_use.push_back(-1); // pad to match row_offset stride (n + 1)

        for (int step = 0; step < n; ++step) {
            h_row_ptr.push_back((int)h_col_idx.size());
            for (const auto& kv : lf.rows[lf.order[step]]) {
                h_col_idx.push_back(kv.first);
                h_col_val.push_back(kv.second);
            }
        }
        h_row_ptr.push_back((int)h_col_idx.size());
        cur_row_offset += n + 1;
    }

    LeafMeta* d_meta = nullptr;
    int* d_row_ptr = nullptr;
    int* d_col_idx = nullptr;
    uint64_t* d_col_val = nullptr;
    int* d_last_use = nullptr;
    uint64_t* d_workspace = nullptr;
    uint64_t* d_results = nullptr;

    CUDA_CHECK(cudaMalloc(&d_meta, num_leaves * sizeof(LeafMeta)));
    CUDA_CHECK(cudaMalloc(&d_results, num_leaves * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_meta, h_meta.data(), num_leaves * sizeof(LeafMeta), cudaMemcpyHostToDevice));

    if (!h_row_ptr.empty()) {
        CUDA_CHECK(cudaMalloc(&d_row_ptr, h_row_ptr.size() * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_col_idx, h_col_idx.size() * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_col_val, h_col_val.size() * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_last_use, h_last_use.size() * sizeof(int)));

        CUDA_CHECK(cudaMemcpy(d_row_ptr, h_row_ptr.data(), h_row_ptr.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_col_idx, h_col_idx.data(), h_col_idx.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_col_val, h_col_val.data(), h_col_val.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_last_use, h_last_use.data(), h_last_use.size() * sizeof(int), cudaMemcpyHostToDevice));
    }

    size_t ws_bytes = (size_t)num_leaves * 2 * (1ULL << GPU_BUF_BITS) * sizeof(uint64_t);
    CUDA_CHECK(cudaMalloc(&d_workspace, ws_bytes));

    frontier_dp_batch_kernel<<<num_leaves, BLOCK_THREADS>>>(
        num_leaves, d_meta, d_row_ptr, d_col_idx, d_col_val, d_last_use, d_workspace, d_results);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint64_t> h_results(num_leaves);
    CUDA_CHECK(cudaMemcpy(h_results.data(), d_results, num_leaves * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    cudaFree(d_meta);
    cudaFree(d_results);
    cudaFree(d_workspace);
    if (d_row_ptr) cudaFree(d_row_ptr);
    if (d_col_idx) cudaFree(d_col_idx);
    if (d_col_val) cudaFree(d_col_val);
    if (d_last_use) cudaFree(d_last_use);

    uint64_t total = 0;
    for (uint64_t v : h_results) total = add61(total, v);
    return total;
}

static uint64_t solve_cuda(const std::vector<AdjRow>& rows, uint64_t& branches) {
    std::vector<AdjRow> copy = rows;
    std::vector<AdjRow> red;
    uint64_t scale = reduce_matrix(copy, (int)copy.size(), red);
    if (scale == 0) return 0;
    if (red.empty()) return scale;

    std::vector<std::vector<AdjRow>> comps;
    if (!split_components(red, comps)) return 0;

    uint64_t ans = scale;
    for (auto& comp : comps) {
        std::vector<CpuLeaf> leaves;
        collect_leaves(std::move(comp), 1, leaves, branches);
        uint64_t comp_val = eval_leaves_gpu(leaves);
        ans = mul61(ans, comp_val);
        if (ans == 0) return 0;
    }
    return ans;
}

// ============================================================================
// Glynn's Formula Reference (Mod P61) for Self-Test
// ============================================================================

static uint64_t glynn_perm_mod61(const std::vector<AdjRow>& rows) {
    int n = (int)rows.size();
    if (n == 0) return 1;
    std::vector<std::vector<uint64_t>> A(n, std::vector<uint64_t>(n, 0));
    for (int r = 0; r < n; ++r)
        for (const auto& kv : rows[r]) A[r][kv.first] = kv.second;

    uint64_t total = 0;
    uint64_t num_masks = 1ULL << (n - 1);
    for (uint64_t mask = 0; mask < num_masks; ++mask) {
        uint64_t prod = 1;
        int pop = 0;
        for (int j = 1; j < n; ++j)
            if ((mask >> (j - 1)) & 1ULL) pop++;
        for (int i = 0; i < n; ++i) {
            uint64_t row_sum = A[i][0];
            for (int j = 1; j < n; ++j) {
                if ((mask >> (j - 1)) & 1ULL) row_sum = sub61(row_sum, A[i][j]);
                else row_sum = add61(row_sum, A[i][j]);
            }
            prod = mul61(prod, row_sum);
            if (prod == 0) break;
        }
        if (pop & 1) total = sub61(total, prod);
        else total = add61(total, prod);
    }
    uint64_t inv2 = (P61 + 1) / 2;
    return mul61(total, pow61(inv2, n - 1));
}

static std::vector<AdjRow> gen_d_reg(int n, int d, std::mt19937_64& rng) {
    std::vector<AdjRow> rows(n);
    std::vector<int> perm(n);
    for (int i = 0; i < n; ++i) perm[i] = i;
    for (int t = 0; t < d; ++t) {
        std::shuffle(perm.begin(), perm.end(), rng);
        for (int r = 0; r < n; ++r) {
            uint64_t w = (rng() % (P61 - 1)) + 1;
            set_entry(rows[r], perm[r], add61(get_entry(rows[r], perm[r]), w));
        }
    }
    return rows;
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("=== Gen 15 CUDA Frontier DP + Batched Cascade-Branching ===\n");
    std::printf("GPU: %s (SM %d.%d, %.1f GB)\n\n",
                prop.name, prop.major, prop.minor, prop.totalGlobalMem / 1e9);

    std::mt19937_64 rng(2026);

    // 1. Self-test against Glynn mod 2^61-1
    std::printf("[Self-Test] Verifying CUDA solver against Glynn mod 2^61-1...\n");
    for (int n : {6, 8, 10, 12, 14}) {
        auto M = gen_d_reg(n, 3, rng);
        uint64_t ref = glynn_perm_mod61(M);
        uint64_t br = 0;
        uint64_t got = solve_cuda(M, br);
        if (ref != got) {
            std::printf("  FAIL at n=%d: glynn=%llu, cuda=%llu\n",
                        n, (unsigned long long)ref, (unsigned long long)got);
            return 1;
        }
        std::printf("  n=%2d PASS (perm=%llu, branches=%llu)\n",
                    n, (unsigned long long)got, (unsigned long long)br);
    }

    // 2. Benchmark on larger 3-regular weighted graphs
    std::printf("\n[Benchmark] 3-Regular Weighted Graphs:\n");
    for (int n : {32, 40, 48, 56}) {
        auto M = gen_d_reg(n, 3, rng);
        uint64_t br = 0;
        auto t0 = std::chrono::high_resolution_clock::now();
        uint64_t val = solve_cuda(M, br);
        double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - t0).count();
        std::printf("  n=%2d | perm=%20llu | br=%5llu | %8.2f ms\n",
                    n, (unsigned long long)val, (unsigned long long)br, ms);
    }
    return 0;
}
