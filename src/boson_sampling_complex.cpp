/*
 * Gen 15 Complex Boson Sampling Sparse Permanent Engine (C++17 + OpenMP)
 * Ported from F_{2^61-1} (src/sparse_permanent.cpp) to std::complex<double>
 * with std::complex<long double> internal accumulation to prevent floating-point cancellation.
 *
 * Features:
 * 1. Iterated zero-division degree-1/2 contractions + connected-component splitting.
 * 2. Flat contiguous vector + active-mask DP for frontier <= 17.
 * 3. OpenMP task-level parallelism across cascade-branching subtrees (depth <= 5).
 * 4. Verification against Glynn's formula on shallow-depth photonic meshes up to n=16.
 * 5. Benchmarks on 3-regular complex matrices up to n=60 with permutation-invariance check.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <queue>
#include <random>
#include <utility>
#include <vector>

using cplx = std::complex<double>;
using cld  = std::complex<long double>;
using AdjRow = std::vector<std::pair<int, cld>>;

struct Stats {
    uint64_t ops = 0;
    uint64_t branches = 0;
};

static inline bool is_zero(const cld& z) {
    return z.real() == 0.0L && z.imag() == 0.0L;
}

static inline cld get_entry(const AdjRow& row, int c) {
    for (const auto& kv : row) if (kv.first == c) return kv.second;
    return cld(0.0L, 0.0L);
}

static inline void set_entry(AdjRow& row, int c, const cld& v) {
    for (size_t i = 0; i < row.size(); ++i) {
        if (row[i].first == c) {
            if (is_zero(v)) { row[i] = row.back(); row.pop_back(); }
            else row[i].second = v;
            return;
        }
    }
    if (!is_zero(v)) row.emplace_back(c, v);
}

static inline void erase_entry(AdjRow& row, int c) {
    for (size_t i = 0; i < row.size(); ++i) {
        if (row[i].first == c) { row[i] = row.back(); row.pop_back(); return; }
    }
}

static cld reduce_matrix(std::vector<AdjRow>& rows, int n_cols, std::vector<AdjRow>& out_rows) {
    int n = (int)rows.size();
    std::vector<AdjRow> col_adj(n_cols);
    for (int r = 0; r < n; ++r)
        for (const auto& kv : rows[r]) col_adj[kv.first].emplace_back(r, kv.second);

    std::vector<uint8_t> ar(n, 1), ac(n_cols, 1);
    int rem_r = n;
    cld scale(1.0L, 0.0L);

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
            if (rows[r].empty()) return cld(0.0L, 0.0L);
            if (rows[r].size() == 1) {
                int c = rows[r][0].first;
                scale *= rows[r][0].second;
                rm_row(r); rm_col(c); changed = true; break;
            }
        }
        if (changed) continue;
        for (int c = 0; c < n_cols; ++c) {
            if (!ac[c]) continue;
            if (col_adj[c].empty()) return cld(0.0L, 0.0L);
            if (col_adj[c].size() == 1) {
                int r = col_adj[c][0].first;
                scale *= col_adj[c][0].second;
                rm_row(r); rm_col(c); changed = true; break;
            }
        }
        if (changed) continue;
        for (int r = 0; r < n; ++r) {
            if (ar[r] && rows[r].size() == 2) {
                int c1 = rows[r][0].first; cld w1 = rows[r][0].second;
                int c2 = rows[r][1].first; cld w2 = rows[r][1].second;
                rm_row(r);
                std::vector<int> aff;
                aff.reserve(col_adj[c1].size() + col_adj[c2].size());
                for (const auto& kv : col_adj[c1]) aff.push_back(kv.first);
                for (const auto& kv : col_adj[c2]) aff.push_back(kv.first);
                std::sort(aff.begin(), aff.end());
                aff.erase(std::unique(aff.begin(), aff.end()), aff.end());
                for (int i : aff) {
                    cld nv = w1 * get_entry(rows[i], c2) + w2 * get_entry(rows[i], c1);
                    set_entry(rows[i], c1, nv); set_entry(col_adj[c1], i, nv);
                }
                rm_col(c2); changed = true; break;
            }
        }
        if (changed) continue;
        for (int c = 0; c < n_cols; ++c) {
            if (ac[c] && col_adj[c].size() == 2) {
                int r1 = col_adj[c][0].first; cld w1 = col_adj[c][0].second;
                int r2 = col_adj[c][1].first; cld w2 = col_adj[c][1].second;
                rm_col(c);
                std::vector<int> aff;
                aff.reserve(rows[r1].size() + rows[r2].size());
                for (const auto& kv : rows[r1]) aff.push_back(kv.first);
                for (const auto& kv : rows[r2]) aff.push_back(kv.first);
                std::sort(aff.begin(), aff.end());
                aff.erase(std::unique(aff.begin(), aff.end()), aff.end());
                for (int j : aff) {
                    cld nv = w1 * get_entry(col_adj[j], r2) + w2 * get_entry(col_adj[j], r1);
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

// Flat-array zero-hash Min-Frontier DP with long double complex accumulation
static cld frontier_dp_flat(const std::vector<AdjRow>& rows, const std::vector<int>& order, Stats& st) {
    int n = (int)rows.size();
    if (n == 0) return cld(1.0L, 0.0L);
    if (n == 1) return rows[0].empty() ? cld(0.0L, 0.0L) : rows[0][0].second;
    if (n == 2) {
        st.ops += 2;
        return get_entry(rows[0], 0) * get_entry(rows[1], 1) +
               get_entry(rows[0], 1) * get_entry(rows[1], 0);
    }

    std::vector<int> last_use(n, -1);
    for (int i = 0; i < n; ++i)
        for (const auto& kv : rows[order[i]]) last_use[kv.first] = i;

    std::vector<int> frontier;
    std::vector<int> col_pos(n, -1);
    std::vector<cld> dp_val(1, cld(1.0L, 0.0L));
    std::vector<uint32_t> dp_keys = {0};

    for (int i = 0; i < n; ++i) {
        const auto& d = rows[order[i]];
        for (const auto& kv : d) {
            if (col_pos[kv.first] == -1) {
                col_pos[kv.first] = (int)frontier.size();
                frontier.push_back(kv.first);
            }
        }
        size_t sz = 1ULL << frontier.size();
        std::vector<cld> nxt_val(sz, cld(0.0L, 0.0L));
        std::vector<uint8_t> nxt_seen(sz, 0);
        std::vector<uint32_t> nxt_keys;
        nxt_keys.reserve(dp_keys.size() * d.size());

        for (uint32_t mask : dp_keys) {
            cld val = dp_val[mask];
            if (is_zero(val)) continue;
            for (const auto& kv : d) {
                uint32_t bit = 1U << col_pos[kv.first];
                if (!(mask & bit)) {
                    uint32_t nm = mask | bit;
                    if (!nxt_seen[nm]) {
                        nxt_seen[nm] = 1;
                        nxt_keys.push_back(nm);
                    }
                    nxt_val[nm] += val * kv.second;
                    st.ops++;
                }
            }
        }

        uint32_t dead_mask = 0;
        std::vector<int> keep_idx;
        for (size_t idx = 0; idx < frontier.size(); ++idx) {
            if (last_use[frontier[idx]] == i) dead_mask |= (1U << idx);
            else keep_idx.push_back((int)idx);
        }

        if (dead_mask) {
            size_t csz = 1ULL << keep_idx.size();
            std::vector<cld> comp_val(csz, cld(0.0L, 0.0L));
            std::vector<uint8_t> comp_seen(csz, 0);
            std::vector<uint32_t> comp_keys;
            comp_keys.reserve(nxt_keys.size());
            for (uint32_t mask : nxt_keys) {
                if ((mask & dead_mask) != dead_mask) continue;
                cld v = nxt_val[mask];
                if (is_zero(v)) continue;
                uint32_t cm = 0;
                for (size_t ni = 0; ni < keep_idx.size(); ++ni)
                    if ((mask >> keep_idx[ni]) & 1U) cm |= (1U << ni);
                if (!comp_seen[cm]) {
                    comp_seen[cm] = 1;
                    comp_keys.push_back(cm);
                }
                comp_val[cm] += v;
            }
            dp_val.swap(comp_val);
            dp_keys.swap(comp_keys);
            std::vector<int> new_f;
            new_f.reserve(keep_idx.size());
            for (int c : frontier) col_pos[c] = -1;
            for (size_t ni = 0; ni < keep_idx.size(); ++ni) {
                int c = frontier[keep_idx[ni]];
                col_pos[c] = (int)ni;
                new_f.push_back(c);
            }
            frontier.swap(new_f);
        } else {
            dp_val.swap(nxt_val);
            dp_keys.swap(nxt_keys);
        }
    }
    return dp_val.empty() ? cld(0.0L, 0.0L) : dp_val[0];
}

static cld solve_hybrid(std::vector<AdjRow> rows, Stats& st, int depth) {
    int n = (int)rows.size();
    std::vector<AdjRow> red;
    cld scale = reduce_matrix(rows, n, red);
    st.ops += std::max(1, n - (int)red.size());
    if (is_zero(scale)) return cld(0.0L, 0.0L);
    if (red.empty()) return scale;

    std::vector<std::vector<AdjRow>> comps;
    if (!split_components(red, comps)) return cld(0.0L, 0.0L);

    cld ans = scale;
    for (auto& comp : comps) {
        int k = (int)comp.size();
        std::vector<int> order;
        int fw = order_min_frontier(comp, order);
        if (k <= 14 || fw <= 15) {
            ans *= frontier_dp_flat(comp, order, st);
            if (is_zero(ans)) return cld(0.0L, 0.0L);
            continue;
        }
        int min_deg = 1e9;
        std::vector<int> cdeg(k, 0);
        for (const auto& d : comp) {
            if ((int)d.size() < min_deg) min_deg = (int)d.size();
            for (const auto& kv : d) cdeg[kv.first]++;
        }
        if ((min_deg == 3 && fw > 15) || fw > 17) {
            int r_star = -1; std::pair<int,int> best_k = {1e9, 1e9};
            for (int r = 0; r < k; ++r) {
                int mc = 1e9;
                for (const auto& kv : comp[r]) mc = std::min(mc, cdeg[kv.first]);
                std::pair<int,int> key = {(int)comp[r].size(), mc};
                if (key < best_k) { best_k = key; r_star = r; }
            }
            int c_star = -1, best_cd = 1e9;
            cld w_star(0.0L, 0.0L);
            for (const auto& kv : comp[r_star]) {
                if (cdeg[kv.first] < best_cd) {
                    best_cd = cdeg[kv.first]; c_star = kv.first; w_star = kv.second;
                }
            }
            st.branches++;

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

            cld v_inc(0.0L, 0.0L), v_exc(0.0L, 0.0L);
            if (depth <= 5 && k >= 22) {
                Stats st_inc, st_exc;
                #pragma omp task shared(v_inc, st_inc) firstprivate(inc)
                { v_inc = w_star * solve_hybrid(std::move(inc), st_inc, depth + 1); }
                #pragma omp task shared(v_exc, st_exc) firstprivate(exc)
                { v_exc = solve_hybrid(std::move(exc), st_exc, depth + 1); }
                #pragma omp taskwait
                st.ops += st_inc.ops + st_exc.ops;
                st.branches += st_inc.branches + st_exc.branches;
            } else {
                v_inc = w_star * solve_hybrid(std::move(inc), st, depth + 1);
                v_exc = solve_hybrid(std::move(exc), st, depth + 1);
            }

            ans *= (v_inc + v_exc);
            if (is_zero(ans)) return cld(0.0L, 0.0L);
        } else {
            ans *= frontier_dp_flat(comp, order, st);
            if (is_zero(ans)) return cld(0.0L, 0.0L);
        }
    }
    return ans;
}

static cplx solve_parallel(const std::vector<AdjRow>& rows, Stats& st) {
    cld res(0.0L, 0.0L);
    #pragma omp parallel
    {
        #pragma omp single
        { res = solve_hybrid(rows, st, 0); }
    }
    return cplx((double)res.real(), (double)res.imag());
}

// Glynn's formula for dense/small n <= 20 verification
static cplx glynn_permanent(const std::vector<std::vector<cplx>>& A) {
    int n = (int)A.size();
    if (n == 0) return cplx(1.0, 0.0);
    uint64_t total = 1ULL << (n - 1);
    long double sum_re = 0.0L, sum_im = 0.0L;
    #pragma omp parallel for reduction(+:sum_re,sum_im) if(n >= 12)
    for (uint64_t mask = 0; mask < total; ++mask) {
        // delta_0 = +1, delta_i = +1 if bit (i-1) is 0 else -1
        int parity = __builtin_popcountll(mask) & 1;
        cld prod(parity ? -1.0L : 1.0L, 0.0L);
        for (int j = 0; j < n; ++j) {
            cld col_sum(A[0][j].real(), A[0][j].imag());
            for (int i = 1; i < n; ++i) {
                cld a_ij(A[i][j].real(), A[i][j].imag());
                if ((mask >> (i - 1)) & 1ULL) col_sum -= a_ij;
                else col_sum += a_ij;
            }
            prod *= col_sum;
        }
        sum_re += prod.real();
        sum_im += prod.imag();
    }
    long double scale = std::ldexp(1.0L, -(n - 1));
    return cplx((double)(sum_re * scale), (double)(sum_im * scale));
}

// Generate a shallow-depth photonic mesh (unitary matrix) of size n x n with `depth` layers
static std::vector<std::vector<cplx>> gen_photonic_mesh(int n, int depth, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> u_ang(0.0, 2.0 * M_PI);
    std::vector<std::vector<cplx>> U(n, std::vector<cplx>(n, cplx(0.0, 0.0)));
    for (int i = 0; i < n; ++i) U[i][i] = cplx(1.0, 0.0);

    for (int d = 0; d < depth; ++d) {
        int offset = d & 1;
        for (int i = offset; i + 1 < n; i += 2) {
            double theta = u_ang(rng);
            double phi   = u_ang(rng);
            double psi   = u_ang(rng);
            double c = std::cos(theta), s = std::sin(theta);
            cplx e_phi(std::cos(phi), std::sin(phi));
            cplx e_psi(std::cos(psi), std::sin(psi));
            // 2x2 unitary acting on rows i, i+1
            cplx u00 = c * e_phi;
            cplx u01 = s * e_psi;
            cplx u10 = -s * std::conj(e_psi);
            cplx u11 = c * std::conj(e_phi);
            for (int col = 0; col < n; ++col) {
                cplx a = U[i][col], b = U[i + 1][col];
                U[i][col]     = u00 * a + u01 * b;
                U[i + 1][col] = u10 * a + u11 * b;
            }
        }
    }
    return U;
}

static std::vector<AdjRow> dense_to_sparse(const std::vector<std::vector<cplx>>& A, double tol = 1e-15) {
    int n = (int)A.size();
    std::vector<AdjRow> rows(n);
    for (int r = 0; r < n; ++r)
        for (int c = 0; c < n; ++c)
            if (std::abs(A[r][c]) > tol)
                rows[r].emplace_back(c, cld(A[r][c].real(), A[r][c].imag()));
    return rows;
}

static std::vector<AdjRow> gen_d_reg_complex(int n, int d, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> u_ang(0.0, 2.0 * M_PI);
    std::vector<AdjRow> rows(n);
    std::vector<int> perm(n);
    for (int i = 0; i < n; ++i) perm[i] = i;
    double amp = 1.0 / std::sqrt((double)d);
    for (int t = 0; t < d; ++t) {
        std::shuffle(perm.begin(), perm.end(), rng);
        for (int r = 0; r < n; ++r) {
            double ang = u_ang(rng);
            cld w(amp * std::cos(ang), amp * std::sin(ang));
            set_entry(rows[r], perm[r], get_entry(rows[r], perm[r]) + w);
        }
    }
    return rows;
}

static std::vector<AdjRow> permute_mat(const std::vector<AdjRow>& rows, std::mt19937_64& rng) {
    int n = (int)rows.size();
    std::vector<int> rp(n), cp(n), invc(n);
    for (int i = 0; i < n; ++i) rp[i] = cp[i] = i;
    std::shuffle(rp.begin(), rp.end(), rng);
    std::shuffle(cp.begin(), cp.end(), rng);
    for (int i = 0; i < n; ++i) invc[cp[i]] = i;
    std::vector<AdjRow> out(n);
    for (int i = 0; i < n; ++i)
        for (const auto& kv : rows[rp[i]]) out[i].emplace_back(invc[kv.first], kv.second);
    return out;
}

int main() {
    std::mt19937_64 rng(2026);
    std::printf("=== Gen 15 Complex Boson Sampling Sparse Permanent (threads=%d) ===\n",
                omp_get_max_threads());

    std::printf("\n--- Part 1: Accuracy vs Glynn's Formula on Shallow-Depth Photonic Meshes (depth=4) ---\n");
    for (int n : {4, 6, 8, 10, 12, 14, 16}) {
        auto U = gen_photonic_mesh(n, 4, rng);
        auto S = dense_to_sparse(U);
        Stats st;
        cplx p_sparse = solve_parallel(S, st);
        cplx p_glynn  = glynn_permanent(U);
        double abs_err = std::abs(p_sparse - p_glynn);
        double rel_err = abs_err / std::max(std::abs(p_glynn), 1e-15);
        std::printf("  n=%2d | |perm|=%.6e | abs_err=%.3e | rel_err=%.3e | %s\n",
                    n, std::abs(p_glynn), abs_err, rel_err, (rel_err < 1e-12 ? "PASS" : "FAIL"));
        if (rel_err >= 1e-12) return 1;
    }

    std::printf("\n--- Part 2: 3-Regular Complex Unitary-Scaled Benchmarks (n=48..60) ---\n");
    for (int n : {48, 56, 60}) {
        int trials = (n <= 56) ? 2 : 1;
        double tot_ops = 0, tot_ms = 0, tot_br = 0;
        for (int t = 0; t < trials; ++t) {
            auto M = gen_d_reg_complex(n, 3, rng);
            Stats st;
            auto t0 = std::chrono::high_resolution_clock::now();
            cplx v1 = solve_parallel(M, st);
            double ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
            tot_ops += st.ops; tot_ms += ms; tot_br += st.branches;
            if (t == 0) {
                Stats st2;
                cplx v2 = solve_parallel(permute_mat(M, rng), st2);
                double rel = std::abs(v1 - v2) / std::max(std::abs(v1), 1e-30);
                if (rel > 1e-11) {
                    std::printf("PERM INVARIANCE FAILED n=%d (rel=%.3e)\n", n, rel);
                    return 1;
                }
            }
        }
        double avg_ops = tot_ops / trials, avg_ms = tot_ms / trials, avg_br = tot_br / trials;
        double ratio = std::pow(2.0, n) / avg_ops;
        std::printf("  n=%2d | avg_ops=%12.1f | vs 2^n: %15.2e x fewer! | br=%7.1f | %8.2f ms | PERM-INV PASS\n",
                    n, avg_ops, ratio, avg_br, avg_ms);
    }
    return 0;
}
