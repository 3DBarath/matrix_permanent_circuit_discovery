#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <queue>
#include <random>
#include <unordered_map>
#include <vector>

namespace sparse_hafnian {

using Matrix = std::vector<std::vector<double>>;

// Brute-force Hafnian for small even n (n <= 14)
double brute_force_hafnian(const Matrix& A) {
    int n = static_cast<int>(A.size());
    if (n == 0) return 1.0;
    if (n % 2 != 0) return 0.0;

    std::vector<bool> used(n, false);
    auto dfs = [&](auto& self, int first_free) -> double {
        while (first_free < n && used[first_free]) {
            ++first_free;
        }
        if (first_free == n) return 1.0;

        used[first_free] = true;
        double sum = 0.0;
        for (int j = first_free + 1; j < n; ++j) {
            if (!used[j] && A[first_free][j] != 0.0) {
                used[j] = true;
                sum += A[first_free][j] * self(self, first_free + 1);
                used[j] = false;
            }
        }
        used[first_free] = false;
        return sum;
    };
    return dfs(dfs, 0);
}

// Brute-force Loop Hafnian for small n (n <= 14)
double brute_force_loop_hafnian(const Matrix& A) {
    int n = static_cast<int>(A.size());
    if (n == 0) return 1.0;

    std::vector<bool> used(n, false);
    auto dfs = [&](auto& self, int first_free) -> double {
        while (first_free < n && used[first_free]) {
            ++first_free;
        }
        if (first_free == n) return 1.0;

        used[first_free] = true;
        double sum = 0.0;
        // Self-loop
        if (A[first_free][first_free] != 0.0) {
            sum += A[first_free][first_free] * self(self, first_free + 1);
        }
        // Pair with j > first_free
        for (int j = first_free + 1; j < n; ++j) {
            if (!used[j] && A[first_free][j] != 0.0) {
                used[j] = true;
                sum += A[first_free][j] * self(self, first_free + 1);
                used[j] = false;
            }
        }
        used[first_free] = false;
        return sum;
    };
    return dfs(dfs, 0);
}

struct SparseGraph {
    int n;
    std::vector<std::vector<std::pair<int, double>>> adj;
};

// Greedy min-frontier vertex ordering
std::vector<int> compute_min_frontier_order(const SparseGraph& g) {
    int n = g.n;
    std::vector<int> order;
    order.reserve(n);
    std::vector<bool> placed(n, false);
    std::vector<int> rem_deg(n, 0);
    for (int i = 0; i < n; ++i) {
        rem_deg[i] = static_cast<int>(g.adj[i].size());
    }

    // Track current frontier: placed vertices with rem_deg > 0
    for (int step = 0; step < n; ++step) {
        int best_v = -1;
        int best_delta = 1e9;
        int best_rem_deg = 1e9;

        for (int v = 0; v < n; ++v) {
            if (placed[v]) continue;
            // Delta in frontier size if we place v now:
            // +1 if v has any unplaced neighbor
            // -1 for each placed neighbor u of v whose rem_deg[u] == 1
            int delta = (rem_deg[v] > 0) ? 1 : 0;
            for (const auto& [u, w] : g.adj[v]) {
                if (placed[u] && rem_deg[u] == 1) {
                    --delta;
                }
            }
            if (delta < best_delta || (delta == best_delta && rem_deg[v] < best_rem_deg)) {
                best_delta = delta;
                best_rem_deg = rem_deg[v];
                best_v = v;
            }
        }

        order.push_back(best_v);
        placed[best_v] = true;
        for (const auto& [u, w] : g.adj[best_v]) {
            if (placed[u]) {
                --rem_deg[u];
                --rem_deg[best_v];
            }
        }
    }
    return order;
}

// Min-Frontier DP on a connected component
double min_frontier_dp(const SparseGraph& g, int* max_frontier_out = nullptr) {
    int n = g.n;
    if (n == 0) return 1.0;
    if (n % 2 != 0) return 0.0;

    std::vector<int> order = compute_min_frontier_order(g);
    std::vector<int> pos(n);
    for (int i = 0; i < n; ++i) pos[order[i]] = i;

    // Build last_neighbor[u] = max pos[v] among neighbors v of u
    std::vector<int> last_neighbor(n, -1);
    for (int u = 0; u < n; ++u) {
        for (const auto& [v, w] : g.adj[u]) {
            last_neighbor[u] = std::max(last_neighbor[u], pos[v]);
        }
    }

    std::vector<int> frontier;
    std::unordered_map<uint64_t, double> dp;
    dp[0] = 1.0;

    int max_f = 0;

    for (int step = 0; step < n; ++step) {
        int v = order[step];
        int f_size = static_cast<int>(frontier.size());
        max_f = std::max(max_f, f_size);

        // Map frontier vertex -> bit index
        std::vector<int> f_idx(n, -1);
        for (int b = 0; b < f_size; ++b) {
            f_idx[frontier[b]] = b;
        }

        // Next frontier: vertices in frontier with last_neighbor > step, plus v if last_neighbor[v] > step
        std::vector<int> next_frontier;
        std::vector<int> keep_bits(f_size, -1);
        for (int b = 0; b < f_size; ++b) {
            if (last_neighbor[frontier[b]] > step) {
                keep_bits[b] = static_cast<int>(next_frontier.size());
                next_frontier.push_back(frontier[b]);
            }
        }
        int v_next_bit = -1;
        if (last_neighbor[v] > step) {
            v_next_bit = static_cast<int>(next_frontier.size());
            next_frontier.push_back(v);
        }

        // Project a mask on `frontier` to `next_frontier`; returns false if any dropped vertex is unmatched
        auto project_mask = [&](uint64_t mask, uint64_t& out_mask) -> bool {
            out_mask = 0;
            for (int b = 0; b < f_size; ++b) {
                if ((mask >> b) & 1ULL) {
                    if (keep_bits[b] == -1) return false;
                    out_mask |= (1ULL << keep_bits[b]);
                }
            }
            return true;
        };

        std::unordered_map<uint64_t, double> next_dp;
        next_dp.reserve(dp.size() * 2 + 16);

        // Precompute backward neighbors of v in frontier
        std::vector<std::pair<int, double>> back_nbrs;
        for (const auto& [u, w] : g.adj[v]) {
            if (pos[u] < step && f_idx[u] != -1) {
                back_nbrs.emplace_back(f_idx[u], w);
            }
        }

        for (const auto& [mask, val] : dp) {
            if (val == 0.0) continue;

            // Option A: leave v unmatched (only if v_next_bit != -1)
            if (v_next_bit != -1) {
                uint64_t nmask = 0;
                if (project_mask(mask, nmask)) {
                    nmask |= (1ULL << v_next_bit);
                    next_dp[nmask] += val;
                }
            }

            // Option B: match v with an unmatched backward neighbor u in frontier
            for (const auto& [ub, w] : back_nbrs) {
                if ((mask >> ub) & 1ULL) {
                    uint64_t cleared = mask & ~(1ULL << ub);
                    uint64_t nmask = 0;
                    if (project_mask(cleared, nmask)) {
                        next_dp[nmask] += val * w;
                    }
                }
            }
        }

        dp = std::move(next_dp);
        frontier = std::move(next_frontier);
    }

    if (max_frontier_out) *max_frontier_out = max_f;
    auto it = dp.find(0);
    return (it != dp.end()) ? it->second : 0.0;
}

// Solve Hafnian with connected-component factorization, degree-1 peeling,
// degree-2 series contraction, and Min-Frontier DP.
double sparse_hafnian(const Matrix& A_in, int* max_frontier_out = nullptr) {
    int n = static_cast<int>(A_in.size());
    if (n == 0) return 1.0;
    if (n % 2 != 0) return 0.0;

    // Build mutable adjacency list (ignoring diagonal)
    std::vector<std::vector<std::pair<int, double>>> adj(n);
    std::vector<bool> active(n, true);
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (A_in[i][j] != 0.0) {
                adj[i].emplace_back(j, A_in[i][j]);
                adj[j].emplace_back(i, A_in[i][j]);
            }
        }
    }

    auto remove_edge = [&](int u, int v) {
        auto& au = adj[u];
        for (size_t i = 0; i < au.size(); ++i) {
            if (au[i].first == v) {
                au[i] = au.back();
                au.pop_back();
                break;
            }
        }
    };

    auto remove_vertex = [&](int u) {
        active[u] = false;
        for (const auto& [v, w] : adj[u]) {
            remove_edge(v, u);
        }
        adj[u].clear();
    };

    auto add_or_update_edge = [&](int u, int v, double w) {
        if (u == v || w == 0.0) return;
        for (auto& [nbr, ew] : adj[u]) {
            if (nbr == v) {
                ew += w;
                for (auto& [nbr2, ew2] : adj[v]) {
                    if (nbr2 == u) {
                        ew2 += w;
                        break;
                    }
                }
                if (ew == 0.0) {
                    remove_edge(u, v);
                    remove_edge(v, u);
                }
                return;
            }
        }
        adj[u].emplace_back(v, w);
        adj[v].emplace_back(u, w);
    };

    double factor = 1.0;

    // Iterative degree-0 check, degree-1 peeling, and degree-2 series contraction
    bool changed = true;
    while (changed) {
        changed = false;
        for (int v = 0; v < n; ++v) {
            if (!active[v]) continue;
            if (adj[v].empty()) {
                return 0.0; // isolated vertex -> no perfect matching
            }
            if (adj[v].size() == 1) {
                int u = adj[v][0].first;
                double w = adj[v][0].second;
                factor *= w;
                remove_vertex(v);
                remove_vertex(u);
                changed = true;
                break;
            }
            if (adj[v].size() == 2) {
                int u = adj[v][0].first;
                double a = adj[v][0].second;
                int w = adj[v][1].first;
                double b = adj[v][1].second;

                // Contract u, v, w into u (removing v and w)
                // Remove edge (u, w) if present, since matching u-w isolates v
                remove_edge(u, w);
                remove_edge(w, u);
                remove_vertex(v);

                // Collect neighbors of w (excluding u, v)
                auto w_nbrs = adj[w];
                // Existing edges of u get multiplied by b
                for (auto& [x, ux_w] : adj[u]) {
                    ux_w *= b;
                    for (auto& [y, xy_w] : adj[x]) {
                        if (y == u) {
                            xy_w *= b;
                            break;
                        }
                    }
                }
                // Add a * w_x for each neighbor x of w
                for (const auto& [x, wx_w] : w_nbrs) {
                    add_or_update_edge(u, x, a * wx_w);
                }
                remove_vertex(w);
                changed = true;
                break;
            }
        }
    }

    // Connected-component factorization
    std::vector<bool> visited(n, false);
    int max_f = 0;
    for (int i = 0; i < n; ++i) {
        if (!active[i] || visited[i]) continue;
        std::vector<int> comp;
        std::queue<int> q;
        q.push(i);
        visited[i] = true;
        while (!q.empty()) {
            int u = q.front();
            q.pop();
            comp.push_back(u);
            for (const auto& [v, w] : adj[u]) {
                if (active[v] && !visited[v]) {
                    visited[v] = true;
                    q.push(v);
                }
            }
        }
        if (comp.size() % 2 != 0) return 0.0;

        int cn = static_cast<int>(comp.size());
        std::vector<int> local_id(n, -1);
        for (int k = 0; k < cn; ++k) local_id[comp[k]] = k;

        SparseGraph sg;
        sg.n = cn;
        sg.adj.resize(cn);
        for (int k = 0; k < cn; ++k) {
            int u = comp[k];
            for (const auto& [v, w] : adj[u]) {
                if (active[v]) {
                    sg.adj[k].emplace_back(local_id[v], w);
                }
            }
        }
        int comp_f = 0;
        factor *= min_frontier_dp(sg, &comp_f);
        max_f = std::max(max_f, comp_f);
        if (factor == 0.0) return 0.0;
    }

    if (max_frontier_out) *max_frontier_out = max_f;
    return factor;
}

// Generate a random 3-regular graph (configuration model with rejection for simple graph)
Matrix random_3_regular(int n, std::mt19937_64& rng) {
    Matrix A(n, std::vector<double>(n, 0.0));
    std::vector<int> stubs;
    stubs.reserve(3 * n);
    while (true) {
        stubs.clear();
        for (int i = 0; i < n; ++i) {
            for (int k = 0; k < 3; ++k) stubs.push_back(i);
        }
        std::shuffle(stubs.begin(), stubs.end(), rng);
        bool ok = true;
        for (int i = 0; i < n; ++i) std::fill(A[i].begin(), A[i].end(), 0.0);
        for (size_t i = 0; i < stubs.size(); i += 2) {
            int u = stubs[i];
            int v = stubs[i + 1];
            if (u == v || A[u][v] != 0.0) {
                ok = false;
                break;
            }
            A[u][v] = 1.0;
            A[v][u] = 1.0;
        }
        if (ok) return A;
    }
}

} // namespace sparse_hafnian

int main() {
    using namespace sparse_hafnian;
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> wdist(0.5, 2.0);

    std::cout << "=== Verification against Brute-Force Hafnian (n = 4, 6, 8, 10) ===\n";
    for (int n : {4, 6, 8, 10}) {
        double max_rel_err = 0.0;
        for (int trial = 0; trial < 20; ++trial) {
            Matrix A(n, std::vector<double>(n, 0.0));
            // Sparse random symmetric matrix (avg degree ~ 2.5 - 3.5)
            std::bernoulli_distribution edist(std::min(1.0, 3.0 / (n - 1)));
            for (int i = 0; i < n; ++i) {
                for (int j = i + 1; j < n; ++j) {
                    if (edist(rng)) {
                        double w = wdist(rng);
                        A[i][j] = A[j][i] = w;
                    }
                }
            }
            double bf = brute_force_hafnian(A);
            double bf_loop = brute_force_loop_hafnian(A); // since diag=0, equals bf
            double sh = sparse_hafnian::sparse_hafnian(A);
            double denom = std::max(1.0, std::abs(bf));
            double rel_err = std::abs(bf - sh) / denom;
            max_rel_err = std::max(max_rel_err, rel_err);
            if (std::abs(bf - bf_loop) / denom > 1e-12) {
                std::cerr << "Mismatch between brute_force_hafnian and brute_force_loop_hafnian!\n";
                return 1;
            }
        }
        std::cout << "n=" << std::setw(2) << n << " | 20 trials | max_rel_err = " << std::scientific << max_rel_err << "\n";
        if (max_rel_err > 1e-9) {
            std::cerr << "VERIFICATION FAILED for n=" << n << "\n";
            return 1;
        }
    }

    std::cout << "\n=== Benchmark on 3-Regular Graphs (n = 24, 28, 32, 36) ===\n";
    for (int n : {24, 28, 32, 36}) {
        double total_ms = 0.0;
        int max_f = 0;
        double last_val = 0.0;
        int trials = 5;
        for (int t = 0; t < trials; ++t) {
            Matrix A = random_3_regular(n, rng);
            int f = 0;
            auto t0 = std::chrono::high_resolution_clock::now();
            last_val = sparse_hafnian::sparse_hafnian(A, &f);
            auto t1 = std::chrono::high_resolution_clock::now();
            total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
            max_f = std::max(max_f, f);
        }
        std::cout << "n=" << std::setw(2) << n
                  << " | avg_time = " << std::fixed << std::setprecision(3) << (total_ms / trials) << " ms"
                  << " | max_frontier = " << max_f
                  << " | sample_haf = " << std::setprecision(0) << last_val << "\n";
    }

    return 0;
}
