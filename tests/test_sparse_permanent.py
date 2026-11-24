import random
from src.sparse_permanent import perm_sparse_exact
from src.tracked_field import glynn_perm_mod, PRIME_61_A, PRIME_61_B


def test_sparse_against_glynn():
    random.seed(42)
    primes = [PRIME_61_A, PRIME_61_B]
    cases = 0

    # 1. Random sparse matrices across sizes n = 3..12
    for n in range(3, 13):
        for p in primes:
            # density 0.3
            A = [[random.randint(0, 5) if random.random() < 0.35 else 0 for _ in range(n)] for _ in range(n)]
            sparse_val, stats = perm_sparse_exact(A, p)
            glynn_val = glynn_perm_mod(A, p)
            assert sparse_val == glynn_val, f"Mismatch at n={n}, prime={p}: {sparse_val} != {glynn_val}"
            cases += 1

    # 2. 3-regular bipartite graphs up to n = 12
    for n in [6, 8, 10, 12]:
        for p in primes:
            A = [[0] * n for _ in range(n)]
            for r in range(n):
                cols = random.sample(range(n), 3)
                for c in cols:
                    A[r][c] = random.randint(1, 9)
            sparse_val, stats = perm_sparse_exact(A, p)
            glynn_val = glynn_perm_mod(A, p)
            assert sparse_val == glynn_val, f"Mismatch on 3-regular n={n}: {sparse_val} != {glynn_val}"
            cases += 1

    print(f"All {cases} verification cases passed bit-exact!")


if __name__ == "__main__":
    test_sparse_against_glynn()
