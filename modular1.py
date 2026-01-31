from math import gcd
from functools import reduce

def crt_pair(a1, m1, a2, m2):
    g = gcd(m1, m2)
    if (a2 - a1) % g != 0:
        raise ValueError("Inconsistent CRT")
    lcm = m1 // g * m2
    t = ((a2 - a1) // g * pow(m1 // g, -1, m2 // g)) % (m2 // g)
    return (a1 + m1 * t) % lcm, lcm


def centered_lift(a, M):
    return a - M if a > M // 2 else a

def mat_vec_mul(A, v):
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]

def is_zero_vector(v):
    return all(x == 0 for x in v)

def rref_mod_p(A, p):
    mat = [[int(x % p) for x in row] for row in A]
    m = len(mat)
    n = len(mat[0]) if m else 0

    pivots = []
    row = 0
    for col in range(n):
        pivot = None
        for r in range(row, m):
            if mat[r][col] % p != 0:
                pivot = r
                break
        if pivot is None:
            continue
        if pivot != row:
            mat[row], mat[pivot] = mat[pivot], mat[row]

        inv = pow(mat[row][col], -1, p)
        mat[row] = [(val * inv) % p for val in mat[row]]

        for r in range(m):
            if r == row:
                continue
            if mat[r][col] % p != 0:
                factor = mat[r][col] % p
                mat[r] = [(mat[r][c] - factor * mat[row][c]) % p for c in range(n)]

        pivots.append(col)
        row += 1
        if row == m:
            break

    return mat, pivots

def nullspace_mod_p_sympy(A, p):
    rref, pivots = rref_mod_p(A, p)
    n = len(A[0]) if A else 0
    pivot_set = set(pivots)
    free_vars = [j for j in range(n) if j not in pivot_set]

    basis = []
    for free_idx in free_vars:
        v = [0] * n
        v[free_idx] = 1
        for row, pivot_col in enumerate(pivots):
            v[pivot_col] = (-rref[row][free_idx]) % p
        basis.append(v)

    return basis, tuple(pivots)

def particular_solution_mod_p(A, b, p):
    aug = [row + [b[i]] for i, row in enumerate(A)]
    rref, pivots = rref_mod_p(aug, p)

    n = len(A[0]) if A else 0
    m = len(A)

    for i in range(m):
        if all((rref[i][j] % p) == 0 for j in range(n)) and (rref[i][n] % p) != 0:
            raise ValueError("No solution modulo p")

    x = [0] * n
    pivot_cols = [c for c in pivots if c < n]
    for row, col in enumerate(pivot_cols):
        x[col] = int(rref[row][n] % p)

    return x

def integer_nullspace(A, primes):
    n = len(A[0]) if A else 0

    modular_bases = []
    nullity = None
    pivot_cols = None
    good_primes = []

    for p in primes:
        try:
            basis, pivots = nullspace_mod_p_sympy(A, p)
        except Exception as e:
            print("Bad prime:", p, "error:", e)
            continue

        if nullity is None:
            nullity = len(basis)
        elif len(basis) != nullity:
            continue  # bad prime

        if pivot_cols is None:
            pivot_cols = pivots
        elif pivots != pivot_cols:
            continue  # bad prime (rank pattern differs)

        modular_bases.append(basis)
        good_primes.append(p)

    if not modular_bases or nullity is None:
        raise RuntimeError("No usable primes")

    # CRT lift
    lifted = []
    for j in range(nullity):
        v = [0] * n
        for i in range(n):
            val = 0
            M = 1
            for p, basis in zip(good_primes, modular_bases):
                val, M = crt_pair(val, M, int(basis[j][i]), p)
            v[i] = centered_lift(val, M)
        lifted.append(v)

    # Exact verification and normalization
    result = []
    for v in lifted:
        if is_zero_vector(mat_vec_mul(A, v)):
            g = reduce(gcd, [int(x) for x in v], 0)
            result.append([int(x) // g for x in v] if g > 1 else v)

    return result

def integer_particular_solution(A, b, primes):
    n = len(A[0]) if A else 0

    vals = [0] * n
    mods = [1] * n

    for p in primes:
        try:
            xp = particular_solution_mod_p(A, b, p)
        except Exception:
            raise RuntimeError("No integer solution exists")

        for i in range(n):
            vals[i], mods[i] = crt_pair(vals[i], mods[i], xp[i], p)

    x = [centered_lift(vals[i], mods[i]) for i in range(n)]
    if mat_vec_mul(A, x) != b:
        raise RuntimeError("Lift failed")

    return x

def eliminate_equalities(A, b, primes):
    N = integer_nullspace(A, primes)
    x0 = integer_particular_solution(A, b, primes)
    return x0, N

def main():
    A = [
        [2, 4, 6],
        [1, 2, 3]
    ]
    b = [0, 0]

    primes = [101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197]

    x0, N = eliminate_equalities(A, b, primes)

    print("x0 =", x0)
    print("Nullspace basis:")
    print(N)

if __name__ == "__main__":
    main()