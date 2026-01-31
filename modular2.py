from math import gcd, isqrt
from functools import reduce
from typing import cast, Optional

import numpy as np
from scipy import sparse
from scipy.sparse import spmatrix, csr_matrix

def crt_pair(a1, m1, a2, m2):
    g = gcd(m1, m2)
    if (a2 - a1) % g != 0:
        raise ValueError("Inconsistent CRT")
    lcm = m1 // g * m2
    t = ((a2 - a1) // g * pow(m1 // g, -1, m2 // g)) % (m2 // g)
    return (a1 + m1 * t) % lcm, lcm


def centered_lift(a, M):
    return a - M if a > M // 2 else a

def lcm(a, b):
    return a // gcd(a, b) * b

def rational_reconstruction(r, m, bound=None):
    r = r % m
    if bound is None:
        bound = isqrt(m // 2)

    a0, a1 = m, r
    b0, b1 = 0, 1

    while abs(a1) > bound:
        q = a0 // a1
        a0, a1 = a1, a0 - q * a1
        b0, b1 = b1, b0 - q * b1

    if b1 == 0 or abs(b1) > bound:
        return None
    if (r * b1 - a1) % m != 0:
        return None
    if b1 < 0:
        a1, b1 = -a1, -b1
    return a1, b1

def is_zero_vector(v):
    return np.all(v == 0)

def rref_mod_p(A, p):
    A_csr = sparse.csr_matrix(A)
    m, n = A_csr.shape

    rows = []
    for i in range(m):
        row = {}
        start, end = A_csr.indptr[i], A_csr.indptr[i + 1]
        for idx, val in zip(A_csr.indices[start:end], A_csr.data[start:end]):
            v = int(val) % p
            if v:
                row[int(idx)] = v
        rows.append(row)

    pivots = []
    pivot_row = 0
    for col in range(n):
        pivot = None
        for r in range(pivot_row, m):
            if rows[r].get(col, 0) % p != 0:
                pivot = r
                break
        if pivot is None:
            continue

        if pivot != pivot_row:
            rows[pivot_row], rows[pivot] = rows[pivot], rows[pivot_row]

        piv_val = rows[pivot_row][col] % p
        inv = pow(piv_val, -1, p)
        rows[pivot_row] = {
            c: (v * inv) % p
            for c, v in rows[pivot_row].items()
            if (v * inv) % p != 0
        }

        for r in range(m):
            if r == pivot_row:
                continue
            factor = rows[r].get(col, 0) % p
            if factor == 0:
                continue
            for c, pv in rows[pivot_row].items():
                val = (rows[r].get(c, 0) - factor * pv) % p
                if val:
                    rows[r][c] = val
                elif c in rows[r]:
                    del rows[r][c]

        pivots.append(col)
        pivot_row += 1
        if pivot_row == m:
            break

    return rows, pivots, n

def nullspace_mod_p(A, p):
    rref, pivots, n = rref_mod_p(A, p)
    pivot_set = set(pivots)
    free_vars = [j for j in range(n) if j not in pivot_set]

    basis = []
    for free_idx in free_vars:
        v = [0] * n
        v[free_idx] = 1
        for row, pivot_col in enumerate(pivots):
            v[pivot_col] = (-rref[row].get(free_idx, 0)) % p
        basis.append(v)

    return basis, tuple(pivots)

def integer_nullspace(A, primes):
    input_is_sparse = sparse.issparse(A)
    if input_is_sparse:
        A_csr = cast(csr_matrix, sparse.csr_matrix(A))
        A_dense: Optional[np.ndarray] = None
    else:
        A_dense = np.asarray(A)
        A_csr = cast(csr_matrix, sparse.csr_matrix(A_dense))

    n = A_csr.shape[1]

    modular_bases = []
    nullity = None
    pivot_cols = None
    good_primes = []

    for p in primes:
        try:
            basis, pivots = nullspace_mod_p(A_csr, p)
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
    modulus = 1
    for p in good_primes:
        modulus *= p

    lifted = []
    for j in range(nullity):
        v = np.zeros((n, 1), dtype=object)
        for i in range(n):
            val = 0
            M = 1
            for p, basis in zip(good_primes, modular_bases):
                val, M = crt_pair(val, M, int(basis[j][i]), p)
            v[i, 0] = val
        lifted.append(v)

    # Exact verification and normalization
    basis_cols = []
    indptr = A_csr.indptr
    indices = A_csr.indices
    data = A_csr.data
    int64_max = np.iinfo(np.int64).max

    for v in lifted:
        fracs = []
        for i in range(n):
            rr = rational_reconstruction(int(v[i, 0]), modulus)
            if rr is None:
                rr = (centered_lift(int(v[i, 0]), modulus), 1)
            fracs.append(rr)

        den = 1
        for _, d in fracs:
            den = lcm(den, d)

        v_int = np.zeros((n, 1), dtype=object)
        for i, (a, d) in enumerate(fracs):
            v_int[i, 0] = a * (den // d)

        if input_is_sparse:
            Av = np.zeros((A_csr.shape[0], 1), dtype=object)
            for i in range(A_csr.shape[0]):
                s = 0
                start, end = indptr[i], indptr[i + 1]
                for idx in range(start, end):
                    s += int(data[idx]) * v_int[indices[idx], 0]
                Av[i, 0] = s
        else:
            assert A_dense is not None
            Av = np.zeros((A_dense.shape[0], 1), dtype=object)
            for i in range(A_dense.shape[0]):
                s = 0
                for j in range(A_dense.shape[1]):
                    s += int(A_dense[i, j]) * v_int[j, 0]
                Av[i, 0] = s

        if is_zero_vector(Av):
            flat = [int(x) for x in v_int.flatten()]
            g = reduce(gcd, flat, 0)
            if g > 1:
                flat = [x // g for x in flat]
            v_norm = np.array(flat, dtype=object).reshape(n, 1)
            if input_is_sparse:
                max_abs = max((abs(x) for x in flat), default=0)
                if max_abs > int64_max:
                    raise OverflowError("Sparse CSR output cannot represent values beyond int64")
                basis_cols.append(np.array(flat, dtype=np.int64).reshape(n, 1))
            else:
                basis_cols.append(v_norm)

    if input_is_sparse:
        if basis_cols:
            return sparse.csr_matrix(np.hstack(basis_cols))
        return sparse.csr_matrix((A_csr.shape[1], 0), dtype=np.int64)

    if basis_cols:
        return np.hstack(basis_cols)
    return np.zeros((A_csr.shape[1], 0), dtype=object)

def main():
    # A = sparse.csr_matrix([
    #     [2, 4, 6],
    #     [1, 2, 3]
    # ])
    # b = np.array([0, 0])
    A = np.array([[6, 1, 3, 3, 0, 0], [0, 0, 0, 0, 2, 1], [0, 0, 4, 1, 0, 2]], dtype=np.int64)
    As = sparse.csr_matrix(A)

    primes = [13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,
        103,107,109,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197]
    primes = primes[20:]
    primes = [673,677,683,691,701,709,719,727,733,739,743,751,757,761,769,773,787,797,809,
        811,821,823,827,829,839,853,857,859,863,877,881,883,887,907,911,919,929,937]

    # Ns = integer_nullspace(As, primes)
    # assert is_zero_vector((As @ Ns).todense())
    # print("Nullspace basis:")
    # print(Ns.todense())

    # N = integer_nullspace(A, primes)
    # assert is_zero_vector(A @ N)
    # print("Nullspace basis:")
    # print(N)

    import gurobi_utils as gu
    # import knapsack_loader as kl
    # instances = kl.generate(1, 2, 20, 5, 10, 1000, seed=42)
    # instance = next(iter(instances))
    # A, _, _, _, _ = gu.get_A_b_c_l_u(instance, keep_sparse=True)
    # Ns = integer_nullspace(A, primes)
    # assert is_zero_vector((A @ Ns).todense())
    # print("Knapsack Nullspace basis:")
    # print(Ns.todense())

    import jsplib_loader as jl
    instances = jl.get_instances()
    problem = instances['abz3']
    instance = problem.as_gurobi_balas_model(use_big_m=True)
    A, _, _, _, _ = gu.get_A_b_c_l_u(instance, keep_sparse=True)
    A = sparse.block_array([[A, sparse.eye(A.shape[0])]]).tocsr()
    Ns = integer_nullspace(A, primes)
    assert is_zero_vector((A @ Ns).todense())
    print("JSP Nullspace basis:")
    print(Ns)

if __name__ == "__main__":
    main()