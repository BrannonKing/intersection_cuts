import knapsack_loader as kl
import gurobipy as gp
import gurobi_utils as gu
import numpy as np
import timeit as ti
import lll_utils as lu

def main():
    models = list(kl.generate(3, 5, 100, 5, 10, 1000, equality=True, seed=43))
    for model in models:
        A, b, c, l, u = gu.get_A_b_c_l_u(model, keep_sparse=False)
        N1 = 10000
        N2 = 100000
        # np.savetxt('jsp_N.txt', N, fmt='%d')
        # print(x_p)

        n = A.shape[1]
        As = np.block(
            [
                [np.eye(n, dtype=np.int64), np.zeros((n, 1), dtype=np.int64)],
                [np.zeros((1, n), dtype=np.int64), np.array([N1])],
                [N2 * A, -N2 * b],
            ]
        ).astype(np.int64, order="C")

        # x_p, N = gu.nullspace_and_offset_via_LLL(As.toarray(), b, N1, N2)
        # assert np.allclose(As @ N, 0)
        last_idx = -1
        last_idx_count = 0

        def check_exit(Bc, Uc, iteration):
            idx = np.argmax(Bc[n, :])
            if idx >= n:
                return False
            # see if idx has been at same location last five iterations:
            nonlocal last_idx, last_idx_count
            if idx == last_idx:
                last_idx_count += 1
            else:
                last_idx = idx
                last_idx_count = 0
            if last_idx_count > 5:
                if np.allclose(A @ Bc[0:n, 0:idx], 0):
                    return True
            return False

        start = ti.default_timer()
        B, U, its = lu.lll_apx(As, iterations=200, early_exit_func=check_exit)
        idx = np.argmax(B[n, :])
        assert idx < n
        N = B[0:n, 0:idx]
        x_p = B[0:n, idx]
        end = ti.default_timer()
        print(f"LLL took {end - start:.4f} seconds. Idx: {idx} / {n}. Iterations: {its}")
        assert np.allclose(A @ N, 0)
        # assert np.allclose(A @ x_p, b) not necessary

        # mdl2 = gu.substitute(model, N, x_p, 'skip')
        # mdl2.params.Presolve = 2 
        # # mdl2.params.DualReductions = 0
        # mdl2.optimize()
        # print('Optimal value after LLL substitution:', mdl2.ObjVal)

if __name__ == "__main__":
    main()