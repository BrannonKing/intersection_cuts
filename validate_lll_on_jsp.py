import jsplib_loader as jl
import gurobipy as gp
import gurobi_utils as gu
import lll_utils as lu
import numpy as np
import scipy.sparse as sps
import timeit as ti


def sparse_allclose(x, y, atol=1e-9, rtol=1e-7):
    """
    allclose for sparse/dense mixes using an infinity-norm residual test.
    """
    if sps.issparse(x) or sps.issparse(y):
        xs = x if sps.issparse(x) else sps.csc_matrix(np.asarray(x))
        ys = y if sps.issparse(y) else sps.csc_matrix(np.asarray(y))

        if xs.shape != ys.shape:
            return False

        # ||x - y||_inf <= atol + rtol * max(||x||_inf, ||y||_inf)
        diff_inf = np.max(np.abs((xs - ys).data)) if (xs - ys).nnz else 0.0
        x_inf = np.max(np.abs(xs.data)) if xs.nnz else 0.0
        y_inf = np.max(np.abs(ys.data)) if ys.nnz else 0.0
        return diff_inf <= (atol + rtol * max(x_inf, y_inf))

    return np.allclose(np.asarray(x), np.asarray(y), atol=atol, rtol=rtol)

def main():
    instance = jl.get_instances()['abz3']
    model = instance.as_gurobi_balas_model(use_big_m=True)
    A, b, c, l, u = gu.get_A_b_c_l_u(model, keep_sparse=True)
    N1 = 100000
    N2 = 1000000
    # np.savetxt('jsp_N.txt', N, fmt='%d')
    # print(x_p)
    senses = [c.Sense for c in model.getConstrs()]
    assert all(s == gp.GRB.GREATER_EQUAL for s in senses)

    A = sps.block_array([[A, -sps.eye(A.shape[0])]], format='csc')

    n = A.shape[1]
    As = sps.block_array(
        [
            [sps.eye(n), sps.csr_array((n, 1))],
            [sps.csr_array((1, n)), sps.csr_array([[N1]])],
            [N2 * A, -N2 * b],
        ],
    format='csc')


    # x_p, N = gu.nullspace_and_offset_via_LLL(As.toarray(), b, N1, N2)
    # assert np.allclose(As @ N, 0)

    last_idx = -1
    last_idx_count = 0

    def check_exit_state(state, iteration):
        idx = state.row_argmax(n)
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
            # Materialize B only when we need the heavier null-space verification.
            Bc = state.B
            if sparse_allclose(A @ Bc[0:n, 0:idx], 0.0):
                return True
        return False

    def check_exit(B, U, iteration):
        idx = B[n].argmax()
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
            # Materialize B only when we need the heavier null-space verification.
            if sparse_allclose(A @ B[0:n, 0:idx], 0.0):
                return True
        return False


    start = ti.default_timer()
    # B, U, iters = lu.lll_apx_sparse(As, iterations=300, early_exit_func=check_exit_state)
    B, U, iters = lu.lll_apx(As.todense(), iterations=300, early_exit_func=check_exit)
    idx = int(np.argmax(B[n, :]))
    assert idx < n, f"Expected x_p column index < {n}, got {idx}"
    N = B[0:n, 0:idx]
    x_p = B[0:n, idx]
    end = ti.default_timer()
    print('LLL iterations:', iters, ' Time (s):', end - start)
    assert sparse_allclose(A @ N, np.zeros((A.shape[0], N.shape[1])))

if __name__ == "__main__":
    main()