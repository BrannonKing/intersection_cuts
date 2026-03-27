import jsplib_loader as jl
import gurobipy as gp
import gurobi_utils as gu
import lll_torch_utils as ltu
import numba as nb
import numpy as np
import scipy.sparse as sps
import timeit as ti
import torch

def main():
    instance = jl.get_instances()['abz4']
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
    Ag = torch.from_numpy(A.todense()).cuda()

    def check_exit(B, U, iteration):
        idx = B[n].argmax().item()  # slow!
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
            Bc = B[0:n, 0:idx]
            res = Ag @ Bc
            if torch.allclose(res, torch.zeros_like(res), atol=1e-6):
                return True
        return False

    start = ti.default_timer()
    B, iters = ltu.lll_apx_torch(As.todense(), iterations=300, early_exit_func=check_exit)
    idx = int(torch.argmax(B[n, :]).item())
    assert idx < n, f"Expected x_p column index < {n}, got {idx}"
    N = B[0:n, 0:idx].numpy()
    x_p = B[0:n, idx].numpy()
    end = ti.default_timer()
    print('LLL iterations:', iters, ' Time (s):', end - start)
    ret = A @ N
    assert np.allclose(ret, np.zeros((A.shape[0], N.shape[1])), atol=1e-6), f"Expected A @ N ≈ 0, got {ret}"    

if __name__ == "__main__":
    main()