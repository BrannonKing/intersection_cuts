import jsplib_loader as jl
import gurobipy as gp
import gurobi_utils as gu
import ntl_wrapper as ntl
import numpy as np
import scipy.sparse as sps
import timeit as ti

def main():
    instance = jl.get_instances()['abz5']
    model = instance.as_gurobi_balas_model(use_big_m=True)
    A, b, c, l, u = gu.get_A_b_c_l_u(model, keep_sparse=True)
    N1 = 100000
    N2 = 1000000
    # np.savetxt('jsp_N.txt', N, fmt='%d')
    # print(x_p)
    senses = [c.Sense for c in model.getConstrs()]
    assert all(s == gp.GRB.GREATER_EQUAL for s in senses)

    As = sps.block_array([[A, -sps.eye(A.shape[0])]], format='csr')

    # x_p, N = gu.nullspace_and_offset_via_LLL(As.toarray(), b, N1, N2)
    # assert np.allclose(As @ N, 0)

    start = ti.default_timer()
    N, x_p, iters = ntl.lll_apx_sparse_early(As, b, 1500, N1, N2)
    end = ti.default_timer()
    print('LLL iterations:', iters, ' Time (s):', end - start)
    assert np.allclose((As @ N).todense(), 0)

    assert np.allclose(As @ x_p, b)

    mdl2 = gu.substitute(model, N, x_p, 'skip')
    # mdl2.params.Presolve = 2 
    mdl2.params.DualReductions = 0
    mdl2.optimize()
    print('Optimal value after LLL substitution:', mdl2.ObjVal)

if __name__ == "__main__":
    main()