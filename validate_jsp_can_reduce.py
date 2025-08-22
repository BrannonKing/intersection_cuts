import fpylll as fpy
import ntl_wrapper as ntl
import dikin_utils as du
import jsplib_loader as jl
import hsnf
import numpy as np
import linetimer as lt
import knapsack_loader as kl

def test_jsp():
    # instance = jl.get_instances()['abz4']
    # model = instance.as_gurobi_balas_model(use_big_m=True)
    instances = kl.generate(1, 20, 50, 5, 10, 1000, equality=False)
    model = next(instances)
    model.update()
    A = model.getA().toarray()
    b = np.array(model.getAttr("RHS")).reshape((-1, 1))
    H = np.hstack((A, b)).astype(np.int64, order='C')

    print("  Before max column norm:", np.linalg.norm(H, axis=0).max())
    with lt.CodeTimer("  LLL time", silent=True) as c2:
        rank, det, U = ntl.lll(H, 9, 10)
        # U = du.lll_fpylll_cols(H, 0.9, verbose=1)
    print("  After max column norm:", np.linalg.norm(H, axis=0).max())
    print("  A norm:", np.linalg.norm(A, axis=0).max())
    print("  AU norm:", np.linalg.norm(A @ U[:-1,:], axis=0).max())
    # Uinv = np.linalg.inv(U)
    # u = np.array(model.getAttr("UB")).reshape((-1, 1))
    # print("  Uu norm:", np.linalg.norm(Uinv @ u, axis=0).max())
    print(f"  LLL took: {c2.took:.2f} ms")


    # B2 = B.copy()
    # rank, det, U = ntl.lll(B2, 75, 100)
    # assert np.allclose(B @ U, B2)

    # B2 = B.copy()
    # U = du.lll_fpylll_rows(B2, 0.75)
    # assert np.allclose(U @ B, B2)

    # B2 = B.copy()
    # U = du.lll_fpylll_cols(B2, 0.999999, verbose=2)
    # assert np.allclose(B @ U, B2)

    # rank, det, U = ntl.lll_left(B2, 75, 100)
    # assert np.allclose(U @ B.T, B2)

    # B2, U = hsnf.column_style_hermite_normal_form(B)
    # assert np.allclose(B @ U, B2)
    # B2, U = hsnf.row_style_hermite_normal_form(B)
    # assert np.allclose(U @ B, B2)
    # print(B2)

    # B2 = B.copy()
    # Q = mgs_orthogonal_cols(B2, None)
    # print("Q:", Q)
    # Q2 = Q.T @ Q
    # Q2 = np.round(Q2, decimals=10)
    # print("Q2:", Q2)

    # B2 = B.copy()
    # U = du.lll_brans_cols(B2, 0.8)
    # assert np.allclose(B @ U, B2)
    # print(B2)

test_jsp()