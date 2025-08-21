import fpylll as fpy
import ntl_wrapper as ntl
import dikin_utils as du
import jsplib_loader as jl
import hsnf
import numpy as np

def test_jsp():
    instance = jl.get_instances()['abz3']
    model = instance.as_gurobi_balas_model(use_big_m=True)
    model.update()
    model.optimize()
    A = model.getA()
    b = np.array(model.getAttr("RHS")).reshape((-1, 1))
    B = np.hstack((A.toarray(), b)).astype(np.int64, order='C')

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