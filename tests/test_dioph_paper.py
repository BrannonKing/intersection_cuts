import hsnf
import numpy as np
import fpylll as fpy
import pytest

pytest.importorskip("ntl_wrapper")

from .. import ntl_wrapper as ntl
from .. import dikin_utils as du

def test_lll_methods_on_diophantine_system():
    """Test various LLL methods on a Diophantine system from literature."""
    # from SOLVING A SYSTEM OF LINEAR DIOPHANTINE EQUATIONS WITH LOWER AND UPPER BOUNDS ON THE VARIABLES
    A = np.array([[6, 1, 3, 3, 0, 0], [0, 0, 0, 0, 2, 1], [0, 0, 4, 1, 0, 2]], dtype=np.int64)
    m, n = A.shape
    b = np.array([[17], [11], [27]], dtype=np.int64)
    N1 = max(np.linalg.norm(b, np.inf).item(), np.linalg.norm(A, np.inf).item()) * 4
    N2 = N1 * 4
    B = np.block([[np.eye(n, dtype=np.int64), np.zeros((n, 1), dtype=np.int64)],
                        [np.zeros((1, n), dtype=np.int64), np.array([N1])],
                        [N2 * A, -N2 * b]]).astype(np.int64, order='C')

    B2 = B.copy()
    rank, det, U = ntl.lll(B2, 75, 100)
    assert np.allclose(B @ U, B2)

    B2 = B.copy()
    U = du.lll_fpylll_rows(B2, 0.75)
    assert np.allclose(U @ B, B2)

    B2 = B.copy()
    U = du.lll_fpylll_cols(B2, 0.75)
    assert np.allclose(B @ U, B2)

    # rank, det, U = ntl.lll_left(B2, 75, 100)
    # assert np.allclose(U @ B.T, B2)

    # B2, U = hsnf.column_style_hermite_normal_form(B)
    # assert np.allclose(B @ U, B2)
    # B2, U = hsnf.row_style_hermite_normal_form(B)
    # assert np.allclose(U @ B, B2)

    B2 = B.copy()
    # Q = mgs_orthogonal_cols(B2, None)
    # print("Q:", Q)
    # Q2 = Q.T @ Q
    # Q2 = np.round(Q2, decimals=10)
    # print("Q2:", Q2)

    U = du.lll_brans_cols(B2, 0.8)
    assert np.allclose(B @ U, B2)
    # Verify result has reasonable norm
    assert np.linalg.norm(B2) < np.linalg.norm(B) * 100, "LLL should not explode norms"

test_paper()