import ntl_wrapper as ntl
import hsnf
import numpy as np
import fpylll as fpy

def lll_fpyll_cols(B, delta=0.75):
    """
    Perform LLL reduction using fpylll.
    :param B: Input matrix to be reduced.
    :param delta: LLL parameter, typically between 0.99 and 0.999.
    :return: Reduced basis matrix.
    """
    B2 = fpy.IntegerMatrix.from_matrix(B.T)
    U = fpy.IntegerMatrix(1, 1)
    print("  Initial norm:", B2[-1].norm())
    # it does rows by default, so we need to transpose it to do columns
    B3 = fpy.LLL.reduction(B2, U=U, delta=delta)
    print("  After norm:", B3[-1].norm())
    result = np.zeros((U.nrows, U.ncols), dtype=np.int64)
    U.to_matrix(result)
    B3.transpose()
    B3.to_matrix(B)
    return result.T

def lll_fpyll_rows(B, delta=0.75):
    """
    Perform LLL reduction using fpylll.
    :param B: Input matrix to be reduced.
    :param delta: LLL parameter, typically between 0.99 and 0.999.
    :return: Reduced basis matrix.
    """
    B2 = fpy.IntegerMatrix.from_matrix(B)
    U = fpy.IntegerMatrix(1, 1)
    print("  Initial norm:", B2[-1].norm())
    B3 = fpy.LLL.reduction(B2, U=U, delta=delta)
    print("  After norm:", B3[-1].norm())
    result = np.zeros((U.nrows, U.ncols), dtype=np.int64)
    U.to_matrix(result)
    B3.to_matrix(B)
    return result

def test_paper():
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
    U = lll_fpyll_rows(B2, 0.75)
    assert np.allclose(U @ B, B2)

    B2 = B.copy()
    U = lll_fpyll_cols(B2, 0.75)
    assert np.allclose(B @ U, B2)

    # rank, det, U = ntl.lll_left(B2, 75, 100)
    # assert np.allclose(U @ B.T, B2)

    # B2, U = hsnf.column_style_hermite_normal_form(B)
    # assert np.allclose(B @ U, B2)
    # B2, U = hsnf.row_style_hermite_normal_form(B)
    # assert np.allclose(U @ B, B2)
    print(B2)

test_paper()