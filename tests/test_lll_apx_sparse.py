import numpy as np
import scipy.sparse as sps

from lll_utils import lll_apx, lll_apx_sparse


def _make_sparse_like_matrix(seed=0, m=24, n=12, p_zero=0.75):
    rng = np.random.default_rng(seed)
    A = rng.integers(-3, 4, size=(m, n)).astype(float)
    A[rng.random((m, n)) < p_zero] = 0.0
    return A


def test_lll_apx_sparse_matches_lll_apx_on_dense_input():
    A = _make_sparse_like_matrix(seed=1)

    B_dense, U_dense, it_dense = lll_apx(A, iterations=4)
    B_sparse, U_sparse, it_sparse = lll_apx_sparse(A, iterations=4)

    assert np.allclose(B_dense, B_sparse)
    assert np.array_equal(U_dense, U_sparse)
    assert it_dense == it_sparse


def test_lll_apx_sparse_accepts_scipy_sparse_input():
    A = _make_sparse_like_matrix(seed=2)
    A_csr = sps.csr_matrix(A)

    B_dense_input, U_dense_input, it_dense_input = lll_apx_sparse(A, iterations=3)
    B_csr_input, U_csr_input, it_csr_input = lll_apx_sparse(A_csr, iterations=3)

    assert np.allclose(B_dense_input, B_csr_input)
    assert np.array_equal(U_dense_input, U_csr_input)
    assert it_dense_input == it_csr_input
