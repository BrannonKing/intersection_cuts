import numpy as np
import pytest

scipy = pytest.importorskip("scipy.linalg")

import scipy.linalg as scl


def _extend_null_space_to_full_basis(A: np.ndarray):
    """Compute null space of A and extend it to an orthonormal n×n basis."""
    N = scl.null_space(A)
    # numpy's complete QR produces an n×n orthonormal basis; works even when N has 0 columns.
    Q_full, _ = np.linalg.qr(N, mode="complete")
    return N, Q_full


def _random_full_row_rank_matrix(rng: np.random.Generator, m: int, n: int) -> np.ndarray:
    assert n >= m
    while True:
        A = rng.integers(-5, 6, size=(m, n))
        if np.linalg.matrix_rank(A) == m:
            return A.astype(float)


@pytest.mark.parametrize("m, n", [(2, 4), (3, 6), (4, 7)])
def test_full_basis_preserves_null_space(m: int, n: int):
    rng = np.random.default_rng(2025 + 31 * m + 17 * n)
    A = _random_full_row_rank_matrix(rng, m, n)

    N, Q_full = _extend_null_space_to_full_basis(A)
    null_dim = n - m

    # The extended basis must be square, orthonormal, and determinant ±1 (up to numerical noise).
    assert Q_full.shape == (n, n)
    assert np.allclose(Q_full.T @ Q_full, np.eye(n), atol=1e-10)
    det = np.linalg.det(Q_full)
    assert pytest.approx(abs(det), abs=1e-10) == 1

    if null_dim:
        # Columns spanning the null space remain in the null space of A.
        residual = A @ Q_full[:, :null_dim]
        assert np.allclose(residual, 0, atol=1e-10)

        # The projection induced by the extended columns matches the original null space projector.
        projector_extended = Q_full[:, :null_dim] @ Q_full[:, :null_dim].T
        projector_original = N @ N.T
        assert np.linalg.norm(projector_extended - projector_original, ord="fro") < 1e-10
    else:
        # No null space: QR should reduce to the identity and N should be empty.
        assert N.size == 0

    # The complementary block should map to an invertible square matrix under A (full row rank).
    orth_complement = Q_full[:, null_dim:]
    mapped = A @ orth_complement
    assert np.linalg.matrix_rank(mapped) == m


@pytest.mark.parametrize("m, n", [(2, 5), (3, 7)])
def test_extended_basis_is_numerically_stable(m: int, n: int):
    rng = np.random.default_rng(4040 + 13 * m + 19 * n)
    A = _random_full_row_rank_matrix(rng, m, n)

    _, Q_full = _extend_null_space_to_full_basis(A)
    null_dim = n - m

    # Check condition numbers to ensure the orthogonal complement is well-behaved.
    if null_dim:
        # A @ complement should have a moderate condition number since A has full row rank.
        complement = Q_full[:, null_dim:]
    else:
        complement = Q_full
    mapped = A @ complement
    # Because mapped is square (m x m), its condition number is defined; guard zero singular values.
    s = np.linalg.svd(mapped, compute_uv=False)
    assert s.min() > 1e-12
    cond = s.max() / s.min()
    assert cond < 1e5
