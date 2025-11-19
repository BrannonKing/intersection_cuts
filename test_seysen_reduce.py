import numpy as np
import pytest

import dikin_utils as du


def _upper_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Return a deterministic upper-triangular matrix derived from ``matrix``.

    Expectations we enforce across all Seysen reducers:
            * Returned transformation matrices are integer-valued and unimodular.
            * Applying the transform to the original R keeps it upper-triangular (within tolerance).
            * The transformed R should not worsen the Gram off-diagonal Frobenius norm.
            * Implementations (`seysen_reduce`, `_iter`, `_blaster`) must agree on the
                resulting unimodular matrix when started from the same upper-triangular input.
    """
    r = np.linalg.qr(matrix.astype(float))[1]
    diag_sign = np.sign(np.diag(r))
    diag_sign[diag_sign == 0.0] = 1.0
    return r * diag_sign[np.newaxis, :]


def _gram_offdiag_norm(mat: np.ndarray) -> float:
    diag = np.diag(np.diag(mat))
    return float(np.linalg.norm(mat - diag))


def _random_upper_triangular(size: int, rng: np.random.Generator) -> np.ndarray:
    base = rng.normal(size=(size, size))
    upper = np.triu(base)
    # Ensure non-zero diagonal to keep matrix non-singular and well-conditioned
    diag = np.diag(upper)
    bump = rng.uniform(low=0.5, high=1.5, size=size)
    upper[np.diag_indices(size)] = np.where(np.abs(diag) < 0.2, bump, diag)
    return upper


def test_seysen_reduce_trivial_identity():
    r = np.array([[3.7]])
    u = du.seysen_reduce(r.copy())
    assert np.array_equal(u, np.array([[1]], dtype=np.int64))


def test_seysen_reduce_reduces_offdiag_norm():
    base = np.array(
        [
            [3.0, 1.0, 0.5],
            [2.0, 2.0, 1.0],
            [0.0, 1.0, 2.0],
        ]
    )
    r = _upper_from_matrix(base)
    u = du.seysen_reduce(r.copy())
    r_reduced = r @ u

    assert np.issubdtype(u.dtype, np.integer)
    assert pytest.approx(abs(np.linalg.det(u)), rel=1e-12, abs=1e-12) == 1.0
    assert np.allclose(r_reduced, np.triu(r_reduced), atol=1e-9)

    original = _gram_offdiag_norm(r.T @ r)
    reduced = _gram_offdiag_norm(r_reduced.T @ r_reduced)
    assert reduced <= original + 1e-9
    assert reduced <= original - 1e-6


@pytest.mark.parametrize(
    "size, seed",
    [(1, 0), (2, 1), (3, 2), (4, 7), (5, 11), (6, 23), (7, 42)],
)
def test_seysen_variants_agree_and_reduce(size: int, seed: int):
    rng = np.random.default_rng(seed)
    r0 = _random_upper_triangular(size, rng)
    baseline = _gram_offdiag_norm(r0.T @ r0)

    # Recursive reference implementation
    r_recursive = r0.copy()
    u_recursive = du.seysen_reduce(r_recursive)

    # Iterative implementation
    r_iter = r0.copy()
    u_iter = du.seysen_reduce_iter(r_iter)

    # Blaster implementation (in-place with supplied U)
    r_blast = r0.copy()
    u_blast = np.eye(size, dtype=np.int64)
    du.seysen_reduce_blaster(r_blast, u_blast)

    assert np.issubdtype(u_recursive.dtype, np.integer)
    assert np.issubdtype(u_iter.dtype, np.integer)
    assert np.issubdtype(u_blast.dtype, np.integer)

    assert np.array_equal(u_recursive, u_iter)
    assert np.array_equal(u_recursive, u_blast)

    u = u_recursive
    det = float(np.linalg.det(u.astype(float)))
    assert pytest.approx(abs(det), rel=1e-9, abs=1e-9) == 1.0

    u_inv = np.round(np.linalg.inv(u)).astype(np.int64)
    assert np.array_equal(u_inv @ u, np.eye(size, dtype=np.int64))

    for label, r_new in {"recursive": r_recursive, "iter": r_iter, "blaster": r_blast}.items():
        assert np.allclose(r_new, r0 @ u, atol=1e-9)
        assert np.allclose(r_new, np.triu(r_new), atol=1e-9), f"{label} output lost triangular form"
        reduced = _gram_offdiag_norm(r_new.T @ r_new)
        assert reduced <= baseline + 1e-9, f"{label} increased Gram off-diagonal norm"


def test_seysen_reduce_iter_matches_recursive_full_rank():
    base = np.array(
        [
            [5, 5, 5, -1, -4],
            [5, -4, -3, -4, -2],
            [0, -4, 3, -3, 3],
            [-2, 3, 5, 5, -3],
            [1, -1, -3, 1, 4],
        ],
        dtype=float,
    )
    r = _upper_from_matrix(base)

    u_recursive = du.seysen_reduce(r.copy())
    u_iter = du.seysen_reduce_iter(r.copy())

    assert np.array_equal(u_iter, u_recursive)

    original = _gram_offdiag_norm(r.T @ r)
    reduced = _gram_offdiag_norm((r @ u_iter).T @ (r @ u_iter))
    assert reduced <= original + 1e-9
    assert reduced <= original - 1e-6


def test_seysen_reduce_blaster_matches_recursive_and_in_place():
    base = np.array(
        [
            [5, 5, 5, -1, -4],
            [5, -4, -3, -4, -2],
            [0, -4, 3, -3, 3],
            [-2, 3, 5, 5, -3],
            [1, -1, -3, 1, 4],
        ],
        dtype=float,
    )
    r = _upper_from_matrix(base)
    r_work = r.copy()
    u_work = np.eye(r.shape[0], dtype=np.int64)

    du.seysen_reduce_blaster(r_work, u_work)

    u_expected = du.seysen_reduce(r.copy())

    assert np.array_equal(u_work, u_expected)
    assert np.allclose(r_work, r @ u_work)
    assert np.allclose(r_work, np.triu(r_work), atol=1e-9)

    original = _gram_offdiag_norm(r.T @ r)
    reduced = _gram_offdiag_norm(r_work.T @ r_work)
    assert reduced <= original + 1e-9
    assert reduced <= original - 1e-6

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(sys.argv))