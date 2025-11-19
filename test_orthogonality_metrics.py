import numpy as np
import pytest

import dikin_utils as du


def test_identity_is_perfect_for_both_orientations():
    mat = np.eye(4)
    assert du.orthogonality_measure_1(mat) == pytest.approx(0.0)
    assert du.orthogonality_measure_1(mat, by_rows=True) == pytest.approx(0.0)
    assert du.orthogonality_measure_2(mat) == pytest.approx(0.0)


def test_shear_off_diagonal_penalty_matches_expectation():
    shear = np.array([[1.0, 1.0], [0.0, 1.0]])
    expected_off_diag = np.sqrt(2)
    assert du.orthogonality_measure_1(shear, include_diagonal=False) == pytest.approx(expected_off_diag)
    assert du.orthogonality_measure_1(shear, by_rows=True, include_diagonal=False) == pytest.approx(expected_off_diag)


def test_deviation_matches_normalised_frobenius():
    mat = np.array([[2.0, 0.0], [0.0, 1.0]])
    norms = np.linalg.norm(mat, axis=0)
    normalised = mat / norms
    expected = du.orthogonality_measure_1(normalised)
    assert du.measure_orthogonality_deviation(mat) == pytest.approx(expected)


def test_measure_orthogonality_infinite_for_degenerate_columns():
    degenerate = np.array([[1.0, 0.0], [0.0, 0.0]])
    assert np.isinf(du.measure_orthogonality(degenerate))


def test_difference_observes_orientation_flag():
    A = np.array([[1.0, 0.0], [0.0, 1.0]])
    B = np.array([[1.0, 0.0], [0.1, 0.9]])
    column_view = du.difference(A, B)
    row_view = du.difference(A.T, B.T, by_rows=True)
    assert row_view == pytest.approx(column_view)


def test_pairwise_hyperplane_angles_handles_rows_and_columns():
    normals = np.array([[1.0, 0.0], [1.0, 1.0]])
    angles_rows = du.pairwise_hyperplane_angles(normals)
    angles_cols = du.pairwise_hyperplane_angles(normals.T, by_rows=False)
    np.testing.assert_allclose(angles_rows, angles_cols)


def test_pairwise_hyperplane_angles_honours_acute_toggle():
    normals = np.array([[1.0, 0.0], [-1.0, 0.0]])
    obtuse_angles = du.pairwise_hyperplane_angles(normals, acute=False)
    acute_angles = du.pairwise_hyperplane_angles(normals, acute=True)
    assert obtuse_angles[0, 1] == pytest.approx(np.pi)
    assert acute_angles[0, 1] == pytest.approx(0.0)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(sys.argv))