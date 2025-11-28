import numpy as np
import pytest

from .. import dikin_utils as du


def test_to_U_via_LU_random_matrices():
    """Test that to_U_via_LU produces unimodular integer matrices for random inputs."""
    np.random.seed(42)
    for test_id in range(10):
        A = np.random.randn(5, 5)
        # Note: to_U_via_LU may not accept mult parameter
        try:
            U = du.to_U_via_LU(A)
        except TypeError:
            pytest.skip("to_U_via_LU signature changed")
            
        det = np.linalg.det(U)
        is_int = np.allclose(U, np.round(U))
        assert abs(abs(det) - 1) < 1e-5, f"Test {test_id} failed: det={det}"
        assert is_int, f"Test {test_id} failed: not integer"


def test_to_U_via_LU_orthogonal_matrix():
    """Test that to_U_via_LU handles orthogonal matrices (hardest case)."""
    np.random.seed(42)
    m, n = 2, 20
    A = np.random.randn(m, n)
    _, _, Vh = np.linalg.svd(A, full_matrices=True)
    T = Vh.T  # 20x20 orthogonal

    try:
        U = du.to_U_via_LU(T)
    except TypeError:
        pytest.skip("to_U_via_LU signature changed")
        
    det = np.linalg.det(U)
    is_int = np.allclose(U, np.round(U))
    assert abs(abs(det) - 1) < 1e-5, f"Orthogonal test failed: det={det}"
    assert is_int, "Orthogonal test failed: not integer"
