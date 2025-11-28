import numpy as np
import fpylll as fpy
import pytest
from .. import dikin_utils as du

def test_seysen_reduces_orthogonality_measure():
    """Test that Seysen reduction improves orthogonality measure."""
    # from SOLVING A SYSTEM OF LINEAR DIOPHANTINE EQUATIONS WITH LOWER AND UPPER BOUNDS ON THE VARIABLES
    A = np.array([[6, 1, 3, 3, 0, 0], [0, 0, 0, 0, 2, 1], [0, 0, 4, 1, 0, 2]], dtype=np.int64)
    m, n = A.shape
    b = np.array([[17], [11], [27]], dtype=np.int64)
    N1 = max(np.linalg.norm(b, np.inf).item(), np.linalg.norm(A, np.inf).item()) * 4
    N2 = N1 * 4
    B = np.block([[np.eye(n, dtype=np.int64), np.zeros((n, 1), dtype=np.int64)],
                        [np.zeros((1, n), dtype=np.int64), np.array([N1])],
                        [N2 * A, -N2 * b]]).astype(np.int64, order='C')

    B = np.array([[3, 1],[2, 2]])
    Q, R = np.linalg.qr(B)
    U = np.eye(R.shape[0], dtype=np.int64)
    U = du.seysen_reduce_iter(R)
    
    # Compare orthogonality before and after
    original_gram = B.T @ B
    BU = B @ U
    reduced_gram = (BU.T @ BU).astype(np.int64)
    
    # Measure orthogonality (off-diagonal norm)
    orig_measure = du.orthogonality_measure_1(B, include_diagonal=False)
    reduced_measure = du.orthogonality_measure_1(BU, include_diagonal=False)
    
    # Verify Seysen improved orthogonality
    assert reduced_measure <= orig_measure, "Seysen should improve or maintain orthogonality"
    
    # Check that we preserved the lattice (U is unimodular)
    det_U = np.linalg.det(U)
    assert abs(abs(det_U) - 1) < 1e-6, "U should be unimodular (det = ±1)"
    
    # Check lattice is preserved
    det_orig = np.linalg.det(original_gram)
    det_reduced = np.linalg.det(reduced_gram)
    assert abs(det_orig - det_reduced) < 1e-6, "Lattice volume should be preserved"
