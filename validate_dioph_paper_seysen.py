import ntl_wrapper as ntl
import dikin_utils as du
import hsnf
import numpy as np
import fpylll as fpy

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

    B = np.array([[3, 1],[2, 2]])
    Q, R = np.linalg.qr(B)
    U = np.eye(R.shape[0], dtype=np.int64)
    U = du.seysen_reduce_iter(R)
    print(U)
    print((Q @ R * 100).astype(np.int64))
    print(B @ U)
    
    # Compare orthogonality before and after
    original_gram = B.T @ B
    BU = B @ U
    reduced_gram = (BU.T @ BU).astype(np.int64)
    
    print("\nOriginal Gram matrix B.T @ B =")
    print(original_gram)
    print("\nReduced basis Gram matrix =")
    print(reduced_gram)
    
    # Measure orthogonality (off-diagonal norm)
    orig_measure = du.orthogonality_measure_1(B, include_diagonal=False)
    reduced_measure = du.orthogonality_measure_1(BU, include_diagonal=False)
    
    print(f"\nOrthogonality measures:")
    print(f"Original: {orig_measure:.2e}")
    print(f"Reduced:  {reduced_measure:.2e}")
    print(f"Improvement factor: {orig_measure / reduced_measure:.2f}")
    
    # Check that we preserved the lattice
    print(f"\nDeterminant check:")
    print(f"Original det(B.T @ B): {np.linalg.det(original_gram):.2e}")
    print(f"Reduced det(reduced.T @ reduced): {np.linalg.det(reduced_gram):.2e}")
    print(f"U determinant: {np.linalg.det(U)}")  # Should be ±1 for unimodular

test_paper()