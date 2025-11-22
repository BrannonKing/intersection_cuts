import numpy as np
import sys
sys.path.insert(0, '/home/brannon/Documents/Research/intersection_cuts')
import dikin_utils as du

# Test 1: Random square matrix
np.random.seed(42)
for test_id in range(10):
    A = np.random.randn(5, 5)
    U = du.to_U_via_LU(A, mult=100)
    det = np.linalg.det(U)
    is_int = np.allclose(U, np.round(U))
    print(f"Test {test_id}: det={det:.6f}, is_int={is_int}, unimodular={abs(abs(det) - 1) < 1e-5}")
    assert abs(abs(det) - 1) < 1e-5, f"Test {test_id} failed: det={det}"
    assert is_int, f"Test {test_id} failed: not integer"

# Test 2: Orthogonal matrix (hardest case)
m, n = 2, 20
A = np.random.randn(m, n)
_, _, Vh = np.linalg.svd(A, full_matrices=True)
T = Vh.T  # 20x20 orthogonal

U = du.to_U_via_LU(T, mult=512)
det = np.linalg.det(U)
is_int = np.allclose(U, np.round(U))
print(f"\nOrthogonal matrix test: det={det:.6f}, is_int={is_int}, unimodular={abs(abs(det) - 1) < 1e-5}")
assert abs(abs(det) - 1) < 1e-5, f"Orthogonal test failed: det={det}"
assert is_int, "Orthogonal test failed: not integer"

print("\n✅ All tests passed!")
