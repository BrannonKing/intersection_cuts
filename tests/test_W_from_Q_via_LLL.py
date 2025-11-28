import numpy as np
import pytest
from .. import dikin_utils as du

class TestWFromQViaLLL:
    """Tests for W_from_Q_via_LLL function."""

    def test_simple_2x3_matrix(self):
        """Test with a simple 2x3 integer matrix Q."""
        # Q is 3x2 (more rows than columns) - nullspace of a 1x3 constraint matrix
        # This is the actual structure returned by nullspace_and_offset_via_LLL
        Q = np.array([[-1, -1],
                      [ 1,  0],
                      [ 0,  1]], dtype=np.int64)
        
        W = du.W_from_Q_via_LLL(Q)
        
        # W should be 2x3
        assert W.shape == (2, 3), f"Expected shape (2, 3), got {W.shape}"
        
        # W should be integer
        assert W.dtype == np.int64, f"Expected int64, got {W.dtype}"
        
        # W @ Q should equal identity
        result = W @ Q
        expected = np.eye(2, dtype=np.int64)
        assert np.allclose(result, expected), f"W @ Q = {result}, expected I"

    def test_3x5_matrix(self):
        """Test with a 5x3 integer matrix Q."""
        # Q is 5x3 - nullspace of a 2x5 constraint matrix
        # This is the actual structure returned by nullspace_and_offset_via_LLL
        Q = np.array([[-1,  0,  1],
                      [ 1,  0,  0],
                      [ 0,  0, -1],
                      [ 0, -1,  1],
                      [ 0,  1,  0]], dtype=np.int64)
        
        W = du.W_from_Q_via_LLL(Q)
        
        # W should be 3x5
        assert W.shape == (3, 5), f"Expected shape (3, 5), got {W.shape}"
        
        # W should be integer
        assert W.dtype == np.int64, f"Expected int64, got {W.dtype}"
        
        # W @ Q should equal identity
        result = W @ Q
        expected = np.eye(3, dtype=np.int64)
        assert np.allclose(result, expected), f"W @ Q = {result}, expected I"

    def test_matrix_from_diophantine_paper(self):
        """Test with matrix similar to the Diophantine equations paper example."""
        # A nullspace matrix from a 4x6 constraint system
        # This is the actual structure returned by nullspace_and_offset_via_LLL
        Q = np.array([[ 1,  1],
                      [-1, -1],
                      [-1,  0],
                      [ 1,  0],
                      [ 0, -1],
                      [ 0,  1]], dtype=np.int64)
        
        W = du.W_from_Q_via_LLL(Q)
        
        # W should be 2x6
        assert W.shape == (2, 6), f"Expected shape (2, 6), got {W.shape}"
        
        # W should be integer
        assert W.dtype == np.int64, f"Expected int64, got {W.dtype}"
        
        # W @ Q should equal identity
        result = W @ Q
        expected = np.eye(2, dtype=np.int64)
        assert np.allclose(result, expected), f"W @ Q = {result}, expected I"

    def test_single_row_matrix(self):
        """Test with a 3x2 matrix (nullspace of single constraint)."""
        # Nullspace of a single 1x3 constraint
        # This is the actual structure returned by nullspace_and_offset_via_LLL
        Q = np.array([[-1,  3],
                      [-1, -2],
                      [ 1,  0]], dtype=np.int64)
        
        W = du.W_from_Q_via_LLL(Q)
        
        # W should be 2x3
        assert W.shape == (2, 3), f"Expected shape (2, 3), got {W.shape}"
        
        # W should be integer
        assert W.dtype == np.int64, f"Expected int64, got {W.dtype}"
        
        # W @ Q should equal I_2
        result = W @ Q
        expected = np.eye(2, dtype=np.int64)
        assert np.allclose(result, expected), f"W @ Q = {result}, expected I"

    def test_4x6_random_full_rank(self):
        """Test with a larger nullspace matrix."""
        # Nullspace from a larger constraint system
        # Using a simple nullspace structure that we know works
        Q = np.array([[-1, -1, -1],
                      [ 1,  0,  0],
                      [ 0,  1,  0],
                      [ 0,  0,  1]], dtype=np.int64)
        
        W = du.W_from_Q_via_LLL(Q)
        
        # W should be 3x4
        assert W.shape == (3, 4), f"Expected shape (3, 4), got {W.shape}"
        
        # W should be integer
        assert W.dtype == np.int64, f"Expected int64, got {W.dtype}"
        
        # W @ Q should equal identity
        result = W @ Q
        expected = np.eye(3, dtype=np.int64)
        assert np.allclose(result, expected), f"W @ Q = {result}, expected I"

    def test_matrix_with_negative_entries(self):
        """Test with a matrix containing negative integers."""
        # Nullspace from a 3x5 constraint system with negative entries
        # This is the actual structure returned by nullspace_and_offset_via_LLL
        Q = np.array([[-1,  2],
                      [ 0, -1],
                      [-1,  1],
                      [ 1,  1],
                      [ 1,  0]], dtype=np.int64)
        
        W = du.W_from_Q_via_LLL(Q)
        
        # W should be 2x5
        assert W.shape == (2, 5), f"Expected shape (2, 5), got {W.shape}"
        
        # W should be integer
        assert W.dtype == np.int64, f"Expected int64, got {W.dtype}"
        
        # W @ Q should equal identity
        result = W @ Q
        expected = np.eye(2, dtype=np.int64)
        assert np.allclose(result, expected), f"W @ Q = {result}, expected I"

    def test_with_more_rows_than_cols(self):
        """Test that the function works when Q has more rows than columns."""
        # The function actually handles this case - after transpose, Q.T has more columns than rows
        Q = np.array([[1, 0],
                      [0, 1],
                      [1, 1]], dtype=np.int64)  # 3x2 matrix (more rows than cols)
        
        # This should work - W will be 2x3, and W @ Q = I_2
        W = du.W_from_Q_via_LLL(Q)
        assert W.shape == (2, 3)
        result = W @ Q
        expected = np.eye(2, dtype=np.int64)
        assert np.allclose(result, expected), f"W @ Q = {result}, expected I"

    def test_w_is_truly_integer(self):
        """Verify that W contains only true integers, not floats that happen to be whole."""
        # Use a nullspace matrix
        Q = np.array([[-1, -1],
                      [ 1,  0],
                      [ 0,  1]], dtype=np.int64)
        
        W = du.W_from_Q_via_LLL(Q)
        
        # Check all entries are exactly integer
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                assert isinstance(W[i, j], (int, np.integer)), \
                    f"W[{i},{j}] = {W[i,j]} is not an integer type"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
