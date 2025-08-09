import ntl_wrapper as ntl
import dikin_utils as du
import hsnf
import numpy as np
import fpylll as fpy

def mgs_orthogonal_cols(B, Q=None, start=0):
    """
    Performs Modified Gram-Schmidt orthogonalization on the columns of a matrix
    without normalizing the resulting vectors.

    :param B: A numpy array where each column is a vector.
    :return: A matrix Q with orthogonal columns.
    """
    # Create a copy to avoid modifying the original matrix
    if Q is None:
        Q = B.astype(np.float64, copy=True)  # could use higher precision here
    else:
        Q[:, start:] = B[:, start:]
    m, n = Q.shape

    for i in range(start, n):
        # The current vector we are orthogonalizing against
        q_i = Q[:, i]
        
        # Calculate the squared L2 norm: ||q_i||^2
        norm_sq = np.dot(q_i.T, q_i)

        # Skip if the vector is a zero vector to avoid division by zero
        if norm_sq < 1e-12: # Using a tolerance for floating-point comparisons
            continue

        # Orthogonalize all subsequent vectors (j > i) against q_i
        for j in range(i + 1, n):
            # Calculate the dot product <Q[:, j], q_i>
            r = np.dot(Q[:, j].T, q_i) / norm_sq
            
            # Subtract the projection of Q[:, j] onto q_i.
            # The projection is (r / ||q_i||^2) * q_i
            Q[:, j] -= r * q_i
            
    return Q


def lll_brans_cols(B, delta=0.75):
    """
    LLL algorithm for column vectors.
    :param B: Input matrix.
    :param delta: Delta parameter for LLL.
    :return: Matrix with reduced columns.
    """
    Q = mgs_orthogonal_cols(B)
    mu = np.zeros((B.shape[1], B.shape[1]), dtype=np.float64)
    def update_mu(st):
        for x in range(st, B.shape[1]):
            for y in range(x):
                denominator = np.dot(Q[:, y], Q[:, y])
                if denominator < 1e-12:  # Handle zero or near-zero denominator
                    mu[x, y] = 0.0
                else:
                    mu[x, y] = np.dot(B[:, x], Q[:, y]) / denominator

    U = np.eye(B.shape[1], dtype=np.int32)
    k = 1
    update_mu(0)
    while k < B.shape[1]:
        start = -1
        for j in range(k - 1, -1, -1):
            if abs(mu[k, j]) > 0.5:
                q = round(mu[k, j])
                B[:, k] -= q * B[:, j]
                U[:, k] -= q * U[:, j]
                start = j
        
        if start >= 0:
            Q = mgs_orthogonal_cols(B, Q, start)
            update_mu(start)
        
        if np.dot(Q[:, k], Q[:, k]) + 1e-12 >= (delta - mu[k, k - 1] ** 2) * np.dot(Q[:, k - 1], Q[:, k - 1]):
            k += 1
        else:
            B[:, [k, k - 1]] = B[:, [k - 1, k]]
            U[:, [k, k - 1]] = U[:, [k - 1, k]]
            Q = mgs_orthogonal_cols(B, Q, k - 1)
            update_mu(k - 1)
            k = max(k - 1, 1)

    return U

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
    U = du.lll_fpylll_rows(B2, 0.75)
    assert np.allclose(U @ B, B2)

    B2 = B.copy()
    U = du.lll_fpylll_cols(B2, 0.75)
    assert np.allclose(B @ U, B2)

    # rank, det, U = ntl.lll_left(B2, 75, 100)
    # assert np.allclose(U @ B.T, B2)

    # B2, U = hsnf.column_style_hermite_normal_form(B)
    # assert np.allclose(B @ U, B2)
    # B2, U = hsnf.row_style_hermite_normal_form(B)
    # assert np.allclose(U @ B, B2)
    print(B2)

    B2 = B.copy()
    # Q = mgs_orthogonal_cols(B2, None)
    # print("Q:", Q)
    # Q2 = Q.T @ Q
    # Q2 = np.round(Q2, decimals=10)
    # print("Q2:", Q2)

    U = lll_brans_cols(B2, 0.8)
    assert np.allclose(B @ U, B2)
    print(B2)

test_paper()