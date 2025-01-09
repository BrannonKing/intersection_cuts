import numpy as np
import scipy.linalg as spl
import scipy.sparse as sps
import scipy.sparse.linalg as spsl

def compute_H(A, b, x):
    # Compute s = b - Ax
    s = b - A @ x

    # Compute the Hessian matrix H
    if isinstance(A, sps.sparray | sps.spmatrix):
        return A.T @ sps.diags(s**(-2)) @ A
    return A.T @ np.diag(s**(-2)) @ A

def compute_V(H):
    if isinstance(H, sps.sparray | sps.spmatrix):
        return spsl.eigs(H)
    # Eigen decomposition of H
    return np.linalg.eigh(H)  # returns (eigenvalues, eigenvectors)

def plot_ellipse(A, b, x):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    H = compute_H(A, b, x)
    lam, V = compute_V(H)
    if isinstance(V, sps.sparray | sps.spmatrix):
        V = V.toarray()
        lam = lam.toarray()    

    # Compute the angle of rotation of the ellipse
    angle = np.degrees(np.arctan2(*V[:, 0][::-1]))

    # Compute the axes lengths of the ellipse
    axis_lengths = 1.0 / np.sqrt(lam)

    # Plot the Dikin ellipsoid
    fig, ax = plt.subplots()

    # Ellipse center at x, axes lengths from eigenvalues, and rotation from eigenvectors
    ell = Ellipse(xy=x, width=2*axis_lengths[0], height=2*axis_lengths[1], angle=angle,
                edgecolor='r', facecolor='none')
    ax.add_patch(ell)

    # Plot the feasible region Ax <= b (bounded by some limits for visualization)
    x_vals = np.linspace(-1, 2, 100)
    y_vals1 = (b[0] - A[0, 0] * x_vals) / A[0, 1]
    y_vals2 = (b[1] - A[1, 0] * x_vals) / A[1, 1]
    y_vals3 = (b[2] - A[2, 0] * x_vals) / A[2, 1]
    ax.plot(x_vals, y_vals1, 'b-', label=r'$x_1 + x_2 \leq 2$')
    ax.plot(x_vals, y_vals2, 'g-', label=r'$-x_1 + 2x_2 \leq 2$')
    ax.plot(x_vals, y_vals3, 'purple', label=r'$2x_1 + x_2 \leq 3$')

    # Plot the chosen point
    ax.plot(*x, 'ro', label='Chosen point $x$')

    # Set plot limits and labels
    ax.set_xlim(-1, 2)
    ax.set_ylim(-1, 2)
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_title('Dikin Ellipsoid')

    # Add legend
    ax.legend()
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    # fig.show()
    # plt.show()
    return fig

def lll_reduction_fpylll(A):
    from fpylll import IntegerMatrix, LLL

    if isinstance(A, np.ndarray | list):
        m = IntegerMatrix.from_matrix(A)  # convert numpy array to IntegerMatrix
    elif isinstance(A, sps.coo_matrix):
        m = IntegerMatrix(*A.shape)
        for i, j, v in zip(A.row, A.col, A.data):
            m._set_entry(i, j, v)
    elif isinstance(A, sps.csr_matrix):
        m = IntegerMatrix(*A.shape)
        for i in range(A.shape[0]):  # Iterate through rows
            for idx in range(A.indptr[i], A.indptr[i+1]):  # Non-zero indices for row i
                j = A.indices[idx]
                m._set_entry(i, j, A.data[idx])
    elif isinstance(A, sps.csc_matrix):
        m = IntegerMatrix(*A.shape)
        for j in range(A.shape[1]):  # Iterate through rows
            for idx in range(A.indptr[j], A.indptr[j+1]):  # Non-zero indices for row i
                i = A.indices[idx]
                m._set_entry(i, j, A.data[idx])
    elif isinstance(A, IntegerMatrix):
        m = A
    else:
        raise NotImplementedError(f"Unsupported matrix type: {type(A)}")
        
    LLL.reduction(m)  # in-place
    
    # replace A data instead?
    B = np.zeros(A.shape, dtype=np.int64)  # ensure m is 64bit
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            B[i, j] = m[i, j]
    return B

def modified_gram_schmidt(B):
    """
    Perform the Modified Gram-Schmidt process on the basis B.
    Returns the orthogonal basis Q and the coefficients R.
    """
    n = B.shape[1]
    Q = np.zeros_like(B, dtype=np.float64)
    R = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        R[i, i] = np.linalg.norm(B[:, i])
        Q[:, i] = B[:, i] / R[i, i]
        for j in range(i + 1, n):
            R[i, j] = np.dot(Q[:, i], B[:, j])
            B[:, j] -= R[i, j] * Q[:, i]

    return R

def modified_gram_schmidt_sparse(B, tol=1e-6):
    """
    Perform Modified Gram-Schmidt (MGS) on a sparse matrix A.
    
    Parameters:
        A (scipy.sparse.csc_matrix): Input sparse matrix of shape (m, n)
    
    Returns:
        Q (scipy.sparse.csc_matrix): Orthonormal basis, sparse matrix of shape (m, n)
        R (numpy.ndarray): Upper triangular matrix, dense matrix of shape (n, n)
    """
    # if not isinstance(B, csc_matrix):
    #     B = csc_matrix(B)
    
    m, n = B.shape
    Q_cols = []
    R = np.zeros((n, n))
    
    for j in range(n):
        # Extract the j-th column of A (kept sparse)
        v = B[:, j]
        
        # Orthogonalization
        for i in range(j):
            qi = Q_cols[i]
            R[i, j] = qi.T @ v  # Inner product
            v -= R[i, j] * qi  # Subtract projection (sparse operation)
        
        # Normalization
        R[j, j] = spsl.norm(v)  # Sparse norm
        if R[j, j] > tol:  # Avoid division by zero
            q = v / R[j, j]  # Normalize (sparse operation)
        else:
            q = sps.csc_matrix((m, 1))  # Zero column if norm is too small
        
        Q_cols.append(q)
    
    # Combine columns into a sparse Q matrix
    # Q = sps.hstack(Q_cols)
    
    return R

def lll_reduction(B, delta=0.75):
    """
    Perform LLL basis reduction on the input basis B.

    Parameters:
    B : numpy.ndarray
        The input basis as a 2D numpy array (columns are basis vectors).
    delta : float
        Lovasz condition parameter, typically 0.75.

    Returns:
    numpy.ndarray
        The reduced basis.
    """
    qr_func = lambda A: np.linalg.qr(A, mode='r')  # use numpy's QR decomposition
    # qr_func = lambda A: spl.qr(A, mode='r', pivoting=True, check_finite=False)  # use scipy's QR decomposition
    if isinstance(B, sps.spmatrix | sps.sparray):
        qr_func = lambda A: modified_gram_schmidt_sparse(A)
    # qr_func = modified_gram_schmidt

    B = B.T
    n = B.shape[1]
    R = qr_func(B)

    k = 1
    while k < n:
        # Size reduction
        for j in range(k - 1, -1, -1):
            mu = R[j, k] / R[j, j]
            if abs(mu) > 0.5:
                B[:, k] -= round(mu) * B[:, j]
                R = qr_func(B)

        # Lovasz condition
        if R[k, k]**2 >= (delta - (R[k - 1, k] / R[k - 1, k - 1])**2) * R[k - 1, k - 1]**2:
            k += 1
        else:
            # Swap
            B[:, [k, k - 1]] = B[:, [k - 1, k]]
            R = qr_func(B)
            k = max(k - 1, 1)

    return B.T

def hermite_normal_form(A: np.ndarray, in_place=False):
    # This is very simplified. It's not the fastest.
    # Look at "Fast computation of Hermite normal forms of random integer matrices" for a better algorithm.
    m, n = A.shape
    H = A.copy() if not in_place else A
    mask = np.ones(m, dtype=bool)  # TODO: can I use a bitmask here instead?
    
    # Perform row operations to get to HNF
    for i in range(min(m, n)):
        # Find the pivot (non-zero element in current column below or at the current row)
        pivot_row = np.argmax(np.abs(H[i:, i])) + i
        if H[pivot_row, i] == 0:
            continue
        
        # Swap rows if pivot not on diagonal
        if pivot_row != i:
            H[[i, pivot_row]] = H[[pivot_row, i]]
        
        # Make the pivot element 1 or -1
        if abs(H[i, i]) != 1:
            gcd = np.gcd.reduce(H[i:, i])
            H[i:, i:] = H[i:, i:] // gcd

        mask[i] = False
        H[mask] -= np.outer(H[mask, i], H[i])
        mask[i] = True

    return H

def row_echelon_form(A, tol=1e-6, in_place=False):
    m, n = A.shape
    A = A.copy() if not in_place else A
    
    for i in range(min(m, n)):
        # Find pivot
        pivot_row = np.argmax(np.abs(A[i:, i])) + i
        if np.isclose(A[pivot_row, i], 0.0, atol=tol):  # Skip if all zeros below
            continue

        # Swap rows
        A[[i, pivot_row]] = A[[pivot_row, i]]

        # Make pivot 1 or -1
        A[i] /= A[i, i]

        # Eliminate below pivot
        # for j in range(i + 1, m):
        #     A[j] -= A[j, i] * A[i]
        A[i + 1:] -= np.outer(A[i + 1:, i], A[i])
    
    return A

def lll_to_unimodular(A: np.ndarray, tol=1e-5):
    det = np.linalg.det(A)  # expensive check
    if np.isclose(abs(det), 1, atol=tol):
        return A
    
    # options:
    # 1. compute hermite normal form
    # 2. compute row echelon form (REF)
    # 3. compute reduced row echelon form (RREF)
    # 4. look at what they gave us and see if it is amenable to diagonal adjustment
    # 5. comput the QR decomposition and just return R (after adjusting its diagonal)
    # 6. compute LU decomposition, adjust diagonals, remultiply them, or just return one side
    return hermite_normal_form(A)

def reverse_interior_point(A, x_optimal, w_optimal, s_optimal, target_distance, max_iterations=100):
    """
    Perform reverse interior point method to move from optimal solution into the interior up to a maximum distance.

    :param A: Constraint matrix
    :param x_optimal: Optimal solution vector
    :param w_optimal: Slacks at the optimal solution
    :param s_optimal: Dual slacks at the optimal solution
    :param min_distance: Distance to travel
    :param max_iterations: Maximum number of iterations to prevent infinite loops
    :return: New point x, y, s, w, and the number of iterations performed
    """
    m, n = A.shape
    x, y, s, w = x_optimal.copy(), np.zeros(m), s_optimal.copy(), w_optimal.copy()

    # Initial duality gap
    mu = np.dot(s, w) / m
    distance_total = 0.0  # Total distance traveled
    print("WARNING! This function is not complete.")
    
    for iteration in range(max_iterations):
        # Increase mu for moving towards the interior
        mu_new = 1.1 * mu  # adjust this based on convergence needs

        # Compute the system for Newton's method
        M = np.block([  # TODO: make this block more efficient, maybe sparse
            [np.zeros((n, n)), A.T, np.zeros((n, m))],
            [A, np.zeros((m, m)), np.eye(m)],
            [np.diag(w / s), np.zeros((m, m)), np.diag(w / s)]
        ])

        rhs = np.concatenate([
            np.zeros(n),  # Ax = 0
            # b - (A @ x + w)  # TODO: this isn't right; use b and y somewhere here
            np.zeros(m),  # A^T y + s = 0
            mu_new - (w / s)  # Centering condition
        ])

        # Solve for delta
        delta = np.linalg.solve(M, rhs)
        delta_x, delta_y, delta_s = np.split(delta, [n, n + m])

        # Compute delta_w from delta_s
        delta_w = -w * delta_s / s

        # Step length alpha
        alpha = 0.99  # Conservative step length, might need adjustment
        step_distance = np.linalg.norm(alpha * delta_x)
        if distance_total + step_distance > target_distance:
            alpha = (target_distance - distance_total) / step_distance
        distance_total += step_distance
        
        # Update variables
        x += alpha * delta_x
        if distance_total >= target_distance:
            break

        y += alpha * delta_y
        s += alpha * delta_s
        w += alpha * delta_w
        
        # Update mu
        mu = np.dot(s, w) / m

    return x, iteration + 1

def reverse_interior_point_gpt(A, b, x_opt, y, target_distance, max_iterations=100, alpha=0.01):
    """
    Perform a reverse interior point walk.

    Parameters:
    - A (numpy.ndarray): Constraint matrix of size (m, n).
    - b (numpy.ndarray): Constraint bounds of size (m,).
    - x_opt (numpy.ndarray): Optimal solution of size (n,).
    - y (numpy.ndarray): Dual variables of size (m,). Get values from Gurobi's Pi attribute.
    - q (int): Number of steps to walk.
    - alpha (float): Fraction of maximum allowable step size (default 0.01).

    Returns:
    - numpy.ndarray: Sequence of interior points of size (q+1, n).
    """
    # Initializations
    m, n = A.shape
    x = x_opt.copy()
    w = b - A @ x  # Initial slacks
    AA = A @ A.T  # Precompute A @ A^T

    for iteration in range(max_iterations):
        # Compute reverse search direction: \Delta x = -A^T y
        delta_x = -A.T @ y

        # Compute maximum step size to maintain feasibility
        alpha_max = min([w[i] / (-A[i] @ delta_x) for i in range(m) if A[i] @ delta_x < 0], default=np.inf)
        step_alpha = min(alpha * alpha_max, alpha_max)

        step_distance = np.linalg.norm(step_alpha * delta_x)
        if distance_total + step_distance > target_distance:
            step_alpha = (target_distance - distance_total) / step_distance
        distance_total += step_distance

        # Update x and w
        x += step_alpha * delta_x
        if distance_total >= target_distance:
            break

        w = b - A @ x

        # Update y to maintain consistency
        delta_y = np.linalg.solve(AA, w)
        
        # Solve A (A^T \Delta y) = b - A x iteratively without forming A @ A^T:
        # delta_y = np.zeros_like(y)
        # for _ in range(100):  # Iterative solver (e.g., Richardson iteration)
        #     delta_y += 0.01 * (w - A @ (A.T @ delta_y))

        y += delta_y

    return x, iteration + 1
