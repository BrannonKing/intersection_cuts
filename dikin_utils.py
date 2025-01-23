import numpy as np
import scipy.linalg as spl
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import scipy.optimize as spo

def compute_H(A, b, x):
    # Compute s = b - Ax
    s = b.flatten() - A @ x.flatten()
    if not np.all(s != 0.0):
        print("Point on boundary!", x, s)
        s[s == 0.0] = 1e-5  # perturb the point slightly

    # Compute the Hessian matrix H
    if isinstance(A, sps.sparray | sps.spmatrix):
        diag1 = sps.diags(s**(-1))
        diag2 = sps.diags(s**(-2))
    else:
        diag1 = np.diags(s**(-1))
        diag2 = np.diags(s**(-2))
    H = A.T @ diag2 @ A
    H2 = diag1 @ A
    # assert np.array_equal(H2.T @ H2, H)
    return H, H2

def compute_V(H):
    # we're assuming that H is symmetric, which the Hessian should be
    if isinstance(H, sps.sparray | sps.spmatrix):
        # return spsl.eigsh(H, k=min(*H.shape))
        H = H.toarray()
    # Eigen decomposition of H
    eigs, eigvecs = np.linalg.eigh(H)  # returns (eigenvalues, eigenvectors)

    # we're expecting to be in the interior of the polytope, which should be convex, having positive curvature
    if not np.all(eigs > 0):
        if np.all(eigs >= -1e-5):
            eigs[eigs < 1e-5] = 1e-5
        else:
            raise ValueError("Negative eigenvalues detected in the Hessian matrix.")

    # eigvecs are normalized coming out of eigh;
    # we need to change them to have the correct length:
    eigvecs /= np.sqrt(eigs)
    return eigvecs

def plot_ellipse(A, b, x, fig=None):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    H, _ = compute_H(A, b, x)
    V = compute_V(H)
    if isinstance(V, sps.sparray | sps.spmatrix):
        V = V.toarray()

    # Compute the angle of rotation of the ellipse
    try:
        angle = np.degrees(np.arctan2(*V[:, 0][::-1][:2]))
    except:
        return fig

    # Compute the axes lengths of the ellipse
    axis_lengths = np.linalg.norm(V, 2, axis=0)

    # Plot the Dikin ellipsoid
    fig, ax = plt.subplots() if fig is None else fig, fig.gca()

    # Ellipse center at x, axes lengths from eigenvalues, and rotation from eigenvectors
    ell = Ellipse(xy=x, width=2*axis_lengths[0], height=2*axis_lengths[1], angle=angle,
                edgecolor='r', facecolor='none')
    ax.add_patch(ell)
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

def CLLL(B):
    """
    Perform LLL algorithm for lattice reduction on a basis matrix B.

    Cong Ling, 2005
    Based on the paper published later:
    Ying Hung Gan, Cong Ling, and Wai Ho Mow, Complex lattice reduction
    algorithm for low-complexity full-diversity MIMO detection,
    IEEE Trans. Signal Processing, vol. 57, pp. 2701-2710, July 2009.

    Parameters:
    B (ndarray): Basis matrix (real-valued only).

    Returns:
    tuple: Reduced basis matrix and the unimodular transformation matrix.
    """
    M = B.shape[1]  # Number of columns
    delta = 0.75  # Reduction parameter

    # Initialize the unimodular transformation matrix
    U = np.eye(M, dtype=np.int32)

    # QR decomposition for Gram-Schmidt orthogonalization
    R = np.linalg.qr(B, mode='r')
    beta = np.abs(np.diag(R)) ** 2
    mu = (R / (np.diag(np.diag(R)) @ np.ones((M, M)))).T

    k = 2
    i_iteration = 0
    max_iterations = 100 * M ** 2

    while i_iteration < max_iterations:
        i_iteration += 1

        # Size reduction
        if abs(mu[k - 1, k - 2]) > 0.5:
            eta = round(mu[k - 1, k - 2])
            B[:, k - 1] -= eta * B[:, k - 2]
            U[:, k - 1] -= eta * U[:, k - 2]
            mu[k - 1, :] -= eta * mu[k - 2, :]

        # Swap if necessary
        if beta[k - 1] < (delta - mu[k - 1, k - 2] ** 2) * beta[k - 2]:
            B[:, [k - 1, k - 2]] = B[:, [k - 2, k - 1]]  # Swap columns
            U[:, [k - 1, k - 2]] = U[:, [k - 2, k - 1]]  # Swap columns in U

            muswap = mu[k - 2, :k - 2].copy()
            mu[k - 2, :k - 2] = mu[k - 1, :k - 2]
            mu[k - 1, :k - 2] = muswap

            old_muk_betak = mu[k:, k - 1] * beta[k - 1]
            old_beta1 = beta[k - 2]
            old_betak = beta[k - 1]
            old_mu = mu[k - 1, k - 2]

            mu[k:, k - 1] = mu[k:, k - 2] - mu[k:, k - 1] * mu[k - 1, k - 2]
            beta[k - 2] = beta[k - 1] + mu[k - 1, k - 2] ** 2 * beta[k - 2]
            beta[k - 1] = old_betak * old_beta1 / beta[k - 2]
            mu[k - 1, k - 2] = old_mu * old_beta1 / beta[k - 2]
            mu[k:, k - 2] = mu[k:, k - 2] * mu[k - 1, k - 2] + old_muk_betak / beta[k - 2]

            k = max(k - 1, 2)
        else:
            for i in range(k - 2, -1, -1):
                if abs(mu[k - 1, i]) > 0.5:
                    eta = round(mu[k - 1, i])
                    B[:, k - 1] -= eta * B[:, i]
                    U[:, k - 1] -= eta * U[:, i]
                    mu[k - 1, :] -= eta * mu[i, :]

            if k < M:
                k += 1
            else:
                break

    if i_iteration >= max_iterations:
        print("Warning: suboptimal CLLL basis")

    return B, U

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

def reverse_interior_point_gpt(A, b, x_opt, y_opt, target_distance, max_iterations=100, alpha=0.1, tol=1e-5):
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
    x = x_opt.copy().reshape(-1, 1)
    y = y_opt.copy().reshape(-1, 1)
    AA = A @ A.T  # Precompute A @ A^T
    distance_total = 0.0
    assert alpha > tol * 2

    # Test one constraint for direction:
    multiplier = -1.0
    if A[0] @ (alpha * A.T @ y) > b[0, 0] + tol:
        multiplier = 1.0

    for iteration in range(max_iterations):
        # Compute reverse search direction: \Delta x = -A^T y
        delta_x = multiplier * (A.T @ y)

        # Compute maximum step size to maintain feasibility
        # alpha_max = min([(w[i, 0] / (-A[i] @ delta_x)).item() for i in range(m) if (A[i] @ delta_x).item() < 0], default=np.inf)
        step_alpha = alpha # min(alpha * alpha_max, alpha_max)

        step_distance = np.linalg.norm(step_alpha * delta_x)
        if distance_total + step_distance > target_distance:
            step_alpha *= (target_distance - distance_total) / step_distance
        distance_total += step_distance

        # Update x and w
        x += step_alpha * delta_x
        if distance_total >= target_distance:
            break

        w = b - A @ x

        # Update y to maintain consistency
        # if isinstance(AA, sps.spmatrix | sps.sparray):
        #     delta_y = spsl.spsolve(AA, w)
        #     assert not np.any(np.isnan(delta_y)), "spsolve failure"
        # else:
        #     delta_y = np.linalg.solve(AA, w)
        
        # Solve A (A^T \Delta y) = b - A x iteratively without forming A @ A^T:
        delta_y = np.zeros_like(y)
        for _ in range(100):  # Iterative solver (e.g., Richardson iteration)
            delta_y += 0.01 * (w - A @ (A.T @ delta_y))

        y += -multiplier * delta_y.reshape(-1, 1)

    return x.flatten(), iteration + 1

def append_bounds_to_matrix(A, b, l, u, infinity=np.inf):
    """
    Append lower and upper bounds to the constraint matrix A using numpy operations.
    """
    # Validate input dimensions
    if A.shape[1] != len(l) or len(l) != len(u):
        raise ValueError("Dimensions of l, u must match the number of columns in A.")
    
    num_vars = A.shape[1]
    
    # Identify which bounds to include
    include_lower = l > -infinity
    include_upper = u < infinity
    ils = include_lower.sum()
    ius = include_upper.sum()

    # Create sparse rows for lower bounds
    lower_rows = sps.csr_matrix((-np.ones(ils), (np.arange(ils), np.where(include_lower)[0])), shape=(ils, num_vars))
    lower_rhs = -l[include_lower]

    # Create sparse rows for upper bounds
    upper_rows = sps.csr_matrix((np.ones(ius), (np.arange(ius), np.where(include_upper)[0])), shape=(ius, num_vars))
    upper_rhs = u[include_upper]

    # Combine the original matrix with the new bounds rows
    A_new = sps.vstack([A, lower_rows, upper_rows])
    b_new = np.hstack([b.flatten(), lower_rhs, upper_rhs])

    return A_new, b_new

def least_squares_interior_grok(A, b, p, l, u, d, infinity):
    # A, _ = append_bounds_to_matrix(A, b, l, u, infinity)  # should make it better, but makes it worse
    try:
        m = spsl.inv(A.T @ A) @ A.T
    except:
        m = spsl.inv(A.T @ A + sps.diags(np.full(A.shape[1], 1e-5))) @ A.T
    ones = np.ones((m.shape[1], 1))
    return p - d * (m @ ones).flatten(), 1  # not using b as we assume p on the constraints

def least_squares_interior(A, b, p, l, u, d, infinity):
    # for reasons beyond me, this doesn't work at all.
    m, n = A.shape
    assert b.shape == (m, 1), p.shape == (m, 1)
    b = b.flatten()
    p = p.flatten()
    finite_l = (l > -infinity)
    fls = finite_l.sum()
    finite_u = (u < infinity)
    fus = finite_u.sum()
    w = np.zeros((2*n + fls + fus,))
    w[:n] = p
    iterations = 0

    # min F(x, s, s_u, s_l) = ||Ax + s - b||² + ||x + s_u - u||² + ||x - s_l - l||² + (||x - p||² - d²)²
    def score(v, *args, **kwargs):
        nonlocal iterations
        iterations += 1
        x = v[:n]
        s = v[n:2*n]
        s_l = v[2*n:2*n + fls]
        s_u = v[2*n + fls:]
        return A @ x + s - b + (s - d)
        # return np.linalg.norm(A @ x + s - b) + np.linalg.norm(x[finite_u] + s_u - u[finite_u]) + \
        #     np.linalg.norm(x[finite_l] - s_l - l[finite_l]) + (np.linalg.norm(x - p) - d)**2 + v[n:].sum()**2
    
    lb = np.zeros_like(w)
    lb[:n] = l
    ub = np.full_like(w, infinity)
    ub[:n] = u
    result = spo.least_squares(score, w, bounds=(lb, ub), xtol=1e-9, ftol=1e-6)
    return result.x[:n], (iterations, result)


def reverse_interior_point_gpt2(A, b, c, l, u, x_start, y_start, target_distance, infinity, is_maximizing=False, max_iterations=100, alpha=0.1, tol=1e-5):
    """
    Interior Point Method for solving LP: min c^T x subject to Ax <= b, l <= x <= u.
    
    Parameters:
        A (ndarray): m x n constraint matrix.
        b (ndarray): m-dimensional vector.
        c (ndarray): n-dimensional cost vector.
        l (ndarray): n-dimensional lower bound vector for x.
        u (ndarray): n-dimensional upper bound vector for x.
        x_start (ndarray): Initial feasible solution for x.
        y_start (ndarray): Initial feasible solution for dual variables.
        tol (float): Convergence tolerance for the duality gap.
        max_iter (int): Maximum number of iterations.
    
    Returns:
        x (ndarray): Optimal primal solution.
        y (ndarray): Optimal dual solution.
    """
    # Dimensions
    m, n = A.shape
    
    # Initialization
    b = b.flatten()
    x = np.copy(x_start)
    y = np.copy(y_start)
    z_l = 1.0 / (x - l)
    z_u = 1.0 / (u - x)

    if is_maximizing:
        c = -c
    
    alpha = 0.99  # Step size scaling factor
    
    # Identify finite bounds
    # finite_l = ~np.isinf(l)
    # finite_u = ~np.isinf(u)
    # finite_l = (l > -infinity)
    # finite_u = (u < infinity)
    l[l < infinity] = infinity
    u[u > infinity] = infinity

    for iteration in range(max_iterations):
        # Compute residuals and duality measure
        r_b = A @ x - b  # primal residual
        r_c = A.T @ y + z_u - z_l + c  # dual residual

        # Form diagonal matrices
        # Only compute for finite bounds
        # X_L_inv = sps.diags(1.0 / (x[finite_l] - l[finite_l]))
        # X_U_inv = sps.diags(1.0 / (u[finite_u] - x[finite_u]))
        # Z_L = sps.diags(z_l[finite_l])
        # Z_U = sps.diags(z_u[finite_u])

        # KKTr = sps.csc_matrix((n, n))
        # if finite_l.any():
        #     KKTr[np.ix_(finite_l, finite_l)] -= X_L_inv @ Z_L
        # if finite_u.any():
        #     KKTr[np.ix_(finite_u, finite_u)] -= X_U_inv @ Z_U

        # # Modify KKT system for reduced variables
        # KKT = sps.block_array([
        #     [sps.csc_matrix((m, m)), A[:, finite_l | finite_u]],
        #     [A[:, finite_l | finite_u].T, KKTr]
        # ])

        X_L_inv = sps.diags(1.0 / (x - l))
        X_U_inv = sps.diags(1.0 / (u - x))
        Z_L = sps.diags(z_l)
        Z_U = sps.diags(z_u)

        # Modify KKT system for reduced variables
        KKT = sps.block_array([
            [sps.csr_matrix((m, m)), A],
            [A.T, -X_L_inv @ Z_L - X_U_inv @ Z_U]
        ], format='csr')
        
        # Right-hand side
        rhs = np.hstack([
            -r_b,
            -r_c - X_L_inv @ Z_L @ r_b - X_U_inv @ Z_U @ r_b
        ])

        # Solve Newton system
        delta = spsl.spsolve(KKT, rhs)
        delta_x = delta[:m]
        delta_y = delta[m:]

        # Compute steps for primal and dual variables
        step_primal = np.min(
            np.minimum(alpha * (u - x) / delta_x, alpha * (x - l) / -delta_x)
        )
        step_dual = np.min(
            np.minimum(alpha * -z_l / delta_x, alpha * z_u / delta_x)
        )
        step = -min(step_primal, step_dual)  # negative for walking backwards

        # Update variables
        x += step * delta_x
        y += step * delta_y
        z_l = 1 / (x - l)
        z_u = 1 / (u - x)

        if np.linalg.norm(x - x_start) >= target_distance:
            break

    else:
        print("Max iterations reached. No convergence.")
    
    return x, y


def smith_normal_form_grok(A):
    m, n = A.shape
    U = np.eye(m, dtype=int)
    V = np.eye(n, dtype=int)
    D = A.copy()

    i, j = 0, 0
    while i < m and j < n:
        # Find non-zero pivot if possible
        non_zero_pivot = np.nonzero(D[i:, j:])[0]
        if non_zero_pivot.size == 0:
            j += 1
            continue
        
        # Swap rows
        row_idx = non_zero_pivot[0] + i
        if row_idx != i:
            D[[i, row_idx]] = D[[row_idx, i]]
            U[[i, row_idx]] = U[[row_idx, i]]
        
        # Swap columns
        col_idx = non_zero_pivot[0] + j
        if col_idx != j:
            D[:, [j, col_idx]] = D[:, [col_idx, j]]
            V[:, [j, col_idx]] = V[:, [col_idx, j]]

        # Make D[i][j] positive if it's negative
        if D[i, j] < 0:
            D[i, :] = -D[i, :]
            U[i, :] = -U[i, :]

        # Eliminate entries above and below the pivot
        for k in range(m):
            if k != i:
                g = np.gcd(D[i, j], D[k, j])
                if g != 0:
                    a = D[k, j] // g
                    b = D[i, j] // g
                    D[k, :] = D[k, :] * b - D[i, :] * a
                    U[k, :] = U[k, :] * b - U[i, :] * a

        # Move to the next pivot
        i += 1
        j += 1

    # Sort diagonal elements (optional, for canonical form)
    diag = np.diag(D)
    indices = np.argsort(diag)
    D = D[indices, :][:, indices]
    U = U[indices, :]
    V = V[:, indices]

    return D, U, V

def extended_gcd(a, b):
    if a == 0:
        return (0, 1)
    else:
        x, y = extended_gcd(b % a, a)
        return (y - (b // a) * x, x)

def smith_normal_form_gemini(A):
    """Computes the Smith Normal Form of an integer matrix A."""
    A = np.array(A, dtype=int)
    m, n = A.shape
    S = A.copy()
    U = np.eye(m, dtype=int)
    V = np.eye(n, dtype=int)

    for i in range(min(m, n)):
        # Find a non-zero pivot
        pivot_row = -1
        for r in range(i, m):
            if S[r, i] != 0:
                pivot_row = r
                break
        if pivot_row == -1:
            continue

        # Swap rows to bring pivot to the top
        S[[i, pivot_row]] = S[[pivot_row, i]]
        U[[i, pivot_row]] = U[[pivot_row, i]]

        # Eliminate elements below the pivot
        for r in range(i + 1, m):
            while S[r, i] != 0:
                q = S[r, i] // S[i, i]
                S[r] -= q * S[i]
                U[r] -= q * U[i]
                S[[i,r]] = S[[r,i]]
                U[[i,r]] = U[[r,i]]

        # Work on columns now
        pivot_col = -1
        for c in range(i,n):
          if S[i,c]!=0:
            pivot_col = c
            break
        if pivot_col == -1:
          continue
        S[:,[i,pivot_col]] = S[:,[pivot_col,i]]
        V[:,[i,pivot_col]] = V[:,[pivot_col,i]]

        for c in range(i+1,n):
          while S[i,c]!=0:
            q = S[i,c]//S[i,i]
            S[:,c] -= q * S[:,i]
            V[:,c] -= q * V[:,i]
            S[:,[i,c]] = S[:,[c,i]]
            V[:,[i,c]] = V[:,[c,i]]
        
        #Ensure positive diagonal
        if S[i,i] < 0:
          S[i,:]*=-1
          U[i,:]*=-1

    return S, U, V

def smith_normal_form(A):
    A = np.array(A, dtype=int)
    m, n = A.shape
    S = A.copy()
    U = np.eye(m, dtype=int)
    V = np.eye(n, dtype=int)

    for i in range(min(m, n)):
        # Bring a non-zero element to the pivot position
        pivot_row = -1
        for r in range(i, m):
            if S[r, i] != 0:
                pivot_row = r
                break
        if pivot_row == -1:
            continue

        S[[i, pivot_row]] = S[[pivot_row, i]]
        U[[i, pivot_row]] = U[[pivot_row, i]]

        # Eliminate elements below the pivot
        for r in range(i + 1, m):
            while S[r, i] != 0:
                q = S[r, i] // S[i, i]
                S[r] -= q * S[i]
                U[r] -= q * U[i]
                S[[i,r]] = S[[r,i]]
                U[[i,r]] = U[[r,i]]

        # Bring a non-zero element to the column pivot position
        pivot_col = -1
        for c in range(i,n):
          if S[i,c]!=0:
            pivot_col = c
            break
        if pivot_col == -1:
          continue
        S[:,[i,pivot_col]] = S[:,[pivot_col,i]]
        V[:,[i,pivot_col]] = V[:,[pivot_col,i]]

        # Eliminate elements to the right of the pivot
        for c in range(i+1,n):
          while S[i,c]!=0:
            q = S[i,c]//S[i,i]
            S[:,c] -= q * S[:,i]
            V[:,c] -= q * V[:,i]
            S[:,[i,c]] = S[:,[c,i]]
            V[:,[i,c]] = V[:,[c,i]]

        # GCD reduction step (Crucial for Smith Normal Form)
        j = i
        while j < min(m, n) - 1:
            if S[i, i] == 0:
                break
            g = np.gcd(S[i, i], S[j + 1, j + 1])
            if g != S[i, i]:
                a = S[i, i]
                b = S[j + 1, j + 1]
                x, y = extended_gcd(a, b)
                S[i, i] = g
                S[j + 1, j + 1] = (a * b) // g
                temp_row = U[i].copy()
                U[i] = x * U[i] + y * U[j+1]
                U[j+1] = (b // g) * (-y) * temp_row + (a // g) * U[j+1]
                temp_col = V[:,i].copy()
                V[:,i] = x*V[:,i] + y*V[:,j+1]
                V[:,j+1] = (b // g) * (-y) * temp_col + (a // g) * V[:,j+1]
            j+=1

        if S[i,i] < 0:
          S[i,:]*=-1
          U[i,:]*=-1

    return S, U, V

# A = np.array([
#     [2, 4, 4, 3],
#     [-6, 6, 12, -9],
#     [10, -4, -16, 2],
#     [1, 11, -7, 1]
# ], dtype=int)

# D, U, V = smith_normal_form(A)
# print("D:\n", D, np.linalg.det(D))
# print("U:\n", U, np.linalg.det(U))
# print("V:\n", V, np.linalg.det(V))
# print("A:\n", np.linalg.inv(U) @ D @ np.linalg.inv(V))
# mid = np.eye(*D.shape)
# # left off: could make it triangular to get a better match
# # mid[1, 1] = -1
# # mid[2, 2] = -1
# for i in range(-10, 10):
#     mid[0, 2] = i
#     A2 = np.linalg.inv(U) @ mid @ np.linalg.inv(V)
#     print(i, "P:\n", A2, np.linalg.norm(A2*2 - A, np.inf))

def to_ref_unimodular(A):
    """
    Converts the matrix A into Row Echelon Form (REF) using only unimodular operations.
    Returns the REF and the unimodular transformation matrix U.

    Parameters:
        A (ndarray): The input matrix (integer entries).

    Returns:
        tuple: (REF matrix, unimodular transformation matrix)
    """
    m, n = A.shape
    U = np.eye(m)  # Start with the identity matrix for transformations
    A = A.copy()  # Work on a copy to preserve the original matrix

    for col in range(min(m, n)):
        # Find the pivot row in the current column
        pivot_row = None
        for row in range(col, m):
            if A[row, col] != 0:
                pivot_row = row
                break

        if pivot_row is None:
            # No pivot in this column, continue to the next column
            continue

        # Swap the pivot row with the current row if necessary
        if pivot_row != col:
            A[[col, pivot_row]] = A[[pivot_row, col]]
            U[[col, pivot_row]] = U[[pivot_row, col]]

        # Normalize the pivot row (make the pivot element 1 or -1)
        pivot_value = A[col, col]
        if pivot_value < 0:
            A[col] = -A[col]
            U[col] = -U[col]

        # Eliminate below the pivot
        for row in range(col + 1, m):
            if A[row, col] != 0:
                # factor = A[row, col] // A[col, col]
                factor = A[row, col] / A[col, col]
                A[row] -= factor * A[col]
                U[row] -= factor * U[col]

    return A, U

# Example usage
# A = np.array([[2,-4, 8], [-6, 12, -14], [10, 20, 22]], dtype=int)
# REF, U = to_ref_unimodular(A)
# print("Original Matrix:\n", A)
# print("REF:\n", REF)
# print("Unimodular Transformation Matrix U:\n", U)
# print("DET of U:", np.linalg.det(U))

# # Verify that U * Original = REF
# print("Verification (U @ A):\n", U @ A)

# print("Verification 2 (U-1 @ REF):\n", np.linalg.inv(U) @ REF)