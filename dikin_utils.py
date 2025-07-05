import numpy as np
import scipy.linalg as spl
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import scipy.optimize as spo
import sparseqr as spqr

def compute_H_small(l, u, x):
    """
    Compute the Hessian matrix H for a small problem with bounds l and u.
    Assumes x is in the interior of the polytope defined by l and u.
    """
    l = l.flatten()
    u = u.flatten()
    x = x.flatten()

    lb = np.round(1.0 / ((x - l)**2), 8)
    ub = np.round(1.0 / ((u - x)**2), 8)

    # Compute the Hessian matrix H
    return np.diag(lb + ub)

def compute_H(A, b, l, u, x):
    # Compute w = b - Ax
    assert x.shape == l.shape and x.shape == u.shape, "x, l, and u must have the same shape."
    w = b - A @ x
    if np.any(w == 0.0):
        raise ValueError("w contains zero elements, which is not allowed in the Dikin ellipsoid computation.")
    
    if np.any(x - l == 0.0) or np.any(u - x == 0.0):
        raise ValueError("x is on the boundary of the polytope defined by l and u, which is not allowed in the Dikin ellipsoid computation.")

    lb = np.round(1.0 / ((x - l)**2), 8)
    ub = np.round(1.0 / ((u - x)**2), 8)

    # Compute the Hessian matrix H
    if isinstance(A, sps.sparray | sps.spmatrix):
        # diag1 = sps.diags_array(w**(-1))
        diag2 = sps.diags_array(w**(-2))
        diag3 = sps.diags_array(lb + ub)
    else:
        # diag1 = np.diags(w**(-1))
        diag2 = np.diagflat(w**(-2))
        diag3 = np.diagflat(lb + ub)

    H = A.T @ diag2 @ A + diag3
    # H2 = diag1 @ A  # this only works for the square root if there are no bounds
    # assert np.array_equal(H2.T @ H2, H)
    return H

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

def plot_ellipse(A, b, l, u, x, fig=None, H=None, offset=None):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    if H is None:
        H = compute_H(A, b, l, u, x)
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

    if offset is not None:
        x += offset

    # Ellipse center at x, axes lengths from eigenvalues, and rotation from eigenvectors
    ell = Ellipse(xy=x.flatten(), width=2*axis_lengths[0], height=2*axis_lengths[1], angle=angle,
                edgecolor='goldenrod', facecolor='none', linewidth=2)
    ax.add_patch(ell)
    return fig, H

def plot_objective(c: np.ndarray, minimizing, fig=None):
    """
    Plot the objective function as a line in the 2D space defined by the first two variables.
    """
    if fig is None:
        fig, ax = plt.subplots()
    else:
        ax = fig.gca()

# Handle case where c might have more than 2 elements
    c = c.flatten()
    if len(c) < 2:
        raise ValueError("Need at least 2 coefficients for 2D plot")
    
    # Use only first two coefficients for 2D plotting
    c1, c2 = c[0], c[1]
    
    # Plot the objective line (level curve c1*x + c2*y = constant)
    x_vals = np.linspace(-5, 5, 100)
    if abs(c2) > 1e-10:  # Avoid division by zero
        y_vals = -(c1 * x_vals) / c2  # For level curve c1*x + c2*y = 0
        ax.plot(x_vals, y_vals, color='purple', label='Objective Level Curve', alpha=0.5)
    else:
        # Vertical line if c2 ≈ 0
        ax.axvline(x=c1, color='purple', label='Objective Level Curve', alpha=0.5)
    
    # Draw gradient arrow
    # The gradient direction is [c1, c2]
    arrow_length = 2.0  # Scale the arrow length
    gradient_norm = np.sqrt(c1**2 + c2**2)
    
    if gradient_norm > 1e-10:  # Avoid division by zero
        # Normalize and scale the gradient
        arrow_dx = arrow_length * c1 / gradient_norm
        arrow_dy = arrow_length * c2 / gradient_norm
        
        if minimizing:
            ax.arrow(arrow_dx, arrow_dy, -arrow_dx, -arrow_dy,
                head_width=0.12, head_length=0.2, fc='red', ec='red',
                label='Gradient Direction')
        else:
            ax.arrow(0, 0, arrow_dx, arrow_dy,
                head_width=0.12, head_length=0.2, fc='red', ec='red',
                label='Gradient Direction')

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


def CLLL_Post(B, delta=0.75, update_B=False, max_iterations=1200):
    """
    Perform LLL algorithm for lattice reduction on a basis matrix B with basis vectors as its columns.

    Cong Ling, 2005
    Based on the paper published later:
    Ying Hung Gan, Cong Ling, and Wai Ho Mow, Complex lattice reduction
    algorithm for low-complexity full-diversity MIMO detection,
    IEEE Trans. Signal Processing, vol. 57, pp. 2701-2710, July 2009.

    Parameters:
    B (ndarray): Basis matrix (real-valued only).

    Returns:
    The unimodular transformation matrix. Also note that B is modified.
    """

    n = B.shape[1]  # Number of columns
    U = np.eye(n, dtype=np.int32)

    # QR decomposition for Gram-Schmidt orthogonalization
    if isinstance(B, sps.spmatrix | sps.sparray):
        _, R, E, _ = spqr.qr(B)  # Capture permutation E
        if update_B:
            B = B[:, E]              # Permute B at the start (Option 1)
        R.resize((n, n))
        diag = R.diagonal()
        beta = diag ** 2
        mu = (R / diag.reshape(-1, 1)).T.tocsr()
    else:
        R = np.linalg.qr(B, mode='r')
        diag = np.diag(R)
        beta = diag ** 2
        mu = (R / diag.reshape(-1, 1)).T

    k = 2
    i_iteration = 0
    while i_iteration < max_iterations:
        i_iteration += 1

        # Size reduction
        if abs(mu[k - 1, k - 2]) > 0.5:
            eta = round(mu[k - 1, k - 2])
            if update_B:
                B[:, k - 1] -= eta * B[:, k - 2]
            U[:, k - 1] -= eta * U[:, k - 2]
            mu[k - 1, :] -= eta * mu[k - 2, :]

        # Swap if necessary
        if beta[k - 1] < (delta - mu[k - 1, k - 2] ** 2) * beta[k - 2]:
            beta[k - 1] += beta[k - 2] * mu[k - 1, k - 2] ** 2
            if update_B:
                B[:, [k - 1, k - 2]] = B[:, [k - 2, k - 1]]
            U[:, [k - 1, k - 2]] = U[:, [k - 2, k - 1]]
            # ... (rest of swap logic remains unchanged)
            k = max(k - 1, 2)
        else:
            for i in range(k - 2, -1, -1):
                if abs(mu[k - 1, i]) > 0.5:
                    eta = round(mu[k - 1, i])
                    if update_B:
                        B[:, k - 1] -= eta * B[:, i]
                    U[:, k - 1] -= eta * U[:, i]
                    mu[k - 1, :] -= eta * mu[i, :]

            if k < n:
                k += 1
            else:
                break

    if i_iteration >= max_iterations:
        print("Warning: suboptimal CLLL basis")

    # Apply inverse permutation to B and U before returning
    if update_B and isinstance(B, sps.spmatrix | sps.sparray):
        E_inv = np.argsort(E)  # Inverse permutation
        B = B[:, E_inv]        # Reorder columns of B back to original order
        U = U[:, E_inv]        # Reorder columns of U accordingly

    return U

def CLLL_Pre(B):
    """
    Perform LLL algorithm for lattice reduction on a basis matrix B.
    Returns U such that U B = B_original. B is not modified, though.
    """
    n = B.shape[1]  # Number of columns
    delta = 0.75  # Reduction parameter

    # Initialize the unimodular transformation matrix
    U = np.eye(n, dtype=np.int32)

    # QR decomposition for Gram-Schmidt orthogonalization
    R = np.linalg.qr(B, mode='r')
    diag = np.diag(R)
    beta = diag ** 2
    mu = (R / diag.reshape(-1, 1)).T

    k = 2
    i_iteration = 0
    max_iterations = 1200
    while i_iteration < max_iterations:
        i_iteration += 1

        # Size reduction (modify U instead of B)
        if abs(mu[k - 1, k - 2]) > 0.5:
            eta = round(mu[k - 1, k - 2])
            U[k - 1, :] -= eta * U[k - 2, :]
            mu[k - 1, :] -= eta * mu[k - 2, :]

        # Swap if necessary
        if beta[k - 1] < (delta - mu[k - 1, k - 2] ** 2) * beta[k - 2]:
            U[[k - 1, k - 2], :] = U[[k - 2, k - 1], :]  # Swap rows of U

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
                    U[k - 1, :] -= eta * U[i, :]
                    mu[k - 1, :] -= eta * mu[i, :]

            if k < n:
                k += 1
            else:
                break

    if i_iteration >= max_iterations:
        print("Warning: suboptimal CLLL basis")

    return U


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


import hsnf

def smith_normal_form(A):
    D, L, R = hsnf.smith_normal_form(A)
    # assert np.allclose(L @ A @ R, D, atol=1e-5)
    return np.linalg.inv(L), D, np.linalg.inv(R)

def to_U_via_SNF(A, mult=1, keep_scale=False):
    SU, SD, SV = smith_normal_form(np.round(A * mult))
    for i in range(SD.shape[0]):
        for j in range(i, SD.shape[1]):
            if i == j:
                SD[i, j] = np.sign(SD[i, j])
                if keep_scale:
                    SD[i, j] *= mult
                continue
            if abs(SD[i, j]) < abs(SD[j, i]):
                SD[i, j] = 0
            else:
                SD[j, i] = 0
    return SU @ SD @ SV

def to_U_via_LU(A, mult=1):
    P, L, U = spl.lu(A, overwrite_a=False)
    # assert np.allclose(P @ L @ U, A, atol=1e-5)
    for i in range(L.shape[0]):
        div = L[i, i]
        if div != 0.0:
            L[i, 0:i+1] /= abs(div)
        div = U[i, i]
        if div != 0.0:
            U[i, i:] /= abs(div)

    s = np.sqrt(mult)
    L = np.round(L * s)
    U = np.round(U * s)
    
    return P @ L @ U

def to_U_via_iteration(A: np.ndarray, swap_threshold=0.75, tol=1e-8):
    n, d = A.shape

    B = A.astype(dtype=float, copy=True)
    U = np.eye(n, dtype=int)

    mp = 0
    while True:
        mp += 1
        if mp > 100:
            print("Warning: suboptimal basis")
            return U, mp
        pass_modified = False

        # --- Phase 1: Size Reduction (your original logic) ---
        # This makes the basis "tidy" by reducing projections.
        for i in range(1, n):
            for j in range(i - 1, -1, -1):
                norm_sq_j = np.dot(B[j], B[j])
                if norm_sq_j < tol:
                    continue
                
                dot_product = np.dot(B[i], B[j])
                q = int(np.round(dot_product / norm_sq_j))

                if q != 0:
                    pass_modified = True
                    B[i] -= q * B[j]
                    U[i] -= q * U[j]

        # --- Phase 2: Heuristic Swap ---
        # This reorders the basis to move shorter vectors first.
        for i in range(1, n):
            norm_i_sq = np.dot(B[i], B[i])
            norm_i_minus_1_sq = np.dot(B[i-1], B[i-1])

            # Heuristic swap condition: if B[i] is significantly shorter
            # than B[i-1], swap them.
            if norm_i_sq < swap_threshold * norm_i_minus_1_sq:
                pass_modified = True
                B[[i, i-1]] = B[[i-1, i]]
                U[[i, i-1]] = U[[i-1, i]]

        # --- Termination Check ---
        # If a full pass of both phases made no changes, we are done.
        if not pass_modified:
            return U, mp

def orthogonality_measure_1(Q):
    QtQ = Q.T @ Q
    deviation = QtQ - np.eye(Q.shape[1])
    return np.linalg.norm(deviation, 'fro')

def orthogonality_measure_2(Q):
    s = np.linalg.svd(Q, compute_uv=False)
    return np.linalg.norm(s - 1.0)  # How far singular values are from 1

def difference(A, B):
    Af = np.linalg.norm(A, ord='fro')
    Bf = np.linalg.norm(B, ord='fro')
    tr = np.abs(np.trace(A.T @ B))
    # distance = np.arccos(tr / (Af * Bf))
    distance = 1 - tr / (Af * Bf)
    return distance

def difference_2(A, B):
    return np.linalg.norm(A - B, 2)

if __name__ == "__main__":
    np.random.seed(42)
    for j in range(10):
        # A = np.random.randint(-10, 11, (5, 5))
        A = np.random.random((5, 5)) * 2 - 1
        i = 16
        while i <= 64:
            U = to_U_via_LU(A, i)
            det = np.linalg.det(U)
            # assert np.isclose(np.abs(np.linalg.det(U)), i, atol=1e-5)

            U2 = to_U_via_SNF(A, i, keep_scale=True)
            det2 = np.linalg.det(U2)
            # assert np.isclose(np.abs(np.linalg.det(U2)), i, atol=1e-5)

            U3 = CLLL_Post(A * (i - 1), max_iterations=200)
            det3 = np.linalg.det(U3)

            d = difference(A, U / i)
            d2 = difference(A, U2 / i)
            d3a = difference(A, U3)
            d3b = difference(A, U3 / i)
            print(j, i, f"{d:.3f}\t{d2:.3f}\t{d3a:.3f}\t{d3b:.3f}\t{det:.1f}\t{det2:.1f}\t{det3:.1f}")
            i *= i

# for NTL LLL:
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <NTL/ZZ.h>
#include <NTL/mat_ZZ.h>
#include <NTL/LLL.h>

# namespace py = pybind11;

# std::tuple<int64_t, int64_t, py::array_t<int64_t, py::array::c_style>> lll(py::array_t<int64_t, py::array::c_style> inout, const long a, const long b) {
#     auto request = inout.request();
#     if (request.ndim != 2)
#         throw std::runtime_error("Input array must be two-dimensional!");
#     if (request.strides[0] % request.strides[1] != 0)
#         throw std::runtime_error("Unexpected stride size 0!");
#     if (request.strides[1] != sizeof(int64_t))
#         throw std::runtime_error("Unexpected stride size 1!");
#     if (request.strides[0] / sizeof(int64_t) != request.shape[1])
#         throw std::runtime_error("Unexpected stride size 2!");

#     NTL::mat_ZZ A;
#     A.SetDims(request.shape[0], request.shape[1]);
#     const auto ptr = static_cast<int64_t*>(request.ptr);
#     for (long i = 0; i < request.shape[0]; ++i)
#         for (long j = 0; j < request.shape[1]; ++j)
#         {
#             NTL::ZZ v;
#             NTL::conv(v, ptr[i * request.shape[1] + j]);
#             A.put(i, j, v);
#         }

#     NTL::ZZ det;
#     NTL::mat_ZZ U;
#     A = NTL::transpose(A);
#     auto rank = NTL::LLL(det, A, U, a, b, 0);
#     A = NTL::transpose(A);
#     U = NTL::transpose(U);
#     // auto rank = NTL::LLL_FP(A);
#     if (U.NumCols() != request.shape[1] || U.NumRows() != request.shape[1])
#         throw std::runtime_error("Unexpected dimensions on U! " + std::to_string(rank) + ", " + std::to_string(U.NumCols())
#             + ", " + std::to_string(U.NumRows()) + ", " + std::to_string(request.shape[0]));

#     py::array_t<int64_t, py::array::c_style> u_ret({request.shape[1], request.shape[1]});
#     auto view_u = u_ret.mutable_unchecked();
#     auto view_inout = inout.mutable_unchecked();
#     for (long i = 0; i < request.shape[0]; ++i)
#         for (long j = 0; j < request.shape[1]; ++j)
#             NTL::conv(view_inout(i, j), A.get(i, j));
#     for (long i = 0; i < request.shape[1]; ++i)
#         for (long j = 0; j < request.shape[1]; ++j)
#             NTL::conv(view_u(i, j), U.get(i, j));

#     int64_t result;
#     NTL::conv(result, det);
#     return {rank, result, u_ret};
# } 

# PYBIND11_MODULE(ntl_wrapper, m) {
#     m.def("lll", &lll, "Call NTL's LLL function.");
# }