from __future__ import annotations
import functools

import ntl_wrapper as ntl
import numpy as np
import scipy.linalg as spl
import scipy.optimize as spo
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import sparseqr as spqr


def extend_null_space_to_full_basis(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the null space of ``A`` and extend it to a full orthonormal basis.

    Returns a tuple ``(N, Q_full)`` where ``N`` contains a basis for the null space
    of ``A`` (possibly empty when the nullity is zero) and ``Q_full`` is an ``n×n``
    orthonormal matrix whose first ``n - m`` columns coincide with ``N``.
    """

    N = spl.null_space(A)
    Q_full, _ = np.linalg.qr(N, mode="complete")
    return N, Q_full


def compute_H_small(l, u, x):
    """
    Compute the Hessian matrix H for a small problem with bounds l and u.
    Assumes x is in the interior of the polytope defined by l and u.
    """
    l = l.flatten()
    u = u.flatten()
    x = x.flatten()

    lb = np.round(1.0 / ((x - l) ** 2), 8)
    ub = np.round(1.0 / ((u - x) ** 2), 8)

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

    lb = np.round(1.0 / ((x - l) ** 2), 8)
    ub = np.round(1.0 / ((u - x) ** 2), 8)

    # Compute the Hessian matrix H
    if isinstance(A, (sps.sparray, sps.spmatrix)):
        # diag1 = sps.diags_array(w**(-1))
        diag2 = sps.diags_array(w ** (-2))
        diag3 = sps.diags_array(lb + ub)
    else:
        # diag1 = np.diags(w**(-1))
        diag2 = np.diagflat(w ** (-2))
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
    ell = Ellipse(xy=x.flatten(), width=2 * axis_lengths[0], height=2 * axis_lengths[1], angle=angle, edgecolor="goldenrod", facecolor="none", linewidth=2)
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
        ax.plot(x_vals, y_vals, color="purple", label="Objective Level Curve", alpha=0.5)
    else:
        # Vertical line if c2 ≈ 0
        ax.axvline(x=c1, color="purple", label="Objective Level Curve", alpha=0.5)

    # Draw gradient arrow
    # The gradient direction is [c1, c2]
    arrow_length = 2.0  # Scale the arrow length
    gradient_norm = np.sqrt(c1**2 + c2**2)

    if gradient_norm > 1e-10:  # Avoid division by zero
        # Normalize and scale the gradient
        arrow_dx = arrow_length * c1 / gradient_norm
        arrow_dy = arrow_length * c2 / gradient_norm

        if minimizing:
            ax.arrow(arrow_dx, arrow_dy, -arrow_dx, -arrow_dy, head_width=0.12, head_length=0.2, fc="red", ec="red", label="Gradient Direction")
        else:
            ax.arrow(0, 0, arrow_dx, arrow_dy, head_width=0.12, head_length=0.2, fc="red", ec="red", label="Gradient Direction")

    return fig


def lll_fpylll_cols_check_reduced(B, delta=0.75, use_bkz=False):
    import fpylll as fpy
    B2 = fpy.IntegerMatrix.from_matrix(B.T)
    # it does rows by default, so we need to transpose it to do columns
    if use_bkz:
        return fpy.BKZ.is_reduced(B2, o=fpy.BKZ.Param(block_size=32))  # , flags=fpy.BKZ.VERBOSE
    return fpy.LLL.is_reduced(B2, delta=delta)


def lll_fpylll_cols(B, delta=0.75, use_bkz=False, verbose=0):
    """
    Perform LLL reduction using fpylll.
    :param B: Input matrix to be reduced.
    :param delta: LLL parameter, typically between 0.99 and 0.999.
    :return: Reduced basis matrix.
    """
    import fpylll as fpy
    B2 = fpy.IntegerMatrix.from_matrix(B.T)
    U = fpy.IntegerMatrix(1, 1)
    if verbose == 1:
        print("  Initial norm:", B2[-1].norm())
    elif verbose == 2:
        for i in range(B2.nrows):
            print(f"  Initial norm at {i}: {B2[i].norm()}")
    # it does rows by default, so we need to transpose it to do columns
    if use_bkz:
        B3 = fpy.BKZ.reduction(B2, U=U, o=fpy.BKZ.Param(block_size=32))  # , flags=fpy.BKZ.VERBOSE
    else:
        B3 = fpy.LLL.reduction(B2, U=U, delta=delta)
    if verbose == 1:
        print("  After norm:", B3[-1].norm())
    elif verbose == 2:
        for i in range(B3.nrows):
            print(f"  After norm at {i}: {B3[i].norm()}")
    result = np.zeros((U.nrows, U.ncols), dtype=object)
    U.to_matrix(result)
    B3.transpose()
    B3.to_matrix(B)
    return result.T


def lll_fpylll_rows(B, delta=0.75):
    """
    Perform LLL reduction using fpylll.
    :param B: Input matrix to be reduced.
    :param delta: LLL parameter, typically between 0.99 and 0.999.
    :return: Reduced basis matrix.
    """
    import fpylll as fpy
    B2 = fpy.IntegerMatrix.from_matrix(B)
    U = fpy.IntegerMatrix(1, 1)
    print("  Initial norm:", B2[-1].norm())
    B3 = fpy.LLL.reduction(B2, U=U, delta=delta)
    print("  After norm:", B3[-1].norm())
    result = np.zeros((U.nrows, U.ncols), dtype=np.int64)
    U.to_matrix(result)
    B3.to_matrix(B)
    return result


def CLLL_Post(B, delta=0.75, update_B=False, max_iterations=1200):
    """
    Perform LLL algorithm for lattice reduction on a basis matrix B with basis vectors as its columns.

    Cong Ling, 2005
    Based on the paper published later:
    Ying Hung Gan, Cong Ling, and Wai Ho Mow, Complex lattice reduction
    algorithm for low-complexity full-diversity MIMO detection,
    IEEE Trans. Signal Processing, vol. 57, pp. 2701-2710, July 2009.

    Parameters:
    ----------
    B (ndarray): Basis matrix (real-valued only).

    Returns:
    -------
    The unimodular transformation matrix. Also note that B is modified.

    """
    n = B.shape[1]  # Number of columns
    U = np.eye(n, dtype=np.int32)

    # QR decomposition for Gram-Schmidt orthogonalization
    if isinstance(B, sps.spmatrix | sps.sparray):
        _, R, E, _ = spqr.qr(B)  # Capture permutation E
        if update_B:
            B = B[:, E]  # Permute B at the start (Option 1)
        R.resize((n, n))
        diag = R.diagonal()
        beta = diag**2
        mu = (R / diag.reshape(-1, 1)).T.tocsr()
    else:
        R = np.linalg.qr(B, mode="r")
        diag = np.diag(R)
        beta = diag**2
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
        B = B[:, E_inv]  # Reorder columns of B back to original order
        U = U[:, E_inv]  # Reorder columns of U accordingly

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
    R = np.linalg.qr(B, mode="r")
    diag = np.diag(R)
    beta = diag**2
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

            muswap = mu[k - 2, : k - 2].copy()
            mu[k - 2, : k - 2] = mu[k - 1, : k - 2]
            mu[k - 1, : k - 2] = muswap

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
        A[i + 1 :] -= np.outer(A[i + 1 :, i], A[i])

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
        M = np.block(
            [  # TODO: make this block more efficient, maybe sparse
                [np.zeros((n, n)), A.T, np.zeros((n, m))],
                [A, np.zeros((m, m)), np.eye(m)],
                [np.diag(w / s), np.zeros((m, m)), np.diag(w / s)],
            ]
        )

        rhs = np.concatenate(
            [
                np.zeros(n),  # Ax = 0
                # b - (A @ x + w)  # TODO: this isn't right; use b and y somewhere here
                np.zeros(m),  # A^T y + s = 0
                mu_new - (w / s),  # Centering condition
            ]
        )

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
    ----------
    - A (numpy.ndarray): Constraint matrix of size (m, n).
    - b (numpy.ndarray): Constraint bounds of size (m,).
    - x_opt (numpy.ndarray): Optimal solution of size (n,).
    - y (numpy.ndarray): Dual variables of size (m,). Get values from Gurobi's Pi attribute.
    - q (int): Number of steps to walk.
    - alpha (float): Fraction of maximum allowable step size (default 0.01).

    Returns:
    -------
    - numpy.ndarray: Sequence of interior points of size (q+1, n).

    """
    # Initializations
    x = x_opt.copy().reshape(-1, 1)
    y = y_opt.copy().reshape(-1, 1)
    # AA = A @ A.T  # Precompute A @ A^T
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
        step_alpha = alpha  # min(alpha * alpha_max, alpha_max)

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
    finite_l = l > -infinity
    fls = finite_l.sum()
    finite_u = u < infinity
    fus = finite_u.sum()
    w = np.zeros((2 * n + fls + fus,))
    w[:n] = p
    iterations = 0

    # min F(x, s, s_u, s_l) = ||Ax + s - b||² + ||x + s_u - u||² + ||x - s_l - l||² + (||x - p||² - d²)²
    def score(v, *args, **kwargs):
        nonlocal iterations
        iterations += 1
        x = v[:n]
        s = v[n : 2 * n]
        s_l = v[2 * n : 2 * n + fls]
        s_u = v[2 * n + fls :]
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
    ----------
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
    -------
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
        KKT = sps.block_array([[sps.csr_matrix((m, m)), A], [A.T, -X_L_inv @ Z_L - X_U_inv @ Z_U]], format="csr")

        # Right-hand side
        rhs = np.hstack([-r_b, -r_c - X_L_inv @ Z_L @ r_b - X_U_inv @ Z_U @ r_b])

        # Solve Newton system
        delta = spsl.spsolve(KKT, rhs)
        delta_x = delta[:m]
        delta_y = delta[m:]

        # Compute steps for primal and dual variables
        step_primal = np.min(np.minimum(alpha * (u - x) / delta_x, alpha * (x - l) / -delta_x))
        step_dual = np.min(np.minimum(alpha * -z_l / delta_x, alpha * z_u / delta_x))
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


def smith_normal_form(A):
    import hsnf  # lame library limited to int64
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


def to_U_via_LU(A, scalar=1.0):
    # we're not sure how to use the scalar here, as the divisions below destroy it.
    # if we scale L and U at scalar * distanceFromDiagonal, we might change the shearing -- the thing we didn't want to change.
    P, L, U = spl.lu(A, overwrite_a=False)
    # assert np.allclose(P @ L @ U, A, atol=1e-5)

    # expect that L has 1s on diagonal with all below filled, U has all to right filled
    for i in range(L.shape[1]):
        div = L[i, i]
        L[i:, i] /= abs(div)

    for i in range(U.shape[0]):
        div = U[i, i]
        U[i, i:] /= abs(div)

    L = np.rint(L)
    U = np.rint(U)

    return P @ L @ U

def to_U_via_iteration2(A: np.ndarray, mult, tol=1e-8):
     # Fallback: Use simple Gaussian elimination approach
    # This doesn't preserve orthogonality well but guarantees unimodularity
    m = A.shape[0]
    
    # Round to get integer matrix
    B = np.rint(A * mult).astype(np.int64)
    
    # Build unimodular transform using elementary row operations
    # Start with identity
    U = np.eye(m, dtype=np.int64)
    
    # Gauss elimination with integer-preserving pivoting
    for k in range(min(m, B.shape[1])):
        # Find best pivot (largest absolute value to minimize error)
        pivot_row = k
        for i in range(k+1, m):
            if abs(B[i, k]) > abs(B[pivot_row, k]):
                pivot_row = i
        
        # Swap rows if needed (unimodular with det = -1)
        if pivot_row != k:
            B[[k, pivot_row]] = B[[pivot_row, k]]
            U[[k, pivot_row]] = U[[pivot_row, k]]
        
        # Eliminate below pivot using integer operations
        if B[k, k] != 0:
            for i in range(k+1, m):
                # q = round(B[i,k] / B[k,k])
                q = int(np.round(B[i, k] / B[k, k]))
                if q != 0:
                    B[i] -= q * B[k]  # Row operation: R_i -= q * R_k
                    U[i] -= q * U[k]  # Apply same to transform
    
    # Normalize signs on diagonal
    for i in range(m):
        if i < B.shape[1] and B[i, i] < 0:
            B[i] *= -1
            U[i] *= -1
    
    return U

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
            norm_i_minus_1_sq = np.dot(B[i - 1], B[i - 1])

            # Heuristic swap condition: if B[i] is significantly shorter
            # than B[i-1], swap them.
            if norm_i_sq < swap_threshold * norm_i_minus_1_sq:
                pass_modified = True
                B[[i, i - 1]] = B[[i - 1, i]]
                U[[i, i - 1]] = U[[i - 1, i]]

        # --- Termination Check ---
        # If a full pass of both phases made no changes, we are done.
        if not pass_modified:
            return U, mp


def _orient_for_orthogonality(M: np.ndarray, by_rows: bool) -> np.ndarray:
    """Return a view where the vectors of interest live in the columns."""
    return M.T if by_rows else M


def _normalise_vectors(oriented: np.ndarray, tol: float = 1e-12) -> np.ndarray | None:
    """Normalise column vectors to unit norm, signalling failure on degenerate columns."""

    if oriented.size == 0:
        return oriented

    norms = np.linalg.norm(oriented, axis=0)
    if np.any(norms <= tol):
        return None

    oriented = oriented / norms
    return oriented


def _gram_matrix(M: np.ndarray, by_rows: bool) -> np.ndarray:
    oriented = _orient_for_orthogonality(M, by_rows)
    return oriented.T @ oriented


def orthogonality_measure_1(
    Q: np.ndarray,
    *,
    by_rows: bool = False,
    include_diagonal: bool = True,
    normalise_vectors: bool = True,
) -> np.floating:
    """Frobenius norm of Gram deviation (optionally normalised to capture pure angles).

    Parameters
    ----------
    Q : ndarray
        Matrix whose column (default) or row vectors will be tested.
    by_rows : bool, optional
        When ``True`` treat the rows of ``Q`` as the vectors instead of columns.
    include_diagonal : bool, optional
        When ``True`` penalise deviations in both angles and lengths. Set to ``False``
        to ignore column/row norms and measure only off-diagonal correlations.
    normalise_vectors : bool, optional
        When ``True`` (default) scale each vector to unit length before measuring,
        making the metric sensitive only to angles. Set to ``False`` to keep the
        original scale-sensitive behaviour.
    """

    oriented = _orient_for_orthogonality(Q, by_rows).astype(float, copy=True)
    if normalise_vectors:
        oriented = _normalise_vectors(oriented)
        if oriented is None:
            return np.inf

    gram = oriented.T @ oriented
    if include_diagonal:
        deviation = gram - np.eye(gram.shape[0])
        return np.linalg.norm(deviation, "fro")

    np.fill_diagonal(gram, 0.0)
    return np.linalg.norm(gram, "fro")


def orthogonality_measure_2(
    Q: np.ndarray,
    *,
    by_rows: bool = False,
    normalise_vectors: bool = True,
) -> np.floating:
    """L2 distance of singular values from 1 with optional vector normalisation."""

    oriented = _orient_for_orthogonality(Q, by_rows).astype(float, copy=True)
    if normalise_vectors:
        oriented = _normalise_vectors(oriented)
        if oriented is None:
            return np.inf

    s = np.linalg.svd(oriented, compute_uv=False)
    return np.linalg.norm(s - 1.0)


def difference(A: np.ndarray, B: np.ndarray, *, by_rows: bool = False) -> float:
    """Cosine-angle based dissimilarity between two vector frames.

    Values are in ``[0, 2]`` with ``0`` meaning the frames agree up to rotation.
    """

    A_oriented = _orient_for_orthogonality(A, by_rows)
    B_oriented = _orient_for_orthogonality(B, by_rows)
    Af = np.linalg.norm(A_oriented, ord="fro")
    Bf = np.linalg.norm(B_oriented, ord="fro")
    if Af == 0 or Bf == 0:
        return np.nan
    tr = np.abs(np.trace(A_oriented.T @ B_oriented))
    return 1 - tr / (Af * Bf)


def difference_2(A: np.ndarray, B: np.ndarray, *, by_rows: bool = False) -> float:
    """Spectral norm of the gap between two frames."""

    A_oriented = _orient_for_orthogonality(A, by_rows)
    B_oriented = _orient_for_orthogonality(B, by_rows)
    return np.linalg.norm(A_oriented - B_oriented, 2)


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
        if norm_sq < 1e-12:  # Using a tolerance for floating-point comparisons
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


def measure_orthogonality_deviation(H: np.ndarray, *, by_rows: bool = False, include_diagonal: bool = False) -> float:
    """Column/row Gram deviation after normalising vector lengths."""

    oriented = _orient_for_orthogonality(H, by_rows).astype(float, copy=True)
    if oriented.size == 0:
        return np.inf

    norms = np.linalg.norm(oriented, axis=0)
    keep = norms > 1e-12
    if not np.any(keep):
        return np.inf
    oriented = oriented[:, keep]
    norms = norms[keep]
    oriented /= norms
    return orthogonality_measure_1(
        oriented,
        include_diagonal=include_diagonal,
        normalise_vectors=False,
    )


def measure_orthogonality(H: np.ndarray, *, by_rows: bool = False) -> float:
    """Log-based orthogonality proxy (scale-sensitive, robust to zeros)."""

    oriented = _orient_for_orthogonality(H, by_rows)
    col_norms = np.linalg.norm(oriented, axis=0, ord=2)

    # Filter out zero columns
    nonzero_norms = col_norms[col_norms > 1e-12]
    if len(nonzero_norms) == 0:
        return np.inf

    if oriented.shape[0] != oriented.shape[1]:
        s = np.linalg.svd(oriented, compute_uv=False)
        nonzero_s = s[s > 1e-12]
        if len(nonzero_s) == 0:
            return np.inf
        log_det = np.sum(np.log(nonzero_s))
    else:
        det = np.linalg.det(oriented)
        if abs(det) < 1e-12:
            return np.inf
        log_det = np.log(abs(det))

    # Use log arithmetic to avoid overflow
    log_prod_norms = np.sum(np.log(nonzero_norms))
    return log_prod_norms - log_det

def seysen_reduce_recursive(R):
    # from https://eprint.iacr.org/2025/774.pdf
    # run QR first and only reduce R, expects columns to be basis vectors
    n = R.shape[0]
    if n == 0 or n != R.shape[1]:
        raise ValueError("R must be a square non-empty matrix")

    if n == 1:
        return np.array([[1]], dtype=np.int64)

    m = n // 2
    U11 = seysen_reduce(R[:m, :m])
    U22 = seysen_reduce(R[m:, m:])

    R[:m, m:] @= U22

    # Compute inv(R11) @ R12 using solve_triangular (since R11 is upper triangular)
    # Y = spl.solve_triangular(R[:m, :m], R[:m, m:], lower=False, check_finite=False)
    # some bug in solve_triangular makes it very slow
    Y = np.linalg.solve(R[:m, :m], R[:m, m:])
    U12 = np.rint(-Y)

    # Update R12 <- R11 @ U'12 + R12
    R[:m, m:] += R[:m, :m] @ U12  # not needed if not updating R in place

    # Construct U
    U = np.zeros((n, n), dtype=np.int64)
    U[:m, :m] = U11
    U[:m, m:] = U11 @ U12
    U[m:, m:] = U22

    return U

def seysen_reduce(R):
    # from https://eprint.iacr.org/2025/774.pdf
    # run QR first and only reduce R, expects columns to be basis vectors
    n = R.shape[0]
    if n == 0 or n != R.shape[1]:
        raise ValueError("R must be a square non-empty matrix")

    if n == 1:
        return np.array([[1]], dtype=np.int64)
    
    if sps.issparse(R):
        U = sps.eye(n, dtype=np.int64, format='csr')
        R = R.tocsc(copy=False)
    else:
        U = np.eye(n, dtype=np.int64)

    seysen_reduce_blaster(R, U)
    return U

@functools.cache
def __reduction_ranges(n):
    """
    Return list of ranges that needs to be reduced.

    More generally, it returns, without using recursion, the list that would be
    the output of the following Python program:

    <<<BEGIN CODE>>>
    def rec_range(n):
        bc, res = [], []
        def F(l, r):
            if l == r:
                return
            if l + 1 == r:
                bc.append(l)
            else:
                m = (l + r) // 2
                F(l, m)
                F(m, r)
                res.append((l, m, r))
        return F(0, n)
    <<<END CODE>>>

    :param n: the length of the array that requires reduction
    :return: pair containing `the base_cases` and `result`.
             `base_cases` is a list of indices `i` such that:
                `i + 1` needs to be reduced w.r.t. `i`.
             `result` is a list of triples `(i, j, k)` such that:
                `[j:k)` needs to be reduced w.r.t. `[i:j)`.
             The guarantee is that for any 0 <= i < j < n:
             1) `i in base_cases && j = i + 1`,
             OR
             2) there is a triple (u, v, w) such that `i in [u, v)` and `j in [v, w)`.
    """
    bit_shift, parts, result, base_cases = 1, 1, [], []
    while parts < n:
        left_bound, left_idx = 0, 0
        for i in range(1, parts + 1):
            right_bound = left_bound + 2 * n

            mid_idx = (left_bound + n) >> bit_shift
            right_idx = right_bound >> bit_shift

            if right_idx > left_idx + 1:
                # Only consider nontrivial intervals
                if right_idx == left_idx + 2:
                    # Return length 2 intervals separately to unroll base case.
                    base_cases.append(left_idx)
                else:
                    # Properly sized interval:
                    result.append((left_idx, mid_idx, right_idx))
            left_bound, left_idx = right_bound, right_idx
        parts *= 2
        bit_shift += 1
    return base_cases, list(reversed(result))


def seysen_reduce_blaster(R, U):
    """
    Perform Seysen's reduction on a matrix R, while keeping track of the transformation matrix U.
    The matrix R is updated along the way.

    :param R: an upper-triangular matrix that will be modified
    :param U: an upper-triangular transformation matrix such that diag(U) = (1, 1, ..., 1).
    :return: Nothing! R is Seysen reduced in place.
    """
    # Assume diag(U) = (1, 1, ..., 1).
    assert R.shape[0] == R.shape[1], "R must be square"
    n = R.shape[0]

    base_cases, ranges = __reduction_ranges(n)
    for i in base_cases:
        U[i, i + 1] = -round(R[i, i + 1] / R[i, i])
        R[i, i + 1] += R[i, i] * U[i, i + 1]

    for i, j, k in ranges:
        # Seysen reduce [j, k) with respect to [i, j).
        #
        #     [R11 R12]      [U11 U12]              [S11 S12]
        # R = [ 0  R22], U = [ 0  U22], S = R · U = [ 0  S22]
        #
        # The previous iteration has computed U11 and U22.
        # Currently, R11 and R22 contain the values of
        # S11 = R11 · U11 and S22 = R22 · U22 respectively.

        # S12' = R12 · U22.
        R[i:j, j:k] @= U[j:k, j:k]

        # U12' = round(-S11^{-1} · S12').
        # U[i:j, j:k] = np.rint(-np.linalg.inv(R[i:j, i:j]) @ R[i:j, j:k]).astype(np.int64)
        # expecting inverse to be less accurate than solve_triangular:
        # X = spl.solve_triangular(R[i:j, i:j], R[i:j, j:k], lower=False, check_finite=False)
        # but solve_triangular is weirdly slow, so we use linalg.solve instead:
        if sps.issparse(R):
            # X = spsl.spsolve_triangular(R[i:j, i:j], R[i:j, j:k], lower=False, overwrite_b=False)
            X = spsl.spsolve(R[i:j, i:j], R[i:j, j:k])
        else:
            X = np.linalg.solve(R[i:j, i:j], R[i:j, j:k])
        U[i:j, j:k] = np.rint(-X).astype(np.int64)
        
        # S12 = S12' + S11 · U12'.
        R[i:j, j:k] += R[i:j, i:j] @ U[i:j, j:k]

        # U12 = U11 · U12'
        U[i:j, j:k] = U[i:j, i:j] @ U[i:j, j:k]

def pairwise_hyperplane_angles(A_active: np.ndarray, *, by_rows: bool = True, acute: bool = True) -> np.ndarray:
    """Return pairwise angles (in radians) between constraint normals.

    Parameters
    ----------
    A_active : ndarray
        Matrix whose vectors represent hyperplane normals.
    by_rows : bool, optional
        Treat rows as normals when ``True`` (default, preserves previous behaviour).
        Set to ``False`` to interpret the columns as normals instead.
    acute : bool, optional
        When ``True`` fold obtuse angles into the acute range via ``abs`` on cosines.
    """

    vectors = _orient_for_orthogonality(A_active, by_rows)
    norms = np.linalg.norm(vectors, axis=0)
    if np.any(norms == 0):
        raise ValueError("Zero normal found.")
    unit = vectors / norms
    C = unit.T @ unit
    C = np.clip(C, -1.0, 1.0)
    if acute:
        C = np.abs(C)
    return np.arccos(C)


def relative_error(target: np.ndarray, approx: np.ndarray) -> float:
    """Compute the relative Frobenius-norm error between two matrices."""
    return float(np.linalg.norm(target - approx, ord="fro") / np.linalg.norm(target, ord="fro"))


def lll_integer_matrix(T: np.ndarray, scale: int) -> tuple[np.ndarray, np.ndarray]:
    """Apply LLL reduction to a scaled integer version of matrix T."""
    integer_scaled = np.round(scale * T).astype(np.int64, order="C")
    rank, _, U = ntl.lll(integer_scaled, 9, 10)
    # assert rank == T.shape[0]
    return U

def lu_integer_matrix(T: np.ndarray):
    return to_U_via_LU(T)

def seysen_integer_matrix(T: np.ndarray, scale: int):
    """Apply Seysen reduction to (possibly rectangular) matrix ``T``.
    
    Supports both dense numpy arrays and scipy sparse matrices.
    """

    rows, cols = T.shape
    P = None
    if rows >= cols:
        if sps.issparse(T):
            _, R, P, rank = spqr.qr(T, economy=True)   # R is cols×cols
            # gives Q*R = A*permutation_vector_to_matrix(P)
            use_qr = rank == cols
        else:
            R = np.linalg.qr(T, mode="r")   # R is cols×cols
            # I'm really trying hard to avoid an expensive rank check here so...
            # heuristic full-rank check
            if np.any(np.abs(np.diag(R)) < 1e-12):
                use_qr = False
            else:
                use_qr = True
    else:
        use_qr = False

    if not use_qr:
        gram = T.T @ T
        if sps.issparse(gram):
            # TODO: check the actual density on this; see if sparse Cholesky is worth it
            gram = gram.toarray()
            P = None
        reg = 0.0
        eye = np.eye(cols, dtype=np.float64)
        while True:
            try:
                R = spl.cholesky(gram + reg * eye, lower=False, check_finite=False)
                break
            except np.linalg.LinAlgError:
                reg = 1e-9 if reg == 0.0 else reg * 10.0
                if reg > 1e-2:
                    raise

    R_scaled = R * scale
    U = seysen_reduce(R_scaled)
    if P is not None:
        perm = sps.eye(cols, dtype=U.dtype, format="csc")[:, P]
        # perm = spqr.permutation_vector_to_matrix(P)
        U = perm @ U
    return U


def cleanup_with_lll(basis: np.ndarray) -> np.ndarray:
    """Apply LLL reduction to clean up an integer basis.
    
    Returns basis @ U where U is the LLL unimodular transform.
    If the rank collapses during rounding, returns the original basis unchanged.
    """
    rank, _, U_obj = ntl.lll(basis.copy(), 9, 10)
    if rank != basis.shape[0]:  # rounding may collapse rank; skip cleanup
        return basis
    U = np.asarray(U_obj, dtype=np.int64) if U_obj.dtype != np.int64 else U_obj
    return basis @ U

def W_from_Q_via_LLL(Q: np.ndarray, verify: bool = True) -> np.ndarray:
    """Compute an integer unimodular matrix W such that W @ Q = I.
    Q need not be square.
    
    Uses the LLL-based Diophantine equation solver for each column.
    See: papers/Lattice Reformulation Cuts.pdf section 4.2.3
    and papers/Solving Diophantine Equations.pdf
    """

    # We start with Q at n x n-m. We need W to hold all vars, so it must need to be n-m x n
    Q = Q.T
    n_m, n = Q.shape

    Q = np.asarray(Q, dtype=np.int64) if Q.dtype != np.int64 else Q
    W = np.zeros((n_m, n), dtype=np.int64)
    N1 = int(max(np.linalg.norm(Q, np.inf).item(), 1) * 100)  # * 2**(m + n) instead of *6
    N2 = N1 * 6

    for i in range(n_m):
        # We need to solve: Q @ w = e_i
        # Using LLL approach from "Solving Diophantine Equations":
        # Construct matrix B = [ I_n   0  ]
        #                      [ 0     N1 ]
        #                      [ N2*A -N2*b ]
        # where A = Q, b = e_i
        
        # Build the augmented matrix for LLL (fresh matrix each iteration):
        # [ I_n        0   ]  <- n rows
        # [ 0          N1  ]  <- 1 row  
        # [ N2*Q    -N2*b  ]  <- m rows
        # where b = e_i (i-th standard basis vector)
        B = np.eye(n + 1 + n_m, n + 1, dtype=np.int64)  # Creates identity in first min(rows,cols) diagonal
        B[n, n] = N1  # Overwrite the (n,n) entry
        B[n+1:n+1+n_m, 0:n] = N2 * Q
        B[n+1 + i, n] = -N2  # Set -N2*b where b is e_i (only i-th entry is 1)
        
        # Apply LLL reduction (B is modified in place)
        rank, det, U = ntl.lll(B, 9, 10)
        
        # The solution is in the first n entries of a column where:
        # - the (n)-th entry equals N1 (indicating coefficient 1 for the "1" variable)
        # - the last m entries are zero (constraints satisfied)
        
        # Look for the column with the particular solution
        found = False
        for col in range(B.shape[1]):
            if B[n, col] == N1:
                # Check if constraints are satisfied (last rows should be zero)
                if np.allclose(B[n+1:, col], 0):
                    w = B[0:n, col]
                    found = True
                    break
        
        if not found:
            raise ValueError(f"LLL failed to find solution for column {i}")
        
        if verify:
            # Verify the solution: Q @ w should equal e_i
            product = Q @ w.reshape(-1, 1)
            expected = np.zeros((n_m, 1), dtype=np.int64)
            expected[i, 0] = 1
            if not np.allclose(product, expected):
                raise ValueError(f"LLL solution verification failed for column {i}: got {product.flatten()}, expected {expected.flatten()}")
        
        W[i, :] = w

    if verify:
        # Final verification
        result = W @ Q.T
        expected = np.eye(n_m, dtype=np.int64)
        if not np.allclose(result, expected):
            raise ValueError(f"Final verification failed: W @ Q.T != I")

    return W
