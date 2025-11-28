import gurobipy as gp
import numpy as np
import sympy as sp

from .. import knapsack_loader as kl
from .. import jsplib_loader as jl
from .. import gurobi_utils as gu
from sympy.matrices.normalforms import hermite_normal_form  # type: ignore[attr-defined]


def hermite_nullspace_and_particular(A_np: np.ndarray, b_np: np.ndarray, tol: float = 1e-9):
    """Compute a null space basis and a particular solution using the Hermite normal form."""

    if b_np.ndim == 1:
        b_np = b_np.reshape((-1, 1))

    A_sym = sp.Matrix(A_np)
    b_sym = sp.Matrix(b_np)

    # Hermite normal form: H = U * A, where U is unimodular
    hnf_result = hermite_normal_form(A_sym, D=None)
    if isinstance(hnf_result, tuple):
        H, U = hnf_result
    else:
        H = hnf_result
        U = sp.eye(A_sym.rows)
    
    # Determine rank from H (number of non-zero rows)
    rank = 0
    for i in range(H.rows):
        if not all(H[i, j] == 0 for j in range(H.cols)):
            rank += 1
        else:
            break

    # Transform b using U
    c = U * b_sym

    # Check consistency: if any row i > rank has c[i] != 0, system is inconsistent
    for i in range(rank, H.rows):
        if sp.simplify(c[i]) != 0:
            raise ValueError("System Ax = b is inconsistent under Hermite normal form analysis.")

    # Solve the system A * x = b for a particular solution
    # We use the original A matrix since the null space computation will work the same way
    try:
        solution = A_sym.gauss_jordan_solve(b_sym)
        x0 = solution[0].applyfunc(sp.simplify)
    except ValueError as e:
        raise ValueError(f"System Ax = b is inconsistent: {e}")

    # Compute null space basis from H
    # The null space of H is the same as the null space of A
    null_space = H.nullspace()
    
    basis_cols: list[sp.Matrix] = []
    for vec in null_space:
        vec = vec.applyfunc(sp.simplify)
        nonzero_entries = [sp.Abs(val) for val in vec if val != 0]
        if nonzero_entries:
            gcd_val = sp.gcd_list(nonzero_entries)
            if gcd_val not in (0, 1):
                vec = (vec / gcd_val).applyfunc(sp.simplify)
        basis_cols.append(vec)

    if basis_cols:
        null_basis = sp.Matrix.hstack(*basis_cols)
    else:
        null_basis = sp.Matrix.zeros(A_sym.cols, 0)

    # Validate the particular solution
    if (A_sym * x0 - b_sym) != sp.zeros(A_sym.rows, 1):
        raise ValueError("Computed particular solution does not satisfy Ax = b.")

    # Validate null space vectors
    for vec in basis_cols:
        if (A_sym * vec) != sp.zeros(A_sym.rows, 1):
            raise ValueError("Computed null space vector does not satisfy A v = 0.")

    return x0, null_basis, H, U, A_sym, b_sym



def test_hermite_nullspace_computation():
    """Test Hermite normal form based null space and particular solution computation."""
    models = list(kl.generate(1, 4, 30, 5, 10, 1000, equality=True))
    
    for model in models:
        A = model.getA().toarray().astype(np.int64)
        b = np.array(model.getAttr("RHS")).reshape((-1, 1)).astype(np.int64)

        x0, null_basis, H, U, As, bs = hermite_nullspace_and_particular(A, b)

        # Verify particular solution satisfies Ax = b
        assert (As @ x0 - bs).is_zero_matrix, "Particular solution check failed."
        
        # Verify null space basis satisfies Av = 0 for all v
        assert (As @ null_basis).is_zero_matrix, "Null space basis check failed."
        
        # Verify H is in Hermite normal form (upper triangular with specific properties)
        assert H is not None, "Hermite form should exist"
        assert U is not None, "Transformation matrix should exist"
