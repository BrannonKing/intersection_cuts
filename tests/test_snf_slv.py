import gurobipy as gp
import numpy as np
import sympy as sp

from .. import knapsack_loader as kl
from .. import jsplib_loader as jl
from .. import gurobi_utils as gu
from sympy.matrices.normalforms import smith_normal_decomp, smith_normal_form  # type: ignore[attr-defined]


def smith_nullspace_and_particular(A_np: np.ndarray, b_np: np.ndarray, tol: float = 1e-9):
    """Compute a null space basis and a particular solution using the Smith normal form."""

    if b_np.ndim == 1:
        b_np = b_np.reshape((-1, 1))

    A_sym = sp.Matrix(A_np)
    b_sym = sp.Matrix(b_np)

    S, U, V = smith_normal_decomp(A_sym)
    # S, U, V = smith_normal_form(A_sym)

    diag_len = min(S.rows, S.cols)
    diag_entries = [S[i, i] for i in range(diag_len)]
    rank = sum(1 for d in diag_entries if d != 0)

    c = U * b_sym
    y0 = sp.zeros(V.cols, 1)

    for i in range(rank):
        d_i = diag_entries[i]
        if d_i == 0:
            break
        y0[i] = sp.simplify(c[i] / d_i)

    for i in range(rank, S.rows):
        if S[i, i] == 0 and sp.simplify(c[i]) != 0:
            raise ValueError("System Ax = b is inconsistent under Smith normal form analysis.")

    x0 = (V * y0).applyfunc(sp.simplify)
    # x1 = A_sym.gauss_jordan_solve(b_sym)[0]
    # lcm_denom = sp.lcm([x.as_numer_denom()[1] for x in x1])
    # x1 = (x1 * lcm_denom).applyfunc(sp.simplify)
    # print("Particular solution x1:")
    # sp.pprint(x1)

    basis_cols: list[sp.Matrix] = []
    for j in range(rank, V.cols):
        vec = sp.Matrix(V[:, j]).applyfunc(sp.simplify)
        nonzero_entries = [sp.Abs(val) for val in vec if val != 0]
        if nonzero_entries:
            gcd_val = sp.gcd_list(nonzero_entries)
            if gcd_val not in (0, 1):
                vec = (vec / gcd_val).applyfunc(sp.simplify)
        basis_cols.append(vec)

    if basis_cols:
        null_basis = sp.Matrix.hstack(*basis_cols)
    else:
        null_basis = sp.Matrix.zeros(V.rows, 0)

    if (A_sym * x0 - b_sym) != sp.zeros(A_sym.rows, 1):
        raise ValueError("Computed particular solution does not satisfy Ax = b.")

    for vec in basis_cols:
        if (A_sym * vec) != sp.zeros(A_sym.rows, 1):
            raise ValueError("Computed null space vector does not satisfy A v = 0.")

    return x0, null_basis, S, U, V, A_sym, b_sym



def test_smith_nullspace_computation():
    """Test Smith normal form based null space and particular solution computation."""
    models = list(kl.generate(1, 4, 30, 5, 10, 1000, equality=True))
    
    for model in models:
        A = model.getA().toarray().astype(np.int64)
        b = np.array(model.getAttr("RHS")).reshape((-1, 1)).astype(np.int64)

        x0, null_basis, S, U, V, As, bs = smith_nullspace_and_particular(A, b)

        # Verify particular solution satisfies Ax = b
        assert (As @ x0 - bs).is_zero_matrix, "Particular solution check failed."
        
        # Verify null space basis satisfies Av = 0 for all v
        assert (As @ null_basis).is_zero_matrix, "Null space basis check failed."
        
        # Verify Smith normal form decomposition exists
        assert S is not None, "Smith normal form should exist"
        assert U is not None and V is not None, "Transformation matrices should exist"


