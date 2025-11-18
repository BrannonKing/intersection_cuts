import knapsack_loader as kl
import jsplib_loader as jl
import gurobipy as gp
import gurobi_utils as gu
import numpy as np
import sympy as sp
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


np.set_printoptions(precision=3, suppress=True, edgeitems=8, linewidth=120)
models = kl.generate(1, 4, 30, 5, 10, 1000, equality=True)
# instances = list(jl.get_instances().values())
# models = [instance.as_gurobi_balas_model(use_big_m=True, env=env) for instance in instances[5:6]]
for model in models:
    name = model.ModelName
    print(f"Model: {name}")
    A = model.getA().toarray().astype(np.int64)
    b = np.array(model.getAttr("RHS")).reshape((-1, 1)).astype(np.int64)

    x0, null_basis, S, U, V, As, bs = smith_nullspace_and_particular(A, b)

    print("Null space basis vectors (columns):")
    sp.pprint(null_basis)

    # diag_len = min(S.rows, S.cols)
    # diag = [sp.simplify(S[i, i]) for i in range(diag_len)]
    # print("Smith normal form diagonal entries:")
    # print(diag)

    # print("Particular solution x0:")
    # sp.pprint(x0)

    assert (As @ x0 - bs).is_zero_matrix, "Particular solution check failed."
    assert (As @ null_basis).is_zero_matrix, "Null space basis check failed."
    print("---")


