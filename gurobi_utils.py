from __future__ import annotations

import ctypes as ct
import pathlib
import platform

import gurobipy as gp
import numpy as np
import scipy.sparse as sp

_libs = pathlib.Path(gp.__file__).parent.rglob("*.dll" if platform.system() == "Windows" else "*.so")
# our DLL is likely the largest library there; we can make this more robust when needed
_likely_gurobi_dll = max(_libs, key=lambda fn: fn.stat().st_size)
_gurobi_dll = ct.CDLL(str(_likely_gurobi_dll))
status_lookup = {getattr(gp.GRB.Status, k): k for k in gp.GRB.Status.__dir__() if "A" <= k[0] <= "Z"}


class GRBsvec(ct.Structure):
    _fields_ = [("len", ct.c_int), ("ind", ct.POINTER(ct.c_int)), ("val", ct.POINTER(ct.c_double))]


ct.pythonapi.PyCapsule_GetPointer.restype = ct.c_void_p
ct.pythonapi.PyCapsule_GetPointer.argtypes = [ct.py_object, ct.c_char_p]
ct.pythonapi.PyCapsule_GetName.restype = ct.c_char_p
ct.pythonapi.PyCapsule_GetName.argtypes = [ct.py_object]

_gurobi_dll.GRBgetBasisHead.argtypes = (ct.c_void_p, ct.POINTER(ct.c_int))
_gurobi_dll.GRBBinvColj.argtypes = (ct.c_void_p, ct.c_int, ct.POINTER(GRBsvec))
_gurobi_dll.GRBBinvRowi.argtypes = (ct.c_void_p, ct.c_int, ct.POINTER(GRBsvec))
_gurobi_dll.GRBgetBasisHead.restype = ct.c_int
_gurobi_dll.GRBBinvColj.restype = ct.c_int
_gurobi_dll.GRBBinvRowi.restype = ct.c_int


def read_basis(m: gp.Model):
    assert m.params.Presolve == 0
    assert m.Status == gp.GRB.OPTIMAL, "We can only pull this data from models solved to optimum."
    data = (ct.c_int * m.NumConstrs)()
    ptr = ct.pythonapi.PyCapsule_GetPointer(m._cmodel, None)
    assert data is not None and ptr != 0
    err = _gurobi_dll.GRBgetBasisHead(ptr, data)
    assert err == 0, f"Error from GRBgetBasisHead: {err}"
    return [data[i] for i in range(m.NumConstrs)]

def read_tableau(m: gp.Model, basis, extra_rows=0, remove_basis_cols=True):
    assert m.Status == gp.GRB.OPTIMAL, "We can only pull this data from models solved to optimum."
    # bases says which var each row goes with (AKA, row_to_var
    # if it's in the basis, we don't want it in our final tableau (so we delete those columns)
    # the first n columns are the variables; columns after that are slacks

    rows = len(basis)
    cols = m.NumVars + m.NumConstrs  # maybe m.NumNZs
    tableau = np.zeros((rows + extra_rows, cols))
    data = GRBsvec()
    data.len = cols
    data.ind = (ct.c_int * cols)()
    data.val = (ct.c_double * cols)()
    ptr = ct.pythonapi.PyCapsule_GetPointer(m._cmodel, None)
    assert ptr != 0
    for row in range(rows):
        # TODO: pass in a list of variables to skip so we don't read unnecessary rows
        err = _gurobi_dll.GRBBinvRowi(ptr, row, data)
        assert err == 0
        indexes = data.ind[: data.len]
        values = data.val[: data.len]
        tableau[row, indexes] = values

    col_to_var_idx = np.arange(tableau.shape[1])
    # Note: Gurobi's GRBBinvRowi returns the B^-1 row. For a >= constraint, the logical 
    # variable is internally <= 0, which corresponds to the standard surplus form's `-s`.
    # Therefore, tableau[i, base] will evaluate to -1 instead of +1. However, since the 
    # substitution s = -logical naturally flips the sign of the entire column, the coefficient
    # of the physical variable s is correctly +1. Therefore, NO rows need to be negated for parity.
    negated_rows = []
    if remove_basis_cols:
        # any basis that is negative needs to negate that row:
        col_to_var_idx = np.delete(col_to_var_idx, basis)  # TODO: mask may be better than delete calls here
        tableau = np.delete(tableau, basis, 1)  # remove any columns in the basis
    assert col_to_var_idx.shape[0] == tableau.shape[1]
    return tableau, col_to_var_idx, negated_rows


def fix_tableau_dirs(m: gp.Model, tableau, col_to_var_idx):
    variables = m.getVars()
    constraints = m.getConstrs()
    for col, j in enumerate(col_to_var_idx):
        if j < len(variables):
            # print("Var INFO:", variables[j].VarName, "VBasis", variables[j].VBasis, "LB", variables[j].LB, "UB", variables[j].UB)
            if variables[j].VBasis == -2:
                tableau[:, col] = variables[j].UB - tableau[:, col]  # might need to be UB - ... ?
                # print("  !Fixing variable", variables[j].VarName, "with UB basis")
            elif variables[j].VBasis == -1 and variables[j].LB != 0.0:  # not sure what to do with VBasis=-3
                print("Warning: LB is nonzero for variable", variables[j].VarName, "LB", variables[j].LB, "UB", variables[j].UB)
        else:
            constraint = constraints[j - len(variables)]
            if constraint.Sense == ">":  # Achterberg said lt and lte are standard; should just need to flip gt
                tableau[:, col] = -tableau[:, col]
    return variables, constraints


def find_cuttable_rows(m: gp.Model, var_to_int_idx):
    tol = m.params.FeasibilityTol
    basis = read_basis(m)
    tableau, col_to_var_idx, negated_rows = read_tableau(m, basis, remove_basis_cols=True)
    variables, _ = fix_tableau_dirs(m, tableau, col_to_var_idx)
    for base, row in zip(basis, -tableau):
        if base not in var_to_int_idx:
            continue
        if np.all(row >= -tol):
            yield variables[base], False
        elif np.all(row <= tol):
            yield variables[base], True


def standardize_lt_to_gt(m: gp.Model):
    m.update()
    flip = ("<", ">")
    to_remove = []
    for constraint in m.getConstrs():  # returns only linear constraints
        if constraint.Sense == flip[0]:
            lhs, rhs, name = m.getRow(constraint), constraint.RHS, constraint.ConstrName
            to_remove.append(constraint)
            m.addLConstr(-lhs, flip[1], -rhs, name + "_rev")
    for tr in to_remove:
        m.remove(tr)
    print(f"   Negated {len(to_remove)} constraints on", m.ModelName)


def standardize_gt_to_lt(m: gp.Model):
    m.update()
    flip = (">", "<")
    to_remove = []
    for constraint in m.getConstrs():  # returns only linear constraints
        if constraint.Sense == flip[0]:
            lhs, rhs, name = m.getRow(constraint), constraint.RHS, constraint.ConstrName
            to_remove.append(constraint)
            m.addLConstr(-lhs, flip[1], -rhs, name + "_rev")
    for tr in to_remove:
        m.remove(tr)
    print(f"   Negated {len(to_remove)} constraints on", m.ModelName)


def standardize_eq_to_gt(m: gp.Model):
    m.update()
    to_remove = []
    for constraint in m.getConstrs():  # returns only linear constraints
        if constraint.Sense == "=":
            lhs, rhs, name = m.getRow(constraint), constraint.RHS, constraint.ConstrName
            to_remove.append(constraint)
            m.addLConstr(-lhs, ">", -rhs, name + "_rev1")
            m.addLConstr(lhs, ">", rhs, name + "_rev2")
    for tr in to_remove:
        m.remove(tr)
    print(f"   Made {len(to_remove)} constraints into <> on", m.ModelName)


def standardize_ub_to_constr(m: gp.Model):
    m.update()
    added = []
    for var in m.getVars():
        if var.UB < gp.GRB.INFINITY:
            added.append(m.addConstr(-var >= -var.UB))
            var.UB = gp.GRB.INFINITY
    print(f"   Standardized {len(added)} upper bounds to be constraints")
    return added


def standardize_lb_to_constr(m: gp.Model):
    m.update()
    added = []
    for var in m.getVars():
        if var.LB > -gp.GRB.INFINITY and var.LB != 0.0:
            added.append(m.addConstr(var >= var.UB))
            var.LB = -gp.GRB.INFINITY
    print(f"   Standardized {len(added)} lower bounds to be constraints")
    return added


def relax_int_or_bin_to_continuous(m: gp.Model, verbose=False):
    m.update()
    relaxed_variables = []
    relaxed_index = {}
    for var in m.getVars():
        if var.VType != gp.GRB.CONTINUOUS:
            if var.VType == gp.GRB.BINARY:
                var.UB = 1
                assert var.LB == 0
            var.VType = gp.GRB.CONTINUOUS
            relaxed_index[var.index] = len(relaxed_variables)
            relaxed_variables.append(var)
    if verbose:
        print(f"   Relaxed {len(relaxed_variables)} variables on", m.ModelName)
    return gp.MVar.fromlist(relaxed_variables), relaxed_index


def nearest_integer(variables: gp.MVar):
    x = variables.X
    x = np.round(x)
    ub = variables.UB
    lb = variables.LB
    return np.clip(x, lb, ub)


def validate_corner(model: gp.Model, basis, tableau, col_to_var, progress=None):
    point = gp.MVar.fromlist(model.getVars()).X  # could this be done just for the relaxed integers, in that space? Prolly
    slacks = [constraint.Slack for constraint in model.getConstrs()]
    point = np.hstack([point, slacks])
    assert tableau.shape[0] == len(basis) and tableau.shape[1] == len(point) - len(basis)
    failures = 0
    A = model.getA().todense()
    tableau = np.append(tableau, np.ones((1, tableau.shape[1])), axis=0)
    lengths = np.linalg.norm(tableau, 2, axis=0)
    basis = basis + [-1]
    constraints = [constraint for constraint in model.getConstrs() if constraint.Sense != "="]
    for i, ray in enumerate(tableau.T):
        if progress is not None:
            next(progress)
        point_shifted = point.copy()
        basis[-1] = col_to_var[i]
        point_shifted[basis] += ray * 0.1 / lengths[i]
        # TODO: optimize this so it runs faster
        for constraint in constraints:
            if constraint.Sense == "<":
                new_lhs = A[constraint.index, :] @ point_shifted[0 : A.shape[1]] - point_shifted[A.shape[1] + constraint.index]
                if new_lhs.item() > constraint.RHS + model.params.FeasibilityTol:
                    print("   Failed validation!", i, constraint, model.getRow(constraint), point_shifted, "<=", constraint.RHS)
                    failures += 1
            elif constraint.Sense == ">":
                new_lhs = A[constraint.index, :] @ point_shifted[0 : A.shape[1]] + point_shifted[A.shape[1] + constraint.index]
                if new_lhs.item() < constraint.RHS - model.params.FeasibilityTol:
                    print("   Failed validation!", i, constraint, model.getRow(constraint), point_shifted, ">=", constraint.RHS)
                    failures += 1
    return failures


def apply_transform(old_model: gp.Model, U: np.ndarray, x0: np.ndarray, basis=None, normalize_Ab=False, mult=1, ignore_bounds=False, env=None):
    """Apply the transformation U to the model."""
    old_model.update()
    # A, b, c, l, u = get_A_b_c_l_u(result) # for debug

    # going to shift it to 0, then apply the transformation, then shift it back (all in one operation):
    # going with this substitution: y=U_inv(x - x0) so x=U(y - x0) + x0
    # and for Ax <= b: AUy <= b + A(Ux0 - x0)
    # and for c^T x: c^T Uy - c^T U x0 + c^T x0
    # and for l <= x <= u: l - x0 <= U (y - x0) <= u - x0  # keep U in the middle or it will mess with the inequality direction

    # Ensure unimodular_matrix has the correct dimensions
    num_vars = old_model.NumVars
    if basis is None and U.shape != (num_vars, num_vars):
        raise ValueError("Unimodular matrix must have dimensions matching the number of variables.")

    # Get original data
    A = old_model.getA()  # Constraint coefficient matrix (as a scipy.sparse matrix)
    b = old_model.getAttr("RHS")  # Right-hand side vector
    sense = old_model.getAttr("Sense")  # Constraint senses (<=, >=, =)
    if not ignore_bounds:
        lb = np.array(old_model.getAttr("LB"))
        ub = np.array(old_model.getAttr("UB"))

    if normalize_Ab:
        for i in range(A.shape[0]):
            if b[i] == 0.0:
                A[i, :] /= np.abs(A[i, :].toarray()).max()
            else:
                A[i, :] /= np.abs(b[i])
                b[i] = np.sign(b[i])

    U_inv = np.linalg.inv(U)
    variables = old_model.getVars()
    if basis is not None:
        assert len(basis) == U.shape[0]
        all_indices = np.arange(num_vars)
        mask = np.ones(all_indices.shape, dtype=bool)
        mask[basis] = False
        non_basis = all_indices[mask]
        new_order = np.concatenate((basis, non_basis))
        variables = [variables[i] for i in new_order]
        x0 = x0[new_order]
        if not ignore_bounds:
            lb = lb[new_order]
            ub = ub[new_order]
        eye = sp.eye(num_vars - len(basis))
        U_inv = sp.block_diag([U_inv, eye], format="csr")
        U = sp.block_diag([U, eye], format="csr")
        # columns of A must also be sorted to match the new basis
        A = A[:, new_order]

    vtypes = []
    for v in variables:
        if v.VType == gp.GRB.BINARY:
            lb[v.index] = 0.0
            ub[v.index] = 1.0
            vtypes.append(gp.GRB.INTEGER)
        else:
            vtypes.append(v.VType)
    vtypes = np.array(vtypes)

    # lb[lb < -gp.GRB.INFINITY] = -gp.GRB.INFINITY  # we can't have infinities when we multiply by 0
    # ub[ub > gp.GRB.INFINITY] = gp.GRB.INFINITY

    # we translate it by x0, do the transform, then transform it back -x0
    x0 = x0.flatten()
    if not ignore_bounds:
        lb -= x0
        ub -= x0

    # Create a new model
    new_model = gp.Model(name=f"{old_model.ModelName}_transformed", env=env)

    # Add new variables y corresponding to the transformed space
    y_vars = new_model.addMVar(num_vars, lb=0 if ignore_bounds else -gp.GRB.INFINITY, vtype=vtypes, name="y")
    if not ignore_bounds:
        Uyx = U @ (y_vars - x0)
        new_model.addConstr(lb <= Uyx * mult, name="lb")
        ub_idx = ub < gp.GRB.INFINITY
        if np.any(ub_idx):
            new_model.addConstr(ub[ub_idx] >= Uyx[ub_idx] * mult, name="ub")

    for idx, variable in enumerate(variables):
        y_vars[idx].VarName = variable.VarName

    A_transformed = A @ U
    b_deduction = A @ (U @ x0 - x0)

    # Add the transformed constraints AUy <= b (or other senses)
    for i in range(A_transformed.shape[0]):
        expr = A_transformed[i, :] @ y_vars * mult
        if sense[i] == "<":
            new_model.addConstr(expr <= b[i] + b_deduction[i], name=f"lt_{i}")
        elif sense[i] == ">":
            new_model.addConstr(expr >= b[i] + b_deduction[i], name=f"gt_{i}")
        elif sense[i] == "=":
            new_model.addConstr(expr == b[i] + b_deduction[i], name=f"eq_{i}")

    # Transform the objective
    obj_coeffs = np.array([v.Obj for v in variables])  # more efficient way?
    obj = old_model.getObjective()
    new_model.setObjective(obj_coeffs @ U @ y_vars * mult - obj_coeffs @ U @ x0 + obj_coeffs @ x0 + obj.getConstant(), old_model.ModelSense)
    new_model._y_vars = y_vars
    new_model._U = U
    new_model._U_inv = U_inv
    new_model._x0 = x0
    new_model._basis = basis
    new_model.update()

    return new_model


def relax_and_shrink(mdl: gp.Model, diagonal_distance, percent_of_diagonal):
    mdl.update()
    relaxed = mdl.copy()
    if relaxed.NumIntVars > 0:
        _, _ = relax_int_or_bin_to_continuous(relaxed)
    relaxed.update()
    if percent_of_diagonal == 0.0:
        return relaxed

    for v in relaxed.getVars():
        if percent_of_diagonal * 2.0 > v.UB - v.LB:
            gap = (v.UB - v.LB) * percent_of_diagonal
            v.LB += gap
            v.UB -= gap
        else:
            if v.LB > -gp.GRB.INFINITY:
                v.LB += percent_of_diagonal
            if v.UB < gp.GRB.INFINITY:
                v.UB -= percent_of_diagonal

    distance = diagonal_distance * percent_of_diagonal
    for c in relaxed.getConstrs():
        lhs = relaxed.getRow(c)
        coeffs = np.array([lhs.getCoeff(i) for i in range(lhs.size())])
        if c.Sense == "<":
            c.RHS -= distance * np.linalg.norm(coeffs) / lhs.size()
        elif c.Sense == ">":
            c.RHS += distance * np.linalg.norm(coeffs) / lhs.size()
    relaxed.update()
    return relaxed


def relax_and_grow(mdl: gp.Model, x0, distance=1):
    mdl.update()
    relaxed = mdl.copy()
    if relaxed.NumIntVars > 0:
        _, _ = relax_int_or_bin_to_continuous(relaxed)
    relaxed.update()

    x0 = x0.flatten()
    for v in relaxed.getVars():
        if v.LB + distance > x0[v.index]:
            v.LB = min(v.LB, x0[v.index]) - distance
        if v.UB - distance < x0[v.index]:
            v.UB = max(v.UB, x0[v.index]) + distance

    to_be = []
    for c in relaxed.getConstrs():
        lhs = relaxed.getRow(c)

        lhs_value = lhs.getConstant()
        for i in range(lhs.size()):
            var = lhs.getVar(i)
            coeff = lhs.getCoeff(i)
            lhs_value += coeff * x0[var.index]

        coeffs = np.array([lhs.getCoeff(i) for i in range(lhs.size())])
        distance *= np.linalg.norm(coeffs) / lhs.size()
        if c.Sense == "<" and lhs_value > c.RHS - distance:
            c.RHS = max(c.RHS, lhs_value) + distance
        elif c.Sense == ">" and lhs_value < c.RHS + distance:
            c.RHS = min(c.RHS, lhs_value) - distance
        else:
            # assert np.isclose(lhs_value, c.RHS, atol=distance*0.5), "Constraint RHS does not match the left-hand side value."
            c.Sense = ">"
            c.RHS = min(c.RHS, lhs_value) - distance
            expr = gp.quicksum(lhs.getCoeff(i) * lhs.getVar(i) for i in range(lhs.size())) + lhs.getConstant()
            to_be.append(expr <= max(c.RHS, lhs_value) + distance)

    relaxed.addConstrs(c for c in to_be)
    relaxed.update()
    return relaxed


def get_A_b_c_l_u(mdl: gp.Model, keep_sparse=False, force_binary=False):
    mdl.update()
    A = mdl.getA()
    if not keep_sparse:
        A = A.toarray()
    b = np.array(mdl.getAttr("RHS")).reshape(-1, 1)
    c = np.array(mdl.getAttr("Obj")).reshape(-1, 1)
    l = np.array(mdl.getAttr("LB")).reshape(-1, 1)
    u = np.array(mdl.getAttr("UB")).reshape(-1, 1)
    if force_binary:
        for v in mdl.getVars():
            if v.VType == gp.GRB.BINARY:
                l[v.index, 0] = 0
                u[v.index, 0] = 1
    return A, b, c, l, u


def substitute(mdl: gp.Model, M: np.ndarray, x0: np.ndarray, sense="<", env=None):
    mdl.update()
    # assert mdl.NumVars == mdl.NumIntVars, "Model must have only integer variables for substitution."
    mdl2 = gp.Model("substituted_" + mdl.ModelName, env=env)
    mdl2.params.LogToConsole = mdl.params.LogToConsole

    y = mdl2.addMVar(shape=(M.shape[1], 1), name="y", lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype="I")

    A, b, c, l, u = get_A_b_c_l_u(mdl, keep_sparse=True, force_binary=True)
    mdl2.setObjective(c.T @ (M @ y + x0)[0:c.shape[0], :] + mdl.ObjCon, mdl.ModelSense)
    if sense == "<":
        mdl2.addConstr(A @ M @ y <= b - A @ x0, name="txA")
    elif sense == ">":
        mdl2.addConstr(A @ M @ y >= b - A @ x0, name="txA")
    elif sense == "=":
        mdl2.addConstr(A @ M @ y == b - A @ x0, name="txA")
    elif sense == "skip":
        pass
    else:
        raise ValueError(f"Invalid sense '{sense}' for substitution. Use '<', '>', or '='.")
    if l.shape[0] < x0.shape[0]:
        l = np.vstack([l, np.zeros((x0.shape[0] - l.shape[0], 1))])  # the slacks are >= 0
    mdl2.addConstr(M @ y + x0 >= l, name="txl")
    mdl2.addConstr((M @ y + x0)[0:u.shape[0], :] <= u, name="txu")
    mdl2.update()

    return mdl2


def relaxed_optimum(model: gp.Model):
    """
    Returns the optimal solution of the relaxed model.
    Assumes the model is a knapsack model with all variables >= 0.
    """
    # relaxed = model.copy()
    # relax_int_or_bin_to_continuous(relaxed)
    relaxed = model.relax()
    relaxed.params.LogToConsole = 0
    relaxed.optimize()
    assert relaxed.status == gp.GRB.Status.OPTIMAL
    return np.array(relaxed.getAttr("X")).reshape((-1, 1)), relaxed.ObjVal


def is_integer_constraint(constraint: gp.Constr, relaxed: gp.Model, int_var_set, tol=1e-6, forceRHS_integer=True):
    """
    Check if a constraint qualifies as an "integer constraint" for GMI cut generation.
    
    A constraint is considered integer if:
    1. All coefficients are integer (or very close to integer)
    2. All variables with non-zero coefficients are in the integer variable set
    
    Note: We do NOT require the RHS to be integer. If all coefficients and variables
    are integer but the RHS is fractional, the constraint can still generate valid
    GMI cuts. In practice, such constraints could be strengthened by rounding the RHS,
    but that's not required for GMI cut generation.
    """
    lhs = relaxed.getRow(constraint)
    
    # Check all coefficients and variables
    for i in range(lhs.size()):
        # Check if coefficient is integer
        if abs(lhs.getCoeff(i) - round(lhs.getCoeff(i))) > tol:
            return False  # must have integer coefficients
        
        # Check if variable is in the integer variable set
        if lhs.getVar(i).index not in int_var_set:
            return False  # all variables must be integer
        
    if forceRHS_integer and abs(constraint.RHS - round(constraint.RHS)) >= tol:
        print("THIS IS NOT EXPECTED: Integer constraint with fractional RHS:", constraint, "RHS =", constraint.RHS)
        return False
    
    return True


def cut_efficacy(cut: gp.LinExpr, x: np.ndarray, b=1.0):
    violation = 0.0
    norm_squared = 0.0

    for i in range(cut.size()):
        coeff = cut.getCoeff(i)
        violation += coeff * x[cut.getVar(i).index, 0]
        norm_squared += coeff * coeff

    violation = max(0, b - violation)  # Convert to violation, assumes ax >= b form
    if norm_squared == 0:
        return 0.0

    return violation / (norm_squared**0.5)  # efficacy

def shift_to_x_gt_0(basis, tableau, col_to_var_idx, variables, constraints, x, relaxed):
    num_vars = len(variables)
    betas = []
    for row_idx, row in enumerate(tableau):
        basis_var_idx = basis[row_idx]
        assert basis_var_idx >= 0
        
        # Start with the original RHS (basic variable value)
        if basis_var_idx < num_vars:
            beta = x[basis_var_idx, 0]
        else:
            constraint_idx = basis_var_idx - num_vars
            con = constraints[constraint_idx]
            # For Gurobi, we need to compute the slack value properly
            # Gurobi's .Slack property returns the feasibility slack (always >= 0)
            # But for GMI cuts, we need the actual slack in tableau form
            con_row = relaxed.getRow(con)
            activity = sum(con_row.getCoeff(j) * con_row.getVar(j).X for j in range(con_row.size()))
            
            if con.Sense == '<':
                # slack = RHS - activity
                beta = con.RHS - activity
            elif con.Sense == '>':
                # slack = activity - RHS
                beta = activity - con.RHS
            else:  # con.Sense == '='
                # For equality constraints, slack should be 0 (but we handle this case)
                beta = 0.0
        betas.append(beta)

    for col_idx in range(tableau.shape[1]):
        var_idx = col_to_var_idx[col_idx].item()
        assert var_idx >= 0
        # For regular variables: check if at upper bound using VBasis
        if var_idx < num_vars:
            vrb = variables[var_idx]
            if vrb.VBasis == -2:  # At upper bound
                # Variable y is at upper bound u, so substitute y' = u - y
                # Coefficient of y' becomes -a (negated)
                tableau[:, col_idx] = -tableau[:, col_idx]
        
        # For slack/surplus variables: adjust for >= constraints
        else:
            # This is a >= constraint with surplus
            constraint_idx = var_idx - num_vars
            con = constraints[constraint_idx]
            if con.Sense == '>':
                # Constraint is >=, surplus is non-basic at 0
                # Negate the coefficient to match GMI convention
                tableau[:, col_idx] = -tableau[:, col_idx]

    return np.array(betas).reshape((-1, 1)), tableau


def make_gmi_cuts(basis, tableau, col_to_var_idx, x,
    int_var_set, variables, constraints, relaxed: gp.Model, W=None,
    tol: float = 1e-6):
    """
    Generate Gomory Mixed Integer (GMI) cuts from the tableau of a Gurobi model.
    
    This function matches the behavior of make_gmi_cuts_highs in highs_utils.py.
    
    Args:
        basis: List of basic variable indices for each row
        tableau: The simplex tableau (after removing basic columns)
        col_to_var_idx: Mapping from tableau columns to variable/slack indices
        x: Current solution vector
        int_var_set: Set of integer variable indices
        variables: List of Gurobi variables
        constraints: List of Gurobi constraints
        relaxed: The Gurobi LP relaxation model
        tol: Tolerance for numerical comparisons
        negated_rows: List of row indices that were negated (ignored, handled internally)
        
    Yields:
        Gurobi constraint objects representing cuts of the form: ax >= b
    """
    num_vars = len(variables)
    frac = lambda a: a - np.floor(a)
    betas, tableau = shift_to_x_gt_0(basis, tableau, col_to_var_idx, variables, constraints, x, relaxed)

    tab2 = []
    if W is not None:
        betas = betas[[ri for ri, b in enumerate(basis) if b < num_vars], 0] # drop the slack rows from betas
        tableau = tableau[[ri for ri, b in enumerate(basis) if b < num_vars], :]  # drop slack rows from tableau
        for row_idx, row in enumerate(W):
            beta = (row @ betas).item()
            if abs(frac(beta)) >= tol:
                tab2.append((beta, row @ tableau))  # assumes basis columns dropped from W already
    else:
        for row_idx, row in enumerate(tableau):
            basis_var_idx = basis[row_idx]
            
            # Skip if basic variable is continuous
            if basis_var_idx < num_vars and basis_var_idx not in int_var_set:
                continue
            
            # Skip if basic slack corresponds to non-integer constraint
            if basis_var_idx >= num_vars:
                constraint_idx = basis_var_idx - num_vars
                if not is_integer_constraint(constraints[constraint_idx], relaxed, int_var_set, tol):
                    continue
            tab2.append((betas[row_idx, 0].item(), row))


    for row_idx, (beta, row) in enumerate(tab2):

        # Now compute f0 from the TRANSFORMED beta
        f0 = frac(beta)
        
        # Skip if nearly integer
        if f0 < tol or f0 > 1 - tol:
            continue

        # Build the cut expression using a dictionary to accumulate coefficients
        cut_dict = {}
        cut_const = 0.0  # Constant term adjustments
        
        for col_idx, aij in enumerate(row):
            var_idx = col_to_var_idx[col_idx].item()
            
            # CRITICAL: Transform tableau coefficient for non-basic variables at bounds
            # The GMI formula assumes all non-basic variables are at their LOWER bound (≥ 0).
            # We need to transform the tableau BEFORE computing GMI coefficients.
            
            # Now compute GMI coefficient from the TRANSFORMED tableau coefficient
            # Standard GMI with RHS = f0 * (1 - f0)
            if var_idx in int_var_set:
                fj = frac(aij)
                if fj <= f0:
                    fj = (1 - f0) * fj
                else:
                    fj = f0 * (1 - fj)
            else:  # continuous variable
                if aij >= 0:
                    fj = (1 - f0) * aij
                else:
                    fj = -f0 * aij

            if abs(fj) < tol or abs(fj - 1) < tol:
                continue

            if var_idx < num_vars:
                vrb = variables[var_idx]
                coeff = fj
                if vrb.VBasis == -1:  # At lower bound
                    # Account for variables that sit at a non-zero lower bound.
                    # The tableau coefficient is built around x_j = lb_j, but the GMI
                    # derivation assumes the shifted variable y_j = x_j - lb_j.
                    # Adjust the accumulated constant so the final cut is written in
                    # terms of the original x_j without cutting off the true optimum.
                    if vrb.LB > -gp.GRB.INFINITY and abs(vrb.LB) > tol:
                        cut_const -= fj * vrb.LB
                elif vrb.VBasis == -2:  # At upper bound
                    # Symmetric adjustment for variables at a finite upper bound when using
                    # the substitution y_j = u_j - x_j. Converting back to x_j introduces
                    # a sign flip on the coefficient while contributing a constant term.
                    if vrb.UB < gp.GRB.INFINITY and abs(vrb.UB) > tol:
                        cut_const += fj * vrb.UB
                    coeff = -fj
                elif vrb.VBasis == -3:
                    print("Warning: Variable", vrb.VarName, "has VBasis -3 (super basic); GMI cut may be invalid.")

                # For variables: add the (possibly sign-adjusted) coefficient.
                cut_dict[var_idx] = cut_dict.get(var_idx, 0.0) + coeff
            else:
                # Slack variable - check if equality or inequality
                constraint_idx = var_idx - num_vars
                con = constraints[constraint_idx]
                
                # Gurobi doesn't have ranged constraints (unlike HiGHS)
                # Sense is either '<', '>', or '='
                if con.Sense == '=':
                    # Equality constraint: skip the slack term entirely
                    # Equality constraints have no meaningful slack - they're always tight.
                    # Including the "slack" (which is just the activity) creates weak cuts.
                    continue
                else:
                    # Inequality constraint: expand implicit slack
                    a_i = relaxed.getRow(con)
                    assert a_i.getConstant() == 0.0
                    
                    # Determine orientation using constraint sense
                    if con.Sense == '<':
                        # slack = RHS - Ax
                        cut_const += fj * con.RHS
                        for j in range(a_i.size()):
                            idx = a_i.getVar(j).index
                            coeff = a_i.getCoeff(j)
                            cut_dict[idx] = cut_dict.get(idx, 0.0) - (fj * coeff)
                    elif con.Sense == '>':
                        # slack = Ax - RHS
                        cut_const -= fj * con.RHS
                        for j in range(a_i.size()):
                            idx = a_i.getVar(j).index
                            coeff = a_i.getCoeff(j)
                            cut_dict[idx] = cut_dict.get(idx, 0.0) + (fj * coeff)
        
        # Compute RHS: f0 * (1 - f0) - cut_const
        # The cut is: sum(cut_dict[i] * x[i]) >= f0*(1-f0) - cut_const
        cut_rhs = f0 * (1 - f0) - cut_const
        
        # Convert dictionary to Gurobi expression
        cut_expr = gp.LinExpr()
        all_integer = True
        for idx in cut_dict:
            if abs(cut_dict[idx]) >= tol:
                if idx not in int_var_set or abs(cut_dict[idx] - round(cut_dict[idx])) >= tol:
                    all_integer = False
                cut_expr.add(variables[idx], cut_dict[idx])

        if all_integer and cut_expr.size() > 0 and abs(cut_rhs - round(cut_rhs)) >= tol:
            # Round RHS for all-integer cuts
            old = cut_rhs
            cut_rhs = np.ceil(cut_rhs - tol)
            print("   Rounded GMI cut RHS from", old, "to", cut_rhs)
        if cut_expr.size() > 0:
            # Gurobi constraint: cut_expr >= cut_rhs
            yield (cut_expr >= cut_rhs)


def run_gmi_cuts(model: gp.Model, rounds=1, W=None, verbose=False, callback=None):
    int_var_idx = {v.index for v in model.getVars() if v.VType in (gp.GRB.INTEGER, gp.GRB.BINARY)}
    relaxed = model.relax()
    relaxed.params.Presolve = 0  # for reading the tableau
    relaxed.params.LogToConsole = 0
    relaxed.optimize()
    assert relaxed.status == gp.GRB.Status.OPTIMAL, "Relaxed model must solve to optimality before GMI cuts."
    if callback is not None:
        callback(relaxed)
    starting_obj = relaxed.ObjVal
    if verbose:
        print(f" GMI round 0 for {model.ModelName}, constraints {model.NumConstrs}, variables {model.NumVars}, integer variables {model.NumIntVars}, start: {starting_obj}")
    for r in range(rounds):
        # basis, tableau, col_to_var_idx, x = transform_to_original_variables(relaxed)
        basis = read_basis(relaxed)
        tableau, col_to_var_idx, negated_rows = read_tableau(relaxed, basis, remove_basis_cols=True)
        variables, constraints = relaxed.getVars(), relaxed.getConstrs()
        W_B = W[:, [b for b in basis if b < relaxed.NumVars]] if W is not None else None

        for nr in negated_rows:
            # print("  Negating row", nr, "in GMI tableau at base", basis[nr])
            tableau[nr, :] = -tableau[nr, :]

        x = np.array(relaxed.X).reshape((-1, 1))
        # if all x are integer, we are done:
        if np.allclose(x[list(int_var_idx), 0], np.round(x[list(int_var_idx), 0]), atol=relaxed.params.FeasibilityTol):
            if verbose:
                print("  All integer variables are integral; stopping GMI cut generation at round", r)
            break
        new_constraints = make_gmi_cuts(basis, tableau, col_to_var_idx, x,
            int_var_idx, variables, constraints, relaxed, W_B,
            tol=relaxed.params.FeasibilityTol
        )
        relaxed.addConstrs(c for c in new_constraints)
        relaxed.optimize()
        if relaxed.status != gp.GRB.Status.OPTIMAL:
            print("  GMI cut generation stopped early due to non-optimal relaxation. Status:", status_lookup.get(relaxed.status, relaxed.status))
            return 0, 0
        if callback is not None:
            callback(relaxed)
        if verbose:
            print(f"  GMI round {r + 1}, obj {relaxed.ObjVal}, constraints {relaxed.NumConstrs}")

    return starting_obj, relaxed


def transform_via_LLL(model: gp.Model, check_gcd=False, verify=True, env=None, reduce_ns=False):
    A = model.getA().toarray()
    b = np.array(model.getAttr("RHS")).reshape(-1, 1)

    m, n = A.shape
    A = A.astype(np.int64)
    b = b.astype(np.int64)
    if verify:
        senses = [con.Sense for con in model.getConstrs()]
        assert all(s == "=" for s in senses), "All constraints must be equalities for LLL solving."

    if check_gcd:
        for i in range(m):
            # find GCD of the row
            gcd = np.gcd.reduce(A[i, :], axis=1).item()
            if gcd > 1:
                print(f"Row GCD: {gcd}")
                if b[i, 0].item() % gcd != 0:
                    raise ValueError("No integer solution exists (b not divisible by GCD)")
                # divide the row by the GCD
                A[i, :] //= gcd
                b[i, 0] //= gcd

    x_p, null_space = nullspace_and_offset_via_LLL(A, b, verify)
    if reduce_ns:
        import ntl_wrapper as ntl
        ntl.lll(null_space, 99, 100)

    mdl2 = substitute(model, null_space, x_p, 'skip', env=env)
    return mdl2

def nullspace_and_offset_via_LLL(A, b, N1 = 0, N2 = 0, verify=False):
    m, n = A.shape
    if N1 == 0:
        N1 = max(np.linalg.norm(b, np.inf).item(), np.linalg.norm(A, np.inf).item()) * 6
    if N2 == 0:
        N2 = N1 * 6
    B = np.block([[np.eye(n, dtype=np.int64), np.zeros((n, 1), dtype=np.int64)],
                        [np.zeros((1, n), dtype=np.int64), np.array([N1])],
                        [N2 * A, -N2 * b]]).astype(np.int64, order='C')
    # B = sp.block_array([[sp.eye(n), sp.csr_array((n, 1))],
    #                     [sp.csr_array((1, n)), N1],
    #                     [N2 * A, -N2 * b]])
    B_red = B.copy()
    import ntl_wrapper as ntl
    rank, det, U = ntl.lll(B_red, 99, 100)
    # x_p_idx = n-m
    x_p_idx = np.argmax(B_red[n] == N1)
    x_p = B_red[0:n, x_p_idx].reshape((-1, 1))
    assert B_red[n, x_p_idx].item() == N1, "---LLL did not preserve N1; something went wrong!"

    if verify:
        assert np.allclose(A @ x_p, b)
    null_space = B_red[0:n, 0:n-m]
    return x_p, null_space
