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
    negated_rows = [i for i, base in enumerate(basis) if tableau[i, base] < -0.5]
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
        _, _ = gu.relax_int_or_bin_to_continuous(relaxed)
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


def get_A_b_c_l_u(mdl: gp.Model, keep_sparse=False):
    mdl.update()
    A = mdl.getA()
    if not keep_sparse:
        A = A.toarray()
    b = np.array(mdl.getAttr("RHS")).reshape(-1, 1)
    c = np.array(mdl.getAttr("Obj")).reshape(-1, 1)
    l = np.array(mdl.getAttr("LB")).reshape(-1, 1)
    u = np.array(mdl.getAttr("UB")).reshape(-1, 1)
    return A, b, c, l, u


def substitute(mdl: gp.Model, M: np.ndarray, x0: np.ndarray, sense="<", env=None):
    mdl.update()
    # assert mdl.NumVars == mdl.NumIntVars, "Model must have only integer variables for substitution."
    mdl2 = gp.Model("substituted_" + mdl.ModelName, env=env)
    y = mdl2.addMVar(shape=(M.shape[1], 1), name="y", lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype="I")

    A, b, c, l, u = get_A_b_c_l_u(mdl, keep_sparse=True)
    mdl2.setObjective(c.T @ (M @ y + x0) + mdl.ObjCon, mdl.ModelSense)
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
    mdl2.addConstr(M @ y + x0 >= l, name="txl")
    mdl2.addConstr(M @ y + x0 <= u, name="txu")

    return mdl2


def relaxed_optimum(model: gp.Model):
    """
    Returns the optimal solution of the relaxed model.
    Assumes the model is a knapsack model with all variables >= 0.
    """
    relaxed = model.copy()
    relax_int_or_bin_to_continuous(relaxed)
    relaxed.params.LogToConsole = 0
    relaxed.optimize()
    if relaxed.status != gp.GRB.Status.OPTIMAL:
        return None
    return np.array(relaxed.getAttr("X")).reshape((-1, 1))


def is_integer_constraint(constraint: gp.Constr, relaxed: gp.Model, int_var_set, tol=1e-6):
    lhs = relaxed.getRow(constraint)
    for i in range(lhs.size()):
        # how should I handle very small coefficients?
        if abs(lhs.getCoeff(i) - round(lhs.getCoeff(i))) > tol:
            return False  # must have integer coefficients
        # if lhs.getVar(i).VType not in (gp.GRB.INTEGER, gp.GRB.BINARY):
        #     return False
        if lhs.getVar(i).index not in int_var_set:
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

def transform_to_original_variables(relaxed: gp.Model):
    variables = relaxed.getVars()
    constraints = relaxed.getConstrs()
    basis = read_basis(relaxed)
    barAn, col_to_var_idx, neg_rows = read_tableau(relaxed, basis, remove_basis_cols=True)
    barB = np.zeros((len(basis),))
    for i, b in enumerate(basis):
        if b < len(variables):
            barB[i] = variables[b].X
        else:
            con = constraints[b - len(variables)]
            barB[i] = con.Slack
    s = [1 if j >= len(variables) or variables[j].VBasis != -2 else -1 for j in col_to_var_idx]
    zBnd = np.zeros((len(s),))
    for i, idx in enumerate(col_to_var_idx):
        if idx < len(variables):
            if variables[idx].VBasis == -2:
                zBnd[i] = variables[idx].UB if variables[idx].UB < gp.GRB.INFINITY else 0 # variables[idx].X
            else:
                zBnd[i] = variables[idx].LB if variables[idx].LB > -gp.GRB.INFINITY else 0 # variables[idx].X

    d = np.zeros_like(barB)
    C = np.zeros_like(barAn)
    for i in range(len(basis)):
        d[i] = barB[i]
        for ji, j in enumerate(col_to_var_idx):
            d[i] += barAn[i, ji] * s[ji] * zBnd[ji]
            C[i, ji] = -barAn[i, ji] * s[ji]

    return basis, C, col_to_var_idx, d.reshape((-1, 1))

    # so we want to drop all the columns corresponding to slack variables.
    # this means that we need to expand the constraints into the original variables.
    # but not just the original variables -- just those that exist in the non-basic set.

    # but this doesn't work if all columns are slack variables.
    # that's because I need at least one column to know the direction of the edge vector.


    # # get all the indices of slack variables in the non-basic set
    # slack_cols = col_to_var_idx[col_to_var_idx >= len(variables)] - len(variables)
    # x_cols = col_to_var_idx[col_to_var_idx < len(variables)]
    
    # # now we're going to substitute out the slack variables to get it in terms of the original variables.
    # # Current state: x_B = d + C @ z, where z = [non-basic original vars; non-basic slacks]
    # M = np.zeros((len(basis), len(variables)))
    # for i in range(len(basis)):
    #     sub = 0.0
    #     for k in slack_cols:
    #         d[i] += C[i, k] * constraints[k].RHS
    #     for j in x_cols:
    #         sub = 0.0
    #         for k in slack_cols:
    #             con = constraints[k]
    #             row = relaxed.getRow(con)
    #             for idx in range(row.size()):
    #                 if row.getVar(idx).index == j:
    #                     sub += C[i, k] * row.getCoeff(idx)
    #                     break
    #         M[i, j] = C[i, j] - sub
                
    # return basis, M, x_cols, d.reshape((-1, 1))


def transform_to_original_variables_try2(relaxed: gp.Model):
    """Recover the standard-form relation between basic and non-basic ORIGINAL variables from a solved (relaxed) model.

    Returns (basis, M, x_col_to_var, d) such that for the current basis we have
        x_B = d - M x_N
    where:
        basis: list[int] indices (into [original vars | slacks]) of basic columns.
        M: matrix shape (len(basis), k) relating k kept non-basic original variables to basic vars.
        x_col_to_var: length-k array mapping columns of M back to original variable indices.
        d: constant vector shape (len(basis), 1) giving basic values when x_N = 0.

    The method:
        1. Reads the current basis and tableau (minus basic columns) via GRB internal API.
        2. Constructs diagonal S to account for bound substitutions (variables basic at upper bound are sign-flipped).
        3. Forms C and d (post shift) in terms of transformed z variables.
        4. Eliminates slack columns to express basics purely as affine function of original non-basic variables.

    Notes / assumptions:
        * Presolve must be disabled (model.Params.Presolve == 0) prior to optimization.
        * Numerical noise in the tableau may require moderate tolerances when validating.
        * Only columns for non-basic original variables retained in x_col_to_var; others were basic and removed.
    """
    variables = relaxed.getVars()
    constraints = relaxed.getConstrs()
    basis = read_basis(relaxed)
    barAn, col_to_var_idx, neg_rows = read_tableau(relaxed, basis, remove_basis_cols=True)
    barB = np.zeros((len(basis), 1))
    for i, b in enumerate(basis):
        if b < len(variables):
            barB[i, 0] = variables[b].X
        else:
            con = constraints[b - len(variables)]
            barB[i, 0] = con.Slack
    S = np.zeros((barAn.shape[1], barAn.shape[1]))
    zBnd = np.zeros((len(variables), 1))
    for i, idx in enumerate(col_to_var_idx):
        if idx < len(variables):
            if variables[idx].VBasis == -2:
                S[i, i] = -1
                zBnd[i, 0] = variables[idx].UB if variables[idx].UB < gp.GRB.INFINITY else variables[idx].X
            else:
                S[i, i] = 1
                zBnd[i, 0] = variables[idx].LB if variables[idx].LB > -gp.GRB.INFINITY else variables[idx].X
        else:
            S[i, i] = 1
            # ZBnd == 0 because it's always at lower bound for slacks

    d = barB - barAn @ (S @ zBnd)
    C = -barAn @ S  # not sure we want to negate this; we can leave that for the users of it
    return basis, C, col_to_var_idx, d

    # # now we're going to substitute out the slack variables to get it in terms of the original variables.
    # # Current state: x_B = d + C @ z, where z = [non-basic original vars; non-basic slacks]
    # Cx = C[:, col_to_var_idx < len(variables)]
    # Cs = C[:, col_to_var_idx >= len(variables)]
    # num_vars = relaxed.NumVars
    
    # if Cs.shape[1] > 0:
    #     # Need to substitute slacks: for constraint i (<=), s_i = b_i - A[i,:] @ x
    #     # But the tableau already expresses ALL variables (basic and non-basic originals) in terms of x_N!
    #     # So we can use the FULL tableau representation to substitute slacks.
        
    #     A = relaxed.getA()
    #     b = np.array(relaxed.getAttr("RHS")).reshape(-1, 1)
    #     slack_cols_mask = col_to_var_idx >= num_vars
    #     slack_rows = col_to_var_idx[slack_cols_mask] - num_vars
    #     x_var_indices = col_to_var_idx[col_to_var_idx < num_vars]
        
    #     # Key insight: ALL original variables can be expressed as linear functions of x_N
    #     # For non-basic originals: they ARE x_N
    #     # For basic originals: x_basic = d + C @ [x_N; s_N]
        
    #     # Build full variable reconstruction: x_full = x0 + M_full @ x_N
    #     # where x0 is the value when x_N = 0
    #     x0_full = np.zeros((num_vars, 1))
    #     M_full = np.zeros((num_vars, len(x_var_indices)))
        
    #     # Non-basic original variables: identity mapping
    #     for i, var_idx in enumerate(x_var_indices):
    #         M_full[var_idx, i] = 1.0
        
    #     # Basic original variables: use tableau rows
    #     for row_idx, basis_idx in enumerate(basis):
    #         if basis_idx < num_vars:  # it's a basic original variable
    #             x0_full[basis_idx, 0] = d[row_idx, 0]
    #             M_full[basis_idx, :] = Cx[row_idx, :]
        
    #     # Now substitute into slacks: s = b - A @ x_full = b - A @ (x0_full + M_full @ x_N)
    #     #                                                 = (b - A @ x0_full) - (A @ M_full) @ x_N
    #     As = A[slack_rows, :].toarray()
    #     bs = b[slack_rows, :]
        
    #     s_constant = bs - As @ x0_full  # (n_slacks, 1)
    #     s_coeff = -As @ M_full          # (n_slacks, n_nonbasic_vars)
        
    #     # Substitute back: x_B = d + Cx @ x_N + Cs @ (s_constant + s_coeff @ x_N)
    #     #                      = (d + Cs @ s_constant) + (Cx + Cs @ s_coeff) @ x_N
    #     d += Cs @ s_constant
    #     M = Cx + Cs @ s_coeff
    # else:
    #     M = Cx
    # # Map each column of M back to the corresponding original variable index
    # # These are exactly the original-variable columns present in the tableau after
    # # removing basis columns, i.e., the entries of col_to_var_idx that refer to vars.
    # x_col_to_var = col_to_var_idx[col_to_var_idx < num_vars]
    # return basis, M, x_col_to_var, d

def make_gmi_cuts(basis, tableau, col_to_var_idx, x, int_var_set, variables, constraints, relaxed: gp.Model, fix_signs=False, tol=1e-6):
    num_vars = len(variables)

    # cuts = []
    for row_idx, row in enumerate(tableau):
        basis_var_idx = basis[row_idx]
        # don't use variables[basis_var_idx].VType == gp.GRB.CONTINUOUS; it's from relaxed model.
        # second realization: we can know that a slack variable can be integer, and this is important.
        # all coefficients must be integer and the variables going with them must be integer.
        # third: we can multiply by f0(1-f0) to clear denominators.
        if basis_var_idx < num_vars and basis_var_idx not in int_var_set:
            continue
        if basis_var_idx >= num_vars and not is_integer_constraint(constraints[basis_var_idx - num_vars], relaxed, int_var_set, tol):
            continue
        f0 = x[basis_var_idx, 0]
        f0 -= np.floor(f0)
        if f0 < tol or f0 > 1 - tol:
            continue  # skip if it's close to an integer

        cut_expr = gp.LinExpr()
        for col_idx, aij in enumerate(row):
            # aij = -aij
            var_idx = col_to_var_idx[col_idx]
            fixing = False
            if fix_signs and aij < -tol:
                aij = -aij
                fixing = True
            fj = aij - np.floor(aij)

            if fj <= f0 and var_idx in int_var_set:
                fj *= 1 - f0
            elif fj > f0 and var_idx in int_var_set:
                fj = (1 - fj) * f0
            elif aij >= 0:
                fj = aij * (1 - f0)
            else:
                fj = -aij * f0

            # if abs(fj) < tol or abs(fj - 1) < tol:
            #     continue

            if var_idx < num_vars:
                cut_expr.add(variables[var_idx], fj)
            else:
                # We need to express this in terms of the original constraint
                # Since slack_i = b_i - a_i^T x, we have:
                con = constraints[var_idx - num_vars]
                a_i = relaxed.getRow(con)
                if con.Sense != ">":
                    for j in range(a_i.size()):
                        cut_expr.add(variables[a_i.getVar(j).index], -fj * a_i.getCoeff(j))
                    cut_expr.addConstant(fj * con.RHS)
                else:
                    for j in range(a_i.size()):
                        cut_expr.add(variables[a_i.getVar(j).index], fj * a_i.getCoeff(j))
                    cut_expr.addConstant(-fj * con.RHS)

        # cs = cut_efficacy(cut_expr, x, b=f0 * (1 - f0))
        if cut_expr.size() > 0:  # Only add non-empty cuts
            cut = -cut_expr <= -f0 * (1 - f0)
            # print("  Found cut", basis_var_idx, cs, cut)
            yield cut


def run_gmi_cuts(model: gp.Model, rounds=1, verbose=False):
    int_var_idx = {v.index for v in model.getVars() if v.VType in (gp.GRB.INTEGER, gp.GRB.BINARY)}
    relaxed = model.relax()
    relaxed.params.Presolve = 0  # for reading the tableau
    relaxed.params.LogToConsole = 0
    relaxed.optimize()
    starting_obj = relaxed.ObjVal
    if verbose:
        print(
            f" GMI round 0 for {model.ModelName}, constraints {model.NumConstrs}, variables {model.NumVars}, integer variables {model.NumIntVars}, start: {starting_obj}"
        )
    for r in range(rounds):
        basis, tableau, col_to_var_idx, x = transform_to_original_variables(relaxed)
        # if all x are integer, we are done:
        if np.allclose(x[list(int_var_idx), 0], np.round(x[list(int_var_idx), 0]), atol=relaxed.params.FeasibilityTol):
            if verbose:
                print("  All integer variables are integral; stopping GMI cut generation at round", r)
            break
        variables, constraints = relaxed.getVars(), relaxed.getConstrs()
        new_constraints = make_gmi_cuts(basis, -tableau, col_to_var_idx, x, int_var_idx, variables, constraints, relaxed, tol=relaxed.params.FeasibilityTol)
        relaxed.addConstrs(c for c in new_constraints)
        relaxed.optimize()
        if relaxed.status != gp.GRB.Status.OPTIMAL:
            print("  GMI cut generation stopped early due to non-optimal relaxation. Status:", status_lookup.get(relaxed.status, relaxed.status))
            return 0, 0, 0

        if verbose:
            print(f"  GMI round {r + 1}, obj {relaxed.ObjVal}, constraints {relaxed.NumConstrs}")

    return starting_obj, relaxed.ObjVal, relaxed.NumConstrs - model.NumConstrs
