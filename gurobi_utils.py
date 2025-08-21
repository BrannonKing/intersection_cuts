import ctypes as ct
import gurobipy as gp
import numpy as np
import platform
import pathlib
_libs = pathlib.Path(gp.__file__).parent.rglob('*.dll' if platform.system() == 'Windows' else '*.so')
# our DLL is likely the largest library there; we can make this more robust when needed
_likely_gurobi_dll = max(_libs, key=lambda fn: fn.stat().st_size)
_gurobi_dll = ct.CDLL(str(_likely_gurobi_dll))
status_lookup = {getattr(gp.GRB.Status, k): k for k in gp.GRB.Status.__dir__() if "A" <= k[0] <= "Z"}


class GRBsvec(ct.Structure):
    _fields_ = [("len", ct.c_int),
                ("ind", ct.POINTER(ct.c_int)),
                ("val", ct.POINTER(ct.c_double))]


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
        indexes = data.ind[:data.len]
        values = data.val[:data.len]
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
                tableau[:, col] = -tableau[:, col]  # might need to be UB - ... ?
            elif variables[j].VBasis == -1:  # not sure what to do with VBasis=-3
                if variables[j].LB != 0.0:
                    print("Warning: LB is nonzero for variable", variables[j].VarName, "LB", variables[j].LB, "UB", variables[j].UB)
        else:
            constraint = constraints[j - len(variables)]
            if constraint.Sense == '>':  # Achterberg said lt and lte are standard; should just need to flip gt
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
    flip = ('<', '>')
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
    flip = ('>', '<')
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
        if constraint.Sense == '=':
            lhs, rhs, name = m.getRow(constraint), constraint.RHS, constraint.ConstrName
            to_remove.append(constraint)
            m.addLConstr(-lhs, '>', -rhs, name + "_rev1")
            m.addLConstr(lhs, '>', rhs, name + "_rev2")
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
    constraints = [constraint for constraint in model.getConstrs() if constraint.Sense != '=']
    for i, ray in enumerate(tableau.T):
        if progress is not None:
            next(progress)
        point_shifted = point.copy()
        basis[-1] = col_to_var[i]
        point_shifted[basis] += ray * 0.1 / lengths[i]
        # TODO: optimize this so it runs faster
        for constraint in constraints:
            if constraint.Sense == '<':
                new_lhs = A[constraint.index, :] @ point_shifted[0:A.shape[1]] - point_shifted[A.shape[1] + constraint.index]
                if new_lhs.item() > constraint.RHS + model.params.FeasibilityTol:
                    print("   Failed validation!", i, constraint, model.getRow(constraint), point_shifted, '<=', constraint.RHS)
                    failures += 1
            elif constraint.Sense == '>':
                new_lhs = A[constraint.index, :] @ point_shifted[0:A.shape[1]] + point_shifted[A.shape[1] + constraint.index]
                if new_lhs.item() < constraint.RHS - model.params.FeasibilityTol:
                    print("   Failed validation!", i, constraint, model.getRow(constraint), point_shifted, '>=', constraint.RHS)
                    failures += 1
    return failures

import scipy.sparse as sp

def apply_transform(old_model: gp.Model, U: np.ndarray, x0: np.ndarray, basis=None, normalize_Ab=False, mult=1, ignore_bounds=False, env=None):
    """Apply the transformation U to the model."""
    old_model.update()
    # A, b, c, l, u = get_A_b_c_l_u(result) # for debug

    # going to shift it to 0, then apply the transformation, then shift it back (all in one operation):
    # going with this substitution: y=U_inv(x - x0) + x0 so x=U(y - x0) + x0
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
        U_inv = sp.block_diag([U_inv, eye], format='csr')
        U = sp.block_diag([U, eye], format='csr')
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
    y_vars = new_model.addMVar(num_vars, lb=0 if ignore_bounds else -gp.GRB.INFINITY, vtype=vtypes, name=f"y")
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
        if sense[i] == '<':
            new_model.addConstr(expr <= b[i] + b_deduction[i], name=f"lt_{i}")
        elif sense[i] == '>':
            new_model.addConstr(expr >= b[i] + b_deduction[i], name=f"gt_{i}")
        elif sense[i] == '=':
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
        if v.UB - v.LB < percent_of_diagonal * 2.0:
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
        if c.Sense == '<':
            c.RHS -= distance * np.linalg.norm(coeffs) / lhs.size()
        elif c.Sense == '>':
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
        if c.Sense == '<' and lhs_value > c.RHS - distance:
            c.RHS = max(c.RHS, lhs_value) + distance
        elif c.Sense == '>' and lhs_value < c.RHS + distance:
            c.RHS = min(c.RHS, lhs_value) - distance
        else:
            # assert np.isclose(lhs_value, c.RHS, atol=distance*0.5), "Constraint RHS does not match the left-hand side value."
            c.Sense = '>'
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
    return A, b, c, l , u

def substitute(mdl: gp.Model, M: np.ndarray, x0: np.ndarray, sense='<', env=None):
    mdl.update()
    # assert mdl.NumVars == mdl.NumIntVars, "Model must have only integer variables for substitution."
    mdl2 = gp.Model("substituted_" + mdl.ModelName, env=env)
    y = mdl2.addMVar(shape=(M.shape[1], 1), name="y", lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype='I')

    A, b, c, l, u = get_A_b_c_l_u(mdl, keep_sparse=True)
    mdl2.setObjective(c.T @ (M @ y + x0) + mdl.ObjCon, mdl.ModelSense)
    if sense == '<':
        mdl2.addConstr(A @ M @ y <= b - A @ x0, name="txA")
    elif sense == '>':
        mdl2.addConstr(A @ M @ y >= b - A @ x0, name="txA")
    elif sense == '=':
        mdl2.addConstr(A @ M @ y == b - A @ x0, name="txA")
    elif sense == 'skip':
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