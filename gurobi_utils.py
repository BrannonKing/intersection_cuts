import ctypes as ct
import gurobipy as gp
import numpy as np
import platform
import pathlib
_libs = pathlib.Path(gp.__file__).parent.rglob('*.dll' if platform.system() == 'Windows' else '*.so')
# our DLL is likely the largest library there; we can make this more robust when needed
_likely_gurobi_dll = max(_libs, key=lambda fn: fn.stat().st_size)
_gurobi_dll = ct.CDLL(str(_likely_gurobi_dll))


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
    assert m.Status == gp.GRB.OPTIMAL, "We can only pull this data from models solved to optimum."
    data = (ct.c_int * m.NumConstrs)()
    ptr = ct.pythonapi.PyCapsule_GetPointer(m._cmodel, None)
    assert ptr != 0
    err = _gurobi_dll.GRBgetBasisHead(ptr, data)
    assert err == 0
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

    col_to_var = np.arange(tableau.shape[1])
    negated_rows = [i for i, base in enumerate(basis) if tableau[i, base] < -0.5]
    if remove_basis_cols:
        # any basis that is negative needs to negate that row:
        col_to_var = np.delete(col_to_var, basis)  # TODO: mask may be better than delete calls here
        tableau = np.delete(tableau, basis, 1)  # remove any columns in the basis
    assert col_to_var.shape[0] == tableau.shape[1]
    return tableau, col_to_var, negated_rows


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


def standardize_ub_to_constr(m: gp.Model):
    m.update()
    cnt = 0
    for var in m.getVars():
        if var.UB < gp.GRB.INFINITY:
            m.addConstr(-var >= -var.UB)
            var.UB = gp.GRB.INFINITY
    print(f"   Standardized {cnt} upper bounds to be constraints")


def standardize_lb_to_constr(m: gp.Model):
    m.update()
    cnt = 0
    for var in m.getVars():
        if var.LB > -gp.GRB.INFINITY and var.LB != 0.0:
            m.addConstr(var >= var.UB)
            var.LB = -gp.GRB.INFINITY
            cnt += 1
    print(f"   Standardized {cnt} lower bounds to be constraints")


def relax_int_or_bin_to_continuous(m: gp.Model):
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
    tableau = np.append(tableau, -np.ones((1, tableau.shape[1])), axis=0)
    lengths = np.linalg.norm(tableau, 2, axis=0)
    basis = basis + [-1]
    constraints = [constraint for constraint in model.getConstrs() if constraint.Sense == '>']
    for i, ray in enumerate(tableau.T):
        if progress is not None:
            next(progress)
        point_shifted = point.copy()
        basis[-1] = col_to_var[i]
        point_shifted[basis] += ray * 0.01 / lengths[i]
        # TODO: optimize this so it runs faster
        for constraint in constraints:
            new_lhs = A[constraint.index, :] @ point_shifted[0:A.shape[1]] - point_shifted[A.shape[1] + constraint.index]
            if new_lhs.item() < constraint.RHS - model.params.FeasibilityTol:
                print("   Failed validation!", i, constraint, model.getRow(constraint), point_shifted, '>=', constraint.RHS)
                failures += 1
    return failures
