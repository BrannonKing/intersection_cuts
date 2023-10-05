import ctypes as ct
import gurobipy as gp
import numpy as np
import os
from importlib import resources
_possible_files = os.listdir(str(resources.files(gp) / '.libs'))
# our DLL is likely the largest library there; we can make this more robust when needed
_shared_lib = max(_possible_files, key=lambda fn: os.stat(resources.files(gp) / '.libs' / fn).st_size)
_gurobi_dll = ct.CDLL(str(resources.files(gp) / '.libs' / _shared_lib))


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
    cur = 0
    for row in range(rows):
        # TODO: pass in a list of variables to skip so we don't read unnecessary rows
        err = _gurobi_dll.GRBBinvRowi(ptr, row, data)
        assert err == 0
        indexes = data.ind[:data.len]
        values = data.val[:data.len]
        tableau[cur, indexes] = values
        cur += 1

    col_to_var = np.arange(tableau.shape[1])
    if remove_basis_cols:
        col_to_var = np.delete(col_to_var, basis)
        tableau = np.delete(tableau, basis, 1)  # remove any columns in the basis
    assert col_to_var.shape[0] == tableau.shape[1]
    return tableau, col_to_var


def standardize_lt_to_gt(m: gp.Model):
    m.update()
    flip = ('<', '>')
    cnt = 0
    to_remove = []
    for constraint in m.getConstrs():  # returns only linear constraints
        if constraint.Sense == flip[0]:
            lhs, sense, rhs, name = m.getRow(constraint), constraint.Sense, constraint.RHS, constraint.ConstrName
            to_remove.append(constraint)
            m.addLConstr(-lhs, flip[1], -rhs, name + "_rev")
            cnt += 1
    for tr in to_remove:
        m.remove(tr)
    print(f"   Negated {cnt} constraints on", m.ModelName)


def relax_int_or_bin_to_continuous(m: gp.Model):
    relaxed_variables = []
    relaxed_index = {}
    for i, var in enumerate(m.getVars()):
        if var.VType != gp.GRB.CONTINUOUS:
            if var.VType == gp.GRB.BINARY:
                var.UB = 1
                assert var.LB == 0
            var.VType = gp.GRB.CONTINUOUS
            relaxed_index[i] = len(relaxed_variables)
            relaxed_variables.append(var)
    print(f"   Relaxed {len(relaxed_variables)} variables on", m.ModelName)
    return gp.MVar.fromlist(relaxed_variables), relaxed_index


def nearest_integer(variables: gp.MVar):
    x = variables.X
    x = np.round(x)
    ub = variables.UB
    lb = variables.LB
    return np.clip(x, lb, ub)
