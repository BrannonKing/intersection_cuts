import gurobipy as gp
# gp.setParam('OutputFlag', 0)  # suppress Gurobi output for this experiment
import gurobi_utils as gu
import numpy as np
import lll_utils as lu

def convert_model_to_eq(model: gp.Model):
    """
    Convert the given Gurobi model to an equivalent one with only equality constraints.
    This is done by introducing slack variables for each inequality constraint.
    """
    new_model = gp.Model()
    for var in model.getVars():
        new_var = new_model.addVar(vtype=var.VType, name=var.VarName, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY)
        if var.LB > -gp.GRB.INFINITY:
            s = new_model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"slack_{var.VarName}")
            new_model.addConstr(new_var - s == var.LB)
        if var.UB < gp.GRB.INFINITY:
            s = new_model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"slack_{var.VarName}")
            new_model.addConstr(new_var + s == var.UB)
    new_model.update()
    variables = new_model.getVars()

    for con in model.getConstrs():
        if con.Sense == gp.GRB.LESS_EQUAL:
            slack = new_model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"slack_{con.ConstrName}")
            expr = gp.LinExpr(slack)
        elif con.Sense == gp.GRB.GREATER_EQUAL:
            slack = new_model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"slack_{con.ConstrName}")
            expr = gp.LinExpr(-slack)
        else:  # equality constraint
            expr = gp.LinExpr()
        row = model.getRow(con)
        for i in range(row.size()):
            var_idx = row.getVar(i).index
            expr.add(variables[var_idx], row.getCoeff(i))
        new_model.addConstr(expr == con.RHS)

    c = model.getObjective()
    assert isinstance(c, gp.LinExpr), "Only linear objectives are supported"
    expr = gp.LinExpr(c.getConstant())
    for i in range(c.size()):
        var_idx = c.getVar(i).index
        expr.add(variables[var_idx], c.getCoeff(i))
    new_model.setObjective(expr, model.ModelSense)
    new_model.update()
    return new_model

def is_integer(x):
    return np.isinf(x) or abs(x - round(x)) < 1e-9

def copy_int_objective_only(orig_model: gp.Model, x0: np.ndarray, N: np.ndarray):
    new_model = gp.Model()
    old_var_to_new_var = {}
    idx = 0
    c = []
    for var in orig_model.getVars():
        if var.VType in (gp.GRB.INTEGER, gp.GRB.BINARY):
            lb = 0 if var.VType == gp.GRB.BINARY else var.LB
            ub = 1 if var.VType == gp.GRB.BINARY else var.UB
            assert is_integer(lb) and is_integer(ub), "Non-integer bounds on integer variable"
            new_model.addVar(vtype=gp.GRB.CONTINUOUS, name=var.VarName, lb=lb, ub=ub)
            old_var_to_new_var[var.index] = idx
            c.append(var.Obj)
            idx += 1
    new_model.update()
    variables = new_model.getVars()
    assert len(variables) == N.shape[1] and len(c) == N.shape[1], "Nums:" + str((len(variables), N.shape[1], len(c)))
    x = gp.MVar.fromlist(variables)
    c = np.array(c).reshape((-1, 1))

    new_model.setObjective(c.T @ N @ x + c.T @ x0, orig_model.ModelSense)
    new_model.update()
    return new_model

def copy_no_int(model: gp.Model, rhs_sub):
    new_model = gp.Model()
    old_var_to_new_var = {}
    idx = 0
    for var in model.getVars():
        if var.VType not in (gp.GRB.INTEGER, gp.GRB.BINARY):
            new_model.addVar(vtype=var.VType, name=var.VarName, lb=var.LB, ub=var.UB)
            old_var_to_new_var[var.index] = idx
            idx += 1
    new_model.update()
    variables = new_model.getVars()
    assert len(variables) > 0

    for con in model.getConstrs():
        assert con.Sense == gp.GRB.EQUAL, "All constraints should be equalities"
        expr = gp.LinExpr()
        row = model.getRow(con)
        for i in range(row.size()):
            var_idx = old_var_to_new_var.get(row.getVar(i).index, -1)
            if var_idx >= 0:
                expr.add(variables[var_idx], row.getCoeff(i))
        for sub in rhs_sub:
            new_model.addConstr(expr == con.RHS - sub, name=con.ConstrName)

    c = model.getObjective()
    assert isinstance(c, gp.LinExpr), "Only linear objectives are supported"
    expr = gp.LinExpr(c.getConstant())  # don't add the constant twice (here instead of in the other obj-only func)
    for i in range(c.size()):
        var_idx = old_var_to_new_var.get(c.getVar(i).index, -1)
        if var_idx >= 0:
            expr.add(variables[var_idx], c.getCoeff(i))
    new_model.setObjective(expr, model.ModelSense)
    new_model.update()
    return new_model

def solve_model(orig_model: gp.Model):
    eq_model = convert_model_to_eq(orig_model)
    A = eq_model.getA()
    int_cols = [i for i, v in enumerate(eq_model.getVars()) if v.VType in (gp.GRB.INTEGER, gp.GRB.BINARY)]
    intA = A[:, int_cols].toarray()
    b = np.array(eq_model.getAttr("RHS")).reshape((-1, 1))

    # remove rows of A that are all zero (and corresponding entries of b)
    nonzero_rows = np.any(intA != 0, axis=1)
    intA = intA[nonzero_rows]
    b = b[nonzero_rows]
    m, n = intA.shape
    chunk_size = n * 2 // 3

    chunks = m // chunk_size + 1
    chunk_size = m // chunks + 1

    rhs_sub = []
    for chunk in range(chunks):
        start_row = chunk * chunk_size
        x0, N = gu.nullspace_and_offset_via_LLL(intA[start_row:start_row + chunk_size], b[start_row:start_row + chunk_size], False)
        rhs_sub.append(intA @ x0)

    objective_only_model = copy_int_objective_only(orig_model, x0, N)
    addition = 0.0
    if objective_only_model is not None:
        objective_only_model.optimize()
        assert objective_only_model.status == gp.GRB.OPTIMAL
        addition = objective_only_model.objVal

    lp_model = copy_no_int(eq_model, rhs_sub)
    lp_model.optimize()
    assert lp_model.status == gp.GRB.OPTIMAL
    return lp_model.objVal + addition

def main():
    import jsplib_loader as jl
    instances = jl.get_instances()
    instance = instances["abz4"]
    model = instance.as_gurobi_balas_model(use_big_m=True)
    obj_val = solve_model(model)
    print(f"Optimal objective value: {obj_val}")

if __name__ == "__main__":
    main()

    

