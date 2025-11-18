import jsplib_loader as jl
import knapsack_loader as kl
import gurobipy as gp
import gurobi_utils as gu
import numpy as np

env = gp.Env()
env.setParam('OutputFlag', 0)
env.start()

def cutter(row_idx: int, tableau: np.ndarray) -> np.ndarray:
    """Create a cutting matrix U for the given row index."""
    m, n = tableau.shape
    U = np.eye(m, dtype=np.int32)
    
    # norm 1 for each column of tableau:
    d = np.linalg.norm(tableau, ord=1, axis=0)

    model = gp.Model(env=env)
    model.params.DualReductions = 0
    model.params.SolutionLimit = 1  # Stop after finding first feasible solution
    x = model.addMVar(shape=(1, m), lb=-100, ub=100, name="x")
    model.setObjective(x.sum(), gp.GRB.MINIMIZE)
    # model.setObjective(x.sum(), gp.GRB.MAXIMIZE)
    model.addConstr(x @ tableau >= d * 0.5, name="nonnegativity")
    # model.addConstr(x @ tableau <= -d * 0.5, name="nonnegativity")
    model.addConstr(x[0, row_idx] == 1, name="pivot_one")
    model.optimize()
    assert model.Status == gp.GRB.OPTIMAL, "Cutting plane LP not optimal: " + gu.status_lookup[model.Status]
    U[row_idx, :] = np.round(x.X)

    return U

# for model in kl.generate(10, 4, 30, 5, 10, 1000, equality=True):
#     name = model.ModelName
instances = jl.get_instances()
for name, instance in instances.items():
    if instance.jobs < 6:
        continue
    model = instance.as_gurobi_balas_model(use_big_m=True, env=env)
    int_var_indices = set(v.index for v in model.getVars() if v.VType != gp.GRB.CONTINUOUS)
    relaxed = model.relax()
    relaxed.params.Presolve = 0
    relaxed.params.LogToConsole = 0
    relaxed.optimize()
    vals = [v.X for v in relaxed.getVars()]
    basis = gu.read_basis(relaxed)
    tableau, col_to_var_idx, negated_rows = gu.read_tableau(relaxed, basis, remove_basis_cols=True)

    print(f"Instance: {name} - {model.ModelName}")

    # cutable rows are those whose basic variable is integer and they have a non-integer solution value:
    cutable = [r for r in range(tableau.shape[0])
               if basis[r] in int_var_indices and
               abs(vals[basis[r]] - round(vals[basis[r]])) > 1e-5]

    slacks = [c.Slack for c in relaxed.getConstrs()]
    b = np.array([(vals[basis[r]] if basis[r] < len(vals) else slacks[basis[r] - len(vals)]) for r in cutable])
    b = b.reshape((-1, 1))
    for c in cutable:
        U = cutter(c, tableau)
        uTab = U @ tableau
        # count the number of rows that are either all nonnegative or all nonpositive:

        zeros = 0
        negs = 0
        poss = 0
        suc = 0
        for r, row in enumerate(uTab):
            for coeff in row:
                if abs(coeff) < 1e-6:
                    zeros += 1
                elif coeff < 0:
                    negs += 1
                else:
                    poss += 1
            if negs == 0 or poss == 0:
                suc += 1

        if suc > 0:
            print("  Successful rows:", suc, "/", tableau.shape[0], "for cutable row", c)
