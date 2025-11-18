import jsplib_loader as jl
import knapsack_loader as kl
import gurobipy as gp
import gurobi_utils as gu

env = gp.Env()
env.setParam('OutputFlag', 0)
env.start()

# for model in kl.generate(10, 4, 30, 5, 10, 1000, equality=True):
#     name = model.ModelName
instances = jl.get_instances()
for name, instance in instances.items():
    model = instance.as_gurobi_balas_model(use_big_m=True, env=env)
    int_var_indices = set(v.index for v in model.getVars() if v.VType != gp.GRB.CONTINUOUS)
    relaxed = model.relax()
    relaxed.params.Presolve = 0
    relaxed.params.LogToConsole = 0
    relaxed.optimize()
    basis = gu.read_basis(relaxed)
    tableau, col_to_var_idx, negated_rows = gu.read_tableau(relaxed, basis, remove_basis_cols=True)

    print(f"Instance: {name} - {model.ModelName}")
    zeros = 0
    negs = 0
    poss = 0
    suc = 0
    per = 0
    for r, row in enumerate(tableau):
        if basis[r] not in int_var_indices:
            continue
        for coeff in row:
            if abs(coeff) < 1e-6:
                zeros += 1
            elif coeff < 0:
                negs += 1
            else:
                poss += 1
        # print(f"  Row: {r}, Pos: {poss}, Neg: {negs}, Zeros: {zeros}")
        if min(negs, poss) <= tableau.shape[0]:
            suc += 1
        if negs == 0 or poss == 0:
            per += 1

    if suc > 0:
        print("  Successful rows:", suc, "/", tableau.shape[0])
    if per > 0:
        print("  Perfect rows:", per, "/", tableau.shape[0])
