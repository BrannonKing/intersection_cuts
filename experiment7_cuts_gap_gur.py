import dikin_utils as du
import io
import numpy as np
import gurobipy as gp
import gurobi_utils as gu
import linetimer as lt
import ntl_wrapper as ntl
import knapsack_loader as kl
import hsnf
import sympy as sp

# Experiment 7b: 
# Generate inequality knapsack instances.
# Measure the solve time in CPLEX.
# LLL(A|b; I|l; -I;u).
# Invert U and use that on objective only.
# Use sympy for c @ U.
# Compare the cuts.

def transform(model: gp.Model, A: np.ndarray, U: np.ndarray, env=None):
    assert model.NumVars == model.NumIntVars
    assert U.shape[0] == U.shape[1] and U.shape[1] == model.NumVars + 1

    c = sp.Matrix(model.getAttr("Obj"))
    Us = sp.Matrix(U[0:-1, :])
    cUs = c.T @ Us
    # get the gcd of the vector cUs -- gcd was always 1
    cUsf = np.array(cUs, dtype=np.int64).reshape((-1, 1))

    senses = np.array(model.getAttr("Sense"))
    assert np.all(senses == gp.GRB.LESS_EQUAL)

    model2 = gp.Model("Transformed " + model.ModelName, env=env)
    # U_inv = np.linalg.inv(U) // can't multiply inequality by a matrix unless it's monomial.
    # y = model2.addMVar((U.shape[0], 1), lb=U_inv @ l, ub=U_inv @ u, vtype='I', name='y')
    y = model2.addMVar((U.shape[0], 1), lb=-gp.GRB.INFINITY, vtype='I', name='y')
    model2.setObjective(cUsf.T @ y + model.ObjCon, model.ModelSense)
    model2.addConstr(A @ y <= 0)
    model2.addConstr(-1 == U[-1, :] @ y)  # generally this just fixes a single variable to -1
    model2.update()
    return model2

def main():
    np.random.seed(42)
    for con_count in [2]:
        for var_count in [20]:
            print(f"Generating instances with {con_count} constraints and {var_count} variables")
            runs = 5
            instances = kl.generate(runs, con_count, var_count, 5, 10, 1000, equality=False)
            before_gaps = []
            after_gaps = []
            for model in instances:
                print("Starting instance", model.ModelName)
                model.params.logToConsole = 0
                model.optimize()

                before, after, cuts = gu.run_gmi_cuts(model, rounds=2, verbose=True)
                print(f"  Cuts: {cuts}, Before: {before}, After: {after}, Opt: {model.ObjVal}")
                before_gaps.append(100 * (before - after) / (before - model.ObjVal))

                A, b, c, l, u = gu.get_A_b_c_l_u(model, False)
                block = np.block([
                    [A, b], 
                    [-np.eye(A.shape[1]), -l],
                    [np.eye(A.shape[1]), u]
                ]).astype(np.int64)

                # print("Block shape:", block.shape)
                # print("  Before max column norm:", np.linalg.norm(block, axis=0).max())
                with lt.CodeTimer("  LLL time", silent=True) as c2:
                    rank, det, U = ntl.lll(block, 9, 10)
                # print("  After max column norm:", np.linalg.norm(block, axis=0).max())
                # print(f"  LLL took: {c2.took:.2f} ms")

                mdl1 = transform(model, block, np.eye(U.shape[0], dtype=np.int64))
                _, after, cuts = gu.run_gmi_cuts(mdl1, rounds=1, verbose=False)
                print(f"  After TFM cuts: {cuts}, After: {after}")

                mdl2 = transform(model, block, U)
                # mdl2.optimize()
                # assert round(mdl2.ObjVal) == round(model.ObjVal)
                _, after, cuts = gu.run_gmi_cuts(mdl2, rounds=1, verbose=False)
                print(f"  After LLL cuts: {cuts}, After: {after}")
                after_gaps.append(100 * (before - after) / (before - model.ObjVal))

            print(f" Average gap closed by GMI cuts before LLL: {np.mean(before_gaps):.3f}%")
            print(f" Average gap closed by GMI cuts after LLL:  {np.mean(after_gaps):.3f}%")
            print()

if __name__ == "__main__":
    main()