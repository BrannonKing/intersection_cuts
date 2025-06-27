import dikin_utils as du
import numpy as np
import gurobipy as gp
import linetimer as lt
import hsnf
import ntl_wrapper as ntl
import knapsack_loader as kl
status_lookup = {getattr(gp.GRB.Status, k): k for k in gp.GRB.Status.__dir__() if "A" <= k[0] <= "Z"}

# Experiment 3: use LLL to convert to Ax = b to UAx=Ub. Works.

# Generating instances with 2 constraints and 15 variables
#  Average original time: 7306.12034840 ms
#  Average transformed time: 2752.63947960 ms

# Generating instances with 3 constraints and 15 variables
#  Average original time: 696268.19003120 ms
#  Average transformed time: 726938.11944260 ms

def solve_via_LLL(A: np.ndarray):
    Am = A.astype(np.int64, copy=True)

    rank, det, U = ntl.lll_left(Am, 9, 10)
    # assert np.allclose(U @ A, A_modified)
    return U

def main():
    np.random.seed(42)
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    averages = {}
    compare_original = True
    for con_count in [2, 3, 4]:
        for var_count in [15]:
            print(f"Generating instances with {con_count} constraints and {var_count} variables")
            runs = 5
            before_times = []
            after_times = []
            instances = kl.generate(runs * 10, con_count, var_count, 5, 10, 1000, equality=True, env=env)
            for model in instances:
                model.params.LogToConsole = 0
                # assumptions on the model: all equality constraints, fully linear objective & constraints, all vars >= 0, maximizing
                A = model.getA().todense()
                b = np.array(model.getAttr("RHS")).reshape((-1, 1))
                c = np.array(model.getAttr("Obj")).reshape((-1, 1))
                ub = np.array(model.getAttr("UB")).reshape((-1, 1))

                if compare_original:
                    with lt.CodeTimer("Original optimization time", silent=True) as c1:
                        model.optimize()
                    if model.status != gp.GRB.Status.OPTIMAL:
                        print("  Skipping as model not optimal: ", status_lookup[model.status])
                        continue
                    before_times.append(c1.took)
                    # print(f"Original objective value: {model.ObjVal}")

                with lt.CodeTimer("  LLL time", silent=True) as c2:
                    U = solve_via_LLL(A)
                if c2.took > 1000:
                    print(f"  LLL took too long: {c2.took:.2f} ms")

                mdl2 = gp.Model(model.ModelName + "_tfrm", env=env)
                z = mdl2.addMVar((model.NumIntVars, 1), lb=0, ub=ub, vtype='I', name='z')
                mdl2.setObjective(c.T @ z, gp.GRB.MAXIMIZE)
                mdl2.addConstr(U @ A @ z == U @ b)
                # mdl2.params.NumericFocus = 3
                mdl2.params.DualReductions = 0
                mdl2.params.LogToConsole = 1
                with lt.CodeTimer("   Transformed optimization time", silent=True) as c1:
                    mdl2.optimize()
                if mdl2.status != gp.GRB.Status.OPTIMAL:
                    print(f"  Skipping as tfm model not optimal: {status_lookup[mdl2.status]}")
                    continue
                after_times.append(c1.took)

                # print("Objective value: ", mdl2.ObjVal)
                if compare_original and not np.allclose(mdl2.ObjVal, model.ObjVal):
                    print(f"Objective values do not match: {mdl2.ObjVal} != {model.ObjVal}")
                if len(after_times) == runs:
                    break
            if compare_original:
                print(f" Average original time: {np.mean(before_times):.8f} ms")
                averages[(con_count, var_count)] = (np.mean(before_times), np.mean(after_times))
            if after_times:
                print(f" Average transformed time: {np.mean(after_times):.8f} ms")
            print()
    print("Averages:", averages)

if __name__ == "__main__":
    main()