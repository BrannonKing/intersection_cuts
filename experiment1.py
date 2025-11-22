import dikin_utils as du
import numpy as np
import gurobipy as gp
import gurobi_utils as gu
import linetimer as lt
import hsnf
import ntl_wrapper as ntl
import knapsack_loader as kl
status_lookup = {getattr(gp.GRB.Status, k): k for k in gp.GRB.Status.__dir__() if "A" <= k[0] <= "Z"}

# Experiment 1: use LLL to convert to Ax <= b. Works well.

def main():
    np.random.seed(42)
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    averages = {}
    compare_original = False
    runs = 5
    for con_count in [2, 3, 4]:
        for var_count in [10, 15, 20, 50, 100]:
            print(f"Generating instances with {con_count} constraints and {var_count} variables")
            before_times = []
            after_times = []
            instances = kl.generate(runs * 10, con_count, var_count, 5, 10, 1000, equality=True, env=env)
            for model in instances:
                model.params.LogToConsole = 0
                # assumptions on the model: all equality constraints, fully linear objective & constraints, all vars >= 0, maximizing
                A = model.getA().todense()
                b = np.array(model.getAttr("RHS")).reshape((-1, 1))

                if compare_original:
                    with lt.CodeTimer("Original optimization time", silent=True) as c1:
                        model.optimize()
                    if model.status != gp.GRB.Status.OPTIMAL:
                        if model.status == gp.GRB.Status.INTERRUPTED:
                            return
                        print("  Skipping as model not optimal: ", status_lookup[model.status])
                        continue
                    before_times.append(c1.took)
                    # print(f"Original objective value: {model.ObjVal}")

                with lt.CodeTimer("  LLL time", silent=True) as c2:
                    mdl2 = gu.solve_via_LLL(model)

                after_times.append(c2.took)

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