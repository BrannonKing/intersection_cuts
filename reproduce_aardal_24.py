from __future__ import annotations

import gurobipy as gp
import numpy as np

import gurobi_utils as gu
import knapsack_loader as kl
import dikin_utils as du
import scipy.linalg as scl

def main():
    np.random.seed(42)
    for con_count in [2, 3, 4]:
        for var_count in [20, 25]:
            runs = 5
            print(f"Generating {runs} instances with {con_count} constraints and {var_count} variables")
            instances = kl.generate(runs, con_count, var_count, 5, 10, 1000, equality=True)
            before_improvements = []
            after_improvements = []
            for model in instances:
                print("  Starting instance", model.ModelName)

                mdl_lll = gu.transform_via_LLL(model, verify=False)
                mdl_lll.optimize()
                assert mdl_lll.status == gp.GRB.Status.OPTIMAL, "Substituted model must solve to optimality."
                actual_obj = mdl_lll.ObjVal

                A = model.getA().toarray()
                b = np.array(model.getAttr("RHS")).reshape(-1, 1)
                x0, Q = gu.nullspace_and_offset_via_LLL(A, b, verify=False)
                # Q = scl.null_space(A) # this doesn't actually work
                assert np.allclose(A @ Q, 0), "LLL transformation incorrect."
                assert Q.shape[1] == A.shape[1] - A.shape[0], "Q does not have correct number of columns."
                W = du.W_from_Q_via_LLL(Q, verify=True)
                # W = np.linalg.pinv(Q) # living dangerous!

                before1, rlxd1 = gu.run_gmi_cuts(model, rounds=50)
                after1 = rlxd1.ObjVal
                before1 -= actual_obj
                after1 -= actual_obj
                before_improvements.append(100 * (before1 - after1) / before1 if before1 != 0 else 0)

                before2, rlxd2 = gu.run_gmi_cuts(model, rounds=50, W=W)
                after2 = rlxd2.ObjVal
                before2 -= actual_obj
                after2 -= actual_obj
                after_improvements.append(100 * (before2 - after2) / before2 if before2 != 0 else 0)

            print(f" Average relative improvement by GMI cuts: {np.mean(before_improvements):.3f}%")
            print(f" Average relative improvement by GMI cuts with W:  {np.mean(after_improvements):.3f}%")
            print()


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, edgeitems=8, linewidth=120)
    gp.setParam("OutputFlag", 0)
    gp.setParam("LogToConsole", 0)
    main()
