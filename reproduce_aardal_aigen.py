from __future__ import annotations

import gurobipy as gp
import numpy as np

import gurobi_utils as gu
import knapsack_loader as kl
import dikin_utils as du

import ntl_wrapper as ntl

def _solve_center_point(model: gp.Model, inset: float) -> np.ndarray:
    A, b, c, l, u = gu.get_A_b_c_l_u(model)
    n = A.shape[1]
    mdlf = gp.Model("Find x0")
    mdlf.params.OutputFlag = 0
    x = mdlf.addMVar(shape=(n, 1), lb=l + inset, ub=u - inset, name="x")
    mdlf.setObjective(c.T @ x, model.ModelSense)
    mdlf.addConstr(A @ x == b)
    mdlf.optimize()
    if mdlf.Status != gp.GRB.OPTIMAL:
        raise RuntimeError("Failed to find an interior feasible point for rounding.")
    return np.array(x.X).reshape((-1, 1))

def main():
    np.random.seed(42)
    for con_count in [2]:
        for var_count in [25]:
            print(f"Generating instances with {con_count} constraints and {var_count} variables")
            runs = 5
            instances = kl.generate(runs, con_count, var_count, 5, 10, 1000, equality=True)
            before_improvements = []
            after_improvements = []
            last_improvements = []
            for model in instances:
                print("Starting instance", model.ModelName)

                mdl_lll = gu.transform_via_LLL(model, verify=False)
                mdl_lll.optimize()
                assert mdl_lll.status == gp.GRB.Status.OPTIMAL, "Substituted model must solve to optimality."
                actual_obj = mdl_lll.ObjVal

                A = model.getA().toarray()
                b = np.array(model.getAttr("RHS")).reshape(-1, 1)
                x0, Q = gu.nullspace_and_offset_via_LLL(A, b, verify=False)
                assert np.allclose(A @ Q, 0), "LLL transformation incorrect."
                assert Q.shape[1] == A.shape[1] - A.shape[0], "Q does not have correct number of columns."
                W = du.W_from_Q_via_LLL(Q, verify=True)

                before1, after1, cuts1 = gu.run_gmi_cuts(model, rounds=5)
                before1 -= actual_obj
                after1 -= actual_obj
                before_improvements.append(100 * (before1 - after1) / before1 if before1 != 0 else 0)

                before2, after2, cuts2 = gu.run_gmi_cuts(model, rounds=5, W=W)
                before2 -= actual_obj
                after2 -= actual_obj
                after_improvements.append(100 * (before2 - after2) / before2 if before2 != 0 else 0)

                x1 = _solve_center_point(model, inset=0.25).flatten()
                D_inv = np.diag(1 / x1)  # can do sqrt(x1)

                # what the generated paper suggested:
                # dat = Q.T @ D_inv.T @ D_inv @ Q
                # dat2 = np.sqrt(dat)
                dat2 = D_inv @ Q
                # dat2 = Q.T @ D_inv.T
                Q_transformed = np.round(dat2 * 1024).astype(np.int64)
                # now do LLL on that to get a U:
                rank, det, U = ntl.lll(Q_transformed, 99, 100)
                U = U.astype(np.int64)
                W = np.linalg.inv(U) @ W

                before3, after3, cuts3 = gu.run_gmi_cuts(model, rounds=5, W=W)
                before3 -= actual_obj
                after3 -= actual_obj
                last_improvements.append(100 * (before3 - after3) / before3 if before3 != 0 else 0)

            print(f" Average relative improvement by GMI cuts: {np.mean(before_improvements):.3f}%")
            print(f" Average relative improvement by GMI cuts with W:  {np.mean(after_improvements):.3f}%")
            print(f" Average relative improvement by GMI cuts with last W:  {np.mean(last_improvements):.3f}%")
            print()


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, edgeitems=8, linewidth=120)
    gp.setParam("OutputFlag", 0)
    gp.setParam("LogToConsole", 0)
    main()
