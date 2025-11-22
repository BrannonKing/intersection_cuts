from __future__ import annotations

from dataclasses import dataclass

import gurobipy as gp
import numpy as np
import scipy.linalg as scl

import gurobi_utils as gu
import knapsack_loader as kl
import dikin_utils as du

# Experiment 11:
# Generate Equality knapsack instances.
# Make a transform for A as the inverse of R, with R from QR decomp.
# Convert that R(-1) into a unimodular matrix U using the LU rounding.
# Transform with resulting U, which is n dim -- no homegeneous coords needed.


def _mean_angle_stats(angle_matrix: np.ndarray) -> tuple[float, float]:
    if angle_matrix.size == 0:
        return 90.0, 0.0
    tri = np.triu_indices(angle_matrix.shape[0], k=1)
    if tri[0].size == 0:
        return 90.0, 0.0
    degrees = np.degrees(angle_matrix[tri])
    return float(degrees.mean()), float(np.abs(degrees - 90.0).mean())


def get_rounderizer(model: gp.Model, inset: float=0.5):
    A = model.getA().toarray()
    # R = np.linalg.qr(A, mode="r")
    # m, n = A.shape
    # if m >= n:
    #     # Full column rank assumed or checked separately
    #     T = np.linalg.inv(R)                 # or solve(R, eye(n))
    # else:  # m < n
    #     T = np.linalg.pinv(R)                # R is m×n, pinv gives least-squares solution

    # _, _, Vh = np.linalg.svd(A, full_matrices=True)
    # T = Vh.T
    # U = du.to_U_via_LU(T)  # or call lu_integer_matrix
    # U = du.lll_integer_matrix(T, 512)  # usually just returns identity
    U = du.seysen_integer_matrix(A, 1)  # usually just returns identity
    # U = du.lll_integer_matrix(A, 1)
    return U

def transform(model: gp.Model, U: np.ndarray):
    assert model.NumVars == model.NumIntVars
    assert U.shape[0] == model.NumVars

    b = np.array(model.getAttr("RHS")).reshape((-1, 1))  # np.zeros((model.NumVars, 1))
    A = model.getA().toarray()

    # find a valid x0 such that A x0 = b and x0 is integer
    # solve it using Gurobi:
    mdlf = gp.Model("Find x0")
    y = mdlf.addMVar(shape=(U.shape[1], 1), vtype='I', lb=0, ub=gp.GRB.INFINITY, name="y")
    x = mdlf.addMVar(shape=(model.NumVars, 1), vtype='I', lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="x")
    mdlf.setObjective(y.sum(), gp.GRB.MINIMIZE)
    mdlf.addConstr(A @ U @ y + A @ x == b)
    mdlf.optimize()
    assert mdlf.Status == gp.GRB.Status.OPTIMAL
    x0 = np.array(x.X).reshape((-1, 1))
    # assert np.allclose(A @ x0, b), "x0 is not feasible!"

    model2 = gu.substitute(model, U, x0, "=")
    # model2.params.Quad = 1
    # model2.params.NumericFocus = 3
    return model2


# next steps:
# 1. write up the three different ways we can handle bound constraints with equality constraints.
# 2. ensure we noted how to prove that orthogonality helps Gomory cuts (and how it transfers from constraints to edges).
# 3. does our knapsack have integer c vector? am I using correct types with sympy?
# 4. run the three different approaches on knapsack and compare cut quantitys and gap closed.
# 5. also measure the density of the constraint matrix.


def main():
    np.random.seed(42)
    compare_original = True
    for con_count in [2]:
        for var_count in [20]:
            print(f"Generating instances with {con_count} constraints and {var_count} variables")
            runs = 3
            instances = kl.generate(runs, con_count, var_count, 5, 10, 1000, equality=True)
            before_improvements = []
            after_improvements = []
            for model in instances:
                print("Starting instance", model.ModelName)
                modelA = model.getA().toarray()
                base_angles = _mean_angle_stats(du.pairwise_hyperplane_angles(modelA, by_rows=False))
                base_measure = du.orthogonality_measure_1(modelA, by_rows=False)
                print(f"  Angles between constraints on original (degrees): {base_angles}, {base_measure}")
                if compare_original:
                    mdl_lll = gu.solve_via_LLL(model, verify=False)
                    actual_obj = mdl_lll.ObjVal
                    print(f"  Ideal objective value: {actual_obj}")

                mdl1 = transform(model, np.eye(model.NumVars, dtype=np.int32))
                mdl1A = mdl1.getA().toarray()
                angles = _mean_angle_stats(du.pairwise_hyperplane_angles(mdl1A, by_rows=False))
                measure = du.orthogonality_measure_2(mdl1A, by_rows=False)
                print(f"  Angles between constraints on base (degrees): {angles}, {measure}")
                before1, after1, cuts1 = gu.run_gmi_cuts(mdl1, rounds=10, verbose=False)
                print(f"  After I GMI cuts: {cuts1}, Before: {before1}, After: {after1}")
                # Measure relative improvement: how much of the initial LP bound was improved
                if compare_original:
                    before1 -= actual_obj
                    after1 -= actual_obj
                before_improvements.append(100 * (before1 - after1) / before1 if before1 != 0 else 0)

                U = get_rounderizer(model)
                # ideal_angles = _mean_angle_stats(du.pairwise_hyperplane_angles(mdl1A @ ideal, by_rows=False))
                # ideal_measure = du.orthogonality_measure_2(mdl1A @ ideal, by_rows=False)
                # print(f"  Ideal transform results (degrees): {ideal_angles}, {ideal_measure}")

                mdl2 = transform(model, U)
                mdl2.params.DualReductions = 0
                mdl2.optimize()
                assert mdl2.status == gp.GRB.Status.OPTIMAL
                assert round(mdl2.ObjVal) == round(actual_obj)
                mdl2A = mdl2.getA().toarray()
                angles = _mean_angle_stats(du.pairwise_hyperplane_angles(mdl2A, by_rows=False))
                measure = du.orthogonality_measure_2(mdl2A, by_rows=False)
                print(f"  Angles between constraints on transformed by U (degrees): {angles}, {measure}")
                before2, after2, cuts2 = gu.run_gmi_cuts(mdl2, rounds=10, verbose=False)
                print(f"  After U GMI cuts: {cuts2}, Before: {before2}, After: {after2}")
                if compare_original:
                    before2 -= actual_obj
                    after2 -= actual_obj
                after_improvements.append(100 * (before2 - after2) / before2 if before2 != 0 else 0)

            print(f" Average relative improvement by GMI cuts before LLL: {np.mean(before_improvements):.3f}%")
            print(f" Average relative improvement by GMI cuts after LLL:  {np.mean(after_improvements):.3f}%")
            print()


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, edgeitems=8, linewidth=120)
    gp.setParam("OutputFlag", 0)
    gp.setParam("LogToConsole", 0)
    main()
