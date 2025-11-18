from __future__ import annotations

import gurobipy as gp
import linetimer as lt
import ntl_wrapper as ntl
import numpy as np
import scipy.linalg as scl

import gurobi_utils as gu
import knapsack_loader as kl
import dikin_utils as du

# Experiment 9:
# Generate Equality knapsack instances.
# Find the ellipsoid rounder without concern for integer point.
# Combine the offset x0 with the rounding transform using homogeneous coordinates.
# LLL(scalar * [H^.5 | H^.5 @ offset]).
# Transform with resulting U, which is n+1 dim.
# That means we have to split U into a 2x2 block matrix with one column on right and one row on bottom.
#
# Use sympy for c @ U.
# Compare the cuts.


def get_rounderizer(model: gp.Model, inset=0.75):
    A, b, c, l, u = gu.get_A_b_c_l_u(model)
    # come inside the bounds by inset.
    # find that h and project it using the null space of A
    # x0 can be feasible for bounds but not Ax=b. That's okay.
    assert np.all(l == 0)
    assert np.all(u >= 2)
    
    # Get the null space of A
    N = scl.null_space(A)  # Shape: (n, n-m) where m is number of constraints
    n = A.shape[1]
    m = A.shape[0]
    assert N.shape == (n, n - m)

    Q, R = np.linalg.qr(N)  # Q is (n, n-m), R is (n-m, n-m)

    # any x0 that satisfies Ax=b and is inside bounds is okay.
    # if it satisfies Ax=b then it will satisfy Ny + x0 = b for some y.
    mdlf = gp.Model("Find x0")
    mdlf.params.OutputFlag = 0
    x = mdlf.addMVar(shape=(n, 1), lb=l+inset, ub=u-inset, name="x")
    mdlf.setObjective(c.T @ x, model.ModelSense)
    mdlf.addConstr(A @ x == b)
    mdlf.optimize()
    assert mdlf.Status == gp.GRB.OPTIMAL
    x0 = np.array(x.X).reshape((-1, 1))

    # find the hessian at x0 given box bounds l and u:
    H = np.diag(1.0 / ((u - x0) * (x0 - l)).flatten())
    
    B = scl.sqrtm(N.T @ H @ N)
    R_inv = np.linalg.inv(R)
    result = B @ R_inv @ Q.T # use R and Q to get back to original space
    return np.hstack([result, result @ x0])


def transform(model: gp.Model, AU: np.ndarray | None, U: np.ndarray, env=None):
    assert model.NumVars == model.NumIntVars
    assert U.shape[0] == U.shape[1] and U.shape[1] == model.NumVars + 1

    A, b, c, l, u = gu.get_A_b_c_l_u(model)
    cUsf = c.astype(int).astype(object).T @ U[:-1, :]

    senses = np.array(model.getAttr("Sense"))
    assert np.all(senses == gp.GRB.EQUAL)

    model2 = gp.Model("Transformed " + model.ModelName, env=env)
    # model2.params.Quad = 1
    # model2.params.NumericFocus = 3
    # U_inv = np.linalg.inv(U) // can't multiply inequality by a matrix unless it's monomial.
    # y = model2.addMVar((U.shape[0], 1), lb=U_inv @ l, ub=U_inv @ u, vtype='I', name='y')
    y = model2.addMVar((U.shape[0], 1), lb=-gp.GRB.INFINITY, vtype="I", name="y")
    model2.setObjective(cUsf @ y + model.ObjCon, model.ModelSense)
    model2.addConstr(A @ U[:-1, :] @ y == b)
    model2.addConstr(U[-1, :] @ y == -1)  # generally this just fixes a single variable to -1
    model2.addConstr(U[:-1, :] @ y >= l, name="lb")
    model2.addConstr(U[:-1, :] @ y <= u, name="ub")
    model2.update()
    return model2


# next steps:
# 1. write up the three different ways we can handle bound constraints with equality constraints.
# 2. ensure we noted how to prove that orthogonality helps Gomory cuts (and how it transfers from constraints to edges).
# 3. does our knapsack have integer c vector? am I using correct types with sympy?
# 4. run the three different approaches on knapsack and compare cut quantitys and gap closed.
# 5. also measure the density of the constraint matrix.


def main():
    np.random.seed(42)
    for con_count in [4]:
        for var_count in [30]:
            print(f"Generating instances with {con_count} constraints and {var_count} variables")
            runs = 3
            instances = kl.generate(runs, con_count, var_count, 5, 10, 1000, equality=True)
            before_improvements = []
            after_improvements = []
            for model in instances:
                print("Starting instance", model.ModelName)
                # model.params.LogToConsole = 0
                # model.optimize()
                angles = du.pairwise_hyperplane_angles(model.getA().toarray())
                print(f"  Angles between constraints on original (degrees): {np.degrees(angles)}")

                before, after, cuts = gu.run_gmi_cuts(model, rounds=10, verbose=False)
                print(f"  Original cuts: {cuts}, Before: {before}, After: {after}")

                mdl1 = transform(model, None, np.eye(model.NumVars + 1, dtype=np.int32))
                angles = du.pairwise_hyperplane_angles(mdl1.getA().toarray())
                print(f"  Angles between constraints on base (degrees): {np.degrees(angles)}")
                before1, after1, cuts1 = gu.run_gmi_cuts(mdl1, rounds=10, verbose=False)
                print(f"  Before LLL but after transform: {cuts1}, Before: {before1}, After: {after1}")
                # Measure relative improvement: how much of the initial LP bound was improved
                before_improvements.append(100 * (before - after) / before if before != 0 else 0)

                # H, x0 = get_rounderizer_bounds_only(model, inset=1)
                H = get_rounderizer(model)
                H = (H * 12).astype(np.int64, order="C")
                with lt.CodeTimer("  LLL time", silent=True) as c2:
                    rank, det, U = ntl.lll(H, 9, 10)

                mdl2 = transform(model, H, U)
                angles = du.pairwise_hyperplane_angles(mdl2.getA().toarray())
                print(f"  Angles between constraints on transformed (degrees): {np.degrees(angles)}")
                # mdl2.optimize()
                # assert round(mdl2.ObjVal) == round(model.ObjVal)
                before2, after2, cuts2 = gu.run_gmi_cuts(mdl2, rounds=10, verbose=True)
                print(f"  After LLL cuts: {cuts2}, Before: {before2}, After: {after2}")
                after_improvements.append(100 * (before2 - after2) / before2 if before2 != 0 else 0)

            print(f" Average relative improvement by GMI cuts before LLL: {np.mean(before_improvements):.3f}%")
            print(f" Average relative improvement by GMI cuts after LLL:  {np.mean(after_improvements):.3f}%")
            print()


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, edgeitems=8, linewidth=120)
    main()
