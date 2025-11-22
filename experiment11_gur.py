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
# Make an ellipsoid at x0 and project that onto the null space of A.
# Find the ellipsoid rounder at x0 without concern for being centered on an integer point.
# The x0 has to match the x0 used for Ny + x0 = b.
# Combine the offset x0 with the rounding transform using homogeneous coordinates.
# LLL(scalar * [H^.5 | H^.5 @ offset]).
# Transform with resulting U, which is n+1 dim.
# That means we have to split U into a 2x2 block matrix with one column on right and one row on bottom.
#
# Use sympy for c @ U.
# Compare the cuts.


@dataclass
class RoundingFrame:
    linear_full: np.ndarray
    homogenized: np.ndarray
    x0: np.ndarray


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


def _build_rounding_frame(model: gp.Model, inset: float) -> RoundingFrame:
    A, _, _, l, u = gu.get_A_b_c_l_u(model)
    assert np.all(l == 0)
    assert np.all(u >= 2)

    N, basis = du.extend_null_space_to_full_basis(A)
    null_dim = N.shape[1]
    x0 = _solve_center_point(model, inset)

    H = np.diag(1.0 / ((u - x0) * (x0 - l)).flatten())
    gram = np.real_if_close(N.T @ H @ N, tol=1e-9)
    sqrt = np.real_if_close(scl.sqrtm(gram), tol=1e-9)
    # sqrt_inv = np.real_if_close(np.linalg.inv(sqrt), tol=1e-9)

    linear_null = N @ sqrt
    row_comp = basis[:, null_dim:]
    linear_full = np.hstack([linear_null, row_comp])

    homogenized = np.block(
        [
            [linear_full, x0],
            [np.zeros((1, linear_full.shape[1])), np.ones((1, 1))],
        ]
    )

    return RoundingFrame(linear_full=linear_full, homogenized=homogenized, x0=x0)


def _strip_homogeneous_columns(matrix: np.ndarray, num_vars: int) -> np.ndarray:
    """Drop the extra column introduced by homogeneous coordinates."""

    if matrix.shape[1] == num_vars + 1:
        return matrix[:, :num_vars]
    return matrix

def _mean_angle_stats(angle_matrix: np.ndarray) -> tuple[float, float]:
    if angle_matrix.size == 0:
        return 90.0, 0.0
    tri = np.triu_indices(angle_matrix.shape[0], k=1)
    if tri[0].size == 0:
        return 90.0, 0.0
    degrees = np.degrees(angle_matrix[tri])
    return float(degrees.mean()), float(np.abs(degrees - 90.0).mean())


def get_rounderizer(model: gp.Model, inset: float=0.5) -> tuple[np.ndarray, RoundingFrame]:
    frame = _build_rounding_frame(model, inset)
    # _, U = du.seysen_integer_matrix(frame.homogenized, 10000)
    U = du.to_U_via_LU(frame.homogenized, 512)
    # U = du.lll_integer_matrix(frame.homogenized, 1024)
    return U, frame


def transform(model: gp.Model, AU: np.ndarray | None, U: np.ndarray):
    assert model.NumVars == model.NumIntVars
    assert U.shape[0] == U.shape[1] and U.shape[1] == model.NumVars + 1

    A, b, c, l, u = gu.get_A_b_c_l_u(model)
    cUsf = c.astype(int).astype(object).T @ U[:-1, :]

    senses = np.array(model.getAttr("Sense"))
    assert np.all(senses == gp.GRB.EQUAL)

    model2 = gp.Model("Transformed " + model.ModelName)
    y = model2.addMVar((U.shape[0], 1), lb=-gp.GRB.INFINITY, vtype="I", name="y")
    model2.setObjective(cUsf @ y + model.ObjCon, model.ModelSense)
    model2.addConstr(A @ U[:-1, :] @ y == b)  # can we use AU here if provided?
    model2.addConstr(U[-1, :] @ y == -1)  # generally this just fixes a single variable to -1
    model2.addConstr(U[:-1, :] @ y >= l, name="lb")
    model2.addConstr(U[:-1, :] @ y <= u, name="ub")
    model2.update()

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
        for var_count in [25]:
            print(f"Generating instances with {con_count} constraints and {var_count} variables")
            runs = 3
            instances = kl.generate(runs, con_count, var_count, 5, 10, 1000, equality=True)
            before_improvements = []
            after_improvements = []
            for model in instances:
                print("Starting instance", model.ModelName)
                modelA = model.getA().toarray()
                base_angles = _mean_angle_stats(du.pairwise_hyperplane_angles(modelA, by_rows=False))
                base_measure = du.orthogonality_measure_2(modelA, by_rows=False)
                print(f"  Angles between constraints on original (degrees): {base_angles}, {base_measure}")
                if compare_original:
                    mdl_lll = gu.solve_via_LLL(model, verify=False)
                    actual_obj = mdl_lll.ObjVal

                mdl1 = transform(model, None, np.eye(model.NumVars + 1, dtype=np.int32))
                mdl1A = _strip_homogeneous_columns(mdl1.getA().toarray(), model.NumVars)
                angles = _mean_angle_stats(du.pairwise_hyperplane_angles(mdl1A, by_rows=False))
                measure = du.orthogonality_measure_2(mdl1A, by_rows=False)
                print(f"  Angles between constraints on base (degrees): {angles}, {measure}")
                before1, after1, cuts1 = gu.run_gmi_cuts(mdl1, rounds=20, verbose=False)
                print(f"  Before LLL but after transform: {cuts1}, Before: {before1}, After: {after1}")
                # Measure relative improvement: how much of the initial LP bound was improved
                if compare_original:
                    before1 -= actual_obj
                    after1 -= actual_obj
                before_improvements.append(100 * (before1 - after1) / before1 if before1 != 0 else 0)

                U, frame = get_rounderizer(model)
                idealA = modelA @ frame.linear_full
                ideal_angles = _mean_angle_stats(du.pairwise_hyperplane_angles(idealA, by_rows=False))
                ideal_measure = du.orthogonality_measure_2(idealA, by_rows=False)
                print(f"  Angles after Dikin rounding frame (degrees): {ideal_angles}, {ideal_measure}")

                mdl2 = transform(model, None, U)
                mdl2A = _strip_homogeneous_columns(mdl2.getA().toarray(), model.NumVars)
                angles = _mean_angle_stats(du.pairwise_hyperplane_angles(mdl2A, by_rows=False))
                measure = du.orthogonality_measure_2(mdl2A, by_rows=False)
                print(f"  Angles between constraints on transformed (degrees): {angles}, {measure}")
                before2, after2, cuts2 = gu.run_gmi_cuts(mdl2, rounds=20, verbose=False)
                print(f"  After LLL cuts: {cuts2}, Before: {before2}, After: {after2}")
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
