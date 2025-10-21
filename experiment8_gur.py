from __future__ import annotations

import gurobipy as gp
import linetimer as lt
import ntl_wrapper as ntl
import numpy as np
import sympy as sp

import dikin_utils as du
import gurobi_utils as gu
import knapsack_loader as kl

# Experiment 8:
# Generate Equality knapsack instances.
# Find the ellipsoid rounder at xr.
# LLL(H^.5).
# Transform with resulting U.
# Use sympy for c @ U.
# Compare the cuts.


def matrix_sqrt(H: np.ndarray):
    eigvals, eigvecs = np.linalg.eigh(H)  # is this better than sqrtm?
    return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T


def get_rounderizer_bounds_only(model: gp.Model, inset=1):
    # x0 can be feasible for bounds but not Ax=b. That's okay.
    u = np.array(model.getAttr("UB")).reshape((-1, 1))
    l = np.array(model.getAttr("LB")).reshape((-1, 1))
    assert np.all(l == 0)
    assert np.all(u >= 2)
    x0 = u - inset
    # find the hessian at x0 give bounds l and u:
    H = np.diag(1.0 / ((u - x0) * (x0 - l)).flatten())
    return matrix_sqrt(H), x0


def get_rounderizer_rift(model: gp.Model, inset=1.5):
    A, b, c, l, u = gu.get_A_b_c_l_u(model)
    rift_model = gp.Model("Rift " + model.ModelName)
    assert model.NumVars == model.NumIntVars
    x = rift_model.addMVar((model.NumIntVars, 1), lb=l, ub=u, vtype="I", name="x")
    for i, row in enumerate(A):
        nm = inset / np.linalg.norm(row)
        rift_model.addConstr(row @ x >= (b[i, 0] - nm))
        rift_model.addConstr(row @ x <= (b[i, 0] + nm))
    rift_model.setObjective(c.T @ x + model.ObjCon, model.ModelSense)
    rift_model.params.LogToConsole = 0
    rift_model.optimize()
    assert rift_model.Status == gp.GRB.OPTIMAL
    x0 = np.array(rift_model.getAttr("X")).reshape((-1, 1))
    A, b, c, l, u = gu.get_A_b_c_l_u(rift_model)
    x0[x0 == 0] = 1
    x0[x0 == u] -= 1
    H = du.compute_H(A, b, l, u, x0)
    return matrix_sqrt(H), x0


def get_rounderizer_double_sub(model: gp.Model):
    # we first find the bounds-based ellipsoid, then we project into the nullspace of A.
    H, x0 = get_rounderizer_bounds_only(model)
    A, b, c, l, u = gu.get_A_b_c_l_u(model)
    # to get the (integer) null space, we're going to use Aardal's trick with the LLL:
    #     N1 = max(np.linalg.norm(b, np.inf).item(), np.linalg.norm(A, np.inf).item()) * 4
    m, n = A.shape
    N1 = max(np.linalg.norm(b, np.inf).item(), np.linalg.norm(A, np.inf).item()) * 6
    N2 = N1 * 4
    B = np.block(
        [
            [np.eye(n, dtype=np.int64), np.zeros((n, 1), dtype=np.int64)],  # fmt: skip
            [np.zeros((1, n), dtype=np.int64), np.array([N1])],
            [N2 * A, -N2 * b],
        ]
    ).astype(np.int64, order="C")

    rank, det, U = ntl.lll(B, 9, 10)  # modifies
    x1 = B[0:n, n - m]
    if B[n, n - m].item() != N1:
        print("---LLL did not preserve the last element; something went wrong!")
    else:
        assert np.allclose(A @ x1, b)
    N = B[0:n, 0 : n - m]
    H = N.T @ H @ N
    N_inv = np.linalg.inv(N)  # yikes! not integer anymore!
    return matrix_sqrt(H), N_inv @ (x1 - x0), (N, x1 - x0)


def transform(model: gp.Model, U: np.ndarray, x0: np.ndarray, N=None, env=None):
    assert model.NumVars == model.NumIntVars
    assert U.shape[0] == U.shape[1] and U.shape[1] == model.NumVars

    A, b, c, l, u = gu.get_A_b_c_l_u(model)
    cs = sp.Matrix(c, dtype=int)
    Us = sp.Matrix(U)
    cUs = cs.T @ Us
    # get the gcd of the vector cUs -- gcd was always 1
    cUsf = np.array(cUs, dtype=np.int64).reshape((-1, 1))

    senses = np.array(model.getAttr("Sense"))
    assert np.all(senses == gp.GRB.EQUAL)

    model2 = gp.Model("Transformed " + model.ModelName, env=env)
    y = model2.addMVar((U.shape[0], 1), lb=-gp.GRB.INFINITY, vtype="I", name="y")

    # if (U is a permutation matrix)
    #     l -= x0
    #     u -= x0
    #     U_inv = Us.inv()
    #     l = U_inv @ l
    #     u = U_inv @ u
    #     set l and u on y
    # else:
    model2.addConstr(U @ y + x0 >= l)
    model2.addConstr(U @ y + x0 <= u)
    model2.setObjective(cUsf.T @ y + c.T @ x0 + model.ObjCon, model.ModelSense)
    model2.addConstr(A @ U @ y == b - A @ x0)
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
    for con_count in [2]:
        for var_count in [20]:
            print(f"Generating instances with {con_count} constraints and {var_count} variables")
            runs = 5
            instances = kl.generate(runs, con_count, var_count, 5, 10, 1000, equality=True)
            before_gaps = []
            after_gaps = []
            for model in instances:
                print("Starting instance", model.ModelName)
                model.params.LogToConsole = 0
                model.optimize()

                before, after, cuts = gu.run_gmi_cuts(model, rounds=5, verbose=False)
                print(f"  Original cuts: {cuts}, Before: {before}, After: {after}, Opt: {model.ObjVal}")

                mdl1 = transform(model, np.eye(model.NumVars, dtype=np.int32), np.zeros((model.NumVars, 1)))
                _, after, cuts = gu.run_gmi_cuts(mdl1, rounds=5, verbose=False)
                print(f"  Before LLL but after transform: {cuts}, After: {after}")
                before_gaps.append(100 * (before - after) / (before - model.ObjVal))

                # H, x0 = get_rounderizer_bounds_only(model, inset=1)
                H, x0 = get_rounderizer_rift(model, inset=1.5)
                H = (H * 128).astype(np.int64, order="C")
                with lt.CodeTimer("  LLL time", silent=True) as c2:
                    rank, det, U = ntl.lll(H, 9, 10)

                mdl2 = transform(model, U, x0)
                mdl2.optimize()
                assert round(mdl2.ObjVal) == round(model.ObjVal)
                _, after, cuts = gu.run_gmi_cuts(mdl2, rounds=5, verbose=False)
                print(f"  After LLL cuts: {cuts}, After: {after}")
                after_gaps.append(100 * (before - after) / (before - model.ObjVal))

            print(f" Average gap closed by GMI cuts before LLL: {np.mean(before_gaps):.3f}%")
            print(f" Average gap closed by GMI cuts after:  {np.mean(after_gaps):.3f}%")
            print()


if __name__ == "__main__":
    main()
