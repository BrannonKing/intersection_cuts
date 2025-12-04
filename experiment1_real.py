import dikin_utils as du
import numpy as np
import gurobipy as gp
import gurobi_utils as gu
import linetimer as lt
import knapsack_loader as kl
import scipy.linalg as scl
status_lookup = {getattr(gp.GRB.Status, k): k for k in gp.GRB.Status.__dir__() if "A" <= k[0] <= "Z"}

# Experiment 1-real: use a null space from scipy
# We know that our ellipsoid is not going to produce integer coefficients. Hence, finding x0-x1 integer seems like a waste.
# Suppose N not in Z and x0==x1:
# First substitution: min c(Ny + x0): A(Ny + x0)=b, 0 <= Ny + x0 <= u, Ny + x0 in Z   -- actually, what is the performance on this? min c(Ny + x0): Ny + x0 == x, 0 <= x <= u, x in Z
# T derived from y N.T H N y: T = H.5 N
# Substitute y = Tz
# Second sub: min c(NTz + x0): 0 <= NTz + x0 <= u, NTz + x0 in Z (assumes Ax0=b and AN=0)


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
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    averages = {}
    compare_original = True
    makeT = False
    runs = 5
    for con_count in [2, 3, 4]:
        for var_count in [10, 15, 20]: # , 50, 100]:
            print(f"Generating instances with {con_count} constraints and {var_count} variables")
            before_times = []
            after_times = []
            instances = kl.generate(runs * 10, con_count, var_count, 5, 10, 1000, equality=True, env=env)
            for model in instances:
                model.params.LogToConsole = 0
                # assumptions on the model: all equality constraints, fully linear objective & constraints, all vars >= 0, maximizing
                A, b, c, l, u = gu.get_A_b_c_l_u(model)

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

                x0 = _solve_center_point(model, inset=0.25)
                assert np.allclose(A @ x0, b), "x0 must be feasible."
                N = scl.null_space(A)
                # N, _ = np.linalg.qr(N)  # orthonormal basis for the null space
                mdl2 = gp.Model(model.ModelName + "_transformed")
                x = mdl2.addMVar(shape=(var_count, 1), lb=l, ub=u, name="x", vtype=gp.GRB.INTEGER)
                y = mdl2.addMVar(shape=(N.shape[1], 1), lb=-gp.GRB.INFINITY, name="y")
                mdl2.setObjective(c.T @ x, model.ModelSense)
                if makeT:
                    # T = (N.T @ H @ N)^(1/2). We compute this using SVD of H^(1/2) @ N.
                    # Let K = H^(1/2) @ N. Then N.T @ H @ N = K.T @ K.
                    # If K = U S V.T, then K.T @ K = V S^2 V.T, and (K.T @ K)^(1/2) = V S V.T.
                    h_sqrt = 1.0 / np.sqrt(((u - x0) * (x0 - l)).flatten())
                    K = h_sqrt[:, None] * N
                    _, s, vh = np.linalg.svd(K, full_matrices=False)
                    T = vh.T @ np.diag(s) @ vh
                    mdl2.addConstr(N @ T @ y + x0 == x)
                else:
                    mdl2.addConstr(N @ y + x0 == x)

                with lt.CodeTimer("  LLL time", silent=True) as c2:
                    mdl2.optimize()
                    assert mdl2.status == gp.GRB.Status.OPTIMAL, "Substituted model must solve to optimality."

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
    gp.setParam("OutputFlag", 0)
    main()