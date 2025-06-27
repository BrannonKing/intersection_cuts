import dikin_utils as du
import numpy as np
import scipy.linalg as sla
import gurobipy as gp
import gurobi_utils as gu
import linetimer as lt
import hsnf
import ntl_wrapper as ntl
import knapsack_loader as kl
status_lookup = {getattr(gp.GRB.Status, k): k for k in gp.GRB.Status.__dir__() if "A" <= k[0] <= "Z"}

# Experiment 2: enlarge the transform using inverse null space. no good.

# Steps:
# 1. Find the null space of the matrix A.
# 2. Find the H
# 3. Find the U that goes with the LLL of our H-1/2. 
# 4. Transform the original problem via variable substition.
# 5. Solve the transformed problem.

def relaxed_optimum(model: gp.Model):
    """
    Returns the optimal solution of the relaxed model.
    Assumes the model is a knapsack model with all variables >= 0.
    """
    relaxed = model.copy()
    gu.relax_int_or_bin_to_continuous(relaxed)
    relaxed.params.LogToConsole = 0
    relaxed.optimize()
    if relaxed.status != gp.GRB.Status.OPTIMAL:
        return None
    return np.array(relaxed.getAttr("X"))

def compute_U_LLL(H: np.ndarray, N):
    """
    Computes the U matrix for the LLL reduced H matrix.
    Assumes H is symmetric and positive definite.
    """
    # Compute the LLL reduced basis of H
    #  H2 = sla.cholesky(H, lower=False)
    # H2 = sla.solve_triangular(H2, np.eye(H.shape[0]), lower=False)
    H2 = sla.sqrtm(H)
    H2 = np.linalg.inv(H2)  # TODO: let's see if we can avoid these two inversions
    H2 = N @ H2 @ N.T  # project to the null space
    H2 = (H2 * 1000).astype(np.int64, order='C')  # scale to avoid numerical issues
    rank, det, U = ntl.lll(H2, 3, 4)
    # if rank < H.shape[0]:
    #     raise ValueError("H is not full rank")
    return np.linalg.inv(U).astype(np.int64)

def main():
    np.random.seed(42)
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    averages = {}
    compare_original = False
    for con_count in [2, 3]:
        for var_count in [20]:
            print(f"Generating instances with {con_count} constraints and {var_count} variables")
            runs = 5
            before_times = []
            after_times = []
            instances = kl.generate(runs * 10, con_count, var_count, 5, 10, 1000, equality=True, env=env)
            for model in instances:
                model.params.LogToConsole = 0
                # assumptions on the model: all equality constraints, fully linear objective & constraints, all vars >= 0, maximizing
                x0 = relaxed_optimum(model)
                if x0 is None:
                    print("  Skipping as relaxed model is infeasible.")
                    continue

                # assuming a lb of 0 that we move to -1
                ub = np.array(model.getAttr("UB")).reshape((-1, 1))
                H = du.compute_H_small(-np.ones_like(ub), ub + 1, x0)

                # we have two choices here:
                # 1. us a smaller variable space with full substitution
                # 2. use a larger variable space with the square of the null space

                # now project it:
                A = model.getA().todense()
                N = sla.null_space(A)

                # option 1: H = N.T @ H @ N, then sub x0 + N z (needs x0 in Z)
                # option 2: H = (N @ N.T) @ H @ (N @ N.T), then sub just U
                # option 2 (below) fails: U is not full rank
                H = N.T @ H @ N
                U = compute_U_LLL(H, N)
                ru = np.linalg.matrix_rank(U)
                print(f"  U has rank {ru} and shape {U.shape}")

                mdl2 = gp.Model(model.ModelName + "_tfrm", env=env)
                z = mdl2.addMVar(model.NumIntVars, lb=-gp.GRB.INFINITY, vtype='I', name='z')
                c = np.array(model.getAttr("Obj")).reshape((-1, 1))
                mdl2.setObjective(c.T @ U @ z, model.ModelSense)
                mdl2.addConstr(A @ U @ z == np.array(model.getAttr("RHS")).reshape((-1, 1)))
                mdl2.addConstr(U @ z >= 0)
                mdl2.addConstr(U @ z <= ub)
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
                break # tmp
            if compare_original:
                print(f" Average original time: {np.mean(before_times):.8f} ms")
                averages[(con_count, var_count)] = (np.mean(before_times), np.mean(after_times))
            if after_times:
                print(f" Average transformed time: {np.mean(after_times):.8f} ms")
            print()
            exit(0)
    print("Averages:", averages)

if __name__ == "__main__":
    main()