import dikin_utils as du
import numpy as np
import gurobipy as gp
import gurobi_utils as gu
import linetimer as lt
import hsnf
import ntl_wrapper as ntl
import knapsack_loader as kl
status_lookup = {getattr(gp.GRB.Status, k): k for k in gp.GRB.Status.__dir__() if "A" <= k[0] <= "Z"}

# Experiment 4: try to just round it. Also, try the hot dog.

def get_rounderizer(A, b, l, u, x, T=None):
    H = du.compute_H(A, b, l, u, x)
    if T is not None:
        H = T @ H @ T.T
    eigvals, eigvecs = np.linalg.eigh(H)  # is this better than sqrtm?
    H2 = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    return H2

def get_slenderizer(c, l1 = 0.25, l2 = 1):
    return l2 * np.eye(c.shape[0]) + (l1 - l2) * (c @ c.T) / (c.T @ c)

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
    return np.array(relaxed.getAttr("X")).reshape((-1, 1))

def main():
    np.random.seed(42)
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    averages = {}
    compare_original = True
    for con_count in [1, 2, 3, 4]:
        for var_count in [10, 15, 20, 25]:
            print(f"Generating instances with {con_count} constraints and {var_count} variables")
            runs = 5
            before_times = []
            after_times = []
            instances = kl.generate(runs * 10, con_count, var_count, 5, 10, 1000, equality=True, env=env)
            for model in instances:
                model.params.LogToConsole = 0
                # assumptions on the model: all equality constraints, fully linear objective & constraints, all vars >= 0, maximizing
                A = model.getA().toarray()
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

                # can I also try it with the rift here? What kind of problems can I solve with the rift?
                # the transform from it won't do anything unless it better aligns the constraints.
                # can I measure the alignment of the starting constraints?!! 
                # Then find a way to make them more aligned?
                # then convert that transform to unimodular form?
                
                # the rounding below doesn't work: x0 isn't feasible for the original model.
                # the cuts that apply to the equality model gain nothing with the slenderizer. It's only for LEQ.
                # because of that, my transform is irrelevant.

                x0 = relaxed_optimum(model)
                x0 = np.floor(x0)
                model2 = gu.relax_and_grow(model, x0, 1)
                A2, b2, c2, l2, u2 = gu.get_A_b_c_l_u(model2)
                T = get_slenderizer(c2)
                H = get_rounderizer(A2, b2, l2, u2, x0)
                H = (H * 128).astype(np.int64, order='C')
                with lt.CodeTimer("  LLL time", silent=True) as c2:
                    rank, det, U = ntl.lll(H, 3, 4)
                if c2.took > 1000:
                    print(f"  LLL took too long: {c2.took:.2f} ms")
                # xp, N = solve_via_snf(A, b)
                # now I have an integer null space and an integer starting solution (that may violate bounds)

                mdl2 = gu.substitute(model, U, x0, '=', env=env)
                # mdl2.params.NumericFocus = 3
                # mdl2.params.DualReductions = 0
                mdl2.params.LogToConsole = 0
                with lt.CodeTimer("   Transformed optimization time", silent=True) as c1:
                    mdl2.optimize()
                if mdl2.status != gp.GRB.Status.OPTIMAL:
                    if mdl2.status == gp.GRB.Status.INTERRUPTED:
                        return
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