import dikin_utils as du
import numpy as np
import gurobipy as gp
import gurobi_utils as gu
import linetimer as lt
import ntl_wrapper as ntl
import knapsack_loader as kl
import sympy as sp
status_lookup = {getattr(gp.GRB.Status, k): k for k in gp.GRB.Status.__dir__() if "A" <= k[0] <= "Z"}

# Experiment 5: 
# Generate inequality knapsack instances.
# Measure the solve time in Gurobi.
# Take their rounderizer.
# Run LLL on that. Measure the orthogonality improvement.

def get_rounderizer(A, b, l, u, x, T=None):
    H = du.compute_H(A, b, l, u, x)
    if T is not None:
        H = T @ H @ T.T
    eigvals, eigvecs = np.linalg.eigh(H)  # is this better than sqrtm?
    H2 = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    return H2

def transform(model: gp.Model, U: np.ndarray, env=None):
    assert model.NumVars == model.NumIntVars
    assert U.shape[0] == U.shape[1] and U.shape[1] == model.NumVars + 1
    model2 = gp.Model("Transformed " + model.ModelName, env=env)
    y = model2.addMVar((U.shape[0], 1), lb=-gp.GRB.INFINITY, vtype='I', name='y')
    U_top = U[0:-1, :].astype(np.float64)
    U_bottom = U[-1, :].astype(np.float64)

    A, b, c, l, u = gu.get_A_b_c_l_u(model, True)
    senses = np.array(model.getAttr("Sense"))
    assert np.all(senses == gp.GRB.LESS_EQUAL)

    model2.setObjective(c.T @ U_top @ y + model.ObjCon, model.ModelSense)
    model2.addConstr(A @ U_top @ y <= b)
    model2.addConstr(-1 == U_bottom @ y)
    model2.addConstr(l <= U_top @ y)
    model2.addConstr(U_top @ y <= u)
    return model2

def main():
    np.random.seed(42)
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 1)
    env.start()
    compare_original = True
    for con_count in [20]:
        for var_count in [75, 100, 125]:
            print(f"Generating instances with {con_count} constraints and {var_count} variables")
            runs = 5
            before_times = []
            after_times = []
            instances = kl.generate(runs, con_count, var_count, 5, 10, 1000, equality=False, env=env)
            for model in instances:
                model.params.LogToConsole = 0
                # assumptions on the model: all equality constraints, fully linear objective & constraints, all vars >= 0, maximizing

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

                x0 = gu.relaxed_optimum(model)
                model2 = gu.relax_and_grow(model, x0, 1)
                A2, b2, c2, l2, u2 = gu.get_A_b_c_l_u(model2)
                H = get_rounderizer(A2, b2, l2, u2, x0)
                H = np.round(np.hstack([H, -H @ x0]) * 10).astype(np.int64, order='C')
                print("  Before max column norm:", np.linalg.norm(H, axis=0).max())
                with lt.CodeTimer("  LLL time", silent=True) as c2:
                    rank, det, U = ntl.lll(H, 9, 10)
                    # U = du.lll_fpylll_cols(H, 0.9, verbose=1)
                print(f"  LLL took: {c2.took:.2f} ms")
                print("  After max column norm:", np.linalg.norm(H, axis=0).max())
                A = model.getA().toarray()
                print("  A norm:", np.linalg.norm(A, axis=1).max())
                print("  A condition:", np.linalg.cond(A))
                print("  A orthogonality:", np.linalg.norm(A @ A.T - np.eye(A.shape[0])))
                AU = sp.Matrix(A) @ sp.Matrix(U[0:-1,:])
                # Use SymPy operations instead of NumPy for AU (a SymPy Matrix)
                print("  AU norm:", max(AU[:, i].norm() for i in range(AU.shape[1])))
                print("  AU condition:", AU.condition_number())
                print("  AU orthogonality:", (AU * AU.T - sp.eye(AU.shape[0])).norm())
                # xp, N = solve_via_snf(A, b)
                # now I have an integer null space and an integer starting solution (that may violate bounds)

                mdl2 = transform(model, U, env=env)
                mdl2.params.NumericFocus = 3
                mdl2.params.DualReductions = 0
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
                # if len(after_times) == runs:
                #     break
            if compare_original:
                print(f" Average original time: {np.mean(before_times):.8f} ms")
            #     averages[(con_count, var_count)] = (np.mean(before_times), np.mean(after_times))
            if after_times:
                print(f" Average transformed time: {np.mean(after_times):.8f} ms")
            print()

if __name__ == "__main__":
    main()