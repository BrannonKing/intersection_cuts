import dikin_utils as du
import numpy as np
import gurobipy as gp
import gurobi_utils as gu
import linetimer as lt
import ntl_wrapper as ntl
import knapsack_loader as kl
import hsnf
import sympy as sp
import cypari2 as cyp
status_lookup = {getattr(gp.GRB.Status, k): k for k in gp.GRB.Status.__dir__() if "A" <= k[0] <= "Z"}
pari = cyp.Pari()

# Experiment 6: 
# Generate inequality knapsack instances.
# Measure the solve time in Gurobi.
# LLL(A|b).
# Invert U and use that on the bounds.

def transform(model: gp.Model, Ab: np.ndarray, U: np.ndarray, env=None):
    assert model.NumVars == model.NumIntVars
    assert U.shape[0] == U.shape[1] and U.shape[1] == model.NumVars + 1
    U_top = U[0:-1, :]

    l = np.array(model.getAttr("LB")).reshape(-1, 1)
    u = np.array(model.getAttr("UB")).reshape(-1, 1)
    c = np.array(model.getAttr("Obj")).reshape((-1, 1))
    senses = np.array(model.getAttr("Sense"))
    assert np.all(senses == gp.GRB.LESS_EQUAL)

    model2 = gp.Model("Transformed " + model.ModelName, env=env)
    # U_inv = np.linalg.inv(U) // can't multiply inequality by a matrix unless it's monomial.
    # y = model2.addMVar((U.shape[0], 1), lb=U_inv @ l, ub=U_inv @ u, vtype='I', name='y')
    y = model2.addMVar((U.shape[0], 1), lb=-gp.GRB.INFINITY, vtype='I', name='y')
    model2.setObjective(c.T @ U_top @ y + model.ObjCon, model.ModelSense)
    # model2.addConstr(Ab @ y <= 0)
    A = model.getA().toarray()
    b = np.array(model.getAttr("RHS")).reshape((-1, 1))
    model2.addConstr(np.hstack([A, b]) @ U @ y <= 0)
    model2.addConstr(-1 == U[-1, :] @ y)
    model2.addConstr(l <= U_top @ y)
    model2.addConstr(U_top @ y <= u)
    return model2

def main():
    np.random.seed(42)
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 1)
    env.start()
    compare_original = True
    for con_count in [5]:
        for var_count in [50]:
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

                Ab = np.hstack((model.getA().toarray(), np.array(model.getAttr("RHS")).reshape((-1, 1)))).astype(np.int64, order='C')
                # H1, U1 = hsnf.column_style_hermite_normal_form(Ab)
                # np.savetxt("H1.csv", H1, fmt='%d')
                # np.savetxt("U1.csv", U1, fmt='%d')
                np.savetxt("dumps/Ab.csv", Ab, fmt='%d')
                print("  Before max column norm:", np.linalg.norm(Ab, axis=0).max())
                with lt.CodeTimer("  LLL time", silent=True) as c2:
                    rank, det, U = ntl.lll(Ab, 9, 10)
                    # pri = pari.Mat(Ab)
                    # U = pri.qflll()
                    # U = du.lll_fpylll_cols(Ab, 0.9, verbose=0)
                print("  After max column norm:", np.linalg.norm(Ab, axis=0).max())
                print(f"  LLL took: {c2.took:.2f} ms")
                # xp, N = solve_via_snf(A, b)
                # now I have an integer null space and an integer starting solution (that may violate bounds)
                np.savetxt("dumps/Abu.csv", Ab, fmt='%d')

                # with lt.CodeTimer("  LLL on U time", silent=True) as c2:
                #     rank, det, U2 = ntl.lll(U, 9, 10)

                # det2 = np.linalg.det(U)
                np.savetxt("dumps/U.csv", U, fmt='%d')
                # get the gcd of each row:
                grU = np.gcd.reduce(U, axis=1)
                np.savetxt("dumps/gcds.csv", grU, fmt='%d')
                Us = sp.Matrix(U)

                Ui = Us.inv()
                np.savetxt("dumps/U1.csv", Ui, fmt='%d')
                np.savetxt("dumps/UU1.csv", np.abs(Ui) @ U, fmt='%d')
                break
                # assert abs(det2) == 1, "U is not unimodular!" + str(det2) + " " + str(det)
                mdl2 = transform(model, Ab, U, env=env)
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