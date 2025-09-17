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

# Experiment 7b: 
# Generate inequality knapsack instances.
# Measure the solve time in Gurobi.
# LLL(A|b; I|l; -I;u).
# Invert U and use that on objective only.
# Use sympy for c @ U.
# Compare the cuts.

def transform(model: gp.Model, A: np.ndarray, U: np.ndarray, env=None):
    assert model.NumVars == model.NumIntVars
    assert U.shape[0] == U.shape[1] and U.shape[1] == model.NumVars + 1

    c = sp.Matrix(model.getAttr("Obj"))
    Us = sp.Matrix(U[0:-1, :])
    cUs = c.T @ Us
    # get the gcd of the vector cUs -- gcd was always 1
    cUsf = np.array(cUs, dtype=np.int64).reshape((-1, 1))

    senses = np.array(model.getAttr("Sense"))
    assert np.all(senses == gp.GRB.LESS_EQUAL)

    model2 = gp.Model("Transformed " + model.ModelName, env=env)
    # U_inv = np.linalg.inv(U) // can't multiply inequality by a matrix unless it's monomial.
    # y = model2.addMVar((U.shape[0], 1), lb=U_inv @ l, ub=U_inv @ u, vtype='I', name='y')
    y = model2.addMVar((U.shape[0], 1), lb=-gp.GRB.INFINITY, vtype='I', name='y')
    model2.setObjective(cUsf.T @ y + model.ObjCon, model.ModelSense)
    model2.addConstr(A @ y <= 0)
    model2.addConstr(-1 == U[-1, :] @ y)  # generally this just fixes a single variable to -1
    return model2

def make_cuts_only(model: gp.Model):
    model.params.Presolve = 0
    model.params.NodeLimit = 1

    model.params.Cuts = 2
    model.params.Heuristics = 0
    model.params.Method = 1  # primal simplex

    model.params.CoverCuts = 0
    model.params.FlowCoverCuts = 0
    model.params.BQPCuts = 0
    model.params.CliqueCuts = 0
    # model.params.DualImpliedCuts = 0
    model.params.FlowCoverCuts = 0
    model.params.FlowPathCuts = 0
    model.params.GUBCoverCuts = 0
    model.params.ImpliedCuts = 0
    model.params.InfProofCuts = 0
    model.params.MIPSepCuts = 0
    model.params.ModKCuts = 0
    # model.params.MxingCuts = 0
    model.params.NetworkCuts = 0
    model.params.ProjImpliedCuts = 0
    model.params.LiftProjectCuts = 0
    model.params.RelaxLiftCuts = 0
    model.params.RLTCuts = 0
    model.params.SubMIPCuts = 0
    model.params.ZeroHalfCuts = 0
    model.params.StrongCGCuts = 2
    model.params.MIRCuts = 2
    model.params.GomoryPasses = 20


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
                # assumptions on the model: all equality constraints, fully linear objective & constraints, all vars >= 0, maximizing

                if compare_original:
                    model.params.LogToConsole = 1
                    make_cuts_only(model)
                    with lt.CodeTimer("Original optimization time", silent=True) as c1:
                        model.optimize()
                    if model.status == gp.GRB.Status.INTERRUPTED:
                        return
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
                A, b, c, l, u = gu.get_A_b_c_l_u(model, False)
                block = np.block([
                    [A, b], 
                    [-np.eye(A.shape[1]), -l],
                    [np.eye(A.shape[1]), u]
                ]).astype(np.int64)

                # H1, U1 = hsnf.column_style_hermite_normal_form(Ab)
                # np.savetxt("H1.csv", H1, fmt='%d')
                # np.savetxt("U1.csv", U1, fmt='%d')
                # np.savetxt("dumps/Ab.csv", block, fmt='%d')
                print("  Before max column norm:", np.linalg.norm(block, axis=0).max())
                with lt.CodeTimer("  LLL time", silent=True) as c2:
                    rank, det, U = ntl.lll(block, 9, 10)
                    # pri = pari.Mat(Ab)
                    # U = pri.qflll()
                    # U = du.lll_fpylll_cols(Ab, 0.9, verbose=0)
                print("  After max column norm:", np.linalg.norm(block, axis=0).max())
                print(f"  LLL took: {c2.took:.2f} ms")
                # xp, N = solve_via_snf(A, b)
                # now I have an integer null space and an integer starting solution (that may violate bounds)
                # np.savetxt("dumps/Abu.csv", block, fmt='%d')
                # np.savetxt("dumps/U.csv", U, fmt='%d')

                mdl2 = transform(model, block, U, env=env)
                # mdl2.params.NumericFocus = 3
                # mdl2.params.DualReductions = 0
                mdl2.params.LogToConsole = 1
                make_cuts_only(mdl2)
                with lt.CodeTimer("   Transformed optimization time", silent=True) as c1:
                    mdl2.optimize()
                if mdl2.status == gp.GRB.Status.INTERRUPTED:
                    return
                after_times.append(c1.took)

                if compare_original:
                    print(f"Resulting gaps: {model.MIPGap} vs {mdl2.MIPGap}")
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