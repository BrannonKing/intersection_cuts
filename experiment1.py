import dikin_utils as du
import numpy as np
import gurobipy as gp
import linetimer as lt
import hsnf
import ntl_wrapper as ntl
import knapsack_loader as kl
status_lookup = {getattr(gp.GRB.Status, k): k for k in gp.GRB.Status.__dir__() if "A" <= k[0] <= "Z"}

def solve_via_LLL(A: np.ndarray, b: np.ndarray, check_gcd=False) -> np.ndarray:
    m, n = A.shape
    A = A.astype(np.int64)
    b = b.astype(np.int64)

    if check_gcd:
        for i in range(m):
            # find GCD of the row
            gcd = np.gcd.reduce(A[i, :], axis=1).item()
            if gcd > 1:
                print(f"Row GCD: {gcd}")
                if b[i, 0].item() % gcd != 0:
                    raise ValueError("No integer solution exists (b not divisible by GCD)")
                # divide the row by the GCD
                A[i, :] //= gcd
                b[i, 0] //= gcd

    N1 = max(np.linalg.norm(b, np.inf).item(), np.linalg.norm(A, np.inf).item()) * 6
    N2 = N1 * 6
    B = np.block([[np.eye(n, dtype=np.int64), np.zeros((n, 1), dtype=np.int64)],
                        [np.zeros((1, n), dtype=np.int64), np.array([N1])],
                        [N2 * A, -N2 * b]]).astype(np.int64, order='C')
    # B = sp.block_array([[sp.eye(n), sp.csr_array((n, 1))],
    #                     [sp.csr_array((1, n)), N1],
    #                     [N2 * A, -N2 * b]])
    B_red = B.copy()
    rank, det, U = ntl.lll(B_red, 3, 4)
    assert B_red[n, n-m].item() == N1
    x_p = B_red[0:n, n-m]
    assert np.allclose(A @ x_p, b)
    null_space = B_red[0:n, 0:n-m]

    return x_p, null_space

def main():
    np.random.seed(42)
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    averages = {}
    for con_count in [1, 2, 3, 4]:
        for var_count in [10, 20, 50, 100]:
            print(f"Generating instances with {con_count} constraints and {var_count} variables")
            runs = 5
            before_times = []
            after_times = []
            instances = kl.generate(runs, con_count, var_count, 5, 10, 1000, equality=True, env=env)
            for model in instances:
                model.params.LogToConsole = 0
                # assumptions on the model: all equality constraints, fully linear objective & constraints, all vars >= 0, maximizing
                A = model.getA().todense()
                b = np.array(model.getAttr("RHS")).reshape((-1, 1))
                c = np.array(model.getAttr("Obj")).reshape((-1, 1))
                ub = np.array(model.getAttr("UB")).reshape((-1, 1))

                with lt.CodeTimer("Original optimization time", silent=True) as c1:
                    model.optimize()
                before_times.append(c1.took)
                if model.status != gp.GRB.Status.OPTIMAL:
                    raise ValueError(f"Original model not optimal: {status_lookup[model.status]}")
                # print(f"Original objective value: {model.ObjVal}")

                with lt.CodeTimer("  LLL time", silent=True) as c2:
                    xp, N = solve_via_LLL(A, b)
                if c2.took > 1000:
                    print(f"  LLL took too long: {c2.took:.2f} ms")
                if np.sum(xp < 0) > 0:
                    print(f"  Number of negative components: {np.sum(xp < 0)}")
                # xp, N = solve_via_snf(A, b)
                # now I have an integer null space and an integer starting solution (that may violate bounds)

                mdl2 = gp.Model(model.ModelName + "_tfrm", env=env)
                z = mdl2.addMVar((N.shape[1], 1), lb=-gp.GRB.INFINITY, vtype='I', name='z')
                mdl2.setObjective(c.T @ xp + c.T @ N @ z, gp.GRB.MAXIMIZE)
                mdl2.addConstr(xp + N@z >= 0)
                mdl2.addConstr(xp + N@z <= ub)
                # mdl2.params.NumericFocus = 3
                # mdl2.params.DualReductions = 0
                mdl2.params.LogToConsole = 0
                with lt.CodeTimer("Transformed optimization time", silent=True) as c1:
                    mdl2.optimize()
                after_times.append(c1.took)
                if mdl2.status != gp.GRB.Status.OPTIMAL:
                    raise ValueError(f"Transformation not optimal: {status_lookup[mdl2.status]}")
                # print("Objective value: ", mdl2.ObjVal)
                if not np.allclose(mdl2.ObjVal, model.ObjVal):
                    print(f"Objective values do not match: {mdl2.ObjVal} != {model.ObjVal}")
            print(f" Average original time: {np.mean(before_times):.8f} ms")
            print(f" Average transformed time: {np.mean(after_times):.8f} ms")
            averages[(con_count, var_count)] = (np.mean(before_times), np.mean(after_times))
            print()
    print("Averages:", averages)

if __name__ == "__main__":
    main()