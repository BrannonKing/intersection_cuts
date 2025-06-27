import dikin_utils as du
import gurobi_utils as gu
import numpy as np
import gurobipy as gp
import linetimer as lt
import ntl_wrapper as ntl
import knapsack_loader as kl
import scipy.linalg as spl
status_lookup = {getattr(gp.GRB.Status, k): k for k in gp.GRB.Status.__dir__() if "A" <= k[0] <= "Z"}

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

def substitute(env, model: gp.Model, U: np.ndarray, x0: np.ndarray):
    """
    Applies the transformation U to the model and returns the transformed model.
    Assumes the model is a knapsack model with all variables >= 0.
    """
    txfm = gp.Model(model.ModelName + "_transformed", env=env)
    A = model.getA().toarray()
    b = np.array(model.getAttr("RHS")).reshape((-1, 1))
    lb = np.array(model.getAttr("LB")).reshape((-1, 1))
    ub = np.array(model.getAttr("UB")).reshape((-1, 1))
    c = np.array(model.getAttr("Obj")).reshape((-1, 1))
    n = A.shape[1]
    assert c.shape == (n, 1), "Objective coefficients must be a column vector"
    y = txfm.addMVar((n, 1), lb=-gp.GRB.INFINITY, vtype='I', name="y")
    txfm.setObjective(c.T @ (U @ y + x0) + model.ObjCon, gp.GRB.MAXIMIZE)
    txfm.addConstr(A @ (U @ y + x0) == b, name="constraints")
    txfm.addConstr(lb <= U @ y + x0, name="lower_bounds")
    txfm.addConstr(U @ y + x0 <= ub, name="upper_bounds")   
    return txfm

def find_U(H: np.ndarray):
    H = H.astype(np.int64, copy=True, order='C')
    rank, det, U = ntl.lll(H, 3, 4)  # modifies H in place
    return U

def main():
    np.random.seed(42)
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    verify = True
    for con_count in [1, 2, 3, 4]:
        for var_count in [20, 30]: # 50, 100]:
            print(f"Generating instances with {con_count} constraints and {var_count} variables")
            runs = 5
            orig_times = []
            txfm_times = []
            instances = kl.generate(runs, con_count, var_count, 5, 10, 1000, equality=True, env=env)
            for model in instances:
                model.params.LogToConsole = 0
                # assumptions on the model: all equality constraints, fully linear objective & constraints, all vars >= 0, maximizing
                x0 = relaxed_optimum(model)
                grown = gu.relax_and_grow(model, x0, 1)
                A = grown.getA().toarray()
                b = np.array(grown.getAttr("RHS")).reshape((-1, 1))
                lb = np.array(grown.getAttr("LB")).reshape((-1, 1))
                ub = np.array(grown.getAttr("UB")).reshape((-1, 1))

                H = du.compute_H(A, b, lb, ub, x0)
                L = np.linalg.cholesky(H)
                H = np.linalg.inv(L.T)
                H = H.real

                U = find_U(H)
                # U2 = np.eye(U.shape[0], dtype=U.dtype)
                # if np.not_equal(U, U2).any():
                #     print(f"U is not the identity matrix: {np.sum(np.not_equal(U, U2))} entries differ")

                # x0 = np.zeros_like(x0, dtype=np.int64)
                # assert np.isclose(abs(np.linalg.det(U)), 1), "U is not unimodular; something went wrong"
                # txfm = gu.apply_transform(model, U, x0)
                # things to try: use the actual optimal point, 
                # figure out what axis-alginment means here, and how to measure it,
                # ensure that our cols/rows are correct, aka, try the transposed U,

                x0 = np.round(x0)
                txfm = substitute(env, model, U, x0)
                # txfm.params.DualReductions = 0
                txfm.params.LogToConsole = 0
                with lt.CodeTimer("  Transformation time", silent=True) as c1:
                    txfm.optimize()
                txfm_times.append(c1.took)
                if txfm.status != gp.GRB.Status.OPTIMAL:
                    print(f"Model {model.ModelName} did not solve to optimality after transformation: {status_lookup[txfm.Status]}")
                    if txfm.Status == gp.GRB.INTERRUPTED:
                        return
                    continue

                if verify:
                    with lt.CodeTimer("  Verification time", silent=True) as c2:
                        model.optimize()
                    orig_times.append(c2.took)
                    c = np.array(model.getAttr("Obj")).reshape((-1, 1))
                    x1 = np.array(txfm.getAttr("X")).reshape((-1, 1))
                    obj = c.T @ (U @ x1 + x0) + model.ObjCon
                    print(f"Name: {model.ModelName}. Actual: {model.ObjVal} in {c2.took:.3f}. Transformed: {obj.item()} in {c1.took:.3f}.")
            print(f"Average transformation time: {np.mean(txfm_times):.3f} ms")
            if verify:
                print(f"  Average original solve time: {np.mean(orig_times):.3f} ms")

            print()

if __name__ == "__main__":
    main()