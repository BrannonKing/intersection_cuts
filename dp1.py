import numpy as np
import gurobipy as gp
import knapsack_loader as kl
import jsplib_loader as jl
import timeit as ti


def ilp_dp(M, b, c, lb, ub, U, worst):
    """
    Solve max c^T x s.t. Mx = b, lb <= x <= ub, x integer
    using implicit DP (shortest path formulation).

    Returns: optimal value or None if infeasible
    """

    b = tuple(b)
    m, n = M.shape

    # start state
    DP = {}
    DP[tuple([0]*m)] = 0

    var_is = list(range(1, n + 1))
    # var_is.reverse()
    # var_is.sort(key=lambda i: -c[i - 1] / np.linalg.norm(M[:, i - 1], ord=1))
    var_is.sort(key=lambda i: ub[i - 1] - lb[i - 1], reverse=True)
    # var_is.sort(key=lambda i: np.linalg.norm(M[:, i - 1], 1))

    for i in var_is:
        m_i = M[:, i - 1]
        c_i = c[i - 1]

        curr = {}
        filtered = clipped = 0
        for k in range(lb[i - 1], ub[i - 1] + 1):
        # for k in range(ub[i - 1], lb[i - 1] - 1, -1):
            for b_prev, cost_prev in DP.items():
                # Use integer arithmetic and tuples to keep state keys exact.
                b_next = tuple(bp + k * mi for bp, mi in zip(b_prev, m_i))

                # where we're at: we need a better plan than using the key for large dimensions.
                # our key is size m, which is fine for small m. 
                # maybe we just hash the first 10 values? 
                # is b_next ever sparse?
                # and we need a structure that works on the GPU too.

                # if (np.abs(b_next) > U).any():
                #     clipped += 1
                #     continue
                # nnz = np.count_nonzero(b_next)
                # print("  State nnz:", nnz, "/", m)
                

                cost = cost_prev - k * c_i
                if cost < worst:
                    clipped += 1
                    continue

                prev_best = curr.get(b_next)
                if prev_best is None or cost < prev_best:
                    curr[b_next] = cost
                else:
                    filtered += 1

        DP = curr
        print(f"DP layer {i}/{n}, states: {len(DP)}, filtered: {filtered}, clipped: {clipped}")

        if not DP:
            return None  # infeasible early exit

    final_cost = DP.get(b, None)
    if final_cost is None:
        return None

    return -final_cost

def better_ubs(instance):
    ubs = []
    relaxed = instance.relax()
    for var in relaxed.getVars():
        relaxed.setObjective(var, gp.GRB.MAXIMIZE)
        relaxed.optimize()
        ubs.append(np.floor(var.X))
        print("UB:", var.VarName, ubs[-1], "vs.", var.UB)
    return np.array(ubs, dtype=int)

def better_bounds(instance):
    lbs = []
    ubs = []
    relaxed = instance.relax()
    relaxed.optimize()
    for var in relaxed.getVars():
        lbs.append(int(max(var.LB, np.floor(var.X) - 3)))
        ubs.append(int(min(var.UB, np.ceil(var.X) + 1)))
    return lbs, ubs
        

def main():
    gp.setParam("OutputFlag", 0)
    instances = kl.generate(1, 2, 12, 2, 6, 30, equality=True, seed=42)
    instance = next(iter(instances))
    instance.optimize()
    assert instance.status == gp.GRB.Status.OPTIMAL
    print("Gurobi optimal value:", instance.ObjVal)
    relaxed = instance.relax()
    relaxed.optimize()
    print("LP relaxation optimal value:", relaxed.ObjVal)
    
    A = instance.getA().toarray()
    b = np.array(instance.getAttr("RHS"), dtype=int)
    c = np.array(instance.getAttr("Obj"), dtype=int)
    lb = np.array(instance.getAttr("LB"), dtype=int)
    ub = np.array(instance.getAttr("UB"), dtype=int)
    # ub = better_ubs(instance)
    lb, ub = better_bounds(instance)
    S = int(max(np.max(np.abs(A)), np.max(np.abs(b))))
    m, n = A.shape
    U = (n + 1) * S * (m * S) ** m
    print("U =", U)
    time_start = ti.default_timer()
    opt_val = ilp_dp(A, b, c, lb, ub, U, -relaxed.ObjVal)
    time_end = ti.default_timer()
    print(f"ILP DP time: {time_end - time_start: .4f} seconds")
    if opt_val is None:
        print("Infeasible")
    else:
        print("Optimal value:", opt_val)

def main_jsp():
    gp.setParam("OutputFlag", 0)
    instances = jl.get_instances()
    instance = instances['abz3'].as_gurobi_balas_model(True)
    instance.optimize()
    assert instance.status == gp.GRB.Status.OPTIMAL
    print("Gurobi optimal value:", instance.ObjVal)
    relaxed = instance.relax()
    relaxed.optimize()
    print("LP relaxation optimal value:", relaxed.ObjVal)

    # verify that all constraint sense are >=
    for constr in instance.getConstrs():
        assert constr.Sense == gp.GRB.GREATER_EQUAL
    
    A = instance.getA().toarray()
    b = np.array(instance.getAttr("RHS"), dtype=int)
    c = np.array(instance.getAttr("Obj"), dtype=int)

    # we have to split this into two problems:
    # (a master that is all integer and a subproblem that is continuous):
    cols_of_integer = [var.index for var in instance.getVars() if var.VType != gp.GRB.CONTINUOUS]
    Ai = A[:, cols_of_integer]
    ci = c[cols_of_integer]
    
    Ar = A[:, [i for i in range(A.shape[1]) if i not in cols_of_integer]]
    cr = c[[i for i in range(A.shape[1]) if i not in cols_of_integer]]

    master = gp.Model()
    xi = master.addMVar(shape=(Ai.shape[1], 1), ub=1, vtype=gp.GRB.BINARY)
    slk = master.addMVar(shape=(b.shape[0], 1), ub=5000, vtype=gp.GRB.INTEGER)
    theta = master.addVar(ub=5000, vtype=gp.GRB.INTEGER)
    master.setObjective(ci @ xi + theta, gp.GRB.MINIMIZE)
    master.addConstr(Ai @ xi + slk >= b)
    master.update()

    # subprob = gp.Model()
    # xr = subprob.addMVar(shape=(Ar.shape[1], 1), lb=0, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
    # subprob.setObjective(cr @ xr, gp.GRB.MINIMIZE)
    # subprob.addConstr(Ar @ xr >= b - Ai @ xi.X)

    # if infeasible, add dual.T @ (b - Ai @ xi) <= 0 cut
    # else add optimality cut: theta >= dual.T @ (b - Ai @ xi)

    lb = [int(var.LB) for var in master.getVars()]
    ub = [int(var.UB) for var in master.getVars()]
    Ao = master.getA().toarray().astype(int)
    bo = np.array(master.getAttr("RHS"), dtype=int)
    co = np.array(master.getAttr("Obj"), dtype=int)

    S = int(max(np.max(np.abs(Ao)), np.max(np.abs(bo))))
    m, n = Ao.shape
    # U = (n + 1) * S * (m * S) ** m
    # print("U =", U, ", S =", S)
    time_start = ti.default_timer()
    opt_val = ilp_dp(Ao, bo, co, lb, ub, None, -relaxed.ObjVal)
    time_end = ti.default_timer()
    print(f"ILP DP time: {time_end - time_start: .4f} seconds")
    if opt_val is None:
        print("Infeasible")
    else:
        print("Optimal value:", opt_val)


if __name__ == "__main__":
    main_jsp()