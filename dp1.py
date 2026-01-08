import numpy as np
import gurobipy as gp
import knapsack_loader as kl


def ilp_dp(M, b, c, lb, ub, U):
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

    for i in range(1, n + 1):
        m_i = M[:, i - 1]
        c_i = c[i - 1]

        curr = dict()
        filtered = clipped = 0
        avg = 0 # np.percentile(list(DP.values()), 40)
        for b_prev, cost_prev in DP.items():
            b_prev = np.array(b_prev, dtype=np.int64)

            for k in range(lb[i - 1], ub[i - 1] + 1):
                b_next = b_prev + k * m_i

                # if (np.abs(b_next) > U).any():
                #     clipped += 1
                #     continue

                b_next = tuple(b_next)
                cost = cost_prev - k * c_i

                if cost < curr.get(b_next, avg):
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

def main():
    gp.setParam("OutputFlag", 0)
    instances = kl.generate(1, 2, 10, 2, 6, 30, equality=True, seed=42)
    instance = next(iter(instances))
    instance.optimize()
    assert instance.status == gp.GRB.Status.OPTIMAL
    print("Gurobi optimal value:", instance.ObjVal)
    
    A = instance.getA().toarray()
    b = np.array(instance.getAttr("RHS"), dtype=int)
    c = np.array(instance.getAttr("Obj"), dtype=int)
    lb = np.array(instance.getAttr("LB"), dtype=int)
    ub = np.array(instance.getAttr("UB"), dtype=int)
    # ub = better_ubs(instance)
    S = int(max(np.max(np.abs(A)), np.max(np.abs(b))))
    m, n = A.shape
    U = (n + 1) * S * (m * S) ** m
    print("U =", U)
    opt_val = ilp_dp(A, b, c, lb, ub, U)
    if opt_val is None:
        print("Infeasible")
    else:
        print("Optimal value:", opt_val)

if __name__ == "__main__":
    main()