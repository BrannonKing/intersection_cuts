import numpy as np
import time
import jsplib_loader as jl
import gurobi_utils as gu
import gurobipy as gp

def split_jsp_problem(problem: jl.JspInstance):
    model = problem.as_gurobi_balas_model(use_big_m=True)
    A, b, c, l, u = gu.get_A_b_c_l_u(model, keep_sparse=True)
    # Replicate the logic from pso_jsp1.py
    for v in model.getVars():
        if v.VType == gp.GRB.BINARY:
            u[v.index] = 1.0
    senses = [con.Sense for con in model.getConstrs()]
    assert all(s == gp.GRB.GREATER_EQUAL for s in senses), "Unexpected constraint sense"
    objective_func = lambda x: (c.T @ x).item()
    return model, objective_func, A, b, c, l, u

def main():
    instances = jl.get_instances()
    problem = instances['abz3'] # Same instance as pso_jsp1.py
    
    print("Preparing model...")
    model, _, A, b, c, l, u = split_jsp_problem(problem)
    
    # 1. Solve LP Relaxation with Gurobi
    print("\n--- Solving LP Relaxation with Gurobi ---")
    lp_model = model.copy()
    for v in lp_model.getVars():
        v.VType = gp.GRB.CONTINUOUS
        # Apply the explicit bounds we extracted
        # Note: Gurobi variables have their own bounds, but let's ensure they match l/u used in PSO
        # split_jsp_problem modified 'u' for binary vars, let's verify if that matches what we want.
        # Ideally we just relax the model in place.
        pass

    lp_model.optimize()
    
    if lp_model.Status == gp.GRB.OPTIMAL:
        print(f"Gurobi LP Optimum: {lp_model.ObjVal}")
        
        # Analyze solution scale
        x_sol = np.array([v.X for v in lp_model.getVars()])
        print(f"Solution range: [{x_sol.min()}, {x_sol.max()}]")
        print(f"Solution mean: {x_sol.mean()}")
        print(f"Indices with values > 10: {np.where(x_sol > 10)[0]}")
        print(f"Values > 10:\n{x_sol[x_sol > 10]}")
        
        # Check initialization scale assumption
        # PSO init is np.random.randn(dim) which is ~[-3, 3]
        if x_sol.max() > 10:
            print("\n[DIAGNOSIS] The solution requires values much larger than the PSO initialization range.")
            print("PSO particles initialized at ~0 might be struggling to reach this region.")
            
    else:
        print(f"Gurobi failed to solve LP. Status: {lp_model.Status}")

    # 2. Check Feasibility of PSO Bounds
    # PSO enforces l <= x <= u.
    # l is from model.LB, u is from model.UB (modified).
    # Let's check if the variables with large values in solution have high enough upper bounds.
    l = l.flatten()
    u = u.flatten()
    
    if lp_model.Status == gp.GRB.OPTIMAL:
        violations = []
        for i, val in enumerate(x_sol):
            if not (l[i] - 1e-5 <= val <= u[i] + 1e-5):
                violations.append((i, l[i], val, u[i]))
        
        if violations:
            print(f"\n[DIAGNOSIS] Gurobi solution violates the bounds extracted for PSO!")
            for v in violations[:10]:
                print(f"Var {v[0]}: LB={v[1]}, Val={v[2]}, UB={v[3]}")
        else:
             print("\n[CONFIRMED] Gurobi solution is within the bounds passed to PSO.")


if __name__ == "__main__":
    main()
