import numpy as np
import time
import gurobipy as gp
gp.setParam('OutputFlag', 0)

import particle_utils as pu
import jsplib_loader as jl
import gurobi_utils as gu

def split_jsp_problem(problem: jl.JspInstance):
    model = problem.as_gurobi_balas_model(use_big_m=True)
    A, b, c, l, u = gu.get_A_b_c_l_u(model, keep_sparse=True)
    integers = [False] * len(l)
    for v in model.getVars():
        if v.VType == gp.GRB.BINARY:
            integers[v.index] = True
            u[v.index] = 1.0
        elif v.VType == gp.GRB.INTEGER:
            integers[v.index] = True

    senses = [con.Sense for con in model.getConstrs()]
    assert all(s == gp.GRB.GREATER_EQUAL for s in senses), "Unexpected constraint sense"

    
    # For scheduling problems, replace infinite upper bounds with a reasonable estimate
    # The bigM value from the model is a good upper bound for start times
    bigM = model._bigM if hasattr(model, '_bigM') else 100000
    u = np.where(np.isinf(u), bigM, u)
    
    # Get LP relaxation solution as a starting hint
    lp_model = model.relax()
    lp_model.optimize()
    assert lp_model.status == gp.GRB.OPTIMAL
    lp_solution = np.array([v.X for v in lp_model.getVars()])
    
    objective = lambda x: (c.T @ x).item()
    return objective, None, None, A, b, l, u, integers, lp_solution

def main():
    instances = jl.get_instances()
    problems = [instances['abz3']] #, instances['abz4'], instances['abz5']]
    num_problems = len(problems)

    print(f"Starting execution of {num_problems} random QP problems...")
    
    for i in range(num_problems):
        np.random.seed(42 + i)  # For reproducibility
        
        objective, A_eq, b_eq, A_ineq, b_ineq, lb, ub, integers, lp_solution = split_jsp_problem(problems[i])
        # Ensure lb and ub are 1D to avoid broadcasting issues
        lb = lb.flatten()
        ub = ub.flatten()
        b_ineq = b_ineq.flatten()
        dims = len(lb)
        
        # Run optimization
        start_time = time.perf_counter()
        best_position, best_value = pu.pso_optimize(
            objective, dims, 
            num_particles=50,      # Increased particle count for better exploration
            max_iterations=300,    # Reduced iterations for faster feedback
            A_eq=A_eq, b_eq=b_eq, 
            A_ineq=A_ineq, b_ineq=b_ineq,
            lb=lb, ub=ub, 
            integers=integers,
            bound_mode='clip',
            verbose=50,
            initial_hint=lp_solution  # Start some particles near LP solution
        )
        end_time = time.perf_counter()
        
        # Analysis
        cont_vio, int_vio = pu.compute_constraint_violation(best_position, A_eq, b_eq, A_ineq, b_ineq, lb, ub, integers)
        
        print(f"\n--- Problem {i+1} Results ---")
        print(f"  Problem Name: {problems[i].name}. Target score: {problems[i].score}")
        print(f"  PSO Best objective value: {best_value:.6f}")
        print(f"  PSO Continuous Violation: {cont_vio:.6e}")
        print(f"  PSO Integer Violation: {int_vio:.6e}")
        
        if A_eq is not None:
             eq_vio = np.linalg.norm(A_eq @ best_position - b_eq)
             print(f"  Equality Violation (L2): {eq_vio:.6f}")
             
        # Check bounds violation
        lb_vio = np.sum(np.maximum(0, lb - best_position))
        ub_vio = np.sum(np.maximum(0, best_position - ub))
        print(f"  Bounds Violation: {lb_vio + ub_vio:.6f}")
        print(f"  PSO Time: {end_time - start_time:.4f}s")


if __name__ == "__main__":
    main()