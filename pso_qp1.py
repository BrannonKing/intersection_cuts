import numpy as np
import time
import particle_utils as pu
import gurobipy as gp

def generate_random_qp(num_dimensions, num_eq, num_ineq):
    """
    Generates a random quadratic programming problem.
    Minimize 0.5 * x^T Q x + c^T x
    Subject to:
        A_eq x = b_eq
        A_ineq x >= b_ineq
        lb <= x <= ub
    """
    # Generate random positive definite Q
    M = np.random.randn(num_dimensions, num_dimensions)
    Q = M.T @ M
    c = np.random.randn(num_dimensions)

    # Generate bounds
    lb = -10 + 20 * np.random.rand(num_dimensions)
    ub = lb + 10 * np.random.rand(num_dimensions)
    
    # Generate a feasible point strictly inside bounds to ensure feasibility
    # This guarantees that at least one solution exists
    x_feas = lb + (ub - lb) * np.random.rand(num_dimensions)
    
    # Generate equality constraints satisfied by x_feas
    if num_eq > 0:
        A_eq = np.random.randn(num_eq, num_dimensions)
        b_eq = A_eq @ x_feas
    else:
        A_eq = None
        b_eq = None
        
    # Generate inequality constraints satisfied by x_feas
    if num_ineq > 0:
        A_ineq = np.random.randn(num_ineq, num_dimensions)
        # Make sure A_ineq @ x_feas >= b_ineq
        # Let b_ineq = A_ineq @ x_feas - slack, where slack >= 0
        slack = np.random.rand(num_ineq)
        b_ineq = A_ineq @ x_feas - slack
    else:
        A_ineq = None
        b_ineq = None

    def objective(x):
        # return c.T @ x
        return 0.5 * x.T @ Q @ x + c.T @ x

    return objective, A_eq, b_eq, A_ineq, b_ineq, lb, ub, Q, c

def solve_with_gurobi(Q, c, A_eq, b_eq, A_ineq, b_ineq, lb, ub):
    """
    Solve the QP using Gurobi.
    Minimize 0.5 * x^T Q x + c^T x
    """

    model = gp.Model("qp")
    model.setParam('OutputFlag', 0)
    
    num_vars = len(c)
    x = model.addMVar(num_vars, lb=lb, ub=ub, name="x")
    
    # Set objective
    obj = 0.5 * x @ Q @ x + c @ x
    model.setObjective(obj, gp.GRB.MINIMIZE)
    
    # Add equality constraints
    if A_eq is not None:
        model.addConstr(A_eq @ x == b_eq, name="eq")
            
    # Add inequality constraints
    if A_ineq is not None:
        model.addConstr(A_ineq @ x >= b_ineq, name="ineq")
            
    model.optimize()
    
    if model.status == gp.GRB.OPTIMAL:
        x_sol = x.X
        return x_sol, model.ObjVal
    else:
        print(f"Gurobi failed to find optimal solution. Status: {model.status}")
        return None, None
            
def main():
    num_problems = 12
    
    print(f"Starting execution of {num_problems} random QP problems...")
    
    for i in range(num_problems):
        np.random.seed(42 + i)  # For reproducibility
        print(f"\n{'='*60}")
        print(f"Solving Random Problem {i+1}/{num_problems}")
        
        dims = np.random.randint(5, 15)
        
        # Ensure we don't have too many equality constraints (need some nullspace)
        max_eq = max(1, dims - 2)
        num_eq = np.random.randint(1, max_eq + 1)
        
        num_ineq = np.random.randint(1, dims * 2)
        
        print(f"  Dimensions: {dims}")
        print(f"  Equality Constraints: {num_eq}") 
        print(f"  Inequality Constraints: {num_ineq}")
        print(f"{'='*60}")
        
        objective, A_eq, b_eq, A_ineq, b_ineq, lb, ub, Q, c = generate_random_qp(dims, num_eq, num_ineq)
        
        # Run optimization
        start_time = time.perf_counter()
        best_position, best_value = pu.pso_optimize(
            objective, dims, 
            num_particles=50,      # Increased particle count for tougher problems
            max_iterations=2000,   # Increased iterations
            A_eq=A_eq, b_eq=b_eq, 
            A_ineq=A_ineq, b_ineq=b_ineq,
            lb=lb, ub=ub, 
            bound_mode='clip',
            verbose=100
        )
        end_time = time.perf_counter()
        
        # Analysis
        violation = pu.compute_constraint_violation(best_position, A_eq, b_eq, A_ineq, b_ineq, lb, ub)
        
        print(f"\n--- Problem {i+1} Results ---")
        print(f"  PSO Best objective value: {best_value:.6f}")
        print(f"  PSO Inequality Violation: {violation:.6e}")
        
        # Verify with Gurobi
        start_gur = time.perf_counter()
        gur_x, gur_obj = solve_with_gurobi(Q, c, A_eq, b_eq, A_ineq, b_ineq, lb, ub)
        end_gur = time.perf_counter()
        
        if gur_obj is not None:
             print(f"  Gurobi objective value:   {gur_obj:.6f}")
             print(f"  Difference (PSO - Gur):   {best_value - gur_obj:.6f}")
             print(f"  Gurobi Time: {end_gur - start_gur:.4f}s")
             
             dist_to_gur = np.linalg.norm(best_position - gur_x)
             print(f"  Distance to Gurobi sol:   {dist_to_gur:.6f}")
        else:
             print("  Gurobi validation skipped/failed.")

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