import numpy as np
import time

def compute_nullspace_basis(A_eq):
    """
    Compute orthonormal basis for the null space of A_eq.
    Returns matrix Z where columns span null(A_eq).
    """
    if A_eq is None or A_eq.shape[0] == 0:
        return None
    
    # SVD: A = U @ S @ Vt
    # Null space is spanned by columns of V corresponding to zero singular values
    U, S, Vt = np.linalg.svd(A_eq, full_matrices=True)
    
    # Tolerance for zero singular values
    tol = max(A_eq.shape) * np.finfo(float).eps * S[0] if len(S) > 0 else 1e-10
    rank = np.sum(S > tol)
    
    # Columns of V (rows of Vt) corresponding to zero singular values
    Z = Vt[rank:].T
    
    return Z if Z.shape[1] > 0 else None

def find_feasible_point(A_eq, b_eq, num_dimensions):
    """
    Find a particular solution to A_eq @ x = b_eq.
    Uses least-squares solution.
    """
    if A_eq is None or b_eq is None:
        return np.zeros(num_dimensions)
    
    # Least squares solution: x = A^+ @ b where A^+ is pseudoinverse
    x_particular = np.linalg.lstsq(A_eq, b_eq, rcond=None)[0]
    return x_particular

def project_point_to_nullspace(x, x_particular, nullspace_basis):
    """
    Project point x onto the equality constraint manifold.
    Returns alpha such that x_projected = x_particular + Z @ alpha is closest to x.
    """
    if nullspace_basis is None:
        return x
    
    # Closest point on manifold: x_proj = x_p + Z @ Z^T @ (x - x_p)
    # Return alpha = Z^T @ (x - x_p)
    return nullspace_basis.T @ (x - x_particular)

def compute_constraint_violation(position, A_eq=None, b_eq=None, A_ineq=None, b_ineq=None, lb=None, ub=None):
    """
    Compute total constraint violation for a point.
    Returns 0.0 if feasible, positive value if infeasible.
    Note: Equality constraints are enforced via null space projection, not measured here.
    """
    violation = 0.0
    
    # Equality constraints are maintained via null space - no violation to measure
    
    if A_ineq is not None and b_ineq is not None:
        # Inequality constraints: Ax >= b (sum of negative violations)
        violation += np.sum(np.maximum(0, b_ineq - A_ineq @ position))

    if lb is not None:
        violation += np.sum(np.maximum(0, lb - position))
    
    if ub is not None:
        violation += np.sum(np.maximum(0, position - ub))
    
    return violation

def is_better(value1, violation1, value2, violation2):
    """
    Compare two points considering feasibility.
    Returns True if point 1 is better than point 2.
    
    Rules:
    - Feasible always beats infeasible
    - Between two infeasible: smaller violation wins
    - Between two feasible: smaller objective wins (minimization)
    """
    feasible1 = violation1 == 0.0
    feasible2 = violation2 == 0.0
    
    if feasible1 and feasible2:
        # Both feasible: compare objective values
        return value1 < value2
    elif feasible1:
        # Point 1 feasible, point 2 infeasible: point 1 wins
        return True
    elif feasible2:
        # Point 1 infeasible, point 2 feasible: point 2 wins
        return False
    else:
        # Both infeasible: compare constraint violations
        return violation1 < violation2

class Particle:
    def __init__(self, alpha, velocity_alpha, x_particular, nullspace_basis):
        """
        Particle working in null-space coordinates.
        
        Args:
            alpha: Reduced coordinates (k-dimensional)
            velocity_alpha: Velocity in reduced coordinates
            x_particular: Particular solution to equality constraints
            nullspace_basis: Null space basis matrix Z (n x k)
        """
        self.alpha = alpha  # Reduced coordinates
        self.velocity_alpha = velocity_alpha  # Velocity in reduced space
        self.x_particular = x_particular
        self.nullspace_basis = nullspace_basis
        
        self.best_alpha = alpha.copy()  # Best reduced coordinates found
        self.best_value = float('inf')  # Best objective value
        self.best_violation = float('inf')  # Best constraint violation
    
    def get_full_position(self):
        """Convert reduced coordinates to full-space position."""
        if self.nullspace_basis is not None:
            return self.x_particular + self.nullspace_basis @ self.alpha
        else:
            return self.alpha
    
    def get_best_full_position(self):
        """Get best position in full space."""
        if self.nullspace_basis is not None:
            return self.x_particular + self.nullspace_basis @ self.best_alpha
        else:
            return self.best_alpha

    def update_velocity(self, global_best_alpha, inertia_weight, cognitive_coeff, social_coeff):
        """Update velocity in reduced coordinates."""
        # Random coefficients in reduced space
        r1 = np.random.random(self.alpha.shape)
        r2 = np.random.random(self.alpha.shape)

        cognitive_velocity = cognitive_coeff * r1 * (self.best_alpha - self.alpha)
        social_velocity = social_coeff * r2 * (global_best_alpha - self.alpha)

        self.velocity_alpha = (inertia_weight * self.velocity_alpha) + cognitive_velocity + social_velocity

    def update_position(self):
        """Update position in reduced coordinates."""
        self.alpha += self.velocity_alpha
    
    def enforce_bounds(self, lb, ub, mode='clip'):
        """
        Enforce variable bounds l <= x <= u.
        
        Args:
            lb: Lower bounds (n-dimensional)
            ub: Upper bounds (n-dimensional)
            mode: 'clip' or 'reflect'
        """
        if lb is None and ub is None:
            return
        
        # Convert to full space
        x = self.get_full_position()
        x_old = x.copy()
        
        # Apply bounds
        if mode == 'clip':
            if lb is not None:
                x = np.maximum(x, lb)
            if ub is not None:
                x = np.minimum(x, ub)
            
            # Project back to null space
            self.alpha = project_point_to_nullspace(x, self.x_particular, self.nullspace_basis)
            
        elif mode == 'reflect':
            # Reflect velocity for violated bounds
            if self.nullspace_basis is not None:
                # Get full-space velocity
                v_full = self.nullspace_basis @ self.velocity_alpha
                
                # Check and reflect each dimension
                if lb is not None:
                    mask_lower = x < lb
                    x[mask_lower] = lb[mask_lower]
                    v_full[mask_lower] *= -1
                
                if ub is not None:
                    mask_upper = x > ub
                    x[mask_upper] = ub[mask_upper]
                    v_full[mask_upper] *= -1
                
                # Project corrected position and velocity back to null space
                self.alpha = project_point_to_nullspace(x, self.x_particular, self.nullspace_basis)
                self.velocity_alpha = self.nullspace_basis.T @ v_full
            else:
                # No null space - work directly in full space
                if lb is not None:
                    mask_lower = x < lb
                    x[mask_lower] = lb[mask_lower]
                    self.velocity_alpha[mask_lower] *= -1
                
                if ub is not None:
                    mask_upper = x > ub
                    x[mask_upper] = ub[mask_upper]
                    self.velocity_alpha[mask_upper] *= -1
                
                self.alpha = x
        else:
            raise ValueError(f"Unknown bound mode: {mode}. Use 'clip' or 'reflect'.")

def pso_optimize(objective_function, num_dimensions, num_particles=30, max_iterations=100, 
                 A_eq=None, b_eq=None, A_ineq=None, b_ineq=None, lb=None, ub=None, bound_mode='clip'):
    """
    PSO optimizer with constraint handling.
    
    Args:
        objective_function: Function to minimize (takes n-dimensional x)
        num_dimensions: Problem dimension
        num_particles: Number of particles in swarm
        max_iterations: Maximum iterations
        A_eq, b_eq: Equality constraints A_eq @ x == b_eq
        A_ineq, b_ineq: Inequality constraints A_ineq @ x >= b_ineq
        lb, ub: Variable bounds l <= x <= u (n-dimensional arrays or None)
        bound_mode: 'clip' or 'reflect' for handling bound violations
    """
    inertia_weight = 0.5
    cognitive_coeff = 1.5
    social_coeff = 1.5

    # Compute null space basis for equality constraints
    nullspace_basis = compute_nullspace_basis(A_eq) if A_eq is not None else None
    
    # Find a particular solution satisfying equality constraints
    x_particular = find_feasible_point(A_eq, b_eq, num_dimensions)
    
    # Determine reduced dimension
    if nullspace_basis is not None:
        reduced_dim = nullspace_basis.shape[1]
    else:
        reduced_dim = num_dimensions
    
    particles = []
    for _ in range(num_particles):
        # Initialize in reduced coordinates
        alpha = np.random.randn(reduced_dim)
        velocity_alpha = np.random.randn(reduced_dim) * 0.1
        
        particles.append(Particle(alpha, velocity_alpha, x_particular, nullspace_basis))

    # Initialize global best
    global_best_alpha = particles[0].alpha.copy()
    global_best_position = particles[0].get_full_position()
    global_best_value = objective_function(global_best_position)
    global_best_violation = compute_constraint_violation(global_best_position, A_eq, b_eq, A_ineq, b_ineq, lb, ub)

    stable_iterations = 0
    
    for i in range(max_iterations):
        prev_best_value = global_best_value

        if i % 10 == 0:
            print(f"Iteration {i}: Best Value = {global_best_value:.6f}, Violation = {global_best_violation:.6f}")
            
        for particle in particles:
            # Evaluate in full space
            current_position = particle.get_full_position()
            current_value = objective_function(current_position)
            current_violation = compute_constraint_violation(current_position, A_eq, b_eq, A_ineq, b_ineq, lb, ub)

            if is_better(current_value, current_violation, particle.best_value, particle.best_violation):
                particle.best_value = current_value
                particle.best_violation = current_violation
                particle.best_alpha = particle.alpha.copy()

            if is_better(current_value, current_violation, global_best_value, global_best_violation):
                global_best_value = current_value
                global_best_violation = current_violation
                global_best_alpha = particle.alpha.copy()
                global_best_position = current_position.copy()

        # Check for convergence
        if global_best_violation == 0.0 and abs(global_best_value - prev_best_value) < 1e-9:
            stable_iterations += 1
            if stable_iterations >= 10:
                print(f"Converged: Stable optimum found at iteration {i} (Value: {global_best_value:.6f})")
                break
        else:
            stable_iterations = 0

        for particle in particles:
            particle.update_velocity(global_best_alpha, inertia_weight, cognitive_coeff, social_coeff)
            particle.update_position()
            particle.enforce_bounds(lb, ub, mode=bound_mode)

    return global_best_position, global_best_value

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
        return 0.5 * x.T @ Q @ x + c.T @ x

    return objective, A_eq, b_eq, A_ineq, b_ineq, lb, ub

def main():
    np.random.seed(42)  # For reproducibility
    num_problems = 12
    
    print(f"Starting execution of {num_problems} random QP problems...")
    
    for i in range(num_problems):
        print(f"\n{'='*60}")
        print(f"Solving Random Problem {i+1}/{num_problems}")
        
        dims = np.random.randint(5, 15)
        
        # Ensure we don't have too many equality constraints (need some nullspace)
        max_eq = max(1, dims - 2)
        num_eq = np.random.randint(1, max_eq + 1)
        
        num_ineq = np.random.randint(1, dims * 2)
        
        print(f"Dimensions: {dims}")
        print(f"Equality Constraints: {num_eq}") 
        print(f"Inequality Constraints: {num_ineq}")
        print(f"{'='*60}")
        
        objective, A_eq, b_eq, A_ineq, b_ineq, lb, ub = generate_random_qp(dims, num_eq, num_ineq)
        
        # Run optimization
        start_time = time.perf_counter()
        best_position, best_value = pso_optimize(
            objective, dims, 
            num_particles=50,      # Increased particle count for tougher problems
            max_iterations=2000,   # Increased iterations
            A_eq=A_eq, b_eq=b_eq, 
            A_ineq=A_ineq, b_ineq=b_ineq,
            lb=lb, ub=ub, 
            bound_mode='clip'
        )
        end_time = time.perf_counter()
        
        # Analysis
        violation = compute_constraint_violation(best_position, A_eq, b_eq, A_ineq, b_ineq)
        
        print(f"\n--- Problem {i+1} Results ---")
        print(f"Best objective value: {best_value:.6f}")
        print(f"Inequality Violation: {violation:.6e}")
        
        if A_eq is not None:
             eq_vio = np.linalg.norm(A_eq @ best_position - b_eq)
             print(f"Equality Violation (L2): {eq_vio:.6e}")
             
        # Check bounds violation
        lb_vio = np.sum(np.maximum(0, lb - best_position))
        ub_vio = np.sum(np.maximum(0, best_position - ub))
        print(f"Bounds Violation: {lb_vio + ub_vio:.6e}")
        print(f"Time: {end_time - start_time:.4f}s")


if __name__ == "__main__":
    main()