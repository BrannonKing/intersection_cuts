import numpy as np
import scipy.sparse as sps

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
    if sps.issparse(A_eq):
        x_particular = sps.linalg.lsqr(A_eq, b_eq)[0]
    else:
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

def compute_constraint_violation(position, A_eq=None, b_eq=None, A_ineq=None, b_ineq=None, lb=None, ub=None, integers=None):
    """
    Compute total constraint violation for a point.
    Returns (continuous_violation, integer_violation) tuple.
    Note: Equality constraints are enforced via null space projection, not measured here.
    """
    continuous_violation = 0.0
    integer_violation = 0.0
    
    # Equality constraints are maintained via null space - no violation to measure
    
    if A_ineq is not None and b_ineq is not None:
        # Inequality constraints: Ax >= b (sum of negative violations)
        # Ensure position is 1D for sparse matrix multiplication
        Ax = A_ineq @ position
        assert Ax.shape == b_ineq.shape, "Dimension mismatch in inequality constraints"
        continuous_violation += np.sum(np.maximum(0, b_ineq - Ax))

    if lb is not None:
        assert lb.shape == position.shape, "Dimension mismatch in lower bounds"
        continuous_violation += np.sum(np.maximum(0, lb - position))
    
    if ub is not None:
        assert ub.shape == position.shape, "Dimension mismatch in upper bounds"
        continuous_violation += np.sum(np.maximum(0, position - ub))

    if integers is not None:
        for i in range(len(position)):
            if integers[i]:
                frac = abs(position[i] - round(position[i]))
                integer_violation += frac
    
    return float(continuous_violation), float(integer_violation)


def repair_continuous_vars(position, A_ineq, b_ineq, lb, ub, integers, objective_function, max_iters=50):
    """
    Given a position with fixed integers, find optimal continuous variables by solving LP.
    Uses scipy.optimize.linprog to minimize objective subject to Ax >= b with fixed integers.
    """
    from scipy.optimize import linprog
    
    pos = position.copy()
    n = len(pos)
    
    # Identify continuous and integer indices
    cont_indices = [i for i in range(n) if not integers[i]]
    int_indices = [i for i in range(n) if integers[i]]
    
    if len(cont_indices) == 0:
        return pos
    
    # Build LP: min c_cont @ x_cont subject to A_cont @ x_cont >= b - A_int @ x_int
    # And lb_cont <= x_cont <= ub_cont
    
    # Get the contribution from fixed integer variables
    if A_ineq is not None:
        if hasattr(A_ineq, 'toarray'):
            A_dense = A_ineq.toarray()
        else:
            A_dense = np.array(A_ineq)
        
        # Extract columns for continuous variables
        A_cont = A_dense[:, cont_indices]
        A_int = A_dense[:, int_indices]
        
        # Fixed contribution from integers
        x_int = pos[int_indices]
        b_adjusted = b_ineq - A_int @ x_int
        
        # For linprog: A_ub @ x <= b_ub, but we have A @ x >= b
        # So we use -A @ x <= -b
        A_ub = -A_cont
        b_ub = -b_adjusted
        
        # Bounds for continuous variables
        bounds = [(lb[i] if lb is not None else None, 
                   ub[i] if ub is not None else None) for i in cont_indices]
        
        # Objective coefficients for continuous variables (minimize objective)
        # The objective is c @ x, extract c for continuous variables
        # For JSP, objective is just c_max which is one of the continuous vars
        c_cont = np.zeros(len(cont_indices))
        for i, idx in enumerate(cont_indices):
            # Check if this variable has an objective coefficient
            test_pos = np.zeros(n)
            test_pos[idx] = 1.0
            c_cont[i] = objective_function(test_pos)
        
        # Solve LP
        try:
            result = linprog(c_cont, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if result.success:
                # Update continuous variables
                for i, idx in enumerate(cont_indices):
                    pos[idx] = result.x[i]
                return pos
        except Exception:
            pass
    
    # Fallback to simple repair if LP fails
    return _simple_repair(position, A_ineq, b_ineq, lb, ub, integers, max_iters)


def _simple_repair(position, A_ineq, b_ineq, lb, ub, integers, max_iters=50):
    """Simple iterative repair (fallback)."""
    pos = position.copy()
    cont_indices = [i for i in range(len(pos)) if not integers[i]]
    
    for _ in range(max_iters):
        if A_ineq is not None:
            Ax = A_ineq @ pos
            violations = b_ineq - Ax
        else:
            violations = np.zeros(1)
            
        lb_vio = np.maximum(0, lb - pos) if lb is not None else np.zeros_like(pos)
        ub_vio = np.maximum(0, pos - ub) if ub is not None else np.zeros_like(pos)
        
        total_vio = np.sum(np.maximum(0, violations)) + np.sum(lb_vio) + np.sum(ub_vio)
        if total_vio < 1e-6:
            break
            
        if A_ineq is not None:
            for c_idx in range(len(violations)):
                if violations[c_idx] > 1e-6:
                    row = A_ineq.getrow(c_idx) if hasattr(A_ineq, 'getrow') else A_ineq[c_idx]
                    if hasattr(row, 'toarray'):
                        row = row.toarray().flatten()
                    
                    for i in cont_indices:
                        if abs(row[i]) > 1e-6:
                            adjustment = violations[c_idx] / row[i] * 0.5
                            new_val = pos[i] + adjustment
                            if lb is not None:
                                new_val = max(new_val, lb[i])
                            if ub is not None:
                                new_val = min(new_val, ub[i])
                            pos[i] = new_val
                            break
        
        if lb is not None:
            pos = np.maximum(pos, lb)
        if ub is not None:
            pos = np.minimum(pos, ub)
    
    return pos


def compute_total_violation(position, A_eq=None, b_eq=None, A_ineq=None, b_ineq=None, lb=None, ub=None, integers=None):
    """Legacy wrapper returning total violation as single float."""
    cont_vio, int_vio = compute_constraint_violation(position, A_eq, b_eq, A_ineq, b_ineq, lb, ub, integers)
    return cont_vio + int_vio


def is_better(value1, cont_vio1, int_vio1, value2, cont_vio2, int_vio2):
    """
    Compare two points considering feasibility with separate continuous and integer violations.
    Returns True if point 1 is better than point 2.
    
    Rules:
    - Continuous feasibility is prioritized over integrality
    - Between two continuously feasible: smaller integer violation wins
    - Between two with same integer feasibility: smaller objective wins
    """
    cont_feasible1 = cont_vio1 < 1e-6
    cont_feasible2 = cont_vio2 < 1e-6
    
    # First priority: continuous constraint feasibility
    if cont_feasible1 and not cont_feasible2:
        return True
    if not cont_feasible1 and cont_feasible2:
        return False
    if not cont_feasible1 and not cont_feasible2:
        # Both infeasible: compare continuous violations
        return cont_vio1 < cont_vio2
    
    # Both continuously feasible - now compare integer feasibility
    int_feasible1 = int_vio1 < 1e-6
    int_feasible2 = int_vio2 < 1e-6
    
    if int_feasible1 and int_feasible2:
        # Both fully feasible: compare objective values
        return value1 < value2
    elif int_feasible1:
        return True
    elif int_feasible2:
        return False
    else:
        # Both have integer violations: prefer smaller violation, tie-break on objective
        if abs(int_vio1 - int_vio2) > 1e-6:
            return int_vio1 < int_vio2
        return value1 < value2

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
        self.best_cont_violation = float('inf')  # Best continuous constraint violation
        self.best_int_violation = float('inf')  # Best integer violation
    
    def get_full_position(self):
        """Convert reduced coordinates to full-space position."""
        if self.nullspace_basis is not None:
            result = self.x_particular + self.nullspace_basis @ self.alpha
            return result
        else:
            return self.alpha
    
    def get_best_full_position(self):
        """Get best position in full space."""
        if self.nullspace_basis is not None:
            result = self.x_particular + self.nullspace_basis @ self.best_alpha
            return result
        else:
            return self.best_alpha

    def update_velocity(self, global_best_alpha, inertia_weight, cognitive_coeff, social_coeff, 
                        integers, integer_gravity=2.0, min_integer_velocity=0.1):
        """
        Update velocity in reduced coordinates with integer gravity wells.
        
        Args:
            global_best_alpha: Global best position in reduced coordinates
            inertia_weight: Standard PSO inertia
            cognitive_coeff: Standard PSO cognitive coefficient
            social_coeff: Standard PSO social coefficient
            integers: Boolean array indicating which dimensions should be integer
            integer_gravity: Strength of attraction toward nearest integer (higher = stronger pull)
            min_integer_velocity: Minimum velocity magnitude when not at an integer
        """
        # Random coefficients in reduced space
        r1 = np.random.random(self.alpha.shape)
        r2 = np.random.random(self.alpha.shape)

        cognitive_velocity = cognitive_coeff * r1 * (self.best_alpha - self.alpha)
        social_velocity = social_coeff * r2 * (global_best_alpha - self.alpha)
        velocity = (inertia_weight * self.velocity_alpha) + cognitive_velocity + social_velocity
        
        # Apply integer gravity wells - creates attraction toward nearest integers
        if integers is not None:
            for i in range(len(velocity)):
                if integers[i]:
                    nearest_int = round(self.alpha[i])
                    dist_to_int = nearest_int - self.alpha[i]  # Signed distance
                    abs_dist = abs(dist_to_int)
                    
                    if abs_dist > 1e-6:  # Not at an integer
                        # Gravity well using a potential that creates strong pull near integers
                        # but also pushes away from the 0.5 boundary (unstable equilibrium)
                        # Force = gravity * sign(dist) * (1 - 2*|frac - 0.5|)^2
                        # This creates steeper wells near integers
                        frac = abs_dist  # fractional part (0 to 0.5 range effectively)
                        # Stronger pull as we get closer to integer, weaker at 0.5
                        well_strength = (1.0 - 2.0 * min(frac, 0.5)) ** 2
                        gravity_force = integer_gravity * np.sign(dist_to_int) * well_strength * abs_dist
                        
                        # Add gravity to velocity  
                        velocity[i] += gravity_force
                        
                        # Ensure minimum velocity toward the integer to prevent stagnation
                        if abs(velocity[i]) < min_integer_velocity:
                            velocity[i] = np.sign(dist_to_int) * min_integer_velocity
                        elif np.sign(velocity[i]) != np.sign(dist_to_int) and abs_dist < 0.25:
                            # Close to integer but moving away - redirect with some randomness
                            velocity[i] = np.sign(dist_to_int) * (abs(velocity[i]) + min_integer_velocity)

        self.velocity_alpha = velocity

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
        
        # Apply bounds
        if mode == 'clip':
            if lb is not None:
                assert lb.shape == x.shape, "Dimension mismatch in lower bounds"
                x = np.maximum(x, lb)
            if ub is not None:
                assert ub.shape == x.shape, "Dimension mismatch in upper bounds"
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
                 A_eq=None, b_eq=None, A_ineq=None, b_ineq=None, lb=None, ub=None, integers=None, 
                 bound_mode='clip', verbose=0, initial_hint=None):
    """
    PSO optimizer with constraint handling and integer gravity wells.
    
    Args:
        objective_function: Function to minimize (takes n-dimensional x)
        num_dimensions: Problem dimension
        num_particles: Number of particles in swarm
        max_iterations: Maximum iterations
        A_eq, b_eq: Equality constraints A_eq @ x == b_eq
        A_ineq, b_ineq: Inequality constraints A_ineq @ x >= b_ineq
        lb, ub: Variable bounds l <= x <= u (n-dimensional arrays or None)
        integers: Boolean array indicating which dimensions should be integer
        bound_mode: 'clip' or 'reflect' for handling bound violations
        initial_hint: Optional initial solution (e.g., from LP relaxation) to seed some particles
    """
    # Adaptive inertia weight: starts high (exploration) and decreases (exploitation)
    inertia_start = 0.9
    inertia_end = 0.4
    cognitive_coeff = 1.5
    social_coeff = 1.5
    
    # Integer gravity parameters - increase over time for stronger convergence
    integer_gravity_start = 2.0
    integer_gravity_end = 10.0
    min_integer_velocity = 0.15
    
    # Snapping threshold - snap to integer when very close
    snap_threshold = 0.05
    
    # Rounding probe: periodically round some particles (lightweight)
    rounding_probe_interval = 100
    rounding_probe_fraction = 0.1
    
    # Local refinement: try snapping individual integer coords to see if it improves
    local_refine_interval = 100

    # Compute null space basis for equality constraints
    nullspace_basis = compute_nullspace_basis(A_eq) if A_eq is not None else None
    
    # Find a particular solution satisfying equality constraints
    x_particular = find_feasible_point(A_eq, b_eq, num_dimensions)
    
    # Determine reduced dimension
    if nullspace_basis is not None:
        reduced_dim = nullspace_basis.shape[1]
    else:
        reduced_dim = num_dimensions
    
    particles: list[Particle] = []
    
    # Determine initialization ranges based on bounds
    # Since alpha is in null space, we can't easily map bounds to alpha generically.
    # However, if reduced_dim == num_dimensions (no equality constraints), we can.
    init_low = np.zeros(reduced_dim)
    init_high = np.ones(reduced_dim) # Default
    
    if nullspace_basis is None and reduced_dim == num_dimensions:
        # Direct mapping (no equality constraints)
        if lb is not None:
             # Use provided LB, handling -inf
            init_low = np.where(np.isneginf(lb), -100.0, lb)
            
        if ub is not None:
            # Use provided UB, handling inf
            # If UB is inf, use max(LB + 100, 100) as heuristic high
            heuristic_high = np.maximum(init_low + 200.0, 200.0) 
            init_high = np.where(np.isinf(ub), heuristic_high, ub)
            
        # Ensure low < high
        init_high = np.maximum(init_high, init_low + 1.0)
    else:
        # With nullspace, reduced coords are abstract. Keep standard normal but maybe wider?
        init_low = -10.0
        init_high = 10.0
    
    # Diversity restart interval - periodically reinitialize worst particles
    diversity_restart_interval = 100
    diversity_restart_fraction = 0.3
        
    for p_idx in range(num_particles):
        # Initialize in reduced coordinates
        if isinstance(init_low, np.ndarray):
            alpha = np.random.uniform(init_low, init_high)
        else:
            alpha = np.random.uniform(init_low, init_high, size=reduced_dim)
        
        # Seed some particles near the LP hint if provided
        if initial_hint is not None and p_idx < num_particles // 4:
            # Project hint to nullspace if needed
            if nullspace_basis is not None:
                alpha = project_point_to_nullspace(initial_hint, x_particular, nullspace_basis)
            else:
                alpha = initial_hint.copy()
            # Add some noise for diversity
            alpha = alpha + np.random.normal(0, 0.5, size=alpha.shape)
            # For hint-seeded particles, also try different roundings of integers
            if integers is not None and p_idx > 0:
                for j in range(len(alpha)):
                    if integers[j]:
                        # Randomly round up or down
                        if np.random.random() < 0.5:
                            alpha[j] = np.floor(alpha[j])
                        else:
                            alpha[j] = np.ceil(alpha[j])
        else:
            # For non-hint particles, start with random binary values for integer vars
            if integers is not None:
                for j in range(len(alpha)):
                    if integers[j]:
                        alpha[j] = float(np.random.randint(0, 2))
            
        # Velocity scale relative to range
        range_width = init_high - init_low
        velocity_alpha = np.random.uniform(-range_width*0.2, range_width*0.2, size=reduced_dim)
        
        particles.append(Particle(alpha, velocity_alpha, x_particular, nullspace_basis))
    
    # Repair a subset of particles at initialization to get good starting points
    if A_ineq is not None:
        num_to_repair = min(10, num_particles // 5)  # Repair 20% up to 10 particles
        for p_idx in range(num_to_repair):
            particle = particles[p_idx]
            pos = particle.get_full_position()
            repaired = repair_continuous_vars(pos, A_ineq, b_ineq, lb, ub, integers, 
                                              objective_function, max_iters=30)
            if nullspace_basis is not None:
                particle.alpha = project_point_to_nullspace(repaired, x_particular, nullspace_basis)
            else:
                particle.alpha = repaired.copy()

    # Initialize global best
    global_best_alpha = particles[0].alpha.copy()
    global_best_position = particles[0].get_full_position()
    global_best_value = objective_function(global_best_position)
    global_best_cont_vio, global_best_int_vio = compute_constraint_violation(
        global_best_position, A_eq, b_eq, A_ineq, b_ineq, lb, ub, integers)

    stable_iterations = 0
    
    for i in range(max_iterations):
        prev_best_value = global_best_value
        
        # Adaptive parameters: interpolate based on iteration progress
        progress = i / max_iterations
        inertia_weight = inertia_start - (inertia_start - inertia_end) * progress
        
        # Apply integer gravity scaled by feasibility, but with a minimum to keep particles moving toward integers
        # Even when infeasible, we want some integer pull to help find feasible integer points
        feasibility_factor = max(0.1, 1.0 - global_best_cont_vio / 100.0)  # Softer gating
        integer_gravity = (integer_gravity_start + (integer_gravity_end - integer_gravity_start) * progress) * feasibility_factor

        if verbose > 0 and i % verbose == 0:
            total_vio = global_best_cont_vio + global_best_int_vio
            print(f"Iter {i}: Val={global_best_value:.1f}, ContVio={global_best_cont_vio:.4f}, "
                  f"IntVio={global_best_int_vio:.4f}, Inertia={inertia_weight:.2f}, IntGrav={integer_gravity:.2f}")
        
        # Rounding probe: periodically round some particles (lightweight)
        # No expensive LP repair - just round and let PSO dynamics handle it
        if i > 0 and i % rounding_probe_interval == 0 and integers is not None:
            num_to_round = max(1, int(num_particles * rounding_probe_fraction))
            probe_indices = np.random.choice(num_particles, size=num_to_round, replace=False)
            for idx in probe_indices:
                particle = particles[idx]
                # Round all integer coordinates
                for j in range(len(particle.alpha)):
                    if integers[j]:
                        particle.alpha[j] = round(particle.alpha[j])
                # Give small velocity to continue exploring
                particle.velocity_alpha = np.random.uniform(-0.3, 0.3, size=len(particle.alpha))
            
        for particle in particles:
            # Snap integer coordinates when very close (before evaluation)
            if integers is not None:
                for j in range(len(particle.alpha)):
                    if integers[j]:
                        nearest_int = round(particle.alpha[j])
                        if abs(particle.alpha[j] - nearest_int) < snap_threshold:
                            particle.alpha[j] = nearest_int
                            particle.velocity_alpha[j] = 0.0  # Stop velocity at snapped position
            
            # Evaluate in full space
            current_position = particle.get_full_position()
            current_value = objective_function(current_position)
            current_cont_vio, current_int_vio = compute_constraint_violation(
                current_position, A_eq, b_eq, A_ineq, b_ineq, lb, ub, integers)

            if is_better(current_value, current_cont_vio, current_int_vio, 
                         particle.best_value, particle.best_cont_violation, particle.best_int_violation):
                particle.best_value = current_value
                particle.best_cont_violation = current_cont_vio
                particle.best_int_violation = current_int_vio
                particle.best_alpha = particle.alpha.copy()

            if is_better(current_value, current_cont_vio, current_int_vio,
                         global_best_value, global_best_cont_vio, global_best_int_vio):
                global_best_value = current_value
                global_best_cont_vio = current_cont_vio
                global_best_int_vio = current_int_vio
                global_best_alpha = particle.alpha.copy()
                global_best_position = current_position.copy()

        # Local refinement: try rounding individual integer coords on the best solution
        if i > 0 and i % local_refine_interval == 0 and integers is not None and global_best_cont_vio < 1e-6:
            # Try rounding each non-integer dimension one at a time
            test_alpha = global_best_alpha.copy()
            improved = False
            for j in range(len(test_alpha)):
                if integers[j] and abs(test_alpha[j] - round(test_alpha[j])) > 1e-6:
                    # Try rounding this dimension
                    old_val = test_alpha[j]
                    for round_dir in [round(old_val), int(old_val), int(old_val) + 1]:
                        test_alpha[j] = round_dir
                        if nullspace_basis is not None:
                            test_pos = x_particular + nullspace_basis @ test_alpha
                        else:
                            test_pos = test_alpha.copy()
                        test_val = objective_function(test_pos)
                        test_cont, test_int = compute_constraint_violation(
                            test_pos, A_eq, b_eq, A_ineq, b_ineq, lb, ub, integers)
                        if is_better(test_val, test_cont, test_int,
                                    global_best_value, global_best_cont_vio, global_best_int_vio):
                            global_best_value = test_val
                            global_best_cont_vio = test_cont
                            global_best_int_vio = test_int
                            global_best_alpha = test_alpha.copy()
                            global_best_position = test_pos.copy()
                            improved = True
                            break
                    if not improved:
                        test_alpha[j] = old_val  # Revert
            
            # If we improved, update a few particles to explore around the new best
            if improved:
                for k in range(min(5, len(particles))):
                    particles[k].alpha = global_best_alpha.copy()
                    particles[k].velocity_alpha = np.random.uniform(-0.5, 0.5, size=len(global_best_alpha))

        # Check for convergence
        # Compute swarm "energy" (average velocity)
        avg_velocity = np.mean([np.linalg.norm(p.velocity_alpha) for p in particles])
        total_violation = global_best_cont_vio + global_best_int_vio

        if total_violation < 1e-6 and abs(global_best_value - prev_best_value) < 1e-5 and avg_velocity < 0.1:
            stable_iterations += 1
            if stable_iterations >= 10:
                print(f"Converged: Stable optimum found at iteration {i} (Value: {global_best_value:.6f})")
                break
        else:
            stable_iterations = 0
        
        # Periodically try LP repair on global best (just once per interval, cheap)
        if i > 0 and i % 200 == 0 and A_ineq is not None and global_best_int_vio < 0.1:
            repaired = repair_continuous_vars(global_best_position, A_ineq, b_ineq, lb, ub, 
                                              integers, objective_function, max_iters=50)
            rep_val = objective_function(repaired)
            rep_cont, rep_int = compute_constraint_violation(repaired, A_eq, b_eq, A_ineq, b_ineq, lb, ub, integers)
            if is_better(rep_val, rep_cont, rep_int, global_best_value, global_best_cont_vio, global_best_int_vio):
                global_best_value = rep_val
                global_best_cont_vio = rep_cont
                global_best_int_vio = rep_int
                global_best_position = repaired.copy()
                if nullspace_basis is not None:
                    global_best_alpha = project_point_to_nullspace(repaired, x_particular, nullspace_basis)
                else:
                    global_best_alpha = repaired.copy()
                if verbose > 0:
                    print(f"  -> LP repair improved: {global_best_value:.1f}")

        for particle in particles:
            particle.update_velocity(global_best_alpha, inertia_weight, cognitive_coeff, social_coeff, 
                                     integers, integer_gravity, min_integer_velocity)
            particle.update_position()
            particle.enforce_bounds(lb, ub, mode=bound_mode)
    
    # Final LP repair on global best
    if A_ineq is not None:
        repaired = repair_continuous_vars(global_best_position, A_ineq, b_ineq, lb, ub, 
                                          integers, objective_function, max_iters=100)
        rep_val = objective_function(repaired)
        rep_cont, rep_int = compute_constraint_violation(repaired, A_eq, b_eq, A_ineq, b_ineq, lb, ub, integers)
        if is_better(rep_val, rep_cont, rep_int, global_best_value, global_best_cont_vio, global_best_int_vio):
            global_best_value = rep_val
            global_best_position = repaired.copy()
            if verbose > 0:
                print(f"Final LP repair improved to: {global_best_value:.1f}")

    return global_best_position, global_best_value
