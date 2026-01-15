import numpy as np
import random as rnd

def aco_mip_optimizer(
    obj_func,          # Objective function: takes np.array x, returns float (minimize)
    viol_func,         # Violation measure: takes np.array x, returns float >=0 (0 if feasible)
    lb,                # List of lower bounds for each variable
    ub,                # List of upper bounds for each variable
    var_types,         # List of 'binary', 'integer', 'real' for each variable
    relaxed_opt=None,  # Optional relaxed optimum np.array
    num_ants=50,       # Number of ants per iteration
    archive_size=50,   # Size of solution archive (k)
    max_iter=200,      # Maximum iterations
    xi=0.85,           # Parameter for sigma calculation (exploration speed)
    penalty_factor=1e6,# Static penalty factor (can be tuned or made adaptive)
    oracle=None,       # Optional oracle value for oracle penalty (estimate of optimal obj)
    seed=42            # Random seed for reproducibility
):
    """
    Ant Colony Optimization for Mixed-Integer Problems with sampling distributions.
    Handles binary, integer, and real variables. Uses penalty for constraints.
    If oracle is provided, uses a simple oracle-inspired penalty; else static.
    
    Returns: best_solution (np.array), best_obj (float)
    """
    np.random.seed(seed)
    stagnation_counter = 0
    max_stagnation = 100
    
    n_vars = len(lb)
    r_vars = np.arange(n_vars)
    assert len(var_types) == n_vars, "var_types must match number of variables"
    
    # Convert bounds to numpy arrays for vectorized operations
    lb = lb.flatten()
    ub = ub.flatten()
    
    # Cache indices for different variable types (ensure integer dtype)
    real_idx = np.array([i for i, vt in enumerate(var_types) if vt == 'real'], dtype=int)
    int_idx = np.array([i for i, vt in enumerate(var_types) if vt == 'integer'], dtype=int)
    binary_idx = np.array([i for i, vt in enumerate(var_types) if vt == 'binary'], dtype=int)
    non_real_idx = np.concatenate([int_idx, binary_idx]) if len(int_idx) + len(binary_idx) > 0 else np.array([], dtype=int)
    
    # Helper to generate initial random solution respecting bounds and types
    def random_solution():
        x = np.zeros(n_vars)
        # Generate all real variables at once
        if len(real_idx) > 0:
            x[real_idx] = np.random.uniform(lb[real_idx], ub[real_idx])
        # Generate all integer/binary variables at once
        if len(non_real_idx) > 0:
            x[non_real_idx] = np.random.randint(lb[non_real_idx].astype(int), ub[non_real_idx].astype(int) + 1)
        return x
    
    # Helper to discretize x based on types (round integers/binaries)
    def discretize(x):
        # Round all non-real variables at once
        if len(non_real_idx) > 0:
            x[non_real_idx] = np.round(x[non_real_idx])
        # Clip binary variables to [0, 1]
        if len(binary_idx) > 0:
            x[binary_idx] = np.clip(x[binary_idx], 0, 1)
        # Clip all non-real variables to their bounds
        if len(non_real_idx) > 0:
            x[non_real_idx] = np.clip(x[non_real_idx], lb[non_real_idx], ub[non_real_idx])
        return x
    
    # Helper to clip to bounds
    def clip_to_bounds(x):
        return np.clip(x, lb, ub)
    
    def initialize_archive():
        archive = []
        
        if relaxed_opt is not None:
            # Add the relaxed optimum itself
            x_rel = clip_to_bounds(relaxed_opt.copy())
            x_rel = discretize(x_rel)  # round integers/binaries
            archive.append(x_rel)
            
            # Add a few small perturbations (diversification)
            for _ in range(min(archive_size // 3, 10)):  # e.g. up to ~1/3 of archive
                pert = x_rel + np.random.normal(0, scale=0.5 + 1e-3*np.random.rand(n_vars))
                pert = clip_to_bounds(pert)
                pert = discretize(pert)
                archive.append(pert)
        
        # Fill the rest with uniform random (or Latin Hypercube if you want better space coverage)
        while len(archive) < archive_size:
            archive.append(random_solution())
        
        return archive[:archive_size]
    
    # Penalized fitness
    def penalized_fitness(x):
        obj = obj_func(x)
        viol = viol_func(x)
        if oracle is not None:
            # Simple oracle penalty: heavy penalty if obj > oracle, else focus on viol
            if obj > oracle:
                penalty = penalty_factor * viol + (obj - oracle)
            else:
                penalty = penalty_factor * viol
            return obj + penalty
        else:
            return obj + penalty_factor * viol
    
    # Initialize archive with random solutions
    archive = initialize_archive()
    fitness = [penalized_fitness(x) for x in archive]
    # Sort archive by fitness (minimize)
    sorted_idx = np.argsort(fitness)
    archive = [archive[i] for i in sorted_idx]
    fitness = [fitness[i] for i in sorted_idx]
    
    best_x = archive[0].copy()
    best_f = obj_func(best_x)  # True obj, not penalized
    
    for iter in range(max_iter):
        # Compute weights: linear based on rank (better = higher weight)
        ranks = np.arange(1, archive_size + 1)
        weights = (archive_size - ranks + 1) / np.sum(ranks)
        
        # Compute sigmas for each kernel (per dimension)
        sigmas = np.zeros((archive_size, n_vars))
        archive_array = np.array(archive)  # Convert to 2D array once for efficiency
        for j in range(n_vars):
            # Average distance to other points in archive for that dim
            # Use vectorized numpy operations instead of list comprehension
            vals = archive_array[:, j]
            pairwise_diffs = np.abs(vals[:, None] - vals)  # Broadcasting for all pairs
            mean_dist = (np.sum(pairwise_diffs) - np.trace(pairwise_diffs)) / (archive_size * (archive_size - 1))
            sigmas[:, j] = xi * mean_dist
            # Enforce min sigma for integers to avoid zero variance
            if var_types[j] != 'real':
                sigmas[:, j] = np.maximum(sigmas[:, j], 1.0)
        
        # Generate new solutions (ants)
        new_solutions = []
        new_fitness = []
        for _ in range(num_ants):
            # Vectorized kernel selection across dimensions
            l_selected = np.random.choice(archive_size, size=n_vars, p=weights)
            mu = archive_array[l_selected, r_vars]
            sigma = sigmas[l_selected, r_vars]
            x_new = np.random.normal(mu, sigma)
            # Clip to bounds and discretize
            x_new = clip_to_bounds(x_new)
            x_new = discretize(x_new)
            new_solutions.append(x_new)
            new_fitness.append(penalized_fitness(x_new))
        
        # Combine archive + new, sort, keep top k
        combined_sols = archive + new_solutions
        combined_fit = fitness + new_fitness
        sorted_idx = np.argsort(combined_fit)
        archive = [combined_sols[i] for i in sorted_idx[:archive_size]]
        fitness = [combined_fit[i] for i in sorted_idx[:archive_size]]
        
        # Update best (using true obj for reporting)
        current_best_f = obj_func(archive[0])
        improved = current_best_f < best_f - 1e-12
        if improved:
            best_x = archive[0].copy()
            best_f = current_best_f
            print(f"Iter {iter+1}: New best obj = {best_f:.6f}")
        
        # Optional: decrease xi over time for more exploitation
        xi *= 0.995  # Gradual reduction

        # print(f"Iter {iter+1}: Best Obj = {best_f:.6f}, Current Best = {current_best_f:.6f}, Xi = {xi:.4f}")
        stagnation_counter = 0 if improved else stagnation_counter + 1
        if (stagnation_counter >= max_stagnation):
            print("Stopping due to stagnation at", iter)
            break

        if (iter & 0x0F) == 0x0F:
            current_max_sigma = np.max(sigmas)
            if current_max_sigma < 1e-5:
                print("Stopping due to low sigma at", iter)
                break
            archive_spread = np.mean(np.std(np.array(archive), axis=0))
            if archive_spread < 1e-4:
                print("Stopping due to low archive spread at", iter)
                break
    
    return best_x, best_f

def test():
    import knapsack_loader as kl
    import gurobi_utils as gu
    gu.gp.setParam('OutputFlag', 0)
    instances = kl.generate(4, 2, 20, 5, 10, 1000, seed=42)
    for instance in instances:
        print("\nTesting instance:", instance.ModelName)
        relaxed = instance.relax()
        relaxed.optimize()
        assert relaxed.Status == gu.gp.GRB.OPTIMAL, "Relaxed problem not optimal"

        relaxed_opt = np.array([v.X for v in relaxed.getVars()])
        A, b, c, l, u = gu.get_A_b_c_l_u(instance, keep_sparse=True)
        vTypeLU = {'C': 'real', 'I': 'integer', 'B': 'binary'}
        var_types = [vTypeLU[v.VType] for v in instance.getVars()]

        objective_func = lambda x: -(c.T @ x).item()  # this aco always minimizes
        def violation_func(x):
            Ax = A @ x
            viol = np.maximum(0, Ax - b)
            return viol.sum()

        best_x, best_f = aco_mip_optimizer(objective_func, violation_func, l, u, var_types, relaxed_opt,
                                           num_ants=50, archive_size=100, max_iter=200, penalty_factor=1e4, oracle=relaxed.ObjVal, seed=42)
        print("  ACO's best optimum:", -best_f)

        mdl_lll = gu.transform_via_LLL(instance, verify=False)
        mdl_lll.optimize()
        print("  Relaxed optimum:", relaxed.ObjVal)
        print("  Ideal optimum:", mdl_lll.ObjVal)

if __name__ == "__main__":
    test()

