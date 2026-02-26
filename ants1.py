import numpy as np

def _aco_make_logger(enabled=False, print_every=10):
    history = []

    def log(iter_idx, stats):
        if not enabled:
            return
        stats = dict(stats)
        stats["iter"] = iter_idx
        history.append(stats)
        if print_every is not None and print_every > 0:
            if (iter_idx % print_every) == 0 or stats.get("stopping", False):
                print(
                    f"[aco] iter={iter_idx:4d} best_obj={stats.get('best_obj'):.6g} "
                    f"best_viol={stats.get('best_viol'):.3g} feas={stats.get('feas_rate'):.2f} "
                    f"penalty={stats.get('penalty'):.3g} xi={stats.get('xi'):.3g} "
                    f"sig(mean/min)={stats.get('sigma_mean'):.3g}/{stats.get('sigma_min'):.3g}"
                )

    def get_history():
        return list(history)

    return log, get_history

def _aco_collect_stats(
    archive,
    fitness,
    sigmas,
    obj_func,
    viol_func,
    feasibility_rate,
    avg_feasibility,
    penalty_factor,
    xi,
    prev_best_obj=None,
    maximize=False,
):
    archive_arr = np.array(archive)
    best_obj = obj_func(archive[0])
    best_viol = viol_func(archive[0])
    sigma_mean = float(np.mean(sigmas)) if sigmas.size else 0.0
    sigma_min = float(np.min(sigmas)) if sigmas.size else 0.0
    sigma_max = float(np.max(sigmas)) if sigmas.size else 0.0
    archive_spread = float(np.mean(np.std(archive_arr, axis=0))) if archive_arr.size else 0.0
    best_improve = None
    if prev_best_obj is not None:
        best_improve = (best_obj - prev_best_obj) if maximize else (prev_best_obj - best_obj)
    return {
        "best_obj": best_obj,
        "best_viol": best_viol,
        "best_fit": fitness[0],
        "best_improve": best_improve,
        "sigma_mean": sigma_mean,
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "archive_spread": archive_spread,
        "feas_rate": feasibility_rate,
        "avg_feas": avg_feasibility,
        "penalty": penalty_factor,
        "xi": xi,
    }

def aco_convergence_report(history, maximize=False, tol=1e-6):
    if not history:
        return "No history to analyze."

    best_series = np.array([h["best_obj"] for h in history], dtype=float)
    deltas = np.diff(best_series)
    if maximize:
        improved = deltas > tol
    else:
        improved = deltas < -tol
    last_improve = np.where(improved)[0]
    last_improve_iter = int(last_improve[-1] + 1) if last_improve.size > 0 else 0
    plateau_len = len(best_series) - 1 - last_improve_iter

    sigma_min_series = np.array([h["sigma_min"] for h in history], dtype=float)
    spread_series = np.array([h["archive_spread"] for h in history], dtype=float)

    report = []
    report.append(f"Iterations: {len(history)}")
    report.append(f"Best obj: {best_series[-1]:.6g}")
    report.append(f"Last improvement at iter: {last_improve_iter}")
    report.append(f"Plateau length: {plateau_len}")
    report.append(f"Final sigma(min/mean): {sigma_min_series[-1]:.3g}/{np.mean([h['sigma_mean'] for h in history]):.3g}")
    report.append(f"Final archive spread: {spread_series[-1]:.3g}")

    if plateau_len > max(10, len(history) // 4):
        report.append("Convergence is early/fast: plateau detected.")
    if sigma_min_series[-1] < 1e-3 or spread_series[-1] < 1e-3:
        report.append("Search may be collapsing: sigma or archive spread very low.")

    return "\n".join(report)

def aco_mip_optimizer(
    obj_func,          # Objective function: takes np.array x, returns float (minimize)
    viol_func,         # Violation measure: takes np.array x, returns float >=0 (0 if feasible)
    lb,                # List of lower bounds for each variable
    ub,                # List of upper bounds for each variable
    var_types,         # List of 'binary', 'integer', 'real' for each variable
    relaxed_opt=None,  # Optional relaxed optimum np.array
    maximize=False,    # Whether to maximize (if True) or minimize (if False)
    num_ants=80,       # Number of ants per iteration
    archive_size=50,   # Size of solution archive (k)
    max_iter=200,      # Maximum iterations
    xi=0.85,           # Parameter for sigma calculation (exploration speed)
    penalty_factor=1e6,# Static penalty factor (can be tuned or made adaptive)
    oracle=None,       # Optional oracle value for oracle penalty (estimate of optimal obj)
    seed=42,           # Random seed for reproducibility
    log_enabled=False, # Toggle logging
    log_every=10,      # Print frequency for logging
    xi_decay=0.9995,    # Slower decay to avoid early collapse
    xi_floor=0.40,      # Lower bound for xi
    explore_rate=0.0,   # Base uniform exploration rate (doesn't work well)
    explore_boost=0.2,  # Exploration rate during plateau
    plateau_patience=12,# Iterations without improvement before boost
    refresh_period=20,  # Archive refresh period (0 disables)
    refresh_frac=0.40   # Fraction of archive to refresh
):
    """
    Ant Colony Optimization for Mixed-Integer Problems with sampling distributions.
    Handles binary, integer, and real variables. Uses penalty for constraints.
    If oracle is provided, uses a simple oracle-inspired penalty; else static.
    
    Returns: best_solution (np.array), best_obj (float), best_viol (float), log_history (list)
    """
    np.random.seed(seed)
    log, get_log = _aco_make_logger(enabled=log_enabled, print_every=log_every)
    stagnation_counter = 0
    max_stagnation = 100
    
    n_vars = len(lb)
    r_vars = np.arange(n_vars)
    assert len(var_types) == n_vars, "var_types must match number of variables"
    
    # Convert bounds to numpy arrays for vectorized operations
    lb = lb.flatten()
    ub = ub.flatten()
    sigma_floor_real = 0.1 * (ub - lb)  # keep exploration scale on continuous dims
    
    # Adaptive penalty state
    penalty_min = penalty_factor * 0.001
    penalty_max = penalty_factor * 1000.0
    feasibility_history = []  # track recent feasibility rates
    feasibility_window = 20  # number of iterations to average over
    
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
            x_rel = clip_to_bounds(relaxed_opt)
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
    
    # Penalized fitness (uses current adaptive penalty)
    def penalized_fitness(x):
        obj = obj_func(x)
        viol = viol_func(x)

        nonlocal penalty_factor
        if oracle is not None:
            # Oracle is only used to scale penalties; it should not flatten the objective.
            # For maximization, keep objective preference and penalize violations.
            if maximize:
                return obj - penalty_factor * viol
            # For minimization, keep objective preference and penalize violations.
            return obj + penalty_factor * viol
        if maximize:
            return obj - penalty_factor * viol
        return obj + penalty_factor * viol
    
    # Initialize archive with random solutions
    archive = initialize_archive()
    fitness = [penalized_fitness(x) for x in archive]
    # Sort archive by fitness (minimize by default, reverse for maximize)
    sorted_idx = np.argsort(fitness)
    if maximize:
        sorted_idx = sorted_idx[::-1]
    archive = [archive[i] for i in sorted_idx]
    fitness = [fitness[i] for i in sorted_idx]
    
    prev_best_obj = None
    no_improve_counter = 0
    explore_rate_current = explore_rate
    for iter in range(max_iter):
        # Plateau-based exploration settings (use previous iteration's status)
        explore_rate_current = explore_boost if no_improve_counter >= plateau_patience else explore_rate
        # Compute weights: tempered softmax on rank (less peaky than linear)
        ranks = np.arange(1, archive_size + 1)
        inv_temp = 3.0 / archive_size
        logits = -inv_temp * ranks
        weights = np.exp(logits - np.max(logits))
        weights = weights / np.sum(weights)
        
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
            # Enforce min sigma to avoid collapse
            # sig_floor = sigma_floor_real[j] if var_types[j] == 'real' else 1.0
            # sigmas[:, j] = np.maximum(sigmas[:, j], sig_floor)

            if var_types[j] != 'real':
                min_sigma_int = max(1, 3.0 * xi)
                sigmas[:, j] = np.maximum(sigmas[:, j], min_sigma_int)
            else:
                sigmas[:, j] = np.maximum(sigmas[:, j], sigma_floor_real[j])
        
        # Generate new solutions (ants)
        new_solutions = []
        for _ in range(num_ants):
            # Vectorized kernel selection across dimensions
            l_selected = np.random.choice(archive_size, size=n_vars, p=weights)
            mu = archive_array[l_selected, r_vars]
            sigma = sigmas[l_selected, r_vars]
            x_new = np.random.normal(mu, sigma)
            # Inject uniform exploration on a subset of dims
            explore_mask = np.random.rand(n_vars) < explore_rate_current
            if np.any(explore_mask):
                x_new[explore_mask] = np.random.uniform(lb[explore_mask], ub[explore_mask])
            # Clip to bounds and discretize
            x_new = discretize(x_new)
            x_new = clip_to_bounds(x_new)
            new_solutions.append(x_new)
        
        # Track feasibility and update global bests
        n_feasible = 0
        for x in new_solutions:
            v_val = viol_func(x)
            if v_val < 1e-4:
                n_feasible += 1
            
        feasibility_rate = n_feasible / num_ants
        feasibility_history.append(feasibility_rate)
        if len(feasibility_history) > feasibility_window:
            feasibility_history.pop(0)

        # Adapt penalty factor based on feasibility rate
        avg_feasibility = np.mean(feasibility_history)
        
        if avg_feasibility < 0.2:
            # Too few feasible solutions: increase penalty to push toward feasibility
            penalty_factor = min(penalty_factor * 1.5, penalty_max)
        elif avg_feasibility > 0.8:
            # Most solutions feasible: can relax penalty to allow more exploration
            penalty_factor = max(penalty_factor * 2 / 3, penalty_min)
        
        # Re-evaluate fitness with updated penalty for fair comparison
        new_fitness = [penalized_fitness(x) for x in new_solutions]
        fitness = [penalized_fitness(x) for x in archive]
        
        # Combine archive + new, sort, keep top k
        combined_sols = archive + new_solutions
        combined_fit = fitness + new_fitness
        sorted_idx = np.argsort(combined_fit)
        if maximize:
            sorted_idx = sorted_idx[::-1]  # reverse for maximization
        archive = [combined_sols[i] for i in sorted_idx[:archive_size]]
        fitness = [combined_fit[i] for i in sorted_idx[:archive_size]]
        
        # Optional: decrease xi over time for more exploitation
        xi = max(xi * xi_decay, xi_floor)  # Gradual reduction with floor

        # Periodic archive refresh to maintain diversity
        if refresh_period and (iter % refresh_period == refresh_period - 1):
            replace = max(1, int(archive_size * refresh_frac))
            for idx in range(archive_size - replace, archive_size):
                archive[idx] = random_solution()
                fitness[idx] = penalized_fitness(archive[idx])
            paired = sorted(zip(fitness, archive), key=lambda t: t[0], reverse=maximize)
            fitness, archive = map(list, zip(*paired))

        # Plateau detection for exploration boost
        prev_best_obj_for_stats = prev_best_obj
        current_best_obj = obj_func(archive[0])
        if prev_best_obj is not None:
            improved = (current_best_obj > prev_best_obj) if maximize else (current_best_obj < prev_best_obj)
            if improved:
                no_improve_counter = 0
            else:
                no_improve_counter += 1
        prev_best_obj = current_best_obj
        # explore_rate_current is updated at loop start

        # Stagnation check is tricky with changing penalty. 
        # Let's just run for max_iter if we are struggling, or use simple checks.
        
        stats = _aco_collect_stats(
            archive,
            fitness,
            sigmas,
            obj_func,
            viol_func,
            feasibility_rate,
            avg_feasibility,
            penalty_factor,
            xi,
            prev_best_obj=prev_best_obj_for_stats,
            maximize=maximize,
        )
        log(iter, stats)

        if (iter & 0x0F) == 0x0F:
            current_max_sigma = np.max(sigmas)
            if current_max_sigma < 1e-5:
                print("Stopping due to low sigma at", iter)
                break
            archive_spread = np.mean(np.std(np.array(archive), axis=0))
            if archive_spread < 1e-4:
                print("Stopping due to low archive spread at", iter)
                break

        # Periodic archive refresh ...
        # if (iter % 20 == 19):
        #     replace = max(1, archive_size // 5)
        #     for idx in range(archive_size - replace, archive_size):
        #         archive[idx] = random_solution()
        #         fitness[idx] = penalized_fitness(archive[idx])
        #     paired = sorted(zip(fitness, archive), key=lambda t: t[0])
        #     fitness, archive = map(list, zip(*paired))
    
    return archive[0], obj_func(archive[0]), viol_func(archive[0]), get_log()

def test():
    import knapsack_loader as kl
    import gurobi_utils as gu
    gu.gp.setParam('OutputFlag', 0)
    use_equality = True
    # disable lazy generator for random seed consistency:
    instances = list(kl.generate(4, 2, 15, 5, 10, 1000, equality=use_equality, seed=43))
    log_enabled = True
    for instance in instances:
        print("\nTesting instance:", instance.ModelName)
        relaxed = instance.relax()
        relaxed.optimize()
        assert relaxed.Status == gu.gp.GRB.OPTIMAL, "Relaxed problem not optimal"

        relaxed_opt = np.array([v.X for v in relaxed.getVars()])
        A, b, c, l, u = gu.get_A_b_c_l_u(instance, keep_sparse=True)
        vTypeLU = {'C': 'real', 'I': 'integer', 'B': 'binary'}
        var_types = [vTypeLU[v.VType] for v in instance.getVars()]

        objective_func = lambda x: (c.T @ x).item()  # positive objective; maximize=True below
        def violation_func(x):
            Ax = A @ x
            if use_equality:
                viol = np.abs(Ax - b)  # equality constraints
            else:
                viol = np.maximum(0, Ax - b) # only <= constraints
            return viol.sum()

        best_x, best_f, best_viol, log_hist = aco_mip_optimizer(
            objective_func,
            violation_func,
            l,
            u,
            var_types,
            relaxed_opt,
            maximize=True,
            num_ants=160,
            archive_size=40,
            max_iter=300,
            penalty_factor=5000,
            xi=0.9,
            oracle=relaxed.ObjVal,
            seed=42,
            log_enabled=log_enabled,
            log_every=20,
        )
        if log_enabled:
            print(aco_convergence_report(log_hist, maximize=True))
        print("  ACO's best optimum:", best_f, "with violation:", best_viol)

        print("  Relaxed optimum:", relaxed.ObjVal)
        if use_equality:
            mdl_lll = gu.transform_via_LLL(instance, verify=False)
            mdl_lll.optimize()
            print("  Ideal optimum:", mdl_lll.ObjVal, ", Gap:", (mdl_lll.ObjVal - best_f) / abs(mdl_lll.ObjVal) * 100, "%")
        else:
            instance.optimize()
            print("  Ideal optimum:", instance.ObjVal, ", Gap:", (instance.ObjVal - best_f) / abs(instance.ObjVal) * 100, "%")


if __name__ == "__main__":
    test()

