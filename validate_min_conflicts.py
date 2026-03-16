import knapsack_loader as kl
import gurobi_utils as gu
import numpy as np

_LARGE_BOUND = 1e30
_TOL = 1e-9


def _normalize_integer_bounds(n, l, u, x_bounds):
    lower_bounds = np.full(n, x_bounds[0], dtype=int)
    upper_bounds = np.full(n, x_bounds[1], dtype=int)

    if l is not None:
        lower_raw = np.asarray(l, dtype=float).reshape(-1)
        if lower_raw.shape[0] != n:
            raise ValueError(f"Expected {n} lower bounds, got {lower_raw.shape[0]}")
        finite_lower = np.abs(lower_raw) < _LARGE_BOUND
        lower_bounds[finite_lower] = np.ceil(lower_raw[finite_lower]).astype(int)

    if u is not None:
        upper_raw = np.asarray(u, dtype=float).reshape(-1)
        if upper_raw.shape[0] != n:
            raise ValueError(f"Expected {n} upper bounds, got {upper_raw.shape[0]}")
        finite_upper = np.abs(upper_raw) < _LARGE_BOUND
        upper_bounds[finite_upper] = np.floor(upper_raw[finite_upper]).astype(int)

    if np.any(lower_bounds > upper_bounds):
        raise ValueError("Infeasible variable bounds after integer rounding")

    return lower_bounds, upper_bounds


def _sample_equality_start(A, b, lower_bounds, upper_bounds, rng, restart_idx):
    x_continuous = np.linalg.pinv(A) @ b
    x_continuous = np.clip(x_continuous, lower_bounds, upper_bounds)

    if restart_idx == 0:
        return np.rint(x_continuous).astype(int)

    jitter_scale = min(3.0, 0.5 + 0.25 * restart_idx)
    x = np.rint(x_continuous + rng.normal(0.0, jitter_scale, size=A.shape[1])).astype(int)
    if restart_idx % 4 == 0:
        mask = rng.random(A.shape[1]) < 0.2
        x[mask] = rng.integers(lower_bounds[mask], upper_bounds[mask] + 1)
    return np.clip(x, lower_bounds, upper_bounds)


def _sample_general_start(A, b, lower_bounds, upper_bounds, rng, restart_idx):
    x_continuous = np.linalg.pinv(A) @ b
    x_continuous = np.clip(x_continuous, lower_bounds, upper_bounds)

    if restart_idx == 0:
        return np.rint(x_continuous).astype(int)

    jitter_scale = min(4.0, 1.0 + 0.35 * restart_idx)
    x = np.rint(x_continuous + rng.normal(0.0, jitter_scale, size=A.shape[1])).astype(int)
    if restart_idx % 3 == 0:
        mask = rng.random(A.shape[1]) < 0.25
        x[mask] = rng.integers(lower_bounds[mask], upper_bounds[mask] + 1)
    return np.clip(x, lower_bounds, upper_bounds)


def _constraint_violation(residual, constraint_type):
    if constraint_type == "=":
        return abs(residual)
    if constraint_type == "<":
        return max(0.0, -residual + 1e-6)
    if constraint_type == ">":
        return max(0.0, residual + 1e-6)
    return 0.0


def _constraint_projected_residual(residual, constraint_type):
    if constraint_type == "=":
        return residual
    if constraint_type == "<":
        return min(0.0, residual)
    if constraint_type == ">":
        return max(0.0, residual)
    return 0.0


def _constraints_satisfied(residuals, constraint_types):
    return all(_constraint_violation(residual, constraint_type) <= _TOL for residual, constraint_type in zip(residuals, constraint_types))


def _select_repair_candidates(A, residuals, constraint_types, max_candidates=8):
    violation_weights = np.array([
        _constraint_violation(residuals[row_idx], constraint_types[row_idx])
        for row_idx in range(A.shape[0])
    ])
    if np.all(violation_weights <= _TOL):
        return np.arange(min(max_candidates, A.shape[1]))
    influence = np.abs(A).T @ violation_weights
    candidate_count = min(max_candidates, A.shape[1])
    return np.argpartition(-influence, candidate_count - 1)[:candidate_count]


def _general_merit(residuals, constraint_types, row_scales, weights):
    projected_residuals = np.array([
        _constraint_projected_residual(residuals[row_idx] / row_scales[row_idx], constraint_types[row_idx])
        for row_idx in range(residuals.shape[0])
    ])
    return float(np.dot(weights, projected_residuals * projected_residuals))


def _best_projected_single_move_general(
    A,
    residuals,
    x,
    lower_bounds,
    upper_bounds,
    row_scales,
    weights,
    constraint_types,
):
    scaled_A = A / row_scales[:, None]
    projected_residuals = np.array([
        _constraint_projected_residual(residuals[row_idx] / row_scales[row_idx], constraint_types[row_idx])
        for row_idx in range(residuals.shape[0])
    ])
    current_merit = float(np.dot(weights, projected_residuals * projected_residuals))
    best_move = None

    for var_idx in range(A.shape[1]):
        col = scaled_A[:, var_idx]
        weighted_col_norm = float(np.dot(weights, col * col))
        if weighted_col_norm <= _TOL:
            continue

        delta_star = float(np.dot(weights, col * projected_residuals) / weighted_col_norm)
        min_delta = int(lower_bounds[var_idx] - x[var_idx])
        max_delta = int(upper_bounds[var_idx] - x[var_idx])
        if min_delta == 0 and max_delta == 0:
            continue

        clipped_delta = int(np.clip(np.rint(delta_star), min_delta, max_delta))
        candidate_deltas = {
            clipped_delta,
            int(np.clip(np.floor(delta_star), min_delta, max_delta)),
            int(np.clip(np.ceil(delta_star), min_delta, max_delta)),
        }
        if min_delta <= -1 <= max_delta:
            candidate_deltas.add(-1)
        if min_delta <= 1 <= max_delta:
            candidate_deltas.add(1)
        if delta_star < min_delta:
            candidate_deltas.add(min_delta)
        if delta_star > max_delta:
            candidate_deltas.add(max_delta)
        candidate_deltas.discard(0)

        for delta in candidate_deltas:
            new_residuals = residuals - A[:, var_idx] * delta
            new_merit = _general_merit(new_residuals, constraint_types, row_scales, weights)
            improvement = current_merit - new_merit
            if best_move is None or improvement > best_move["improvement"]:
                best_move = {
                    "type": "single",
                    "var_idx": var_idx,
                    "delta": delta,
                    "residuals": new_residuals,
                    "improvement": improvement,
                }

    return best_move


def _equality_merit(residuals, row_scales, weights):
    scaled_residuals = residuals / row_scales
    return float(np.dot(weights, scaled_residuals * scaled_residuals))


def _best_projected_single_move(A, residuals, x, lower_bounds, upper_bounds, row_scales, weights):
    scaled_residuals = residuals / row_scales
    scaled_A = A / row_scales[:, None]
    current_merit = _equality_merit(residuals, row_scales, weights)
    best_move = None

    for var_idx in range(A.shape[1]):
        col = scaled_A[:, var_idx]
        weighted_col_norm = float(np.dot(weights, col * col))
        if weighted_col_norm <= _TOL:
            continue

        delta_star = float(np.dot(weights, col * scaled_residuals) / weighted_col_norm)
        min_delta = int(lower_bounds[var_idx] - x[var_idx])
        max_delta = int(upper_bounds[var_idx] - x[var_idx])
        if min_delta == 0 and max_delta == 0:
            continue

        clipped_delta = int(np.clip(np.rint(delta_star), min_delta, max_delta))
        candidate_deltas = {
            clipped_delta,
            int(np.clip(np.floor(delta_star), min_delta, max_delta)),
            int(np.clip(np.ceil(delta_star), min_delta, max_delta)),
        }
        if min_delta <= -1 <= max_delta:
            candidate_deltas.add(-1)
        if min_delta <= 1 <= max_delta:
            candidate_deltas.add(1)
        if delta_star < min_delta:
            candidate_deltas.add(min_delta)
        if delta_star > max_delta:
            candidate_deltas.add(max_delta)
        candidate_deltas.discard(0)

        for delta in candidate_deltas:
            new_residuals = residuals - A[:, var_idx] * delta
            new_merit = _equality_merit(new_residuals, row_scales, weights)
            improvement = current_merit - new_merit
            if best_move is None or improvement > best_move["improvement"]:
                best_move = {
                    "type": "single",
                    "var_idx": var_idx,
                    "delta": delta,
                    "residuals": new_residuals,
                    "improvement": improvement,
                }

    return best_move


def _best_projected_pair_move(
    A,
    residuals,
    x,
    lower_bounds,
    upper_bounds,
    row_scales,
    weights,
    candidate_vars=None,
    max_pair_span=3,
):
    scaled_A = A / row_scales[:, None]
    scaled_residuals = residuals / row_scales
    current_merit = _equality_merit(residuals, row_scales, weights)
    best_move = None
    if candidate_vars is None:
        candidate_vars = range(A.shape[1])
    else:
        candidate_vars = list(candidate_vars)

    for first_pos, first_idx in enumerate(candidate_vars[:-1]):
        min_first = int(lower_bounds[first_idx] - x[first_idx])
        max_first = int(upper_bounds[first_idx] - x[first_idx])
        if min_first == 0 and max_first == 0:
            continue

        for second_idx in candidate_vars[first_pos + 1 :]:
            min_second = int(lower_bounds[second_idx] - x[second_idx])
            max_second = int(upper_bounds[second_idx] - x[second_idx])
            if min_second == 0 and max_second == 0:
                continue

            pair_cols = scaled_A[:, [first_idx, second_idx]]
            gram = pair_cols.T @ (weights[:, None] * pair_cols)
            rhs = pair_cols.T @ (weights * scaled_residuals)

            try:
                delta_star = np.linalg.solve(gram, rhs)
            except np.linalg.LinAlgError:
                delta_star, *_ = np.linalg.lstsq(gram, rhs, rcond=None)

            center_first = int(np.clip(np.rint(delta_star[0]), min_first, max_first))
            center_second = int(np.clip(np.rint(delta_star[1]), min_second, max_second))

            if max_first - min_first <= 2 * max_pair_span:
                first_candidates = range(min_first, max_first + 1)
            else:
                low = max(min_first, center_first - max_pair_span)
                high = min(max_first, center_first + max_pair_span)
                first_candidates = range(low, high + 1)

            if max_second - min_second <= 2 * max_pair_span:
                second_candidates = range(min_second, max_second + 1)
            else:
                low = max(min_second, center_second - max_pair_span)
                high = min(max_second, center_second + max_pair_span)
                second_candidates = range(low, high + 1)

            for delta_first in first_candidates:
                for delta_second in second_candidates:
                    if delta_first == 0 and delta_second == 0:
                        continue
                    new_residuals = residuals - A[:, first_idx] * delta_first - A[:, second_idx] * delta_second
                    new_merit = _equality_merit(new_residuals, row_scales, weights)
                    improvement = current_merit - new_merit
                    if best_move is None or improvement > best_move["improvement"]:
                        best_move = {
                            "type": "pair",
                            "first_idx": first_idx,
                            "second_idx": second_idx,
                            "delta_first": delta_first,
                            "delta_second": delta_second,
                            "residuals": new_residuals,
                            "improvement": improvement,
                        }

    return best_move


def _best_exact_pair_move_2d(A, residuals, x, lower_bounds, upper_bounds, row_scales, weights):
    current_merit = _equality_merit(residuals, row_scales, weights)
    best_move = None
    num_vars = A.shape[1]

    for first_idx in range(num_vars - 1):
        min_first = int(lower_bounds[first_idx] - x[first_idx])
        max_first = int(upper_bounds[first_idx] - x[first_idx])
        if min_first == 0 and max_first == 0:
            continue

        for second_idx in range(first_idx + 1, num_vars):
            min_second = int(lower_bounds[second_idx] - x[second_idx])
            max_second = int(upper_bounds[second_idx] - x[second_idx])
            if min_second == 0 and max_second == 0:
                continue

            pair_matrix = A[:, [first_idx, second_idx]]
            det = pair_matrix[0, 0] * pair_matrix[1, 1] - pair_matrix[0, 1] * pair_matrix[1, 0]
            if abs(det) <= _TOL:
                continue

            delta_star = np.linalg.solve(pair_matrix, residuals)
            first_candidates = {
                int(np.clip(np.floor(delta_star[0]), min_first, max_first)),
                int(np.clip(np.ceil(delta_star[0]), min_first, max_first)),
                int(np.clip(np.rint(delta_star[0]), min_first, max_first)),
            }
            second_candidates = {
                int(np.clip(np.floor(delta_star[1]), min_second, max_second)),
                int(np.clip(np.ceil(delta_star[1]), min_second, max_second)),
                int(np.clip(np.rint(delta_star[1]), min_second, max_second)),
            }

            for delta_first in first_candidates:
                for delta_second in second_candidates:
                    if delta_first == 0 and delta_second == 0:
                        continue
                    new_residuals = residuals - pair_matrix @ np.array([delta_first, delta_second])
                    new_merit = _equality_merit(new_residuals, row_scales, weights)
                    improvement = current_merit - new_merit
                    if best_move is None or improvement > best_move["improvement"]:
                        best_move = {
                            "type": "pair",
                            "first_idx": first_idx,
                            "second_idx": second_idx,
                            "delta_first": delta_first,
                            "delta_second": delta_second,
                            "residuals": new_residuals,
                            "improvement": improvement,
                        }

    return best_move


def _try_exact_repair_2d(A, residuals, x, lower_bounds, upper_bounds, candidate_vars):
    rounded_residuals = np.rint(residuals).astype(int)
    if not np.all(np.abs(residuals - rounded_residuals) <= _TOL):
        return None

    movable_vars = []
    delta_ranges = []
    for var_idx in candidate_vars:
        min_delta = int(lower_bounds[var_idx] - x[var_idx])
        max_delta = int(upper_bounds[var_idx] - x[var_idx])
        if min_delta == 0 and max_delta == 0:
            continue
        movable_vars.append(var_idx)
        delta_ranges.append(range(min_delta, max_delta + 1))

    if not movable_vars:
        return None

    split_idx = len(movable_vars) // 2
    left_vars = movable_vars[:split_idx]
    right_vars = movable_vars[split_idx:]
    left_ranges = delta_ranges[:split_idx]
    right_ranges = delta_ranges[split_idx:]
    target = tuple(int(value) for value in rounded_residuals)
    left_states = {}

    def build_states(vars_subset, ranges_subset, pos, contrib, deltas, sink):
        if pos == len(vars_subset):
            key = (int(contrib[0]), int(contrib[1]))
            total_adjustment = sum(abs(delta) for delta in deltas.values())
            if key not in sink or total_adjustment < sink[key][0]:
                sink[key] = (total_adjustment, deltas.copy())
            return

        var_idx = vars_subset[pos]
        column = A[:, var_idx].astype(int)
        for delta in ranges_subset[pos]:
            if delta == 0:
                build_states(vars_subset, ranges_subset, pos + 1, contrib, deltas, sink)
                continue
            deltas[var_idx] = delta
            build_states(vars_subset, ranges_subset, pos + 1, contrib + column * delta, deltas, sink)
            deltas.pop(var_idx)

    build_states(left_vars, left_ranges, 0, np.zeros(2, dtype=int), {}, left_states)

    best_adjustment = None
    best_delta_map = None

    def search_right(pos, contrib, deltas):
        nonlocal best_adjustment, best_delta_map
        if pos == len(right_vars):
            complement = (target[0] - int(contrib[0]), target[1] - int(contrib[1]))
            if complement not in left_states:
                return
            adjustment = left_states[complement][0] + sum(abs(delta) for delta in deltas.values())
            if best_adjustment is None or adjustment < best_adjustment:
                best_adjustment = adjustment
                best_delta_map = left_states[complement][1].copy()
                best_delta_map.update(deltas)
            return

        var_idx = right_vars[pos]
        column = A[:, var_idx].astype(int)
        for delta in right_ranges[pos]:
            if delta == 0:
                search_right(pos + 1, contrib, deltas)
                continue
            deltas[var_idx] = delta
            search_right(pos + 1, contrib + column * delta, deltas)
            deltas.pop(var_idx)

    search_right(0, np.zeros(2, dtype=int), {})

    if best_delta_map is None:
        return None

    return {
        "type": "exact",
        "delta_map": best_delta_map,
        "residuals": np.zeros_like(residuals),
        "improvement": np.inf,
    }


def _try_local_discrete_repair(
    A,
    residuals,
    x,
    lower_bounds,
    upper_bounds,
    constraint_types,
    candidate_vars,
    delta_radius=3,
):
    candidate_vars = list(candidate_vars)
    movable_vars = []
    delta_ranges = []
    for var_idx in candidate_vars:
        min_delta = int(lower_bounds[var_idx] - x[var_idx])
        max_delta = int(upper_bounds[var_idx] - x[var_idx])
        if min_delta == 0 and max_delta == 0:
            continue
        low = max(min_delta, -delta_radius)
        high = min(max_delta, delta_radius)
        if low > high:
            continue
        movable_vars.append(var_idx)
        delta_ranges.append(range(low, high + 1))

    if not movable_vars:
        return None

    split_idx = len(movable_vars) // 2
    left_vars = movable_vars[:split_idx]
    right_vars = movable_vars[split_idx:]
    left_ranges = delta_ranges[:split_idx]
    right_ranges = delta_ranges[split_idx:]
    left_states = []

    def build_states(vars_subset, ranges_subset, pos, contrib, deltas, sink):
        if pos == len(vars_subset):
            sink.append((contrib.copy(), deltas.copy()))
            return

        var_idx = vars_subset[pos]
        column = A[:, var_idx]
        for delta in ranges_subset[pos]:
            if delta == 0:
                build_states(vars_subset, ranges_subset, pos + 1, contrib, deltas, sink)
                continue
            deltas[var_idx] = delta
            build_states(vars_subset, ranges_subset, pos + 1, contrib + column * delta, deltas, sink)
            deltas.pop(var_idx)

    build_states(left_vars, left_ranges, 0, np.zeros(A.shape[0], dtype=float), {}, left_states)

    best_adjustment = None
    best_delta_map = None
    best_residuals = None

    def search_right(pos, contrib, deltas):
        nonlocal best_adjustment, best_delta_map, best_residuals
        if pos == len(right_vars):
            for left_contrib, left_deltas in left_states:
                total_contrib = left_contrib + contrib
                candidate_residuals = residuals - total_contrib
                if not _constraints_satisfied(candidate_residuals, constraint_types):
                    continue
                adjustment = sum(abs(delta) for delta in left_deltas.values()) + sum(abs(delta) for delta in deltas.values())
                if best_adjustment is None or adjustment < best_adjustment:
                    best_adjustment = adjustment
                    best_delta_map = left_deltas.copy()
                    best_delta_map.update(deltas)
                    best_residuals = candidate_residuals.copy()
            return

        var_idx = right_vars[pos]
        column = A[:, var_idx]
        for delta in right_ranges[pos]:
            if delta == 0:
                search_right(pos + 1, contrib, deltas)
                continue
            deltas[var_idx] = delta
            search_right(pos + 1, contrib + column * delta, deltas)
            deltas.pop(var_idx)

    search_right(0, np.zeros(A.shape[0], dtype=float), {})

    if best_delta_map is None:
        return None

    return {
        "type": "exact",
        "delta_map": best_delta_map,
        "residuals": best_residuals,
        "improvement": np.inf,
    }


def _apply_discrete_move(x, move):
    if move["type"] == "single":
        x[move["var_idx"]] += move["delta"]
    elif move["type"] == "exact":
        for var_idx, delta in move["delta_map"].items():
            x[var_idx] += delta
    else:
        x[move["first_idx"]] += move["delta_first"]
        x[move["second_idx"]] += move["delta_second"]


def _solve_equalities(A, b, lower_bounds, upper_bounds, max_iters, rng):
    row_scales = np.linalg.norm(A, axis=1)
    row_scales[row_scales < 1.0] = 1.0

    best_x = None
    best_residuals = None
    best_abs_residual = np.inf
    weights = np.ones(A.shape[0], dtype=float)
    restart_idx = 0
    total_iters = 0
    stall_iters = 0
    x = _sample_equality_start(A, b, lower_bounds, upper_bounds, rng, restart_idx)
    residuals = b - A @ x

    while total_iters < max_iters:
        abs_residual = float(np.abs(residuals).sum())
        if abs_residual < best_abs_residual:
            best_abs_residual = abs_residual
            best_x = x.copy()
            best_residuals = residuals.copy()

        if np.all(np.abs(residuals) <= _TOL):
            return x, True, {"iterations": total_iters, "restarts": restart_idx, "residuals": residuals.copy()}

        if A.shape[0] == 2 and abs_residual <= 32:
            scaled_A = A / row_scales[:, None]
            gradients = np.abs(scaled_A.T @ (weights * (residuals / row_scales)))
            candidate_count = min(8, A.shape[1])
            candidate_vars = np.argpartition(-gradients, candidate_count - 1)[:candidate_count]
            exact_move = _try_exact_repair_2d(A, residuals, x, lower_bounds, upper_bounds, candidate_vars)
            if exact_move is not None:
                _apply_discrete_move(x, exact_move)
                residuals = exact_move["residuals"]
                return x, True, {"iterations": total_iters, "restarts": restart_idx, "residuals": residuals.copy()}

        single_move = _best_projected_single_move(A, residuals, x, lower_bounds, upper_bounds, row_scales, weights)
        if single_move is not None and single_move["improvement"] > _TOL:
            _apply_discrete_move(x, single_move)
            residuals = single_move["residuals"]
            stall_iters = 0
        else:
            scaled_residuals = np.abs(residuals / row_scales)
            if A.shape[0] == 2:
                pair_move = _best_exact_pair_move_2d(A, residuals, x, lower_bounds, upper_bounds, row_scales, weights)
            else:
                scaled_A = A / row_scales[:, None]
                gradients = np.abs(scaled_A.T @ (weights * (residuals / row_scales)))
                candidate_count = min(8, A.shape[1])
                candidate_vars = np.argpartition(-gradients, candidate_count - 1)[:candidate_count]
                pair_move = _best_projected_pair_move(
                    A,
                    residuals,
                    x,
                    lower_bounds,
                    upper_bounds,
                    row_scales,
                    weights,
                    candidate_vars=candidate_vars,
                )

            if pair_move is not None and pair_move["improvement"] > _TOL:
                _apply_discrete_move(x, pair_move)
                residuals = pair_move["residuals"]
                stall_iters = 0
            else:
                weights += 0.5 + scaled_residuals / (scaled_residuals.max() + _TOL)
                stall_iters += 1
            if stall_iters >= 12:
                restart_idx += 1
                x = _sample_equality_start(A, b, lower_bounds, upper_bounds, rng, restart_idx)
                residuals = b - A @ x
                weights.fill(1.0)
                stall_iters = 0

        total_iters += 1

    return best_x, False, {"iterations": total_iters, "restarts": restart_idx, "residuals": best_residuals}


def solve_linear_constraints(A, b, constraint_types, l=None, u=None, x_bounds=(-100, 100), 
                             max_iters=10000, noise=0.1, step_size=1, seed=None):
    """
    Min-Conflicts variant for Ax = b, Ax < b, Ax > b in Z^n.
    
    A: (m, n) matrix
    b: (m,) vector
    constraint_types: list of strings ['=', '<', '>']
    """
    m, n = A.shape
    b = np.asarray(b, dtype=float).reshape(-1)
    if b.shape[0] != m:
        raise ValueError(f"Expected {m} RHS values, got {b.shape[0]}")

    constraint_types = np.asarray(constraint_types).reshape(-1)
    if constraint_types.shape[0] != m:
        raise ValueError(f"Expected {m} constraint types, got {constraint_types.shape[0]}")

    lower_bounds, upper_bounds = _normalize_integer_bounds(n, l, u, x_bounds)
    rng = np.random.default_rng(seed)

    if np.all(constraint_types == '='):
        return _solve_equalities(A, b, lower_bounds, upper_bounds, max_iters, rng)

    row_scales = np.linalg.norm(A, axis=1)
    row_scales[row_scales < 1.0] = 1.0
    restart_idx = 0
    x = _sample_general_start(A, b, lower_bounds, upper_bounds, rng, restart_idx)
    
    # weights for each constraint (the 'Breakout' mechanism)
    weights = np.ones(m, dtype=float)
    stall_iters = 0
    
    # Initial residuals: r = b - Ax
    # For '=', we want r = 0
    # For '<', we want Ax < b => b - Ax > 0 => r > 0
    # For '>', we want Ax > b => b - Ax < 0 => r < 0
    current_Ax = A @ x
    residuals = b - current_Ax
    best_x = x.copy()
    best_residuals = residuals.copy()
    best_penalty = np.inf

    for i in range(max_iters):
        # 1. Identify violated constraints
        penalties = np.array([_constraint_violation(residuals[j], constraint_types[j]) for j in range(m)])
        violated_indices = np.where(penalties > 0)[0]
        total_penalty = float(penalties.sum())

        if total_penalty < best_penalty:
            best_penalty = total_penalty
            best_x = x.copy()
            best_residuals = residuals.copy()
        
        if len(violated_indices) == 0:
            stats = {"iterations": i, "restarts": 0, "residuals": residuals.copy()}
            return x, True, stats

        if total_penalty <= 128 or stall_iters >= 4:
            candidate_vars = _select_repair_candidates(A, residuals, constraint_types, max_candidates=8)
            repair_move = _try_local_discrete_repair(
                A,
                residuals,
                x,
                lower_bounds,
                upper_bounds,
                constraint_types,
                candidate_vars,
                delta_radius=4,
            )
            if repair_move is not None:
                _apply_discrete_move(x, repair_move)
                residuals = repair_move["residuals"]
                stats = {"iterations": i, "restarts": restart_idx, "residuals": residuals.copy()}
                return x, True, stats

        move = _best_projected_single_move_general(
            A,
            residuals,
            x,
            lower_bounds,
            upper_bounds,
            row_scales,
            weights,
            constraint_types,
        )

        if move is not None and move["improvement"] > _TOL:
            _apply_discrete_move(x, move)
            residuals = move["residuals"]
            stall_iters = 0
        elif rng.random() < noise:
            candidate_vars = _select_repair_candidates(A, residuals, constraint_types, max_candidates=min(6, n))
            var_idx = int(rng.choice(candidate_vars))
            feasible_moves = [
                move for move in (-step_size, step_size)
                if lower_bounds[var_idx] <= x[var_idx] + move <= upper_bounds[var_idx]
            ]
            if feasible_moves:
                best_move = int(rng.choice(feasible_moves))
                x[var_idx] += best_move
                residuals -= A[:, var_idx] * best_move
                stall_iters = 0
            else:
                weights[violated_indices] += 0.5
                stall_iters += 1
        else:
            # STUCK: Increase weights of violated constraints and restart periodically.
            weights[violated_indices] += 0.5 + penalties[violated_indices] / (penalties.max() + _TOL)
            stall_iters += 1
            if stall_iters >= 12:
                restart_idx += 1
                x = _sample_general_start(A, b, lower_bounds, upper_bounds, rng, restart_idx)
                residuals = b - A @ x
                weights.fill(1.0)
                stall_iters = 0

    stats = {"iterations": max_iters, "restarts": restart_idx, "residuals": best_residuals.copy()}
    return best_x, False, stats

def main2():
    # --- Example Usage ---
    # 2x + 3y = 10
    # x - y > 2
    # x, y in Z
    A = np.array([[2, 3], [1, -1]])
    b = np.array([10, 2])
    types = ['=', '>']
    l = np.array([[0], [0]])
    u = np.array([[10], [10]])

    sol, success, stats = solve_linear_constraints(A, b, types, l=l, u=u, seed=0)
    print(f"Success: {success}, Iterations: {stats['iterations']}, Solution: {sol}")

def main():
    model = list(kl.generate(1, 3, 30, 5, 10, 1000, equality=True, seed=43))[0]
    A, b, c, l, u = gu.get_A_b_c_l_u(model)
    constraint_types = model.getAttr("Sense")
    sol, success, stats = solve_linear_constraints(
        A,
        b,
        constraint_types,
        l=l,
        u=u,
        max_iters=100000,
        seed=0,
    )
    print(
        f"Success: {success}, Iterations: {stats['iterations']}, "
        f"Residuals: {stats['residuals']}, Solution: {sol}"
    )

if __name__ == "__main__":
    main()