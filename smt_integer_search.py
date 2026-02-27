import math
import pysat.solvers as sat
from pysat.formula import IDPool
from pysat.pb import PBEnc, EncType
import gurobipy as gp
import gurobi_utils as gu
import numpy as np
from typing import Optional

# break down the MIP into a SAT part and an LP part:
# LP part: f(b) = \min_x \{c_x^T x : A_x x \le d - A_b b\}
# by strong duality: f(b) = \lambda^T (d - A_b b)
# define: \delta_i = \lambda^T A_{b,i}
# then f(b) = \gamma - \sum_i \delta_i b_i
# by (negative) derivative of the delta_i, we can improve the objective:
# if \delta_i > 0, then we can set b_i = 1 to improve the objective

# loop:
#     SAT proposes assignment integer variables

#     run LP under those assumptions:
#         to check feasibility
#         to compute bounds

#     if infeasible:
#         derive Farkas certificate from duality
#         learn clause to exclude this solution and add it to the SAT solver
#         continue

#     store the current solution if it is the best one so far.
#     learn a cut to cut off the current solution, assuming it is not optimal.
#     learn the SAT assumptions to improve the objective, if possible.
#     order those by largest |delata_i| first, to get the most improvement in the objective.
#     also add a cut to ensure that we improve our objective


def _add_pb_constraint(
    solver: sat.Solver,
    vpool: IDPool,
    lits: list[int],
    weights: list[int],
    bound: int,
    sense: str,
) -> bool:
    """Add a PB constraint to the solver, handling negative weights and bounds.
    sense is '<', '>', or '='.
    Returns True if the constraint was added, False if it is trivially satisfied,
    and raises ValueError if it is trivially unsatisfiable.
    """
    pos_lits = []
    pos_weights = []
    adj_bound = bound

    for l, w in zip(lits, weights):
        if w == 0:
            continue
        if w < 0:
            pos_lits.append(-l)
            pos_weights.append(-w)
            adj_bound += -w
        else:
            pos_lits.append(l)
            pos_weights.append(w)

    if not pos_lits:
        if sense == '<' and 0 > adj_bound:
            raise ValueError(f"Infeasible fixed constraint: 0 <= {adj_bound}")
        elif sense == '>' and 0 < adj_bound:
            raise ValueError(f"Infeasible fixed constraint: 0 >= {adj_bound}")
        elif sense == '=' and 0 != adj_bound:
            raise ValueError(f"Infeasible fixed constraint: 0 == {adj_bound}")
        return False

    if sense == '<':
        if adj_bound < 0:
            raise ValueError(f"Infeasible PB constraint: sum >= 0 <= {adj_bound}")
        if adj_bound >= sum(pos_weights):
            return False  # trivially satisfied
        enc = PBEnc.leq(lits=pos_lits, weights=pos_weights, bound=adj_bound, vpool=vpool, encoding=EncType.best)
    elif sense == '>':
        if adj_bound <= 0:
            return False  # trivially satisfied
        if adj_bound > sum(pos_weights):
            raise ValueError(f"Infeasible PB constraint: sum <= {sum(pos_weights)} >= {adj_bound}")
        enc = PBEnc.geq(lits=pos_lits, weights=pos_weights, bound=adj_bound, vpool=vpool, encoding=EncType.best)
    elif sense == '=':
        if adj_bound < 0 or adj_bound > sum(pos_weights):
            raise ValueError(f"Infeasible PB constraint: sum == {adj_bound}")
        enc = PBEnc.equals(lits=pos_lits, weights=pos_weights, bound=adj_bound, vpool=vpool, encoding=EncType.best)
    else:
        raise ValueError(f"Unknown sense: {sense}")

    solver.append_formula(enc.clauses)
    return True


def _make_lp_subproblem(
    model: gp.Model,
    discrete_vals: dict[int, int],
    discrete_indices: list[int],
    lp: Optional[gp.Model] = None,
) -> gp.Model:
    """Create/update an LP subproblem by fixing discrete variables to proposed values.

    discrete_vals maps Gurobi var.index -> integer value.
    discrete_indices is the ordered list of discrete var indices (positions in getVars()).
    If ``lp`` is provided, it is reused and only bounds/fixings are updated.
    """
    if lp is None:
        lp = model.copy()
        lp.params.OutputFlag = 0
        lp.params.InfUnbdInfo = 1  # needed to get Farkas duals on infeasibility
        lp.params.Method = 1  # use dual simplex to get better Farkas duals
        lp_vars = lp.getVars()
        lp._discrete_lp_vars = [lp_vars[i] for i in discrete_indices]
        n = len(lp._discrete_lp_vars)
        lp.setAttr("VType", lp._discrete_lp_vars, [gp.GRB.CONTINUOUS] * n)
    # Use positional indexing: var.index == position in getVars() after copy.
    # Batch all attribute writes with setAttr to avoid per-variable Python overhead.
    vals = [float(discrete_vals[i]) for i in discrete_indices]
    lp.setAttr("LB", lp._discrete_lp_vars, vals)
    lp.setAttr("UB", lp._discrete_lp_vars, vals)
    lp.update()
    return lp


def _add_benders_feasibility_cut(
    solver: sat.Solver,
    vpool: IDPool,
    discrete_encodings: list[tuple[int, list[int], list[int]]],
    model_rhs: np.ndarray,
    A_d,  # scipy sparse (n_constrs x n_discrete)
    farkas_duals: np.ndarray,
    scale: float = 1e4,
) -> bool:
    """Derive and add a Benders feasibility cut from a Gurobi FarkasDual certificate.

    Gurobi FarkasDual convention: for infeasible LP, the certificate satisfies
        sum_j delta_j * x_j > rhs_float    (proves infeasibility for x^*)
    where delta_j = sum_i y_i * A[i,j] and rhs_float = sum_i y_i * b_i.

    To EXCLUDE infeasible assignments, we add the NEGATED constraint:
        sum_j delta_j * x_j <= rhs_float
    Equivalently (multiply by -1 for PBEnc.geq):
        sum_j (-delta_j) * x_j >= -rhs_float

    Returns True if a non-trivial cut was added.
    """
    y = np.asarray(farkas_duals)
    rhs_float = float(y @ model_rhs)
    delta_arr = np.asarray(A_d.T @ y).ravel()

    adjusted_rhs_float = rhs_float
    for j, (L, _, _) in enumerate(discrete_encodings):
        adjusted_rhs_float -= delta_arr[j] * L

    neg_rhs_scaled_float = -adjusted_rhs_float * scale
    neg_rhs_scaled = int(math.floor(neg_rhs_scaled_float + 1e-6))

    pos_lits, pos_weights = [], []
    for j, (_, lits, weights) in enumerate(discrete_encodings):
        d = delta_arr[j]
        for lit, w in zip(lits, weights):
            a_scaled = float(-d * w) * scale
            if np.isclose(a_scaled, 0.0):
                continue
            weight_int = int(math.ceil(a_scaled - 1e-6))
            pos_lits.append(lit)
            pos_weights.append(weight_int)

    if not pos_lits:
        return False

    try:
        _add_pb_constraint(solver, vpool, pos_lits, pos_weights, neg_rhs_scaled, '>')
        return True
    except ValueError:
        return False

def _add_benders_optimality_cut(
    solver: sat.Solver,
    vpool: IDPool,
    discrete_encodings: list[tuple[int, list[int], list[int]]],
    model_rhs: np.ndarray,
    discrete_obj_arr: np.ndarray,
    A_d,  # scipy sparse (n_constrs x n_discrete)
    lp: gp.Model,
    upper_bound: float,
    guard_lit: Optional[int] = None,
    scale: float = 16384,
) -> tuple[bool, list[int]]:
    """Derive and add a Benders optimality cut from LP dual variables.

    From LP duality: f(b') >= c_b^T b' + lambda^T(d - A_b b')
                           = gamma + sum_j delta_j * b'_j   for all b'

    where gamma = lambda^T d  and  delta_j = c_{b,j} - (A_b^T lambda)_j.

    We add the SAT constraint: gamma + sum_j delta_j * b_j < upper_bound,
    i.e., sum_j delta_j * b_j < upper_bound - gamma,
    to prune any binary assignment whose lower bound cannot beat the incumbent.

    Returns True if a non-trivial cut was added.
    """
    # Batch-fetch dual variables in a single C-level call, then use numpy
    pi = np.array(lp.getAttr("Pi"))

    # gamma = lambda^T d = pi . model_rhs
    gamma = float(pi @ model_rhs)

    # delta_j = c_{b,j} - (A_b^T pi)_j  for each discrete variable j (sparse matvec)
    delta_arr = discrete_obj_arr - np.asarray(A_d.T @ pi).ravel()

    adjusted_rhs_float = upper_bound - gamma
    for j, (L, _, _) in enumerate(discrete_encodings):
        adjusted_rhs_float -= delta_arr[j] * L

    # Cut: sum_j delta_j * b_j < upper_bound - gamma  (strict, so LEQ with -1 after scaling)
    # Use a conservative integerization to avoid over-pruning feasible assignments
    # due to floating-point/scaling artifacts near the optimum.
    bound_scaled = math.ceil(adjusted_rhs_float * scale - 1e-6) - 1

    pos_lits, pos_weights = [], []
    assumptions = []
    for j, (_, lits, weights) in enumerate(discrete_encodings):
        d = delta_arr[j]
        for lit, w in zip(lits, weights):
            coeff_float = d * w * scale
            coeff = int(math.floor(coeff_float + 1e-6))
            if coeff == 0:
                continue
            # prefer b_j=1 if δ<0, b_j=0 if δ>0 (minimizes the lower bound)
            assumptions.append(lit if coeff < 0 else -lit)
            pos_lits.append(lit)
            pos_weights.append(coeff)

    if not pos_lits:
        return False, assumptions

    norm_lits: list[int] = []
    norm_weights: list[int] = []
    adj_bound = bound_scaled
    for lit, weight in zip(pos_lits, pos_weights):
        if weight == 0:
            continue
        if weight < 0:
            norm_lits.append(-lit)
            norm_weights.append(-weight)
            adj_bound += -weight
        else:
            norm_lits.append(lit)
            norm_weights.append(weight)

    if not norm_lits:
        return False, assumptions
    if adj_bound < 0:
        return False, assumptions
    if adj_bound >= sum(norm_weights):
        return False, assumptions

    enc = PBEnc.leq(
        lits=norm_lits,
        weights=norm_weights,
        bound=adj_bound,
        vpool=vpool,
        encoding=EncType.best,
    )
    if guard_lit is None:
        solver.append_formula(enc.clauses)
    else:
        guarded = [[-guard_lit] + clause for clause in enc.clauses]
        solver.append_formula(guarded)
    return True, assumptions


def solve_smt(model: gp.Model, initial_lower_bound: float):
    model.update()

    # ------------------------------------------------------------------
    # 1. Assign SAT literals to discrete variables (1-based, positive ints)
    # ------------------------------------------------------------------
    discrete_encoding_by_col: dict[int, tuple[int, list[int], list[int]]] = {}
    next_lit = 1
    for var in model.getVars():
        if var.VType in (gp.GRB.BINARY, gp.GRB.INTEGER):
            L = int(math.ceil(var.LB))
            U = int(math.floor(var.UB))
            if L > U:
                print(f"Infeasible bounds for {var.VarName}: [{var.LB}, {var.UB}]")
                return
            if L == U:
                discrete_encoding_by_col[var.index] = (L, [], [])
            else:
                diff = U - L
                K = diff.bit_length()
                lits = []
                weights = []
                for k in range(K):
                    lits.append(next_lit)
                    weights.append(1 << k)
                    next_lit += 1
                discrete_encoding_by_col[var.index] = (L, lits, weights)

    # vpool allocates auxiliary variables well above the primary literals
    vpool = IDPool(start_from=next_lit)
    solver = sat.Solver()
    added_pb_constraints = 0

    # Add upper bound constraints for integer variables
    for col, (L, lits, weights) in discrete_encoding_by_col.items():
        if not lits:
            continue
        var = model.getVars()[col]
        U = int(math.floor(var.UB))
        diff = U - L
        if sum(weights) > diff:
            try:
                if _add_pb_constraint(solver, vpool, lits, weights, diff, '<'):
                    added_pb_constraints += 1
            except ValueError as e:
                print(f"Infeasible upper bound for {var.VarName}: {e}")
                return

    # ------------------------------------------------------------------
    # Precompute sparse discrete submatrix and RHS/obj vectors once.
    # These are reused every Benders iteration to avoid O(nnz) Python loops.
    # ------------------------------------------------------------------
    _model_constrs = model.getConstrs()
    model_rhs = np.array(model.getAttr("RHS", _model_constrs))
    model_A = model.getA()  # scipy sparse (n_constrs x n_vars)
    discrete_indices = list(discrete_encoding_by_col.keys())
    discrete_encodings = [discrete_encoding_by_col[i] for i in discrete_indices]
    A_d = model_A[:, discrete_indices]  # sparse (n_constrs x n_discrete)
    all_obj = np.array(model.getAttr("Obj", model.getVars()))
    discrete_obj_arr = all_obj[discrete_indices]

    # ------------------------------------------------------------------
    # 2. Add ONLY pure-discrete constraints to the SAT solver.
    #    Mixed constraints (discrete + continuous) are NOT encoded here.
    #    They are enforced implicitly through Benders cuts learned at
    #    runtime: when the LP subproblem is infeasible for a proposed
    #    discrete assignment, we derive a purely-discrete Farkas cut and
    #    add it to the SAT solver so that region is never revisited.
    # ------------------------------------------------------------------
    for constraint in model.getConstrs():
        row = model.getRow(constraint)
        has_continuous = any(
            row.getVar(i).VType not in (gp.GRB.BINARY, gp.GRB.INTEGER) for i in range(row.size())
        )
        if has_continuous:
            continue  # mixed row — handled via Benders cuts at solve time

        discrete_row_indices = [
            i for i in range(row.size()) if row.getVar(i).VType in (gp.GRB.BINARY, gp.GRB.INTEGER)
        ]
        if not discrete_row_indices:
            continue

        lits = []
        weights = []
        adjusted_rhs = constraint.RHS
        
        for i in discrete_row_indices:
            var = row.getVar(i)
            c_i = int(row.getCoeff(i))
            L, var_lits, var_weights = discrete_encoding_by_col[var.index]
            adjusted_rhs -= c_i * L
            for lit, w in zip(var_lits, var_weights):
                lits.append(lit)
                weights.append(c_i * w)
                
        adjusted_rhs = int(round(adjusted_rhs))

        if not lits:
            # All variables are fixed, just check if constraint is satisfied
            if constraint.Sense == "<" and 0 > adjusted_rhs:
                print(f"Infeasible fixed constraint: 0 <= {adjusted_rhs}")
                return
            elif constraint.Sense == ">" and 0 < adjusted_rhs:
                print(f"Infeasible fixed constraint: 0 >= {adjusted_rhs}")
                return
            elif constraint.Sense == "=" and 0 != adjusted_rhs:
                print(f"Infeasible fixed constraint: 0 == {adjusted_rhs}")
                return
            continue

        try:
            if _add_pb_constraint(solver, vpool, lits, weights, adjusted_rhs, constraint.Sense):
                added_pb_constraints += 1
        except ValueError as e:
            print(f"Infeasible constraint {constraint.ConstrName}: {e}")
            return

    print(f"Added {added_pb_constraints} pure-discrete PB constraints to SAT solver.")
    print(f"Discrete variables: {len(discrete_encoding_by_col)}")

    # ------------------------------------------------------------------
    # 3. Benders loop
    # ------------------------------------------------------------------
    best_obj: Optional[float] = None
    best_discrete_vals: Optional[dict[int, int]] = None
    iteration = 0
    guards = 0
    assumptions = []
    lp_subproblem: Optional[gp.Model] = None
    lower_bound = float(initial_lower_bound)
    target_bound: Optional[float] = None
    active_target_guard: Optional[int] = None
    gap_tolerance = 1.0

    while True:
        iteration += 1
        solver.set_phases(assumptions)  # guide SAT search towards improving the objective
        solve_assumptions: list[int] = []
        if active_target_guard is not None:
            solve_assumptions.append(active_target_guard)
        success = solver.solve(assumptions=solve_assumptions)
        if not success:
            print(f"[iter {iteration}] SAT solver exhausted — problem is infeasible.")
            break

        # Extract proposed discrete assignment from SAT model
        sat_model = solver.get_model()
        if sat_model is None:
            print(f"[iter {iteration}] SAT solver returned no model, terminating.")
            break
        sat_set = set(sat_model)
        discrete_vals = {}
        for col_idx, (L, lits, weights) in discrete_encoding_by_col.items():
            val = L
            for lit, w in zip(lits, weights):
                if lit in sat_set:
                    val += w
            discrete_vals[col_idx] = val

        # Solve LP subproblem with discrete variables fixed
        lp = _make_lp_subproblem(model, discrete_vals, discrete_indices, lp_subproblem)
        lp_subproblem = lp
        lp.optimize()

        status = lp.Status
        if status == gp.GRB.INFEASIBLE:
            if target_bound is not None:
                lower_bound = max(lower_bound, target_bound)
                active_target_guard = None

            # Benders feasibility cut: derive from Farkas dual and add to SAT.
            # Also always add a no-good clause to guarantee this exact assignment
            # is never reproposed (integer rounding in PB encoding can leave the
            # current point on the wrong side of the cut).
            farkas = np.array(lp.getAttr("FarkasDual"))
            added = _add_benders_feasibility_cut(
                solver, vpool, discrete_encodings, model_rhs, A_d, farkas
            )
            if not added:
                no_good = []
                for _, lits, _ in discrete_encodings:
                    for lit in lits:
                        no_good.append(-lit if lit in sat_set else lit)
                solver.add_clause(no_good)
            
            continue

        if status in (gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL):
            obj = lp.ObjVal
            if (iteration & 511) == 0:  # print every 512 iterations to avoid spamming the console
                print(f"[iter {iteration}] LP feasible, obj = {obj:.4f}, best_obj = {best_obj}")
            if best_obj is None or obj < best_obj:
                best_obj = obj
                best_discrete_vals = dict(discrete_vals)
                print(f"  -> [iter {iteration}] New best objective: {best_obj:.4f}")

            while True:

                if best_obj - lower_bound <= gap_tolerance:
                    print(
                        f"[iter {iteration}] Binary-search gap closed: "
                        f"lower={lower_bound:.4f}, upper={best_obj:.4f}"
                    )
                    break

                # Use binary-search bounds after first feasible point is found.
                # lower_bound starts from the relaxation value and best_obj is always
                # the best known feasible objective (upper bound).
                target_bound = 0.5 * (lower_bound + best_obj)
        
                guard_lit = vpool.id(f"target_cut_{guards}")
                guards += 1
                cut_added, assumptions = _add_benders_optimality_cut(
                    solver,
                    vpool,
                    discrete_encodings,
                    model_rhs,
                    discrete_obj_arr,
                    A_d,
                    lp,
                    target_bound,
                    guard_lit=guard_lit,
                )
                if cut_added:
                    # using assumptions as we don't have a way to remove a cut if it turns out to be too aggressive
                    target_ok = solver.solve(assumptions=[guard_lit])
                    if target_ok:
                        active_target_guard = guard_lit
                        break
                    else:
                        lower_bound = max(lower_bound, target_bound)
                        print(
                            f"[iter {iteration}] Target bound {target_bound:.4f} is infeasible, "
                            f"updating lower_bound to {lower_bound:.4f}"
                        )
                        active_target_guard = None
                else:
                    break

            if best_obj - lower_bound <= gap_tolerance:
                break

            if not cut_added:
                no_good = []
                for _, lits, _ in discrete_encodings:
                    for lit in lits:
                        no_good.append(-lit if lit in sat_set else lit)
                solver.add_clause(no_good)
            
            continue

        # Unbounded or other status — skip
        print(f"[iter {iteration}] LP status {status}, skipping.")
        no_good = []
        for _, lits, _ in discrete_encodings:
            for lit in lits:
                no_good.append(-lit if lit in sat_set else lit)
        solver.add_clause(no_good)

    print(f"\nFinished after {iteration} iterations.")
    if best_obj is not None:
        print(f"Best objective found: {best_obj:.4f}")
        if best_discrete_vals is not None and len(best_discrete_vals) <= 40:
            print(f"Best discrete assignment: {best_discrete_vals}")
    else:
        print("No feasible solution found.")


if __name__ == "__main__":
    import jsplib_loader as jl

    instance = jl.get_instances()["abz5"]
    model: gp.Model = instance.as_gurobi_balas_model(use_big_m=True, all_int=True)
    model = model.presolve()

    relaxed = model.relax()
    relaxed.optimize()
    assert relaxed.Status == gp.GRB.OPTIMAL, "LP relaxation is infeasible or unbounded"
    lower_bound = relaxed.ObjVal

    solve_smt(model, lower_bound)
