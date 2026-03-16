import math
import time
import pysat.solvers as sat
from pysat.formula import IDPool
from pysat.pb import PBEnc, EncType
from pysat.card import CardEnc, EncType as CardEncType
import gurobipy as gp
import numpy as np
from dataclasses import dataclass
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


@dataclass(frozen=True)
class IntegerVarEncoding:
    col_index: int
    values: np.ndarray
    lits: np.ndarray


def _is_integral_var(var: gp.Var) -> bool:
    return var.VType in (gp.GRB.BINARY, gp.GRB.INTEGER)


def _build_integer_var_encodings(
    model: gp.Model,
    max_onehot_domain_size: int = 16_000,
) -> tuple[list[IntegerVarEncoding], int]:
    """Create one-hot SAT encodings for every bounded integer/binary variable.

    For each integral variable x with integer domain [lb, ub], create literals
    z_v for each v in [lb, ub] and enforce exactly-one(z_v).

    This one-hot encoding is only practical for moderate domain sizes.
    """
    encodings: list[IntegerVarEncoding] = []
    next_lit = 1
    for var in model.getVars():
        if not _is_integral_var(var):
            continue

        lb, ub = float(var.LB), float(var.UB)
        if math.isinf(lb) or math.isinf(ub):
            raise ValueError(
                f"Integral variable {var.VarName} must have finite bounds to be SAT-encoded."
            )

        lb_i = int(math.ceil(lb - 1e-9))
        ub_i = int(math.floor(ub + 1e-9))
        if lb_i > ub_i:
            raise ValueError(
                f"Integral variable {var.VarName} has empty integer domain [{lb}, {ub}]."
            )

        domain_size = ub_i - lb_i + 1
        if domain_size > max_onehot_domain_size:
            raise ValueError(
                f"Integral variable {var.VarName} has one-hot domain size {domain_size}, "
                f"which exceeds the configured limit {max_onehot_domain_size}. "
                "Tighten bounds first (FBBT/OBBT/presolve on original model) or use a non-one-hot encoding."
            )

        values = np.arange(lb_i, ub_i + 1, dtype=np.int64)
        lits = np.arange(next_lit, next_lit + len(values), dtype=np.int64)
        next_lit += len(values)

        encodings.append(IntegerVarEncoding(var.index, values, lits))

    return encodings, next_lit


def _add_exactly_one_constraints(
    solver: sat.Solver,
    vpool: IDPool,
    int_encodings: list[IntegerVarEncoding],
) -> int:
    """Add exactly-one constraints for each integer variable encoding."""
    added = 0
    for enc in int_encodings:
        lits = enc.lits.tolist()
        if len(lits) == 1:
            solver.add_clause([lits[0]])
            added += 1
            continue

        eq = CardEnc.equals(
            lits=lits,
            bound=1,
            vpool=vpool,
            encoding=CardEncType.seqcounter,
        )
        solver.append_formula(eq.clauses)
        added += 1
    return added


def _encode_signed_leq(
    solver: sat.Solver,
    vpool: IDPool,
    lits: list[int],
    coeffs: list[int],
    bound: int,
) -> bool:
    """Encode sum_i coeffs[i] * lits[i] <= bound with signed integer coeffs."""
    pos_lits: list[int] = []
    pos_weights: list[int] = []
    adj_bound = int(bound)

    for lit, coeff in zip(lits, coeffs):
        if coeff == 0:
            continue
        if coeff > 0:
            pos_lits.append(int(lit))
            pos_weights.append(int(coeff))
        else:
            pos_lits.append(-int(lit))
            pos_weights.append(int(-coeff))
            adj_bound += int(-coeff)

    if not pos_lits:
        if 0 <= adj_bound:
            return False
        solver.add_clause([])
        return True

    enc = PBEnc.leq(
        lits=pos_lits,
        weights=pos_weights,
        bound=adj_bound,
        vpool=vpool,
        encoding=EncType.best,
    )
    solver.append_formula(enc.clauses)
    return True


def _encode_signed_geq(
    solver: sat.Solver,
    vpool: IDPool,
    lits: list[int],
    coeffs: list[int],
    bound: int,
) -> bool:
    """Encode sum_i coeffs[i] * lits[i] >= bound with signed integer coeffs."""
    neg_coeffs = [-c for c in coeffs]
    return _encode_signed_leq(solver, vpool, lits, neg_coeffs, -bound)


def _make_lp_subproblem(
    model: gp.Model,
    integer_vals: dict[int, int],
    integer_indices: list[int],
    lp: Optional[gp.Model] = None,
) -> gp.Model:
    """Create/update an LP subproblem by fixing integer variables to proposed values.

    integer_vals maps Gurobi var.index -> integer assignment.
    integer_indices is the ordered list of integer var indices (positions in getVars()).
    If ``lp`` is provided, it is reused and only bounds/fixings are updated.
    """
    if lp is None:
        lp = model.copy()
        lp.params.OutputFlag = 0
        lp.params.InfUnbdInfo = 1  # needed to get Farkas duals on infeasibility
        lp.params.Method = 1  # use dual simplex to get better Farkas duals
        lp_vars = lp.getVars()
        lp._integer_lp_vars = [lp_vars[i] for i in integer_indices]
        n = len(lp._integer_lp_vars)
        lp.setAttr("VType", lp._integer_lp_vars, [gp.GRB.CONTINUOUS] * n)
    # Use positional indexing: var.index == position in getVars() after copy.
    # Batch all attribute writes with setAttr to avoid per-variable Python overhead.
    vals = [float(integer_vals[i]) for i in integer_indices]
    lp.setAttr("LB", lp._integer_lp_vars, vals)
    lp.setAttr("UB", lp._integer_lp_vars, vals)
    lp.update()
    return lp


def _add_benders_feasibility_cut(
    solver: sat.Solver,
    vpool: IDPool,
    int_encodings: list[IntegerVarEncoding],
    model_rhs: np.ndarray,
    A_i,  # scipy sparse (n_constrs x n_integer_vars)
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
    # delta_j = sum_i y_i * A[i, j]  for each integer variable j (sparse matvec)
    rhs_float = float(y @ model_rhs)
    delta_arr = np.asarray(A_i.T @ y).ravel()

    # The cut to add to SAT is: sum_j (-delta_j) * x_j >= -rhs_float.
    # For validity after integerization, first rewrite to positive-weight literals,
    # then use: weight <- ceil(weight*scale), bound <- floor(bound*scale).
    # This can only weaken the GEQ cut (never tighten it), so feasible points are
    # never excluded due to rounding.
    neg_rhs_scaled_float = -rhs_float * scale

    # PBEnc.geq requires positive weights; transform negative-coefficient terms:
    #   a_j * x_j with a_j < 0
    # = |a_j| * (1 - x_j) - |a_j|,
    # so move -|a_j| to RHS: RHS += |a_j| and flip the literal.
    pos_lits, pos_weights = [], []
    for d, enc in zip(delta_arr, int_encodings):
        for lit, value in zip(enc.lits, enc.values):
            a_scaled = float(-d * float(value)) * scale
            if a_scaled == 0.0:
                continue
            if a_scaled > 0.0:
                pos_lits.append(int(lit))
                pos_weights.append(int(math.ceil(a_scaled)))
            else:
                abs_a_scaled = -a_scaled
                pos_lits.append(-int(lit))
                pos_weights.append(int(math.ceil(abs_a_scaled)))
                neg_rhs_scaled_float += abs_a_scaled

    neg_rhs_scaled = int(math.floor(neg_rhs_scaled_float))

    if not pos_lits:
        return False

    # If bound <= 0 the GEQ constraint is trivially satisfied (LHS >= 0 always)
    if neg_rhs_scaled <= 0:
        return False

    enc = PBEnc.geq(
        lits=pos_lits,
        weights=pos_weights,
        bound=neg_rhs_scaled,
        vpool=vpool,
        encoding=EncType.best,
    )
    solver.append_formula(enc.clauses)
    return True


def _add_benders_optimality_cut(
    solver: sat.Solver,
    vpool: IDPool,
    int_encodings: list[IntegerVarEncoding],
    model_rhs: np.ndarray,
    integer_obj_arr: np.ndarray,
    A_i,  # scipy sparse (n_constrs x n_integer_vars)
    lp: gp.Model,
    upper_bound: float,
    scale: float = 1024,
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

    # delta_j = c_{i,j} - (A_i^T pi)_j  for each integer variable j (sparse matvec)
    delta_arr = integer_obj_arr - np.asarray(A_i.T @ pi).ravel()

    # Cut: sum_j delta_j * b_j < upper_bound - gamma  (strict, so LEQ with -1 after scaling)
    rhs_float = upper_bound - gamma
    # Use ceiling so the scaled integer cut is never looser than the true cut.
    bound_scaled = math.ceil(rhs_float * scale) - 1

    # PBEnc.leq requires positive weights; handle negative-coefficient terms:
    #   delta_j * b_j  with delta_j < 0
    # = |delta_j| * (1 - b_j) - |delta_j|
    # => move -|delta_j| to the RHS: bound += |delta_j|, use negated literal.
    pos_lits, pos_weights = [], []
    assumptions = []
    for d, enc in zip(delta_arr, int_encodings):
        value_costs = d * enc.values.astype(float)
        best_idx = int(np.argmin(value_costs))
        best_lit = int(enc.lits[best_idx])
        assumptions.extend([-int(lit) for lit in enc.lits if int(lit) != best_lit])
        assumptions.append(best_lit)

        for lit, value in zip(enc.lits, enc.values):
            coeff = int(math.floor(float(d * float(value)) * scale))
            if coeff == 0:
                continue
            if coeff > 0:
                pos_lits.append(int(lit))
                pos_weights.append(coeff)
            else:
                pos_lits.append(-int(lit))
                pos_weights.append(-coeff)
                bound_scaled += (-coeff)  # coeff < 0, so += |coeff|

    # set_phases is a polarity map; order doesn't matter, so no need to sort or slice.

    if not pos_lits:
        return False, assumptions

    enc = PBEnc.leq(
        lits=pos_lits,
        weights=pos_weights,
        bound=bound_scaled,
        vpool=vpool,
        encoding=EncType.best,
    )
    solver.append_formula(enc.clauses)
    return True, assumptions


def _sat_model_to_integer_vals(
    sat_model: list[int] | None,
    int_encodings: list[IntegerVarEncoding],
) -> tuple[dict[int, int], dict[int, int]]:
    """Convert a PySAT model to integer assignments and selected literals.

    Returns:
      - assignments: col_index -> chosen integer value
      - selected_lits: col_index -> SAT literal corresponding to chosen value
    """
    if sat_model is None:
        return {}, {}
    # Build a set of positive literals; checking `lit in sat_set` is O(1) and
    # avoids constructing an intermediate abs->sign dict.
    sat_set = set(sat_model)
    assignments: dict[int, int] = {}
    selected_lits: dict[int, int] = {}
    for enc in int_encodings:
        chosen_lit = int(enc.lits[0])
        chosen_val = int(enc.values[0])
        for lit, value in zip(enc.lits, enc.values):
            if int(lit) in sat_set:
                chosen_lit = int(lit)
                chosen_val = int(value)
                break
        assignments[enc.col_index] = chosen_val
        selected_lits[enc.col_index] = chosen_lit
    return assignments, selected_lits


def solve_smt(
    model: gp.Model,
    max_onehot_domain_size: int = 10_000,
    encode_static_pure_integer_rows: bool = False,
    static_row_term_limit: int = 2000,
    static_weight_limit: int = 10_000_000,
):

    # ------------------------------------------------------------------
    # 1. Assign SAT literals to bounded integer/binary variables
    # ------------------------------------------------------------------
    int_encodings, next_lit = _build_integer_var_encodings(
        model,
        max_onehot_domain_size=max_onehot_domain_size,
    )
    int_encoding_by_col = {enc.col_index: enc for enc in int_encodings}

    # vpool allocates auxiliary variables well above the primary literals
    vpool = IDPool(start_from=next_lit)
    solver = sat.Solver()
    added_exactly_one = _add_exactly_one_constraints(solver, vpool, int_encodings)
    added_pb_constraints = 0

    # ------------------------------------------------------------------
    # Precompute sparse integer submatrix and RHS/obj vectors once.
    # These are reused every Benders iteration to avoid O(nnz) Python loops.
    # ------------------------------------------------------------------
    _model_constrs = model.getConstrs()
    model_rhs = np.array(model.getAttr("RHS", _model_constrs))
    model_A = model.getA()  # scipy sparse (n_constrs x n_vars)
    integer_indices = [enc.col_index for enc in int_encodings]
    A_i = model_A[:, integer_indices]  # sparse (n_constrs x n_integer)
    all_obj = np.array(model.getAttr("Obj", model.getVars()))
    integer_obj_arr = all_obj[integer_indices]

    # ------------------------------------------------------------------
    # 2. Add ONLY pure-integer constraints to the SAT solver.
    #    Mixed constraints (integer + continuous) are NOT encoded here.
    #    They are enforced implicitly through Benders cuts learned at
    #    runtime: when the LP subproblem is infeasible for a proposed
    #    integer assignment, we derive a purely-integer Farkas cut and
    #    add it to the SAT solver so that region is never revisited.
    # ------------------------------------------------------------------
    skipped_nonintegral_rows = 0
    skipped_large_rows = 0
    skipped_large_weight_rows = 0
    skipped_continuous_rows = 0
    static_start = time.time()

    total_sat_literals = sum(len(enc.lits) for enc in int_encodings)
    print(
        f"Added {added_exactly_one} exactly-one constraints for integer vars "
        f"({len(int_encodings)} vars, {total_sat_literals} one-hot literals)."
    )

    encode_static_rows = encode_static_pure_integer_rows and (total_sat_literals <= 50_000)

    for row_idx, constraint in enumerate(model.getConstrs(), start=1):
        row = model.getRow(constraint)
        has_continuous = any(
            not _is_integral_var(row.getVar(i)) for i in range(row.size())
        )
        if has_continuous:
            skipped_continuous_rows += 1
            continue  # mixed row — handled via Benders cuts at solve time

        if row.size() == 0:
            continue

        expanded_terms = 0
        for i in range(row.size()):
            expanded_terms += len(int_encoding_by_col[row.getVar(i).index].lits)

        if not encode_static_rows or expanded_terms > static_row_term_limit:
            skipped_large_rows += 1
            continue

        lits: list[int] = []
        coeffs: list[int] = []
        can_encode_row = True
        row_weight_overflow = False
        row_nonintegral = False

        for i in range(row.size()):
            var = row.getVar(i)
            coeff = float(row.getCoeff(i))
            enc = int_encoding_by_col[var.index]
            for lit, value in zip(enc.lits, enc.values):
                c = coeff * float(value)
                c_rounded = int(round(c))
                if abs(c - c_rounded) > 1e-6:
                    can_encode_row = False
                    row_nonintegral = True
                    break
                if abs(c_rounded) > static_weight_limit:
                    can_encode_row = False
                    row_weight_overflow = True
                    break
                lits.append(int(lit))
                coeffs.append(c_rounded)
            if not can_encode_row:
                break

        rhs = float(constraint.RHS)
        rhs_rounded = int(round(rhs))
        if abs(rhs - rhs_rounded) > 1e-6:
            can_encode_row = False
            row_nonintegral = True

        if not can_encode_row:
            if row_weight_overflow:
                skipped_large_weight_rows += 1
            if row_nonintegral:
                skipped_nonintegral_rows += 1
            continue

        if constraint.Sense == "<":
            added = _encode_signed_leq(solver, vpool, lits, coeffs, rhs_rounded)
            if added:
                added_pb_constraints += 1
        elif constraint.Sense == ">":
            added = _encode_signed_geq(solver, vpool, lits, coeffs, rhs_rounded)
            if added:
                added_pb_constraints += 1
        elif constraint.Sense == "=":
            added_leq = _encode_signed_leq(solver, vpool, lits, coeffs, rhs_rounded)
            added_geq = _encode_signed_geq(solver, vpool, lits, coeffs, rhs_rounded)
            if added_leq or added_geq:
                added_pb_constraints += 1

    print(f"Added {added_pb_constraints} pure-integer PB constraints to SAT solver.")
    print(
        f"Static row encoding: {'enabled' if encode_static_rows else 'disabled'}"
        f" (term limit={static_row_term_limit}, weight limit={static_weight_limit})"
    )
    print(
        f"Skipped rows -> mixed:{skipped_continuous_rows}, large:{skipped_large_rows},"
        f" large-weight:{skipped_large_weight_rows}, non-integral:{skipped_nonintegral_rows}"
    )
    if skipped_nonintegral_rows > 0 or skipped_large_rows > 0 or skipped_large_weight_rows > 0:
        print(
            "Some pure-integer rows were not statically encoded in SAT;"
            " they are handled dynamically by Benders cuts."
        )
    print(f"Static encoding time: {time.time() - static_start:.2f}s")

    # ------------------------------------------------------------------
    # 3. Benders loop
    # ------------------------------------------------------------------
    best_obj: Optional[float] = None
    best_integer_vals: Optional[dict[int, int]] = None
    iteration = 0
    assumptions = []
    lp_subproblem: Optional[gp.Model] = None

    while True:
        iteration += 1
        solver.set_phases(assumptions)  # guide SAT search towards improving the objective
        success = solver.solve()
        if not success:
            print(f"[iter {iteration}] SAT solver exhausted — problem is infeasible.")
            break

        # Extract proposed integer assignment from SAT model
        sat_model = solver.get_model()
        integer_vals, selected_lits = _sat_model_to_integer_vals(sat_model, int_encodings)

        # Solve LP subproblem with integer variables fixed
        lp = _make_lp_subproblem(model, integer_vals, integer_indices, lp_subproblem)
        lp_subproblem = lp
        lp.optimize()

        status = lp.Status
        if status == gp.GRB.INFEASIBLE:
            # Benders feasibility cut: derive from Farkas dual and add to SAT.
            # Also always add a no-good clause to guarantee this exact assignment
            # is never reproposed (integer rounding in PB encoding can leave the
            # current point on the wrong side of the cut).
            farkas = np.array(lp.getAttr("FarkasDual"))
            added = _add_benders_feasibility_cut(
                solver, vpool, int_encodings, model_rhs, A_i, farkas
            )
            if not added:
                no_good = [-lit for lit in selected_lits.values()]
                solver.add_clause(no_good)
            
            continue

        if status in (gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL):
            obj = lp.ObjVal
            if (iteration & 511) == 0:  # print every 512 iterations to avoid spamming the console
                print(f"[iter {iteration}] LP feasible, obj = {obj:.4f}, best_obj = {best_obj}")
            if best_obj is None or obj < best_obj:
                best_obj = obj
                best_integer_vals = dict(integer_vals)
                print(f"  -> [iter {iteration}] New best objective: {best_obj:.4f}")

            # Benders optimality cut: prune assignments whose lower bound >= incumbent.
            # f(b') >= gamma + sum_j delta_j * b'_j  for all feasible b',
            # so we add: sum_j delta_j * b_j < best_obj  to SAT.
            cut_added, assumptions = _add_benders_optimality_cut(
                solver,
                vpool,
                int_encodings,
                model_rhs,
                integer_obj_arr,
                A_i,
                lp,
                best_obj,
            )
            if not cut_added:
                no_good = [-lit for lit in selected_lits.values()]
                solver.add_clause(no_good)
            
            continue

        # Unbounded or other status — skip
        print(f"[iter {iteration}] LP status {status}, skipping.")
        no_good = [-lit for lit in selected_lits.values()]
        solver.add_clause(no_good)

    print(f"\nFinished after {iteration} iterations.")
    if best_obj is not None:
        print(f"Best objective found: {best_obj:.4f}")
        print(f"Best integer assignment: {best_integer_vals}")
    else:
        print("No feasible solution found.")

if __name__ == "__main__":
    import jsplib_loader as jl

    instance = jl.get_instances()["abz3"]
    model: gp.Model = instance.as_gurobi_balas_model(use_big_m=True, all_int=True)
    model = model.presolve()
    solve_smt(model)