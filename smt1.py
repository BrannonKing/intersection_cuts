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


def _make_lp_subproblem(
    model: gp.Model, binary_vals: dict[int, int], binary_indices: list[int]
) -> gp.Model:
    """Create an LP subproblem by fixing binary variables to proposed values.

    binary_vals maps Gurobi var.index -> 0 or 1.
    binary_indices is the ordered list of binary var indices (positions in getVars()).
    Returns a new Model (LP relaxation with binary vars fixed).
    """
    lp = model.copy()
    lp.Params.OutputFlag = 0
    lp.Params.InfUnbdInfo = 1  # needed to get Farkas duals on infeasibility
    # Use positional indexing: var.index == position in getVars() after copy.
    # Batch all attribute writes with setAttr to avoid per-variable Python overhead.
    lp_vars = lp.getVars()
    binary_lp_vars = [lp_vars[i] for i in binary_indices]
    vals = [float(binary_vals[i]) for i in binary_indices]
    n = len(binary_lp_vars)
    lp.setAttr("VType", binary_lp_vars, [gp.GRB.CONTINUOUS] * n)
    lp.setAttr("LB", binary_lp_vars, vals)
    lp.setAttr("UB", binary_lp_vars, vals)
    lp.update()
    return lp


def _add_benders_feasibility_cut(
    solver: sat.Solver,
    vpool: IDPool,
    binary_lits_arr: np.ndarray,
    model_rhs: np.ndarray,
    A_b,  # scipy sparse (n_constrs x n_binary)
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
    # delta_j = sum_i y_i * A[i, j]  for each binary variable j (sparse matvec)
    rhs_float = float(y @ model_rhs)
    delta_arr = np.asarray(A_b.T @ y).ravel()

    # The cut to add to SAT is: sum_j (-delta_j) * x_j >= -rhs_float
    # Use floor for the bound so the integer cut is never tighter than the true cut
    # (avoids incorrectly excluding feasible binary assignments due to rounding).
    neg_rhs_scaled = math.floor(-rhs_float * scale)

    # PBEnc.geq requires positive weights; transform negative-coefficient terms:
    #   (-delta_j) * x_j  with (-delta_j) < 0
    # = |delta_j| * (1 - x_j) - |delta_j|
    # => move -|delta_j| to RHS: RHS += |delta_j|, use negated literal.
    pos_lits, pos_weights = [], []
    for lit, d in zip(binary_lits_arr, delta_arr):
        coeff = int(round(-d * scale))
        if coeff == 0:
            continue
        if coeff > 0:
            pos_lits.append(int(lit))
            pos_weights.append(coeff)
        else:
            # coeff < 0: substitute x_j = 1 - (1-x_j), flip literal, adjust RHS
            pos_lits.append(-int(lit))
            pos_weights.append(-coeff)
            neg_rhs_scaled -= coeff  # coeff < 0, so -= coeff adds |coeff|

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
    binary_lits_arr: np.ndarray,
    model_rhs: np.ndarray,
    binary_obj_arr: np.ndarray,
    A_b,  # scipy sparse (n_constrs x n_binary)
    lp: gp.Model,
    upper_bound: float,
    scale: float = 1e4,
) -> (bool, list[int]):
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

    # delta_j = c_{b,j} - (A_b^T pi)_j  for each binary variable j (sparse matvec)
    delta_arr = binary_obj_arr - np.asarray(A_b.T @ pi).ravel()

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
    for lit, d in zip(binary_lits_arr, delta_arr):
        coeff = int(round(d * scale))
        if coeff == 0:
            continue
        # prefer b_j=1 if δ<0, b_j=0 if δ>0 (minimizes the lower bound)
        # assumptions.append(int(lit) if coeff < 0 else -int(lit))  # makes it worse
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


def _sat_model_to_binary_vals(
    sat_model: list[int] | None, binary_lit_by_col: dict[int, int]
) -> dict[int, int]:
    """Convert a PySAT model (list of signed literals) to a dict col_index -> {0,1}."""
    if sat_model is None:
        return {}
    # Build a set of positive literals; checking `lit in sat_set` is O(1) and
    # avoids constructing an intermediate abs->sign dict.
    sat_set = set(sat_model)
    return {
        col_idx: (1 if lit in sat_set else 0)
        for col_idx, lit in binary_lit_by_col.items()
    }


def solve_smt(model: gp.Model):
    model.update()

    # ------------------------------------------------------------------
    # 1. Assign SAT literals to binary variables (1-based, positive ints)
    # ------------------------------------------------------------------
    binary_lit_by_col: dict[int, int] = {}
    next_lit = 1
    for var in model.getVars():
        if var.VType == gp.GRB.BINARY:
            binary_lit_by_col[var.index] = next_lit
            next_lit += 1

    # vpool allocates auxiliary variables well above the primary literals
    vpool = IDPool(start_from=next_lit)
    solver = sat.Solver()
    added_pb_constraints = 0

    # ------------------------------------------------------------------
    # Precompute sparse binary submatrix and RHS/obj vectors once.
    # These are reused every Benders iteration to avoid O(nnz) Python loops.
    # ------------------------------------------------------------------
    _model_constrs = model.getConstrs()
    model_rhs = np.array(model.getAttr("RHS", _model_constrs))
    model_A = model.getA()  # scipy sparse (n_constrs x n_vars)
    binary_indices = list(binary_lit_by_col.keys())
    binary_lits_arr = np.array([binary_lit_by_col[i] for i in binary_indices])
    A_b = model_A[:, binary_indices]  # sparse (n_constrs x n_binary)
    all_obj = np.array(model.getAttr("Obj", model.getVars()))
    binary_obj_arr = all_obj[binary_indices]

    # ------------------------------------------------------------------
    # 2. Add ONLY pure-binary constraints to the SAT solver.
    #    Mixed constraints (binary + continuous) are NOT encoded here.
    #    They are enforced implicitly through Benders cuts learned at
    #    runtime: when the LP subproblem is infeasible for a proposed
    #    binary assignment, we derive a purely-binary Farkas cut and
    #    add it to the SAT solver so that region is never revisited.
    # ------------------------------------------------------------------
    for constraint in model.getConstrs():
        row = model.getRow(constraint)
        has_continuous = any(
            row.getVar(i).VType != gp.GRB.BINARY for i in range(row.size())
        )
        if has_continuous:
            continue  # mixed row — handled via Benders cuts at solve time

        binary_row_indices = [
            i for i in range(row.size()) if row.getVar(i).VType == gp.GRB.BINARY
        ]
        if not binary_row_indices:
            continue

        coeffs = [int(row.getCoeff(i)) for i in binary_row_indices]
        lits = [binary_lit_by_col[row.getVar(i).index] for i in binary_row_indices]
        rhs = int(round(constraint.RHS))

        if constraint.Sense == "<":
            enc = PBEnc.leq(lits=lits, weights=coeffs, bound=rhs, vpool=vpool, encoding=EncType.best)
            solver.append_formula(enc.clauses)
            added_pb_constraints += 1
        elif constraint.Sense == ">":
            enc = PBEnc.geq(lits=lits, weights=coeffs, bound=rhs, vpool=vpool, encoding=EncType.best)
            solver.append_formula(enc.clauses)
            added_pb_constraints += 1
        elif constraint.Sense == "=":
            enc = PBEnc.equals(lits=lits, weights=coeffs, bound=rhs, vpool=vpool, encoding=EncType.best)
            solver.append_formula(enc.clauses)
            added_pb_constraints += 1

    print(f"Added {added_pb_constraints} pure-binary PB constraints to SAT solver.")
    print(f"Binary variables: {len(binary_lit_by_col)}")

    # ------------------------------------------------------------------
    # 3. Benders loop
    # ------------------------------------------------------------------
    best_obj: Optional[float] = None
    best_binary_vals: Optional[dict[int, int]] = None
    iteration = 0
    assumptions = []

    while True:
        iteration += 1
        solver.set_phases(assumptions)  # guide SAT search towards improving the objective
        success = solver.solve()
        if not success:
            print(f"[iter {iteration}] SAT solver exhausted — problem is infeasible.")
            break

        # Extract proposed binary assignment from SAT model
        sat_model = solver.get_model()
        binary_vals = _sat_model_to_binary_vals(sat_model, binary_lit_by_col)

        # Solve LP subproblem with binary variables fixed
        lp = _make_lp_subproblem(model, binary_vals, binary_indices)
        lp.optimize()

        status = lp.Status
        if status == gp.GRB.INFEASIBLE:
            # Benders feasibility cut: derive from Farkas dual and add to SAT.
            # Also always add a no-good clause to guarantee this exact assignment
            # is never reproposed (integer rounding in PB encoding can leave the
            # current point on the wrong side of the cut).
            farkas = np.array(lp.getAttr("FarkasDual"))
            added = _add_benders_feasibility_cut(
                solver, vpool, binary_lits_arr, model_rhs, A_b, farkas
            )
            if not added:
                no_good = [
                    -lit if binary_vals[col] == 1 else lit
                    for col, lit in binary_lit_by_col.items()
                ]
                solver.add_clause(no_good)
            print(f"  LP infeasible — added Benders cut: {added}")
            continue

        if status in (gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL):
            obj = lp.ObjVal
            print(f"[iter {iteration}] LP feasible, obj = {obj:.4f}")
            if best_obj is None or obj < best_obj:
                best_obj = obj
                best_binary_vals = dict(binary_vals)
                print(f"  -> New best objective: {best_obj:.4f}")

            # Benders optimality cut: prune assignments whose lower bound >= incumbent.
            # f(b') >= gamma + sum_j delta_j * b'_j  for all feasible b',
            # so we add: sum_j delta_j * b_j < best_obj  to SAT.
            cut_added, assumptions = _add_benders_optimality_cut(
                solver, vpool, binary_lits_arr, model_rhs, binary_obj_arr, A_b, lp, best_obj
            )
            if not cut_added:
                no_good = [
                    -lit if binary_vals[col] == 1 else lit
                    for col, lit in binary_lit_by_col.items()
                ]
                solver.add_clause(no_good)
            print(f"  -> Benders optimality cut added: {cut_added}")
            continue

        # Unbounded or other status — skip
        print(f"[iter {iteration}] LP status {status}, skipping.")
        no_good = [
            -lit if binary_vals[col] == 1 else lit
            for col, lit in binary_lit_by_col.items()
        ]
        solver.add_clause(no_good)

    print(f"\nFinished after {iteration} iterations.")
    if best_obj is not None:
        print(f"Best objective found: {best_obj:.4f}")
        print(f"Best binary assignment: {best_binary_vals}")
    else:
        print("No feasible solution found.")

if __name__ == "__main__":
    import jsplib_loader as jl

    instance = jl.get_instances()["abz4"]
    model: gp.Model = instance.as_gurobi_balas_model(use_big_m=True)
    solve_smt(model)