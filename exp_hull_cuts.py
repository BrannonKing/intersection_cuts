import gurobipy as gp
gp.setParam('OutputFlag', 0)  # suppress Gurobi output for this experiment
import gurobi_utils as gu
import numpy as np
import scipy.spatial as spatial
import lll_utils as lu


def are_in_half_plane(vectors, tol):
    # 0. Filter out 0-length vectors
    magnitudes = np.linalg.norm(vectors, axis=1)
    vectors = vectors[magnitudes > tol]
    
    if len(vectors) <= 1:
        print("  Only", len(vectors), "nonzero vectors; treating as in a half-plane.")
        return False, None, None

    # 1. Convert to angles in the range [-pi, pi]
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])  # TODO: this can probably be computed without trig
    
    # 2. Get sorted indices to map back to original vectors
    sorted_indices = np.argsort(angles)
    sorted_angles = angles[sorted_indices]  # TODO: speedup to not use this directly
    
    # 3. Compute differences between adjacent angles
    # np.diff gets gaps between 1st and 2nd, 2nd and 3rd, etc.
    gaps = np.diff(sorted_angles)
    
    # 4. Check the wrap-around gap (between the last and first angle)
    # The total circle is 2*pi.
    wrap_around_gap = 2 * np.pi - (sorted_angles[-1] - sorted_angles[0])
    
    # Find the largest gap
    # If the largest gap is >= pi, they are in a half-plane
    max_gap_idx = np.argmax(gaps)
    if gaps[max_gap_idx] < wrap_around_gap:
        # The wrap-around gap is the empty space (from last to first)
        # The cone containing the vectors is from the first to the last
        v1 = vectors[sorted_indices[0]]
        v2 = vectors[sorted_indices[-1]]
        max_gap = wrap_around_gap
    else:
        # The gap is between max_gap_idx and max_gap_idx + 1
        # The cone containing the vectors starts at max_gap_idx + 1 and wraps around to max_gap_idx
        v1 = vectors[sorted_indices[max_gap_idx + 1]]
        v2 = vectors[sorted_indices[max_gap_idx]]
        max_gap = gaps[max_gap_idx]
        
    if max_gap > np.pi + tol:
        assert v1[0]*v2[1] - v1[1]*v2[0] != 0
        return True, v1, v2
    return False, v1, v2

from math import sqrt, floor, ceil

def wedge_side_points(
    v1: tuple[float, float],
    v2: tuple[float, float],
    p0: tuple   [float, float],
    num_steps: int = 16,
):
    """
    Enumerate integer lattice points along each ray on the interior
    side of the wedge formed by v1 and v2 from origin p0.

    The interior is the side where the angle between the rays is < 180°,
    i.e. the side determined by the sign of cross(v1, v2).

    Returns dict with keys 'v1' and 'v2', each a list of (x, y) ints.
    """
    v1x, v1y = v1
    v2x, v2y = v2
    p0x, p0y = p0

    # cross(v1, v2) > 0 → v2 is to the left of v1 (CCW wedge)
    # cross(v1, v2) < 0 → v2 is to the right of v1 (CW wedge)
    winding = v1x * v2y - v1y * v2x
    assert winding != 0, "Rays must not be parallel or opposing"

    # The interior side of v1 is whichever side v2 lies on.
    # sign(cross(v1, d)) == sign(winding)  →  d is on v2's side of v1.
    # Similarly for v2: interior side is whichever side v1 lies on,
    # which is the *opposite* winding sign.
    side_of_v1 = 1 if winding > 0 else -1   # interior side for v1
    side_of_v2 = -side_of_v1                 # interior side for v2

    def collect(vx, vy, ox, oy, side):
        # ox,oy = the other ray's direction
        # A point is only emitted if it passes both halfplane tests
        other_side = -side  # interior side of the other ray
        mag = sqrt(vx*vx + vy*vy)
        result = []
        for i in range(num_steps):
            t = i / mag
            rx, ry = p0x + vx*t, p0y + vy*t
            for gx in range(floor(rx), ceil(rx)+1):
                for gy in range(floor(ry), ceil(ry)+1):
                    dx, dy = gx - p0x, gy - p0y
                    cross_this  =  vx*dy -  vy*dx
                    cross_other = ox*dy - oy*dx
                    if cross_this * side >= 0 and cross_other * other_side >= 0:
                        pt = (gx, gy)
                        result.append(pt)
        return result

    return collect(v1x, v1y, v2x, v2y, side_of_v1), collect(v2x, v2y, v1x, v1y, side_of_v2)

def visible_hull_facets(all_pts, px):
    hull = spatial.ConvexHull(all_pts)

    visible_facets = []
    for simplex, equation in zip(hull.simplices, hull.equations):
        normal = equation[:2]   # 2D: the outward-facing normal
        offset = equation[2]    # hull equation: normal @ x + offset = 0
        # positive value => px is on the outside (visible) side
        if normal @ px + offset > 0:
            p1 = all_pts[simplex[0]]
            p2 = all_pts[simplex[1]]
            visible_facets.append((p1, p2))
    return visible_facets

def build_cut_expr(normal, bvar1, bvar2, rhs, variables, constraints, relaxed):
    cut_expr = gp.LinExpr()
    cut_expr.addTerms(normal[0], variables[bvar1])
    if bvar2 < relaxed.NumVars:
        cut_expr.addTerms(normal[1], variables[bvar2])
    else:
        con = constraints[bvar2 - relaxed.NumVars]
        a_i = relaxed.getRow(con)
        if con.Sense == '<':
            rhs -= normal[1] * con.RHS
            for j in range(a_i.size()):
                cut_expr.addTerms(-normal[1] * a_i.getCoeff(j), a_i.getVar(j))
        elif con.Sense == '>':
            rhs += normal[1] * con.RHS
            for j in range(a_i.size()):
                cut_expr.addTerms(normal[1] * a_i.getCoeff(j), a_i.getVar(j))
    return cut_expr, rhs

def make_cuts(
    basis,
    tableau,
    col_to_var_idx,
    x,
    int_var_idx,
    variables,
    constraints,
    relaxed,
    tol,
    verbose=False,
    max_simple_cuts=200,
):
    tableau_orig = tableau.copy()
    # Shift the tableau so that all non-basic variables are treated as >= 0
    # This guarantees that the tableau columns represent rays pointing into the feasible region.
    betas, tableau = gu.shift_to_x_gt_0(basis, tableau, col_to_var_idx, variables, constraints, x, relaxed)

    int_row_indices = [i for i, b in enumerate(basis) if b < relaxed.NumVars and b in int_var_idx]
    U_row = None
    if int_row_indices:
        import lll_utils
        tableau_int = tableau[int_row_indices, :]
        # B_val, U_val = lll_utils.lll_apx(tableau_int.T)
        B_val, U_val = lll_utils.lll(tableau_int.T)
        U_candidate = U_val.T
        U_int = np.rint(U_candidate)
        use_transform = np.allclose(U_candidate, U_int, atol=1e-9)
        if use_transform:
            det = round(np.linalg.det(U_int))
            use_transform = abs(det) == 1
        if use_transform:
            U_row = U_int
            tableau[int_row_indices, :] = B_val.T
            betas[int_row_indices, :] = U_row @ betas[int_row_indices, :]
            # Now the x_B variables in these rows are y = U_row @ x_{B, int}
            # We need to compute px using the new betas
        else:
            U_row = np.eye(len(int_row_indices))
            if verbose:
                print("  Skipping non-unimodular/non-integer LLL row transform for validity")

    def build_cut_expr_transformed(normal, bi1, bi2, rhs):
        cut_expr = gp.LinExpr()
        adjusted_rhs = rhs
        
        # Determine weights for all integer rows
        w_int = np.zeros(len(int_row_indices))
        k1 = int_row_indices.index(bi1)
        w_int += normal[0] * U_row[k1, :]
        
        bvar2 = basis[bi2]
        bvar2_is_int = bvar2 in int_var_idx
        if bvar2_is_int:
            k2 = int_row_indices.index(bi2)
            w_int += normal[1] * U_row[k2, :]
            
        for ki, bi in enumerate(int_row_indices):
            w = w_int[ki]
            if abs(w) > 1e-12:
                cut_expr.addTerms(w, variables[basis[bi]])
                
        if not bvar2_is_int:
            w2 = normal[1]
            if abs(w2) > 1e-12:
                if bvar2 < relaxed.NumVars:
                    cut_expr.addTerms(w2, variables[bvar2])
                else:
                    con = constraints[bvar2 - relaxed.NumVars]
                    a_i = relaxed.getRow(con)
                    if con.Sense == '<':
                        adjusted_rhs -= w2 * con.RHS
                        for j in range(a_i.size()):
                            cut_expr.addTerms(-w2 * a_i.getCoeff(j), a_i.getVar(j))
                    elif con.Sense == '>':
                        adjusted_rhs += w2 * con.RHS
                        for j in range(a_i.size()):
                            cut_expr.addTerms(w2 * a_i.getCoeff(j), a_i.getVar(j))
                            
        return cut_expr, adjusted_rhs

    def cut_violation_at_current_solution(cut_expr, rhs):
        lhs = 0.0
        for i in range(cut_expr.size()):
            lhs += cut_expr.getCoeff(i) * x[cut_expr.getVar(i).index, 0]
        return rhs - lhs

    # for each pair of basis row that is supposed to be integer but is not,
    # see if all those 2D vectors are on the same side of some line through the origin.
    # if so, enumerate some integer points along each ray and find their convex hull.
    # take the line of the convex hull closest to the solution and add it as a cut.

    cut_records = []
    simple_cut_candidates = []
    stats = {"hull": 0, "simple": 0, "split": 0}

    for bi1, bvar1 in enumerate(basis):
        if bvar1 >= relaxed.NumVars or bvar1 not in int_var_idx:
            continue
        
        # verify the transformed basic variable is fractional:
        if abs(betas[bi1, 0] - round(betas[bi1, 0])) < tol:
            continue

        for bi2, bvar2 in enumerate(basis[bi1 + 1:], bi1 + 1):
            bvar2_is_int = bvar2 in int_var_idx

            # the ray directions are -tableau since x_{B} = beta - tableau * x_N
            vectors = -tableau[[bi1, bi2], :]
            in_half_plane, v1, v2 = are_in_half_plane(vectors.T, tol)
            if not in_half_plane:
                if verbose:
                    print("  Basis rows", bi1, "and", bi2, "are not in a half-plane; skipping.")
                    # run the split cut
                continue
            assert v1 is not None and v2 is not None

            px = (betas[bi1, 0], betas[bi2, 0])

            if bvar2_is_int:
                pts1, pts2 = wedge_side_points(v1, v2, px, num_steps=16)

                for p1, p2 in visible_hull_facets(pts1 + pts2, px):
                    assert p1 != px and p2 != px, "The solution point should not be a vertex of the hull."
                    # The cut is the line through p1 and p2. We want the one that cuts off px.
                    # The normal vector is perpendicular to the line from p1 to p2.
                    normal = np.array([p2[1] - p1[1], p1[0] - p2[0]])
                    if np.dot(normal, np.array(px) - p1) > 0:
                        normal = -normal
                    normal = normal / np.linalg.norm(normal)
                    rhs = np.dot(normal, p1)
                    if np.dot(normal, px) - rhs >= -tol:
                        print("  WARNING: Facet through", p1, "and", p2, "does not cut off solution", px)
                        continue  # this facet does not cut off the current solution

                    if verbose:
                        print("  Adding cut from facet through", p1, "and", p2, "with normal", normal, "and rhs", rhs)

                    # now add the constraint in terms of the original variables:
                    cut_expr, adjusted_rhs = build_cut_expr_transformed(normal, bi1, bi2, rhs)
                    if cut_violation_at_current_solution(cut_expr, adjusted_rhs) <= tol:
                        if verbose:
                            print("  Skipping hull cut: not violated at current LP solution after mapping")
                        continue
                    cut_records.append(
                        (
                            "hull",
                            cut_expr,
                            adjusted_rhs,
                            {
                                "p1": p1,
                                "p2": p2,
                                "bi1": bi1,
                                "bi2": bi2,
                                "bvar1": bvar1,
                                "bvar2": bvar2,
                                "px": px,
                            },
                        )
                    )
                    stats["hull"] += 1
            else:
                # Strongest intersection cut when x is integer and y is continuous.
                # Equivalent to a split cut on x <= floor(px[0]) or x >= ceil(px[0]).
                split_axis_tol = max(10.0 * tol, 1e-9)
                geom_tol = max(100.0 * tol, 1e-8)

                def fallback_vertical_split():
                    # Conservative fallback: add one side of the split disjunction directly.
                    left_violation = px[0] - np.floor(px[0])
                    right_violation = np.ceil(px[0]) - px[0]
                    if right_violation >= left_violation:
                        n = np.array([1.0, 0.0])
                        p = (float(np.ceil(px[0])), px[1])
                    else:
                        n = np.array([-1.0, 0.0])
                        p = (float(np.floor(px[0])), px[1])
                    return n, p

                def get_t(vx):
                    if vx > split_axis_tol:
                        return (np.ceil(px[0]) - px[0]) / vx
                    elif vx < -split_axis_tol:
                        return (np.floor(px[0]) - px[0]) / vx
                    else:
                        return float('inf')
                
                t1 = get_t(v1[0])
                t2 = get_t(v2[0])
                
                simple_case = "finite"
                if np.isinf(t1) and np.isinf(t2):
                    simple_case = "both_inf"
                    # Both rays are vertical and x is strictly fractional.
                    # The cone contains no feasible integer x. Cut off px with vertical line.
                    normal = np.array([1.0, 0.0])
                    p1 = (np.ceil(px[0]), px[1])
                    p2 = p1
                elif np.isinf(t1):
                    simple_case = "t1_inf"
                    p2 = (px[0] + t2 * v2[0], px[1] + t2 * v2[1])
                    # One ray is parallel to split boundaries; use vertical split facet directly.
                    xk = float(np.round(p2[0]))
                    normal = np.array([1.0, 0.0]) if xk >= px[0] else np.array([-1.0, 0.0])
                    p1 = p2
                elif np.isinf(t2):
                    simple_case = "t2_inf"
                    p1 = (px[0] + t1 * v1[0], px[1] + t1 * v1[1])
                    # One ray is parallel to split boundaries; use vertical split facet directly.
                    xk = float(np.round(p1[0]))
                    normal = np.array([1.0, 0.0]) if xk >= px[0] else np.array([-1.0, 0.0])
                    p2 = p1
                else:  # case = "finite"
                    x1_target = float(np.ceil(px[0]) if v1[0] > 0 else np.floor(px[0]))
                    x2_target = float(np.ceil(px[0]) if v2[0] > 0 else np.floor(px[0]))
                    p1 = (x1_target, px[1] + t1 * v1[1])
                    p2 = (x2_target, px[1] + t2 * v2[1])
                    normal = np.array([p2[1] - p1[1], p1[0] - p2[0]])
                    if np.linalg.norm(normal) < geom_tol:
                        normal, p1 = fallback_vertical_split()
                        p2 = p1
                
                if np.linalg.norm(normal) > tol:
                    normal = normal / np.linalg.norm(normal)
                
                if np.dot(normal, np.array(px) - p1) > 0:
                    normal = -normal
                    
                rhs = np.dot(normal, p1)
                if np.dot(normal, px) - rhs >= -tol:
                    continue

                if verbose:
                    print("  Adding cut from facet through", p1, "and", p2, "with normal", normal, "and rhs", rhs)

                cut_expr, adjusted_rhs = build_cut_expr_transformed(normal, bi1, bi2, rhs)
                # Keep simple cuts slightly conservative to avoid numeric over-tightening.
                adjusted_rhs -= geom_tol
                # Strength is measured in the actual model space to avoid selecting ineffective cuts.
                strength = cut_violation_at_current_solution(cut_expr, adjusted_rhs)
                if strength > tol:
                    simple_cut_candidates.append(
                        (
                            strength,
                            cut_expr,
                            adjusted_rhs,
                            p1,
                            p2,
                            normal,
                            rhs,
                            {
                                "p1": p1,
                                "p2": p2,
                                "bi1": bi1,
                                "bi2": bi2,
                                "bvar1": bvar1,
                                "bvar2": bvar2,
                                "px": px,
                                "v1": (v1[0], v1[1]),
                                "v2": (v2[0], v2[1]),
                                "t1": t1,
                                "t2": t2,
                                "simple_case": simple_case,
                            },
                        )
                    )

    if simple_cut_candidates:
        simple_cut_candidates.sort(key=lambda c: c[0], reverse=True)
        selected_simple = simple_cut_candidates[:max_simple_cuts]
        for _, cut_expr, adjusted_rhs, p1, p2, normal, rhs, meta in selected_simple:
            if verbose:
                print("  Adding cut from facet through", p1, "and", p2, "with normal", normal, "and rhs", rhs)
            cut_records.append(("simple", cut_expr, adjusted_rhs, meta))
        stats["simple"] = len(selected_simple)
        if verbose and len(simple_cut_candidates) > len(selected_simple):
            print(
                "  Kept",
                len(selected_simple),
                "strongest simple cuts out of",
                len(simple_cut_candidates),
            )

    if not cut_records:
        # Fallback split cuts: generate GMI cuts from transformed integer rows.
        # W combines original integer basic rows into transformed rows y = W x_B,int.
        W_split = None
        if U_row is not None:
            var_row_indices = [ri for ri, b in enumerate(basis) if b < relaxed.NumVars]
            row_to_var_pos = {ri: pos for pos, ri in enumerate(var_row_indices)}
            W_split = np.zeros((len(int_row_indices), len(var_row_indices)))
            for k in range(len(int_row_indices)):
                for j in range(len(int_row_indices)):
                    coeff = U_row[k, j]
                    if abs(coeff) > 1e-12:
                        W_split[k, row_to_var_pos[int_row_indices[j]]] = coeff

        split_cuts = list(
            gu.make_gmi_cuts(
                basis,
                tableau_orig,
                col_to_var_idx,
                x,
                int_var_idx,
                variables,
                constraints,
                relaxed,
                W=W_split,
                tol=tol,
                return_expr_rhs=True,
            )
        )
        for cut_expr, cut_rhs in split_cuts:
            cut_records.append(("split", cut_expr, cut_rhs, {}))
        stats["split"] += len(split_cuts)
        if verbose and split_cuts:
            print("  Adding", len(split_cuts), "transformed split cuts")

    return cut_records, stats


def run_cuts(
    model: gp.Model,
    rounds=1,
    verbose=False,
    callback=None,
    max_simple_cuts=200,
    known_opt_obj=None,
    debug_track_invalid=False,
):
    int_var_idx = {v.index for v in model.getVars() if v.VType in (gp.GRB.INTEGER, gp.GRB.BINARY)}
    l_int_var_idx = list(int_var_idx)
    relaxed = model.relax()
    relaxed.params.Presolve = 0  # for reading the tableau
    relaxed.params.LogToConsole = 0
    relaxed.optimize()
    assert relaxed.status == gp.GRB.Status.OPTIMAL, "Relaxed model must solve to optimality before cuts."
    if callback is not None:
        callback(relaxed)
    starting_obj = relaxed.ObjVal
    if verbose:
        print(f" Cutter round 0 for {model.ModelName}, constraints {model.NumConstrs}, variables {model.NumVars}, integer variables {model.NumIntVars}, start: {starting_obj}")

    def cut_lhs_at_solution(cut_expr):
        lhs = 0.0
        for i in range(cut_expr.size()):
            lhs += cut_expr.getCoeff(i) * relaxed.X[cut_expr.getVar(i).index]
        return lhs

    def summarize_cut(cut_expr, rhs, family, idx, meta):
        terms = []
        for i in range(cut_expr.size()):
            coeff = cut_expr.getCoeff(i)
            if abs(coeff) > 1e-10:
                terms.append((abs(coeff), coeff, cut_expr.getVar(i).VarName))
        terms.sort(reverse=True)
        top = ", ".join(f"{c:+.6g}*{name}" for _, c, name in terms[:12])
        print(
            f"  OFFENDING CUT idx={idx} family={family} rhs={rhs:.12g} "
            f"lhs_at_lp={cut_lhs_at_solution(cut_expr):.12g} terms={cut_expr.size()}"
        )
        if top:
            print("   Top terms:", top)
        if meta:
            print("   Meta:", meta)

    for r in range(rounds):
        # basis, tableau, col_to_var_idx, x = transform_to_original_variables(relaxed)
        basis = gu.read_basis(relaxed)
        tableau, col_to_var_idx, negated_rows = gu.read_tableau(relaxed, basis, remove_basis_cols=True)
        variables, constraints = relaxed.getVars(), relaxed.getConstrs()

        for nr in negated_rows:
            # print("  Negating row", nr, "in GMI tableau at base", basis[nr])
            tableau[nr, :] = -tableau[nr, :]

        x = np.array(relaxed.X).reshape((-1, 1))
        # if all x are integer, we are done:
        if np.allclose(x[l_int_var_idx, 0], np.round(x[l_int_var_idx, 0]), atol=relaxed.params.FeasibilityTol):
            if verbose:
                print(f"  All integer variables are integral for {model.ModelName}; stopping cut generation at round {r}\n")
            break
        cut_records, stats = make_cuts(
            basis,
            tableau,
            col_to_var_idx,
            x,
            int_var_idx,
            variables,
            constraints,
            relaxed,
            tol=relaxed.params.FeasibilityTol,
            verbose=verbose,
            max_simple_cuts=max_simple_cuts,
        )
        if len(cut_records) == 0:
            if verbose:
                print("  No cuts generated at round", r, "; stopping.")
            break

        added_constraints = [relaxed.addConstr(cut_expr >= rhs) for _, cut_expr, rhs, _ in cut_records]
        relaxed.optimize()
        if relaxed.status != gp.GRB.Status.OPTIMAL:
            print("  Cut generation stopped early due to non-optimal relaxation. Status:", gu.status_lookup.get(relaxed.status, relaxed.status))
            return 0, 0

        if known_opt_obj is not None and relaxed.ObjVal > known_opt_obj + relaxed.params.FeasibilityTol:
            print(
                f"  WARNING: LP bound crossed known optimum at round {r + 1}: "
                f"obj={relaxed.ObjVal} > known={known_opt_obj}"
            )
            if debug_track_invalid:
                # Roll back round cuts and add one-by-one to isolate the first offending cut.
                for con in added_constraints:
                    relaxed.remove(con)
                relaxed.update()
                relaxed.optimize()
                if relaxed.status != gp.GRB.Status.OPTIMAL:
                    print("  Failed to restore pre-round LP while debugging.")
                    return starting_obj, relaxed

                added_seq = []
                for idx, (family, cut_expr, rhs, meta) in enumerate(cut_records):
                    con = relaxed.addConstr(cut_expr >= rhs)
                    added_seq.append(con)
                    relaxed.optimize()
                    if relaxed.status != gp.GRB.Status.OPTIMAL:
                        print(
                            f"  Debug isolation stopped: non-optimal after cut idx {idx}, "
                            f"family {family}, status {gu.status_lookup.get(relaxed.status, relaxed.status)}"
                        )
                        summarize_cut(cut_expr, rhs, family, idx, meta)
                        return starting_obj, relaxed
                    if relaxed.ObjVal > known_opt_obj + relaxed.params.FeasibilityTol:
                        summarize_cut(cut_expr, rhs, family, idx, meta)
                        return starting_obj, relaxed

                print("  Could not isolate a single offending cut; invalidity may be from cut interaction.")
                return starting_obj, relaxed

        if callback is not None:
            callback(relaxed)
        print(f"  Cutter round {r + 1}, obj {relaxed.ObjVal}, constraints {relaxed.NumConstrs}, added: hull={stats['hull']}, simple={stats['simple']}, split={stats['split']}")

    return starting_obj, relaxed


def main():
    import example_loader as el
    import plot_utils as pu

    wants_plots = False

    for ex in el.get_instances().values():
        model = ex.as_gurobi_model()
        plotter: pu.PlotterBase = None
        last_count = model.NumConstrs

        def plotter_callback(mdl):
            nonlocal last_count, plotter
            if mdl.NumConstrs > last_count and plotter is not None:
                for c in mdl.getConstrs()[last_count:]:
                    plotter.add_constraint(c)
            else:
                plotter = pu.create(mdl)
                if plotter is not None:
                    plotter.add_ball(1.5)
        run_cuts(model, rounds=15, verbose=True, callback=(plotter_callback if wants_plots else None))
        if plotter is not None:
            plotter.render()

if __name__ == "__main__":
    main()