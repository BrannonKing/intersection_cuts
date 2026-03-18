import gurobipy as gp
gp.setParam('OutputFlag', 0)  # suppress Gurobi output for this experiment
import gurobi_utils as gu
import numpy as np
import scipy.spatial as spatial


def are_in_half_plane(vectors, tol):
    # 0. Filter out 0-length vectors
    magnitudes = np.linalg.norm(vectors, axis=1)
    vectors = vectors[magnitudes > tol]
    
    if len(vectors) <= 1:
        print("  Only", len(vectors), "nonzero vectors; treating as in a half-plane.")
        return False, None, None

    # 1. Convert to angles in the range [-pi, pi]
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    
    # 2. Get sorted indices to map back to original vectors
    sorted_indices = np.argsort(angles)
    sorted_angles = angles[sorted_indices]
    
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

def make_cuts(basis, tableau, col_to_var_idx, x, int_var_idx, variables, constraints, relaxed, tol, verbose=False):

    # Shift the tableau so that all non-basic variables are treated as >= 0
    # This guarantees that the tableau columns represent rays pointing into the feasible region.
    betas, tableau = gu.shift_to_x_gt_0(basis, tableau, col_to_var_idx, variables, constraints, x, relaxed)

    # for each pair of basis row that is supposed to be integer but is not,
    # see if all those 2D vectors are on the same side of some line through the origin.
    # if so, enumerate some integer points along each ray and find their convex hull.
    # take the line of the convex hull closest to the solution and add it as a cut.

    new_constraints = []
    stats = {"hull": 0, "simple": 0, "split": 0}

    for bi1, bvar1 in enumerate(basis):
        if bvar1 >= relaxed.NumVars or bvar1 not in int_var_idx:
            continue
        # verify bvar1 is fractional:
        if abs(x[bvar1, 0] - round(x[bvar1, 0])) < tol:
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
                    cut_expr, adjusted_rhs = build_cut_expr(normal, bvar1, bvar2, rhs, variables, constraints, relaxed)
                    new_constraints.append(cut_expr >= adjusted_rhs)
                    stats["hull"] += 1
            else:
                # Strongest intersection cut when x is integer and y is continuous.
                # Equivalent to a split cut on x <= floor(px[0]) or x >= ceil(px[0]).
                def get_t(vx):
                    if vx > tol:
                        return (np.ceil(px[0]) - px[0]) / vx
                    elif vx < -tol:
                        return (np.floor(px[0]) - px[0]) / vx
                    else:
                        return float('inf')
                
                t1 = get_t(v1[0])
                t2 = get_t(v2[0])
                
                if np.isinf(t1) and np.isinf(t2):
                    # Both rays are vertical and x is strictly fractional.
                    # The cone contains no feasible integer x. Cut off px with vertical line.
                    normal = np.array([1.0, 0.0])
                    p1 = (np.ceil(px[0]), px[1])
                    p2 = p1
                elif np.isinf(t1):
                    p2 = (px[0] + t2 * v2[0], px[1] + t2 * v2[1])
                    normal = np.array([v1[1], -v1[0]])
                    if np.dot(normal, np.array(px) - p2) > 0:
                        normal = -normal
                    p1 = p2
                elif np.isinf(t2):
                    p1 = (px[0] + t1 * v1[0], px[1] + t1 * v1[1])
                    normal = np.array([v2[1], -v2[0]])
                    if np.dot(normal, np.array(px) - p1) > 0:
                        normal = -normal
                    p2 = p1
                else:
                    p1 = (px[0] + t1 * v1[0], px[1] + t1 * v1[1])
                    p2 = (px[0] + t2 * v2[0], px[1] + t2 * v2[1])
                    normal = np.array([p2[1] - p1[1], p1[0] - p2[0]])
                    if np.linalg.norm(normal) < tol:
                        normal = np.array([1.0 if v1[0] > 0 else -1.0, 0.0])
                
                if np.linalg.norm(normal) > 0:
                    normal = normal / np.linalg.norm(normal)
                
                if np.dot(normal, np.array(px) - p1) > 0:
                    normal = -normal
                    
                rhs = np.dot(normal, p1)
                if np.dot(normal, px) - rhs >= -tol:
                    continue

                if verbose:
                    print("  Adding cut from facet through", p1, "and", p2, "with normal", normal, "and rhs", rhs)

                cut_expr, adjusted_rhs = build_cut_expr(normal, bvar1, bvar2, rhs, variables, constraints, relaxed)
                new_constraints.append(cut_expr >= adjusted_rhs)
                stats["simple"] += 1

    if not new_constraints:
        # run the split cut (an intersection cut):
        to_cut = [(row, base) for row, base in enumerate(basis) if base < relaxed.NumVars and base in int_var_idx and not np.isclose(x[base, 0], round(x[base, 0]), atol=tol)]
        for row, base in to_cut:
            cut_expr = gp.LinExpr()
            cut_rhs = 1.0
            for col, ray in enumerate(tableau.T):
                rr = ray[row]
                if np.isclose(rr, 0, atol=tol):
                    continue
                xv = x[base, 0]
                f0 = xv % 1.0
                
                var_idx = col_to_var_idx[col]
                is_int = var_idx in int_var_idx
                
                if is_int:
                    fj = rr % 1.0
                    scale = fj / f0 if fj <= f0 else (1.0 - fj) / (1.0 - f0)
                else:
                    scale = rr / f0 if rr > 0 else -rr / (1.0 - f0)

                if var_idx < relaxed.NumVars:
                    vrb = variables[var_idx]
                    if vrb.VBasis == -1: # Non-basic at lower bound
                        cut_expr.addTerms(scale, vrb)
                        if vrb.LB > -gp.GRB.INFINITY and abs(vrb.LB) > tol:
                            cut_rhs += scale * vrb.LB
                    elif vrb.VBasis == -2: # Non-basic at upper bound
                        cut_expr.addTerms(-scale, vrb)
                        cut_rhs -= scale * vrb.UB
                    else:
                        cut_expr.addTerms(scale, vrb)
                else:
                    # Substitute slack variable using the original constraint
                    con = constraints[var_idx - relaxed.NumVars]
                    a_i = relaxed.getRow(con)
                    if con.Sense == '<':
                        cut_rhs -= scale * con.RHS
                        for j in range(a_i.size()):
                            cut_expr.addTerms(-scale * a_i.getCoeff(j), a_i.getVar(j))
                    elif con.Sense == '>':
                        cut_rhs += scale * con.RHS
                        for j in range(a_i.size()):
                            cut_expr.addTerms(scale * a_i.getCoeff(j), a_i.getVar(j))

            new_constraints.append(cut_expr >= cut_rhs)
            stats["split"] += 1
            if verbose:
                print("  Adding split cut for basis row", row, "variable", base)

    return new_constraints, stats


def run_cuts(model: gp.Model, rounds=1, verbose=False, callback=None):
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
        new_constraints, stats = make_cuts(basis, tableau, col_to_var_idx, x,
            int_var_idx, variables, constraints, relaxed, tol=relaxed.params.FeasibilityTol, verbose=verbose
        )
        if (len(new_constraints) == 0):
            if verbose:
                print("  No cuts generated at round", r, "; stopping.")
            break
        relaxed.addConstrs(c for c in new_constraints)
        relaxed.optimize()
        if relaxed.status != gp.GRB.Status.OPTIMAL:
            print("  Cut generation stopped early due to non-optimal relaxation. Status:", gu.status_lookup.get(relaxed.status, relaxed.status))
            return 0, 0
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