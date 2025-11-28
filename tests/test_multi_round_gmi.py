#!/usr/bin/env python3
"""Test multiple rounds of GMI cuts comparing HiGHS vs Gurobi."""

import gurobipy as gp
import highspy as hp
import numpy as np

from .. import highs_utils as hu
from .. import gurobi_utils as gu
from .. import example_loader

def test_multi_round_gmi(example_name="2DbottomLeft", rounds=2):
    print("=" * 80)
    print(f"MULTI-ROUND GMI CUT TEST: {example_name} ({rounds} rounds)")
    print("=" * 80)
    
    # Get the example
    instances = example_loader.get_instances()
    example = instances[example_name]
    gur_model = example.as_gurobi_model()
    
    print(f"\nOriginal Model: {gur_model.ModelName}")
    print(f"Variables: {gur_model.NumVars}, Constraints: {gur_model.NumConstrs}")
    print(f"Integer variables: {sum(1 for v in gur_model.getVars() if v.VType in (gp.GRB.INTEGER, gp.GRB.BINARY))}")
    
    # Display the model
    for v in gur_model.getVars():
        print(f"  {v.VarName}: [{v.LB}, {v.UB}], type={v.VType}")
    print(f"\nObjective: {gur_model.getObjective()}, sense={'MIN' if gur_model.ModelSense == 1 else 'MAX'}")
    for c in gur_model.getConstrs():
        print(f"  {c.ConstrName}: {gur_model.getRow(c)} {c.Sense} {c.RHS}")
    
    # Prepare integer variable set
    int_var_set = {v.index for v in gur_model.getVars() if v.VType in (gp.GRB.INTEGER, gp.GRB.BINARY)}
    
    # ========================================================================
    # HIGHS - Multiple rounds
    # ========================================================================
    print("\n" + "=" * 80)
    print("HIGHS - MULTIPLE ROUNDS OF GMI CUTS")
    print("=" * 80)
    
    h = hu.gur_to_highs(gur_model, relaxed=True)
    h.setOptionValue("output_flag", False)
    
    # Track objectives and cuts per round
    highs_objectives = []
    highs_cuts_per_round = []
    highs_total_constraints = []
    
    for r in range(rounds):
        print(f"\n--- HiGHS Round {r} ---")
        
        # Solve
        status = h.run()
        if h.getModelStatus() != hp.HighsModelStatus.kOptimal:
            print(f"ERROR: HiGHS solve failed with status {h.getModelStatus()}")
            break
        
        obj_val = h.getInfo().objective_function_value
        num_constrs = h.numConstrs
        highs_objectives.append(obj_val)
        highs_total_constraints.append(num_constrs)
        
        print(f"Objective: {obj_val:.10f}")
        print(f"Constraints: {num_constrs}")
        
        solution_h = h.getSolution()
        x_h = np.array(solution_h.col_value)
        slack_h = np.array(solution_h.row_value)
        
        print(f"Solution:")
        for i in range(len(x_h)):
            if i < gur_model.NumVars:
                print(f"  var_{i}: {x_h[i]:.10f}")
        
        # Check if integer variables are integral
        int_vars_fractional = False
        for var_idx in int_var_set:
            if abs(x_h[var_idx] - np.round(x_h[var_idx])) > 1e-6:
                int_vars_fractional = True
                print(f"  var_{var_idx} is fractional: {x_h[var_idx]:.10f}")
        
        if not int_vars_fractional:
            print("All integer variables are integral; stopping")
            highs_cuts_per_round.append(0)
            break
        
        # Generate cuts
        basis_h, var_status_h, con_status_h = hu.read_basis(h)
        tableau_h, col_to_var_idx_h = hu.read_tableau(h, basis_h, remove_basis_cols=True)
        
        cuts_h = list(hu.make_gmi_cuts_highs(
            basis_h, var_status_h, con_status_h, tableau_h, col_to_var_idx_h,
            x_h, slack_h, int_var_set, h, tol=1e-6
        ))
        
        highs_cuts_per_round.append(len(cuts_h))
        print(f"Generated {len(cuts_h)} cuts")
        
        if len(cuts_h) == 0:
            print("No cuts generated; stopping")
            break
        
        # Display cuts
        for i, (indices, coeffs, rhs) in enumerate(cuts_h):
            print(f"  Cut {i}: ", end="")
            terms = [f"{coeffs[j]:.6f}*x{indices[j]}" for j in range(len(indices))]
            print(" + ".join(terms) + f" >= {rhs:.6f}")
            
            # Check violation
            violation = rhs - sum(coeffs[j] * x_h[indices[j]] for j in range(len(indices)))
            print(f"    Violation: {violation:.10f}")
        
        # Add cuts to model
        for indices, coeffs, rhs in cuts_h:
            h.addRow(rhs, hp.kHighsInf, len(indices), indices, coeffs)
    
    # ========================================================================
    # GUROBI - Multiple rounds
    # ========================================================================
    print("\n" + "=" * 80)
    print("GUROBI - MULTIPLE ROUNDS OF GMI CUTS")
    print("=" * 80)
    
    relaxed_g = gur_model.relax()
    relaxed_g.params.Presolve = 0
    relaxed_g.params.LogToConsole = 0
    
    # Track objectives and cuts per round
    gurobi_objectives = []
    gurobi_cuts_per_round = []
    gurobi_total_constraints = []
    
    for r in range(rounds):
        print(f"\n--- Gurobi Round {r} ---")
        
        # Solve
        relaxed_g.optimize()
        if relaxed_g.status != gp.GRB.Status.OPTIMAL:
            print(f"ERROR: Gurobi solve failed with status {relaxed_g.status}")
            break
        
        obj_val = relaxed_g.ObjVal
        num_constrs = relaxed_g.NumConstrs
        gurobi_objectives.append(obj_val)
        gurobi_total_constraints.append(num_constrs)
        
        print(f"Objective: {obj_val:.10f}")
        print(f"Constraints: {num_constrs}")
        
        x_g = np.array(relaxed_g.X).reshape((-1, 1))
        
        print(f"Solution:")
        for v in relaxed_g.getVars():
            if v.index < gur_model.NumVars:
                print(f"  {v.VarName}: {v.X:.10f}")
        
        # Check if integer variables are integral
        int_vars_fractional = False
        for var_idx in int_var_set:
            if abs(x_g[var_idx, 0] - np.round(x_g[var_idx, 0])) > 1e-6:
                int_vars_fractional = True
                print(f"  var_{var_idx} is fractional: {x_g[var_idx, 0]:.10f}")
        
        if not int_vars_fractional:
            print("All integer variables are integral; stopping")
            gurobi_cuts_per_round.append(0)
            break
        
        # Generate cuts
        basis_g = gu.read_basis(relaxed_g)
        tableau_g, col_to_var_idx_g, negated_rows_g = gu.read_tableau(relaxed_g, basis_g, remove_basis_cols=True)
        
        variables_g = relaxed_g.getVars()
        constraints_g = relaxed_g.getConstrs()
        
        cuts_g = list(gu.make_gmi_cuts(
            basis_g, tableau_g, col_to_var_idx_g, x_g,
            int_var_set, variables_g, constraints_g, relaxed_g,
            tol=1e-6, negated_rows=negated_rows_g
        ))
        
        gurobi_cuts_per_round.append(len(cuts_g))
        print(f"Generated {len(cuts_g)} cuts")
        
        if len(cuts_g) == 0:
            print("No cuts generated; stopping")
            break
        
        # Display cuts
        for i, cut in enumerate(cuts_g):
            print(f"  Cut {i}: {cut}")
            if hasattr(cut, 'getLHS'):
                lhs = cut.getLHS()
                rhs = cut.getRHS()
                # Check violation
                violation = rhs - sum(lhs.getCoeff(j) * lhs.getVar(j).X for j in range(lhs.size()))
                print(f"    Violation: {violation:.10f}")
        
        # Add cuts to model
        relaxed_g.addConstrs(c for c in cuts_g)
    
    # ========================================================================
    # COMPARISON
    # ========================================================================
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    print(f"\nRounds completed:")
    print(f"  HiGHS:  {len(highs_objectives)} rounds")
    print(f"  Gurobi: {len(gurobi_objectives)} rounds")
    
    print(f"\nObjective values by round:")
    for r in range(max(len(highs_objectives), len(gurobi_objectives))):
        h_obj = highs_objectives[r] if r < len(highs_objectives) else "N/A"
        g_obj = gurobi_objectives[r] if r < len(gurobi_objectives) else "N/A"
        
        if isinstance(h_obj, float) and isinstance(g_obj, float):
            diff = abs(h_obj - g_obj)
            match = "✓" if diff < 1e-6 else "✗"
            print(f"  Round {r}: HiGHS={h_obj:.10f}, Gurobi={g_obj:.10f}, diff={diff:.10e} {match}")
        else:
            print(f"  Round {r}: HiGHS={h_obj}, Gurobi={g_obj}")
    
    print(f"\nCuts generated per round:")
    for r in range(max(len(highs_cuts_per_round), len(gurobi_cuts_per_round))):
        h_cuts = highs_cuts_per_round[r] if r < len(highs_cuts_per_round) else "N/A"
        g_cuts = gurobi_cuts_per_round[r] if r < len(gurobi_cuts_per_round) else "N/A"
        
        if isinstance(h_cuts, int) and isinstance(g_cuts, int):
            match = "✓" if h_cuts == g_cuts else "✗"
            print(f"  Round {r}: HiGHS={h_cuts}, Gurobi={g_cuts} {match}")
        else:
            print(f"  Round {r}: HiGHS={h_cuts}, Gurobi={g_cuts}")
    
    print(f"\nTotal constraints by round:")
    for r in range(max(len(highs_total_constraints), len(gurobi_total_constraints))):
        h_cons = highs_total_constraints[r] if r < len(highs_total_constraints) else "N/A"
        g_cons = gurobi_total_constraints[r] if r < len(gurobi_total_constraints) else "N/A"
        
        if isinstance(h_cons, int) and isinstance(g_cons, int):
            match = "✓" if h_cons == g_cons else "✗"
            print(f"  Round {r}: HiGHS={h_cons}, Gurobi={g_cons} {match}")
        else:
            print(f"  Round {r}: HiGHS={h_cons}, Gurobi={g_cons}")
    
    # Final verdict
    print("\n" + "=" * 80)
    all_match = True
    
    if len(highs_objectives) != len(gurobi_objectives):
        print("⚠️  Different number of rounds completed!")
        all_match = False
    
    for r in range(min(len(highs_objectives), len(gurobi_objectives))):
        if abs(highs_objectives[r] - gurobi_objectives[r]) > 1e-6:
            print(f"⚠️  Objectives differ at round {r}!")
            all_match = False
        
        if r < len(highs_cuts_per_round) and r < len(gurobi_cuts_per_round):
            if highs_cuts_per_round[r] != gurobi_cuts_per_round[r]:
                print(f"⚠️  Number of cuts differ at round {r}!")
                all_match = False
    
    if all_match:
        print("✅ SUCCESS: HiGHS and Gurobi GMI cuts match perfectly across all rounds!")
    else:
        print("❌ FAILURE: Differences detected between HiGHS and Gurobi!")
    
    return all_match


if __name__ == "__main__":
    example_name = sys.argv[1] if len(sys.argv) > 1 else "2DbottomLeft"
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    
    success = test_multi_round_gmi(example_name, rounds)
    sys.exit(0 if success else 1)
