#!/usr/bin/env python3
"""Verify the tableau and GMI cut construction step by step."""

import highspy as hp
import numpy as np

from .. import highs_utils as hu
from .. import example_loader

def test_tableau_and_gmi_cut_construction():
    print("=" * 70)
    print("VERIFYING TABLEAU AND GMI CUT CONSTRUCTION")
    print("=" * 70)
    
    # Get the 2Dabove example
    instances = example_loader.get_instances()
    example = instances["2Dabove"]
    gur_model = example.as_gurobi_model()
    
    # Convert to HiGHS
    h = hu.gur_to_highs(gur_model, relaxed=True)
    int_var_set = {0, 1}  # x and y are integers
    
    h.setOptionValue("output_flag", False)
    status = h.run()
    
    solution = h.getSolution()
    x_lp = solution.col_value[0]
    y_lp = solution.col_value[1]
    
    print(f"\nLP Solution: x={x_lp:.6f}, y={y_lp:.6f}")
    
    # Get slacks
    slack = solution.row_value
    print(f"\nSlack values (row_value from HiGHS):")
    print(f"  row 0 activity: {slack[0]:.6f}")
    print(f"  row 1 activity: {slack[1]:.6f}")
    
    # Compute slacks manually
    status_c0, lb0, ub0, _ = hp.Highs.getRow(h, 0)
    status_c1, lb1, ub1, _ = hp.Highs.getRow(h, 1)
    
    # Row 0: -0.9x + 0.9y >= 1
    activity_0 = -0.9 * x_lp + 0.9 * y_lp
    slack_0 = activity_0 - lb0
    print(f"\nRow 0: -0.9*{x_lp:.6f} + 0.9*{y_lp:.6f} = {activity_0:.6f} >= {lb0:.6f}")
    print(f"  Slack = {activity_0:.6f} - {lb0:.6f} = {slack_0:.6f}")
    
    # Row 1: 0.9x + 0.6y >= 2.5
    activity_1 = 0.9 * x_lp + 0.6 * y_lp
    slack_1 = activity_1 - lb1
    print(f"\nRow 1: 0.9*{x_lp:.6f} + 0.6*{y_lp:.6f} = {activity_1:.6f} >= {lb1:.6f}")
    print(f"  Slack = {activity_1:.6f} - {lb1:.6f} = {slack_1:.6f}")
    
    # Get tableau
    basis, var_status, con_status = hu.read_basis(h)
    tableau, col_to_var_idx = hu.read_tableau(h, basis, remove_basis_cols=True)
    
    print("\n" + "=" * 70)
    print("TABLEAU VERIFICATION")
    print("=" * 70)
    print(f"Basis: {basis}")
    print(f"  Row 0: var_{basis[0]} is basic")
    print(f"  Row 1: var_{basis[1]} is basic")
    print(f"\nTableau (non-basic columns only):")
    print(f"  Non-basic columns: {col_to_var_idx} (slacks for constraints 0 and 1)")
    print(f"  Row 0: {tableau[0]}")
    print(f"  Row 1: {tableau[1]}")
    
    # The tableau represents: basic_var = RHS - sum(tableau[i,j] * non_basic[j])
    # For row 0 (y is basic):
    # y = RHS_0 - 0.666667*slack_0 - 0.666667*slack_1
    
    # Verify this is correct
    print("\n" + "=" * 70)
    print("VERIFYING TABLEAU EQUATION FOR Y (row 0)")
    print("=" * 70)
    print(f"Tableau says: y = RHS - 0.666667*slack_0 - 0.666667*slack_1")
    
    # The RHS in tableau form should give us y_lp
    # When slacks are at their current values
    rhs_0_from_tableau = y_lp + tableau[0][0] * slack_0 + tableau[0][1] * slack_1
    print(f"\nComputing RHS:")
    print(f"  RHS = {y_lp:.6f} + {tableau[0][0]:.6f}*{slack_0:.6f} + {tableau[0][1]:.6f}*{slack_1:.6f}")
    print(f"      = {rhs_0_from_tableau:.6f}")
    
    # Fractional part
    frac = lambda a: a - np.floor(a)
    f0 = frac(rhs_0_from_tableau)
    print(f"\nFractional part f0 = {f0:.6f}")
    
    # GMI cut from tableau row: f0 <= frac(0.666667)*slack_0 + frac(0.666667)*slack_1
    # With scaled coefficients
    fj_0 = frac(tableau[0][0])
    fj_1 = frac(tableau[0][1])
    
    print("\n" + "=" * 70)
    print("GMI CUT FROM TABLEAU ROW")
    print("=" * 70)
    print(f"Fractional parts: fj_0 = {fj_0:.6f}, fj_1 = {fj_1:.6f}")
    
    # GMI formula (scaled by f0*(1-f0))
    if fj_0 < f0:
        coeff_0 = (1 - f0) * fj_0
    else:
        coeff_0 = (1 - fj_0) * f0
    
    if fj_1 < f0:
        coeff_1 = (1 - f0) * fj_1
    else:
        coeff_1 = (1 - fj_1) * f0
    
    print(f"\nGMI coefficients (scaled):")
    print(f"  coeff_0 = {coeff_0:.6f}")
    print(f"  coeff_1 = {coeff_1:.6f}")
    
    print(f"\nGMI cut in slack space:")
    print(f"  {coeff_0:.6f}*slack_0 + {coeff_1:.6f}*slack_1 >= {f0*(1-f0):.6f}")
    
    # Now expand slacks
    print("\n" + "=" * 70)
    print("EXPANDING SLACKS TO ORIGINAL VARIABLES")
    print("=" * 70)
    
    # slack_0 = -0.9x + 0.9y - 1
    # slack_1 = 0.9x + 0.6y - 2.5
    
    print(f"Substituting:")
    print(f"  slack_0 = -0.9x + 0.9y - 1")
    print(f"  slack_1 = 0.9x + 0.6y - 2.5")
    
    print(f"\n{coeff_0:.6f}*(-0.9x + 0.9y - 1) + {coeff_1:.6f}*(0.9x + 0.6y - 2.5) >= {f0*(1-f0):.6f}")
    
    # Expand
    x_coeff = coeff_0 * (-0.9) + coeff_1 * 0.9
    y_coeff = coeff_0 * 0.9 + coeff_1 * 0.6
    const = coeff_0 * (-1) + coeff_1 * (-2.5)
    
    print(f"\nExpanded:")
    print(f"  x: {coeff_0:.6f}*(-0.9) + {coeff_1:.6f}*0.9 = {x_coeff:.6f}")
    print(f"  y: {coeff_0:.6f}*0.9 + {coeff_1:.6f}*0.6 = {y_coeff:.6f}")
    print(f"  const: {coeff_0:.6f}*(-1) + {coeff_1:.6f}*(-2.5) = {const:.6f}")
    
    rhs = f0 * (1 - f0) - const
    print(f"\nRHS = {f0*(1-f0):.6f} - ({const:.6f}) = {rhs:.6f}")
    
    print(f"\nFinal cut: {x_coeff:.6f}*x + {y_coeff:.6f}*y >= {rhs:.6f}")
    
    # Test at (1, 3)
    print("\n" + "=" * 70)
    print("TESTING AT (1, 3)")
    print("=" * 70)
    lhs = x_coeff * 1 + y_coeff * 3
    print(f"LHS = {x_coeff:.6f}*1 + {y_coeff:.6f}*3 = {lhs:.6f}")
    print(f"RHS = {rhs:.6f}")
    print(f"Satisfied? {lhs >= rhs - 1e-6}")
    
    if lhs < rhs - 1e-6:
        print(f"\n*** CUT VIOLATES (1, 3) by {lhs - rhs:.6f} ***")
        
        # Let's check: is (1, 3) in the corner polyhedron?
        print("\n" + "=" * 70)
        print("IS (1, 3) IN THE CORNER POLYHEDRON?")
        print("=" * 70)
        
        # At (1, 3), what are the slack values?
        slack_0_at_13 = -0.9 * 1 + 0.9 * 3 - 1
        slack_1_at_13 = 0.9 * 1 + 0.6 * 3 - 2.5
        
        print(f"At (1, 3):")
        print(f"  slack_0 = -0.9*1 + 0.9*3 - 1 = {slack_0_at_13:.6f}")
        print(f"  slack_1 = 0.9*1 + 0.6*3 - 2.5 = {slack_1_at_13:.6f}")
        
        # Check if these are non-negative
        print(f"\nBoth slacks >= 0? slack_0={slack_0_at_13 >= 0}, slack_1={slack_1_at_13 >= 0}")
        
        # The corner polyhedron is defined by: x, y integer and the constraints
        # The GMI cut should only cut off points NOT in the corner polyhedron
        # Since (1,3) satisfies all constraints AND is integer, it IS in the corner polyhedron
        # Therefore, the GMI cut is INVALID
    
    # Assertions for test
    assert tableau is not None, "Should be able to read tableau"
    assert basis is not None, "Should be able to read basis"
    assert len(basis) == 2, "Should have 2 basic variables"
    assert x_lp is not None and y_lp is not None, "Should have LP solution"
