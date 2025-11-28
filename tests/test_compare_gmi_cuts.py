#!/usr/bin/env python3
"""Compare HiGHS and Gurobi GMI cuts on 2DbottomLeft example to find numeric issues."""

import gurobipy as gp
import highspy as hp
import numpy as np

from .. import highs_utils as hu
from .. import gurobi_utils as gu
from .. import example_loader

def compare_gmi_cuts(example_name="2DbottomLeft"):
    print("=" * 70)
    print(f"COMPARING GMI CUTS: {example_name}")
    print("=" * 70)
    
    # Get the example
    instances = example_loader.get_instances()
    example = instances[example_name]
    gur_model = example.as_gurobi_model()
    
    print(f"\nModel: {gur_model.ModelName}")
    print(f"Variables: {gur_model.NumVars}")
    print(f"Constraints: {gur_model.NumConstrs}")
    
    # Display the model
    for v in gur_model.getVars():
        print(f"  {v.VarName}: {v.LB} <= x <= {v.UB}, type={v.VType}")
    
    print(f"\nObjective: {gur_model.getObjective()}, sense={gur_model.ModelSense}")
    
    for c in gur_model.getConstrs():
        print(f"  {c.ConstrName}: {gur_model.getRow(c)} {c.Sense} {c.RHS}")
    
    # Prepare for GMI cuts
    int_var_set = {v.index for v in gur_model.getVars() if v.VType in (gp.GRB.INTEGER, gp.GRB.BINARY)}
    
    print("\n" + "=" * 70)
    print("HIGHS GMI CUTS")
    print("=" * 70)
    
    # Convert to HiGHS and solve
    h = hu.gur_to_highs(gur_model, relaxed=True)
    h.setOptionValue("output_flag", False)
    status = h.run()
    
    if h.getModelStatus() != hp.HighsModelStatus.kOptimal:
        print(f"ERROR: HiGHS solve failed with status {h.getModelStatus()}")
        return
    
    solution_h = h.getSolution()
    x_h = np.array(solution_h.col_value)
    slack_h = np.array(solution_h.row_value)
    
    print(f"\nLP Solution (HiGHS):")
    for i, val in enumerate(x_h):
        print(f"  var_{i}: {val:.10f}")
    
    print(f"\nSlack values (HiGHS):")
    for i, val in enumerate(slack_h):
        print(f"  constraint_{i}: {val:.10f}")
    
    # Get basis and tableau (HiGHS)
    basis_h, var_status_h, con_status_h = hu.read_basis(h)
    tableau_h, col_to_var_idx_h = hu.read_tableau(h, basis_h, remove_basis_cols=True)
    
    print(f"\nBasis (HiGHS): {basis_h}")
    print(f"Var status (HiGHS): {[var_status_h[i] for i in range(len(x_h))]}")
    print(f"Con status (HiGHS): {[con_status_h[i] for i in range(len(slack_h))]}")
    print(f"\nTableau (HiGHS):")
    print(f"  Non-basic columns: {col_to_var_idx_h}")
    for i, row in enumerate(tableau_h):
        print(f"  Row {i} (basis var {basis_h[i]}): {row}")
    
    # Generate GMI cuts (HiGHS)
    highs_cuts = list(hu.make_gmi_cuts_highs(
        basis_h, var_status_h, con_status_h, tableau_h, col_to_var_idx_h, 
        x_h, slack_h, int_var_set, h, tol=1e-6
    ))
    
    print(f"\nGenerated {len(highs_cuts)} GMI cuts (HiGHS):")
    for i, (indices, coeffs, rhs) in enumerate(highs_cuts):
        print(f"\nCut {i}:")
        print(f"  Indices: {indices}")
        print(f"  Coeffs:  {[f'{c:.10f}' for c in coeffs]}")
        print(f"  RHS:     {rhs:.10f}")
        # Check violation
        violation = rhs - sum(coeffs[j] * x_h[indices[j]] for j in range(len(indices)))
        print(f"  Violation: {violation:.10f}")
    
    print("\n" + "=" * 70)
    print("GUROBI GMI CUTS")
    print("=" * 70)
    
    # Create relaxed Gurobi model
    relaxed_g = gur_model.relax()
    relaxed_g.params.Presolve = 0
    relaxed_g.params.LogToConsole = 0
    relaxed_g.optimize()
    
    if relaxed_g.status != gp.GRB.Status.OPTIMAL:
        print(f"ERROR: Gurobi solve failed with status {relaxed_g.status}")
        return
    
    x_g = np.array(relaxed_g.X).reshape((-1, 1))
    
    print(f"\nLP Solution (Gurobi):")
    for v in relaxed_g.getVars():
        print(f"  {v.VarName}: {v.X:.10f}, VBasis={v.VBasis}")
    
    print(f"\nSlack values (Gurobi):")
    for c in relaxed_g.getConstrs():
        # Manually compute activity
        row = relaxed_g.getRow(c)
        activity = sum(row.getCoeff(j) * row.getVar(j).X for j in range(row.size()))
        
        # Compute tableau slack based on constraint sense
        if c.Sense == '<':
            tableau_slack = c.RHS - activity
        elif c.Sense == '>':
            tableau_slack = activity - c.RHS
        else:
            tableau_slack = 0.0
        
        print(f"  {c.ConstrName}: Gurobi.Slack={c.Slack:.10f}, Tableau.Slack={tableau_slack:.10f}, CBasis={c.CBasis}, Activity={activity:.10f}, RHS={c.RHS:.10f}, Sense={c.Sense}")
    
    # Get basis and tableau (Gurobi)
    basis_g = gu.read_basis(relaxed_g)
    tableau_g, col_to_var_idx_g, negated_rows_g = gu.read_tableau(relaxed_g, basis_g, remove_basis_cols=True)
    
    print(f"\nBasis (Gurobi): {basis_g}")
    print(f"Negated rows (Gurobi): {negated_rows_g}")
    print(f"\nTableau (Gurobi):")
    print(f"  Non-basic columns: {col_to_var_idx_g}")
    for i, row in enumerate(tableau_g):
        print(f"  Row {i} (basis var {basis_g[i]}): {row}")
    
    # Generate GMI cuts (Gurobi)
    variables_g = relaxed_g.getVars()
    constraints_g = relaxed_g.getConstrs()
    
    gurobi_cuts = list(gu.make_gmi_cuts(
        basis_g, tableau_g, col_to_var_idx_g, x_g,
        int_var_set, variables_g, constraints_g, relaxed_g,
        tol=1e-6, negated_rows=negated_rows_g
    ))
    
    print(f"\nGenerated {len(gurobi_cuts)} GMI cuts (Gurobi):")
    for i, cut in enumerate(gurobi_cuts):
        print(f"\nCut {i}: {cut}")
        # Extract coefficients
        if hasattr(cut, 'getLHS'):
            lhs = cut.getLHS()
            rhs = cut.getRHS()
            sense = cut.getSense()
            print(f"  Sense: {sense}")
            print(f"  RHS: {rhs:.10f}")
            print(f"  LHS size: {lhs.size()}")
            for j in range(lhs.size()):
                var = lhs.getVar(j)
                coeff = lhs.getCoeff(j)
                print(f"    {var.VarName}: {coeff:.10f}")
            # Check violation
            violation = rhs - sum(lhs.getCoeff(j) * lhs.getVar(j).X for j in range(lhs.size()))
            print(f"  Violation: {violation:.10f}")
    
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"HiGHS generated {len(highs_cuts)} cuts")
    print(f"Gurobi generated {len(gurobi_cuts)} cuts")
    
    if len(highs_cuts) != len(gurobi_cuts):
        print("\n⚠️  WARNING: Different number of cuts generated!")
    
    # Compare basis
    print(f"\nBasis comparison:")
    print(f"  HiGHS:  {basis_h}")
    print(f"  Gurobi: {basis_g}")
    if basis_h != basis_g:
        print("  ⚠️  Different basis!")
    
    # Compare solutions
    print(f"\nSolution comparison:")
    max_diff = np.max(np.abs(x_h - x_g.flatten()))
    print(f"  Max difference in solution: {max_diff:.10e}")
    if max_diff > 1e-6:
        print("  ⚠️  Solutions differ significantly!")
        for i in range(len(x_h)):
            diff = abs(x_h[i] - x_g[i, 0])
            if diff > 1e-8:
                print(f"    var_{i}: HiGHS={x_h[i]:.10f}, Gurobi={x_g[i,0]:.10f}, diff={diff:.10e}")


if __name__ == "__main__":
    import sys
    example_name = sys.argv[1] if len(sys.argv) > 1 else "2DbottomLeft"
    compare_gmi_cuts(example_name)
