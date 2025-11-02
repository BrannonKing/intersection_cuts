#!/usr/bin/env python3
"""Trace how constraints with fractional RHS values are created and added."""

import sys
sys.path.insert(0, '/home/brannon/Documents/Research/intersection_cuts')

import gurobipy as gp
import numpy as np
import gurobi_utils as gu
import example_loader

def trace_constraints():
    print("=" * 80)
    print("TRACING CONSTRAINT CREATION AND RHS VALUES")
    print("=" * 80)
    
    # Get the example
    instances = example_loader.get_instances()
    example = instances["2DbottomLeft"]
    gur_model = example.as_gurobi_model()
    
    print(f"\nOriginal Model: {gur_model.ModelName}")
    print(f"Variables: {gur_model.NumVars}, Constraints: {gur_model.NumConstrs}")
    
    # Show original constraints
    print("\nOriginal constraints:")
    for c in gur_model.getConstrs():
        row = gur_model.getRow(c)
        print(f"  {c.ConstrName}: {row} {c.Sense} {c.RHS}")
        print(f"    RHS = {c.RHS}, is_integer = {abs(c.RHS - round(c.RHS)) < 1e-6}")
        for j in range(row.size()):
            var = row.getVar(j)
            coeff = row.getCoeff(j)
            print(f"      {var.VarName}: coeff={coeff}, is_integer_coeff={abs(coeff - round(coeff)) < 1e-6}, VType={var.VType}")
    
    # Create relaxation and solve
    int_var_set = {v.index for v in gur_model.getVars() if v.VType in (gp.GRB.INTEGER, gp.GRB.BINARY)}
    relaxed = gur_model.relax()
    relaxed.params.Presolve = 0
    relaxed.params.LogToConsole = 0
    relaxed.optimize()
    
    print("\n" + "=" * 80)
    print("AFTER ROUND 0 - GENERATING GMI CUTS")
    print("=" * 80)
    
    x = np.array(relaxed.X).reshape((-1, 1))
    basis = gu.read_basis(relaxed)
    tableau, col_to_var_idx, negated_rows = gu.read_tableau(relaxed, basis, remove_basis_cols=True)
    
    variables = relaxed.getVars()
    constraints = relaxed.getConstrs()
    
    # Generate cuts WITHOUT the RHS filter
    print("\nGenerating cuts (without RHS filter):")
    frac = lambda a: a - np.floor(a)
    
    for row_idx, row in enumerate(tableau):
        basis_var_idx = basis[row_idx]
        
        if basis_var_idx < len(variables) and basis_var_idx not in int_var_set:
            continue
        
        # Don't check is_integer_constraint yet - generate the cut anyway
        if basis_var_idx < len(variables):
            beta = x[basis_var_idx, 0]
            print(f"\n  Row {row_idx}: basis_var={basis_var_idx} (variable), beta={beta:.10f}")
        else:
            constraint_idx = basis_var_idx - len(variables)
            con = constraints[constraint_idx]
            con_row = relaxed.getRow(con)
            activity = sum(con_row.getCoeff(j) * con_row.getVar(j).X for j in range(con_row.size()))
            
            if con.Sense == '<':
                beta = con.RHS - activity
            elif con.Sense == '>':
                beta = activity - con.RHS
            else:
                beta = 0.0
            
            print(f"\n  Row {row_idx}: basis_var={basis_var_idx} (slack for constraint {constraint_idx})")
            print(f"    Constraint: {con.ConstrName}, Sense={con.Sense}, RHS={con.RHS}")
            print(f"    Activity={activity:.10f}, beta={beta:.10f}")
        
        f0 = frac(beta)
        
        if f0 < 1e-6 or f0 > 1 - 1e-6:
            print(f"    f0={f0:.10f} -> SKIPPED (nearly integer)")
            continue
        
        print(f"    f0={f0:.10f} -> CUT WOULD BE GENERATED")
        
        # Now check if it's an integer constraint
        if basis_var_idx >= len(variables):
            constraint_idx = basis_var_idx - len(variables)
            con = constraints[constraint_idx]
            con_row = relaxed.getRow(con)
            
            print(f"    Checking if integer constraint:")
            print(f"      RHS = {con.RHS}, is_integer = {abs(con.RHS - round(con.RHS)) < 1e-6}")
            
            all_integer_coeffs = True
            all_integer_vars = True
            for j in range(con_row.size()):
                var = con_row.getVar(j)
                coeff = con_row.getCoeff(j)
                is_int_coeff = abs(coeff - round(coeff)) < 1e-6
                is_int_var = var.index in int_var_set
                
                if not is_int_coeff:
                    all_integer_coeffs = False
                if not is_int_var:
                    all_integer_vars = False
                
                print(f"        {var.VarName}: coeff={coeff:.6f}, is_int_coeff={is_int_coeff}, is_int_var={is_int_var}")
            
            print(f"      All integer coeffs: {all_integer_coeffs}")
            print(f"      All integer vars: {all_integer_vars}")
            print(f"      RHS is integer: {abs(con.RHS - round(con.RHS)) < 1e-6}")
            
            if all_integer_coeffs and all_integer_vars:
                print(f"    -> This IS an integer constraint (ignoring RHS)")
                if abs(con.RHS - round(con.RHS)) >= 1e-6:
                    print(f"    -> But RHS is fractional: {con.RHS}")
                    print(f"    -> Your point: Gurobi should have already strengthened this!")
                    print(f"    -> Let's check if this is a constraint WE added (a GMI cut)")
    
    # Now actually generate and add the cuts
    cuts = list(gu.make_gmi_cuts(
        basis, tableau, col_to_var_idx, x,
        int_var_set, variables, constraints, relaxed,
        tol=1e-6, negated_rows=negated_rows
    ))
    
    print("\n" + "=" * 80)
    print(f"GMI CUTS GENERATED (with current filter): {len(cuts)}")
    print("=" * 80)
    
    for i, cut in enumerate(cuts):
        print(f"\nCut {i}: {cut}")
        if hasattr(cut, 'getLHS'):
            lhs = cut.getLHS()
            rhs = cut.getRHS()
            print(f"  RHS: {rhs:.10f}, is_integer: {abs(rhs - round(rhs)) < 1e-6}")
            print(f"  Coefficients:")
            for j in range(lhs.size()):
                var = lhs.getVar(j)
                coeff = lhs.getCoeff(j)
                print(f"    {var.VarName}: {coeff:.10f}, is_integer: {abs(coeff - round(coeff)) < 1e-6}")
    
    print("\n" + "=" * 80)
    print("KEY INSIGHT")
    print("=" * 80)
    print("\nThe GMI cuts we generate have:")
    print("  - Integer variables only")
    print("  - Fractional coefficients (e.g., 0.7, 0.4)")
    print("  - Fractional RHS (e.g., -1.4, -2.3)")
    print("\nThese cuts represent VALID constraints, but they are NOT 'integer constraints'")
    print("in the sense that their coefficients are not all integers.")
    print("\nYour question: Should we generate GMI cuts from these constraints in round 2+?")
    print("\nThe answer depends on the GMI theory:")
    print("  - GMI cuts require the tableau row to come from an integer equation")
    print("  - An 'integer equation' means: all coeffs are integer AND all vars are integer AND RHS is integer")
    print("  - Our round 1 cuts have fractional coeffs, so they're NOT integer equations")
    print("  - Therefore, they should NOT generate GMI cuts in subsequent rounds")
    print("\nHowever, you raise a good point about strengthening:")
    print("  - IF all coeffs were integer and all vars were integer but RHS was fractional,")
    print("  - THEN we could strengthen by rounding the RHS appropriately")
    print("  - But that's not the case here - the coefficients themselves are fractional")


if __name__ == "__main__":
    trace_constraints()
