import cplex as cp
import numpy as np

def relaxed_copy(model: cp.Cplex):
    """
    Create a relaxed copy of a CPLEX model where all integer and binary variables are made continuous.
    """
    relaxed = cp.Cplex(model)

    # Relax integer and binary variables
    for i, var_type in enumerate(relaxed.variables.get_types()):
        if var_type in [relaxed.variables.type.binary, relaxed.variables.type.integer]:
            if var_type == relaxed.variables.type.binary:
                # Ensure binary variables have correct bounds [0,1]
                relaxed.variables.set_upper_bounds(i, 1.0)
                relaxed.variables.set_lower_bounds(i, 0.0)

            # Change to continuous
            relaxed.variables.set_types(i, relaxed.variables.type.continuous)

    # Set problem type to LP
    relaxed.set_problem_type(relaxed.problem_type.LP)
    return relaxed

def read_basis(model: cp.Cplex):
    """
    Read the basis information from a CPLEX model.
    Returns a list indicating which variable/constraint is basic for each row.
    """
    # Get basis status for variables and constraints
    num_vars = model.variables.get_num()
    num_constrs = model.linear_constraints.get_num()
    
    # Get variable basis status
    var_basis, constr_basis = model.solution.basis.get_basis()  # [var_basis, constr_basis]
    
    # Build basis list - which variable is basic for each constraint row
    basic_vars = []
    
    # Find basic variables
    for i, status in enumerate(var_basis):
        if status == model.solution.basis.status.basic:
            basic_vars.append(i)

    # Find basic slack variables
    for i, status in enumerate(constr_basis):
        if status == model.solution.basis.status.basic:
            basic_vars.append(num_vars + i)  # Slack variables come after regular variables

    return basic_vars  # Return only as many as there are constraints


def read_tableau(model: cp.Cplex, basis, extra_rows=0, remove_basis_cols=True):
    """
    Read the simplex tableau from a CPLEX model.
    
    Note: CPLEX doesn't provide direct tableau access like Gurobi.
    This is a simplified implementation that reconstructs key information.
    
    Args:
        model: CPLEX model
        basis: Basis information from read_basis()
        extra_rows: Number of extra rows to allocate
        remove_basis_cols: Whether to remove basis columns from the tableau
        
    Returns:
        tableau: The tableau matrix (approximated)
        col_to_var_idx: Mapping from tableau column to variable index
        negated_rows: List of rows that need to be negated
    """
    tableau = model.solution.advanced.binvrow()
    tableau = np.array(tableau)
    tab2 = model.solution.advanced.binvarow()
    tab2 = np.array(tab2)

    # generate the index of columns to variable indexes:
    num_vars = model.variables.get_num()
    num_constrs = model.linear_constraints.get_num()
    total_cols = num_vars + num_constrs  # Including slack variables
    col_to_var_idx = np.arange(tableau.shape[1])
    negated_rows = [i for i, base in enumerate(basis) if tableau[i, base] < -0.5]
    if remove_basis_cols:
        col_to_var_idx = np.delete(col_to_var_idx, basis)  # TODO: mask may be better than delete calls here
        tableau = np.delete(tableau, basis, 1)  # remove any columns in the basis

    return tableau, col_to_var_idx, negated_rows

def fix_tableau_dirs(model: cp.Cplex, tableau, col_to_var_idx):
    """
    Fix tableau directions based on variable bounds and constraint senses.
    """
    num_vars = model.variables.get_num()
    
    # Get variable information
    var_names = model.variables.get_names()
    var_types = model.variables.get_types()
    var_lb = model.variables.get_lower_bounds()
    var_ub = model.variables.get_upper_bounds()
    
    # Get constraint information  
    constr_senses = model.linear_constraints.get_senses()
    constr_names = model.linear_constraints.get_names()
    
    for col, j in enumerate(col_to_var_idx):
        if j < num_vars:
            # Handle variables
            # Get basis status
            var_basis = model.solution.basis.get_basis()[0]
            if var_basis[j] == model.solution.basis.status.at_upper_bound:
                tableau[:, col] = -tableau[:, col]
            # Additional logic for variable bounds can be added here
        else:
            # Handle slack variables (constraints)
            constr_idx = j - num_vars
            if constr_idx < len(constr_senses):
                if constr_senses[constr_idx] == 'G':  # Greater than constraint
                    tableau[:, col] = -tableau[:, col]
    
    return var_names, constr_names


def relax_int_or_bin_to_continuous(model: cp.Cplex, verbose=False):
    """
    Relax integer and binary variables to continuous.
    
    Returns:
        relaxed_variables: List of relaxed variable indices
        relaxed_index: Mapping from original index to relaxed list index
    """
    relaxed_variables = []
    relaxed_index = {}
    
    # Get current variable types
    var_types = model.variables.get_types()
    num_vars = len(var_types)
    
    for i, var_type in enumerate(var_types):
        if var_type in [model.variables.type.binary, model.variables.type.integer]:
            if var_type == model.variables.type.binary:
                # Ensure binary variables have correct bounds [0,1]
                model.variables.set_upper_bounds(i, 1.0)
                model.variables.set_lower_bounds(i, 0.0)
            
            # Change to continuous
            model.variables.set_types(i, model.variables.type.continuous)
            relaxed_index[i] = len(relaxed_variables)
            relaxed_variables.append(i)
    
    if verbose:
        print(f"   Relaxed {len(relaxed_variables)} variables")
    
    return relaxed_variables, relaxed_index


def make_gmi_cuts(basis, tableau, col_to_var_idx, x, orig_model: cp.Cplex, relaxed_model: cp.Cplex, tol=1e-6):
    """
    Generate Gomory Mixed Integer (GMI) cuts from the optimal tableau for CPLEX.
    
    Args:
        basis: Basis information
        tableau: Tableau matrix  
        col_to_var_idx: Column to variable index mapping
        x: Current solution values
        orig_model: Original CPLEX model with integer variables
        relaxed_model: Relaxed CPLEX model
        tol: Numerical tolerance
        
    Returns:
        cuts: List of constraint data for GMI cuts
    """
    cuts = []
    
    # Identify integer variables from original model
    orig_var_types = orig_model.variables.get_types()
    int_vars = {i for i, vtype in enumerate(orig_var_types) 
                if vtype in [orig_model.variables.type.integer, orig_model.variables.type.binary]}
    
    num_vars = orig_model.variables.get_num()
    num_constrs = orig_model.linear_constraints.get_num()
    
    # Get current solution values
    if hasattr(x, 'shape') and len(x.shape) > 1:
        x_vals = x.flatten()
    else:
        x_vals = np.array(x)
    
    # Extend solution with slack values
    x_extended = np.zeros(num_vars + num_constrs)
    x_extended[:min(len(x_vals), num_vars)] = x_vals[:min(len(x_vals), num_vars)]
    
    # Calculate slack values
    try:
        slack_values = relaxed_model.solution.get_linear_slacks()
        for i, slack in enumerate(slack_values):
            if i < num_constrs:
                x_extended[num_vars + i] = slack
    except:
        pass  # If we can't get slacks, continue with zeros
    
    for row_idx, row in enumerate(tableau):
        if row_idx >= len(basis):
            continue
            
        basis_var_idx = basis[row_idx]
        if basis_var_idx not in int_vars:
            continue
            
        # Calculate fractional part of the basic variable
        if basis_var_idx < len(x_extended):
            basic_var_value = x_extended[basis_var_idx]
        else:
            continue
            
        f0 = basic_var_value - np.floor(basic_var_value)
        if f0 < tol or f0 > 1 - tol:
            continue  # Skip if close to integer
        
        # Build cut coefficients
        cut_vars = []
        cut_coeffs = []
        
        for col_idx, coeff in enumerate(row):
            if abs(coeff) <= tol:
                continue
                
            if col_idx >= len(col_to_var_idx):
                continue
                
            var_idx = col_to_var_idx[col_idx]
            
            # Handle structural variables
            if var_idx < num_vars:
                if var_idx in int_vars:
                    # For integer variables, use GMI formula
                    if var_idx < len(x_extended):
                        var_value = x_extended[var_idx]
                        fj = var_value - np.floor(var_value)
                        
                        if coeff >= 0:
                            cut_coeff = coeff * (1 - fj) / (1 - f0)
                        else:
                            cut_coeff = -coeff * fj / f0
                    else:
                        cut_coeff = coeff
                else:
                    # For continuous variables
                    cut_coeff = min(0, coeff)
                    
                if abs(cut_coeff) > tol:
                    cut_vars.append(var_idx)
                    cut_coeffs.append(cut_coeff)
        
        if len(cut_vars) > 0:
            # Create cut data: variables, coefficients, sense, and RHS
            cut_data = {
                'vars': cut_vars,
                'coeffs': cut_coeffs,
                'sense': 'G',  # Greater than or equal
                'rhs': 1.0
            }
            cuts.append(cut_data)
    
    return cuts


def add_cuts_to_model(model: cp.Cplex, cuts):
    """
    Add GMI cuts to a CPLEX model.
    
    Args:
        model: CPLEX model
        cuts: List of cut data from make_gmi_cuts
    """
    for i, cut in enumerate(cuts):
        model.linear_constraints.add(
            lin_expr=[cp.SparsePair(cut['vars'], cut['coeffs'])],
            senses=cut['sense'],  # Single string, not list
            rhs=[cut['rhs']],
            names=[f"gmi_cut_{i}"]
        )


def run_gmi_cuts(orig_model: cp.Cplex, rounds=1, verbose=False):
    """
    Run multiple rounds of GMI cut generation on a CPLEX model.
    
    Args:
        orig_model: Original CPLEX model with integer variables
        rounds: Number of cutting rounds
        verbose: Whether to print progress information
        
    Returns:
        starting_obj: Objective value before cuts
        final_obj: Objective value after cuts  
        num_cuts_added: Total number of cuts added
    """
    # Create relaxed copy
    relaxed = cp.Cplex(orig_model)
    
    # Relax integer variables
    relax_int_or_bin_to_continuous(relaxed, verbose=verbose)
    
    if not verbose:
        relaxed.set_log_stream(None)
        relaxed.set_error_stream(None)
        relaxed.set_warning_stream(None)
        relaxed.set_results_stream(None)
    relaxed.solve()
    
    if relaxed.solution.get_status() != relaxed.solution.status.optimal:
        raise ValueError("Relaxed model could not be solved to optimality")
    
    starting_obj = relaxed.solution.get_objective_value()
    
    if verbose:
        print(f" GMI round 0 for model, constraints {relaxed.linear_constraints.get_num()}, "
              f"variables {relaxed.variables.get_num()}, start obj: {starting_obj}")
    
    total_cuts_added = 0
    
    for r in range(rounds):
        # Read basis and tableau
        try:
            basis = read_basis(relaxed)
            tableau, col_to_var_idx, negated_rows = read_tableau(relaxed, basis, remove_basis_cols=True)
        except Exception as e:
            if verbose:
                print(f"  Could not read tableau in round {r+1}: {e}")
            break
        
        # Get current solution
        x = relaxed.solution.get_values()
        
        # Generate cuts
        cuts = make_gmi_cuts(basis, tableau, col_to_var_idx, x, orig_model, relaxed, tol=1e-6)
        
        if len(cuts) == 0:
            if verbose:
                print(f"  No GMI cuts found in round {r+1}.")
            break
        
        # Add cuts to model
        add_cuts_to_model(relaxed, cuts)
        total_cuts_added += len(cuts)
        
        # Re-solve
        relaxed.solve()
        
        if relaxed.solution.get_status() != relaxed.solution.status.optimal:
            if verbose:
                print("  GMI cut generation stopped early due to non-optimal relaxation.")
            break
        
        if verbose:
            print(f"  GMI round {r+1}, obj {relaxed.solution.get_objective_value()}, "
                  f"constraints {relaxed.linear_constraints.get_num()}, cuts added: {len(cuts)}")
    
    final_obj = relaxed.solution.get_objective_value()
    
    return starting_obj, final_obj, total_cuts_added


def find_cuttable_rows(model: cp.Cplex, int_var_indices):
    """
    Find rows in the tableau that can be used for cut generation.
    
    Args:
        model: CPLEX model
        int_var_indices: Set of integer variable indices
        
    Yields:
        Tuples of (variable_index, needs_negation)
    """
    if model.solution.get_status() != model.solution.status.optimal:
        return
    
    try:
        basis = read_basis(model)
        tableau, col_to_var_idx, negated_rows = read_tableau(model, basis, remove_basis_cols=True)
        
        tol = 1e-6  # Default tolerance
        
        for i, (base, row) in enumerate(zip(basis, tableau)):
            if base not in int_var_indices:
                continue
            
            if np.all(row >= -tol):
                yield base, False
            elif np.all(row <= tol):
                yield base, True
                
    except Exception:
        # If tableau reading fails, return empty
        return

def get_A_b_c_l_u(model: cp.Cplex, keep_sparse=False):
    """
    Extract constraint matrix A, RHS b, objective c, and bounds l, u from a CPLEX model.

    Args:
        model: CPLEX model
        keep_sparse: Whether to keep sparse format (not implemented for CPLEX)

    Returns:
        A: Constraint matrix (numpy array)
        b: RHS vector
        c: Objective coefficients
        l: Lower bounds
        u: Upper bounds
    """
    num_vars = model.variables.get_num()
    num_constrs = model.linear_constraints.get_num()

    # Get constraint matrix
    A = np.zeros((num_constrs, num_vars))
    for i in range(num_constrs):
        row = model.linear_constraints.get_rows(i)
        for j, coeff in zip(row.ind, row.val):
            A[i, j] = coeff

    # Get RHS vector and handle constraint senses
    b = np.array(model.linear_constraints.get_rhs()).reshape((-1, 1))

    # Get objective coefficients
    c = np.array(model.objective.get_linear()).reshape((-1, 1))

    # Get bounds
    l = np.array(model.variables.get_lower_bounds()).reshape((-1, 1))
    u = np.array(model.variables.get_upper_bounds()).reshape((-1, 1))

    return A, b, c, l, u

# def standardize():
#     # Convert to <= form by negating >= constraints
#     b = np.zeros(num_constrs)
#     for i, (rhs_val, sense) in enumerate(zip(rhs, senses)):
#         if sense == 'L':  # <=
#             b[i] = rhs_val
#         elif sense == 'G':  # >=, convert to <=
#             b[i] = -rhs_val
#             A[i, :] = -A[i, :]
#         elif sense == 'E':  # =, keep as is
#             b[i] = rhs_val
#         else:
#             raise ValueError(f"Unknown constraint sense: {sense}")


from typing import List, Dict, Any
def generate_gmi_cuts(model: cp.Cplex, solution) -> List[Dict[str, Any]]:
    """
    Generates Gomory Mixed-Integer (GMI) cuts for a solved LP relaxation.

    This function inspects the optimal tableau of a solved LP relaxation. For each
    basic variable that is required to be integer but has a fractional value,
    it generates a GMI cut.

    Args:
        model: The original CPLEX model with integer variable type information
        solution: The solution object from a relaxed copy of the model

    Returns:
        A list of generated cuts. Each cut is a dictionary containing the
        cut's linear expression (as a cplex.SparsePair) and its
        right-hand side, ready to be added to a model.
    """
    # Tolerance for floating-point comparisons
    TOLERANCE = 1e-6

    # --- 1. Get Model and Solution Information ---
    lp_solution_values = solution.get_values()
    var_names = model.variables.get_names()
    var_types = model.variables.get_types()
    num_vars = model.variables.get_num()
    num_constraints = model.linear_constraints.get_num()

    # Get basis information from the solution
    cstat, rstat = solution.basis.get_basis()

    # Original RHS vector 'b' from the original model
    rhs_vector = np.array(model.linear_constraints.get_rhs())

    # Identify indices of integer and continuous variables for quick lookup
    integer_var_indices = {
        i for i, v_type in enumerate(var_types)
        if v_type in [model.variables.type.integer, model.variables.type.binary]
    }

    # Identify basic variables
    basic_indices = [i for i, stat in enumerate(cstat) if stat == solution.basis.status.basic]

    generated_cuts = []

    # --- 2. Iterate Through Basic Variables to Find Source Rows ---
    for row_idx, basis_var_idx in enumerate(basic_indices):
        if row_idx >= num_constraints:
            break

        # Check if the basic variable should be an integer
        if basis_var_idx not in integer_var_indices:
            continue

        # Check if its value is fractional
        if basis_var_idx >= len(lp_solution_values):
            continue

        var_value = lp_solution_values[basis_var_idx]
        # Use floor consistently for fractional part calculation
        if abs(var_value - np.floor(var_value + 0.5)) < TOLERANCE:
            continue

        # This row is a candidate for a GMI cut
        # --- 3. Reconstruct the Tableau Row ---

        # Get the row_idx-th row of the basis inverse matrix (B^-1)
        binv_row = np.array(solution.advanced.binvrow(row_idx))

        # Calculate the tableau RHS: b_bar = (B^-1 * b)_{row_idx}
        tableau_rhs = np.dot(binv_row, rhs_vector)

        # Calculate fractional part using floor (always rounds down)
        f0 = tableau_rhs - np.floor(tableau_rhs)

        # If fractional part is negligible, skip
        if f0 < TOLERANCE or (1.0 - f0) < TOLERANCE:
            continue

        cut_expr_var_indices = []
        cut_expr_coeffs = []

        # --- 4. Calculate Cut Coefficients for All Non-Basic Variables ---
        for j in range(num_vars):
            if j == basis_var_idx:
                continue  # Skip basic variable

            # Check if variable is basic
            if j in basic_indices:
                continue  # Skip other basic variables

            # Get the j-th column from the original constraint matrix A
            col_coeffs = []
            for i in range(num_constraints):
                coeff = model.linear_constraints.get_coefficients(i, j)
                col_coeffs.append(coeff)

            col_vector = np.array(col_coeffs)

            # Calculate tableau coefficient: a_bar = (B^-1 * A_j)_{row_idx}
            tableau_coeff = np.dot(binv_row, col_vector)
            if cstat[j] == solution.basis.status.at_upper_bound:
                tableau_coeff = -tableau_coeff

            # Apply GMI formula to get the cut coefficient
            if j in integer_var_indices:
                # GMI for integer non-basic variables
                # Calculate fractional part using floor (always rounds down)
                f_j = tableau_coeff - np.floor(tableau_coeff)

                if f_j <= f0 + TOLERANCE:
                    cut_coeff = f_j
                else:
                    cut_coeff = (f0 * (1.0 - f_j)) / (1.0 - f0)
            else:
                # GMI for continuous non-basic variables
                if tableau_coeff >= 0:
                    cut_coeff = tableau_coeff
                else:
                    cut_coeff = (-f0 * tableau_coeff) / (1.0 - f0)

            if abs(cut_coeff) > TOLERANCE:
                cut_expr_var_indices.append(j)
                cut_expr_coeffs.append(cut_coeff)

        # --- 5. Calculate Cut Coefficients for Non-Basic Slack Variables ---
        # For slack variables, we substitute them out using slack_i = b_i - A_i * x
        # So a coefficient on slack_i becomes a coefficient on -A_i * x plus b_i on RHS
        slack_rhs_adjustment = 0.0

        for slack_idx in range(num_constraints):
            if rstat[slack_idx] == solution.basis.status.basic:
                continue  # Skip basic slack variables

            # For a slack variable, its column in [A|I] is a unit vector
            # The tableau coefficient is just the corresponding element in binv_row
            tableau_coeff = binv_row[slack_idx]

            # Slacks are continuous, so apply the continuous variable formula
            if tableau_coeff >= 0:
                cut_coeff = tableau_coeff
            else:
                cut_coeff = (-f0 * tableau_coeff) / (1.0 - f0)

            if abs(cut_coeff) > TOLERANCE:
                # Substitute slack variable: slack_i = b_i - A_i * x
                # So cut_coeff * slack_i becomes cut_coeff * (b_i - A_i * x)
                # = cut_coeff * b_i - cut_coeff * A_i * x

                # Add cut_coeff * b_i to RHS adjustment
                constraint_rhs = model.linear_constraints.get_rhs()[slack_idx]
                slack_rhs_adjustment += cut_coeff * constraint_rhs

                # Add -cut_coeff * A_i * x to the cut coefficients
                constraint_row = model.linear_constraints.get_rows(slack_idx)
                for var_idx, coeff in zip(constraint_row.ind, constraint_row.val):
                    # Find if this variable is already in our cut
                    if var_idx in [cut_expr_var_indices[k] for k in range(len(cut_expr_var_indices))]:
                        existing_idx = cut_expr_var_indices.index(var_idx)
                        cut_expr_coeffs[existing_idx] -= cut_coeff * coeff
                    else:
                        # Variable not in cut yet, add it
                        cut_expr_var_indices.append(var_idx)
                        cut_expr_coeffs.append(-cut_coeff * coeff)

        # --- 6. Store the Generated Cut ---
        if cut_expr_var_indices:
            # Filter out coefficients that are too small after substitution
            filtered_indices = []
            filtered_coeffs = []
            for idx, coeff in zip(cut_expr_var_indices, cut_expr_coeffs):
                if abs(coeff) > TOLERANCE:
                    filtered_indices.append(idx)
                    filtered_coeffs.append(coeff)

            if filtered_indices:
                # Adjust RHS with slack variable substitution
                adjusted_rhs = f0 - slack_rhs_adjustment

                cut = {
                    'source_variable': var_names[basis_var_idx],
                    'lin_expr': cp.SparsePair(ind=filtered_indices, val=filtered_coeffs),
                    'rhs': adjusted_rhs,
                    'sense': 'G'  # Greater than or equal to
                }
                generated_cuts.append(cut)

    return generated_cuts


def main():
    """
    Sets up and solves a small MILP's LP relaxation, then generates GMI cuts.
    """
    # --- Create a sample MILP problem ---
    # Maximize z = 4*x1 + x2
    # Subject to:
    #   c1:  x1 + 2*x2 <= 5
    #   c2:  2*x1 + x2 <= 6
    #   x1, x2 are non-negative integers

    model = cplex.Cplex()
    model.objective.set_sense(model.objective.sense.maximize)

    # Add variables
    var_names = ["x1", "x2"]
    model.variables.add(names=var_names, types="II", obj=[4.0, 1.0])

    # Add constraints
    constraints = [
        cplex.SparsePair(ind=["x1", "x2"], val=[1.0, 2.0]),
        cplex.SparsePair(ind=["x1", "x2"], val=[2.0, 1.0])
    ]
    rhs = [5.0, 6.0]
    constraint_names = ["c1", "c2"]
    model.linear_constraints.add(lin_expr=constraints, senses="LL", rhs=rhs, names=constraint_names)

    # --- Solve the LP relaxation ---
    # Temporarily change variable types to continuous
    model.variables.set_types([(i, model.variables.type.continuous) for i in range(len(var_names))])

    try:
        model.solve()
        print("✅ LP relaxation solved successfully.")
        solution = model.solution
        print(f"Objective Value: {solution.get_objective_value():.4f}")
        for i, name in enumerate(var_names):
            print(f"  {name} = {solution.get_values(i):.4f}")
    except cplex.exceptions.CplexError as e:
        print(f"Solver failed: {e}")
        return

    # --- Generate and print GMI cuts ---
    print("\n🔍 Generating GMI cuts...")
    gmi_cuts = generate_gmi_cuts(model, solution)

    if not gmi_cuts:
        print("No GMI cuts were generated.")
    else:
        print(f"\nGenerated {len(gmi_cuts)} GMI cut(s):")
        for i, cut in enumerate(gmi_cuts):
            source_var = cut['source_variable']
            expr = cut['lin_expr']
            rhs = cut['rhs']

            cut_str_parts = []
            for var, coeff in zip(expr.ind, expr.val):
                cut_str_parts.append(f"{coeff:.4f} * {var}")

            cut_str = " + ".join(cut_str_parts)
            print(f"\n--- Cut {i + 1} (from source variable {source_var}) ---")
            print(f"  {cut_str} >= {rhs:.4f}")


if __name__ == "__main__":
    main()
