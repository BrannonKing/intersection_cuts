import gurobipy as gp
import highspy as hp
from highspy.highs import highs_var, highs_cons, highs_linear_expression
import numpy as np
import scipy.sparse as sp

class HighsVar(highs_var):
    def __init__(self, v: highs_var):
        super().__init__(v.index, v.highs)

    @property
    def X(self) -> float:
        rd = self.highs.getSolution()
        return rd.col_value[self.index]
    
    @property
    def VarName(self) -> str:
        status, name = self.highs.getColName(self.index)
        if status != hp.HighsStatus.kOk:
            return f"var_{self.index}"
        return name
    
    @property
    def LB(self) -> float:
        status, cost, lb, ub, nnz = self.highs.getCol(self.index)
        assert status == hp.HighsStatus.kOk
        return lb
    
    @property
    def UB(self) -> float:
        status, cost, lb, ub, nnz = self.highs.getCol(self.index)
        assert status == hp.HighsStatus.kOk
        return ub
    
    @property
    def VType(self) -> str:
        status, integ = self.highs.getColIntegrality(self.index)
        assert status == hp.HighsStatus.kOk
        if integ == hp.HighsVarType.kInteger or integ == hp.HighsVarType.kImplicitInteger or integ == hp.HighsVarType.kSemiInteger:
            status, cost, lb, ub, nnz = self.highs.getCol(self.index)
            assert status == hp.HighsStatus.kOk
            if lb == 0.0 and ub == 1.0:
                return 'B'
            return 'I'
        elif integ == hp.HighsVarType.kContinuous or integ == hp.HighsVarType.kSemiContinuous:
            # Semi-continuous: either 0 or continuous in [lb, ub]
            return 'C'
        else:
            print(f"Warning: Unknown variable type in HiGHS: {integ}")
            return 'C'  # Default to continuous for unknown types

class HighsConstr(highs_cons):
    def __init__(self, c: highs_cons):
        super().__init__(c.index, c.highs)

    @property
    def ConstrName(self) -> str:
        status, name = self.highs.getRowName(self.index)
        if status != hp.HighsStatus.kOk:
            return f"constr_{self.index}"
        return name
    
    @property
    def RHS(self) -> float:
        # Call the original HiGHS getRow method (not the overridden one)
        status, lb, ub, nnz = hp.Highs.getRow(self.highs, self.index)
        assert status == hp.HighsStatus.kOk
        if ub < hp.kHighsInf:
            return ub
        else:
            return lb

    @property
    def Sense(self) -> str:
        # Call the original HiGHS getRow method (not the overridden one)
        status, lb, ub, nnz = hp.Highs.getRow(self.highs, self.index)
        assert status == hp.HighsStatus.kOk
        if lb == ub:
            return '='
        elif lb > -hp.kHighsInf and ub < hp.kHighsInf:
            print("Ranged constraint detected.")
            return 'R'  # ranged
        elif ub < hp.kHighsInf:
            return '<'
        else:
            return '>'
        
class RowWrapper:
    def __init__(self, indexes: np.ndarray, coeffs: np.ndarray, getVarFunc):
        self.indexes = indexes
        self.coeffs = coeffs
        self.getVar = getVarFunc

    def size(self) -> int:
        return len(self.indexes)

    def getCoeff(self, i: int) -> float:
        return self.coeffs[i]

class HighsCompat(hp.Highs):
    def __init__(self, model_name: str = ""):
        super().__init__()
        self.ModelName = model_name

    @property
    def NumVars(self) -> int:
        return self.numVariables
    
    @property
    def NumConstrs(self) -> int:
        return self.numConstrs
    
    @property
    def NumIntVars(self) -> int:
        return sum(1 for i in range(self.numVariables) if self.getColIntegrality(i) == hp.HighsVarType.kInteger)
    
    def update(self):
        pass

    def getVars(self):
        vbs = super().getVariables()
        return [HighsVar(v) for v in vbs]
    
    def getConstrs(self):
        cs = super().getConstrs()
        return [HighsConstr(c) for c in cs]

    def getRow(self, constraint):
        if isinstance(constraint, int):
            # If given an index directly, use it
            status, indices, values = self.getRowEntries(constraint)
        else:
            # If given a constraint object, get its index
            status, indices, values = self.getRowEntries(constraint.index)
        assert status == hp.HighsStatus.kOk
        # The lambda receives an index into the indices array, not a variable index
        return RowWrapper(indices, values, lambda i: HighsVar(highs_var(indices[i].item(), self)))
        

def gur_to_highs(gur_model: gp.Model, relaxed=False) -> HighsCompat:
    hig_model = HighsCompat(gur_model.ModelName)
    objSense = hp.ObjSense.kMinimize if gur_model.ModelSense == gp.GRB.MINIMIZE else hp.ObjSense.kMaximize
    
    # Get objective constant (default to 0 if not present)
    gur_obj = gur_model.getObjective()
    obj_const = gur_obj.getConstant() if hasattr(gur_obj, 'getConstant') else 0.0
    hig_model.changeObjectiveOffset(obj_const)
    hig_model.changeObjectiveSense(objSense)
    
    for v in gur_model.getVars():
        var_type = hp.HighsVarType.kContinuous if relaxed or v.VType == 'C' else hp.HighsVarType.kInteger
        lb, ub = (v.LB, v.UB) if v.VType != 'B' else (0.0, 1.0)
        if lb <= -gp.GRB.INFINITY:
            lb = -hp.kHighsInf
        if ub >= gp.GRB.INFINITY:
            ub = hp.kHighsInf
        
        hig_model.addVariable(lb=lb, ub=ub, obj=v.Obj, type=var_type)

    for c in gur_model.getConstrs():
        lb = -hp.kHighsInf
        ub = hp.kHighsInf
        if c.Sense == '<':
            ub = c.RHS
        elif c.Sense == '>':
            lb = c.RHS
        elif c.Sense == '=':
            lb = c.RHS
            ub = c.RHS
        else:
            raise Exception(f"Unknown constraint sense: {c.Sense}")

        row = gur_model.getRow(c)
        indexes = np.array([row.getVar(i).index for i in range(row.size())], dtype=np.int32)
        coeffs = np.array([row.getCoeff(i) for i in range(row.size())])
        hig_model.addRow(lb, ub, len(indexes), indexes, coeffs)

    return hig_model


def read_basis(h: hp.Highs):
    """
    Read the basis from a solved HiGHS model.
    Returns a list of basic variable indices for each row (constraint).
    
    In HiGHS:
    - Basic variables have status kBasic
    - Non-basic variables are at their bounds (kLower, kUpper, kZero)
    - Row slacks can also be basic
    
    The basis is represented as a list where basis[i] is the index of the
    basic variable for row i. Variable indices [0, numVars) are actual variables,
    indices [numVars, numVars+numConstrs) are slack variables.
    """
    basis_obj = h.getBasis()
    assert basis_obj.valid
    
    num_vars = h.numVariables
    col_status = basis_obj.col_status
    assert len(col_status) == num_vars
    row_status = basis_obj.row_status
    
    # Build row-aligned list of basic variables using HiGHS helper
    status, basic_vars = h.getBasicVariables()
    assert status == hp.HighsStatus.kOk
    assert len(basic_vars) == h.numConstrs
    # -1 means slack 0 is basic, -2 means slack 2, etc:
    basis = [(v if v >= 0 else num_vars-v-1) for v in basic_vars]
    
    return basis, col_status, row_status


def read_tableau(h: hp.Highs, basis, remove_basis_cols=True):
    """
    Read the tableau from a HiGHS model using getReducedRow and getBasisInverseRow.
    
    The tableau represents the system in standard form after simplex transformations.
    Each row i represents: x_B[i] = ... (expressed in terms of non-basic variables)
    
    Uses getReducedRow(i) to efficiently get row i of B^{-1}*A (coefficients for variables)
    and getBasisInverseRow(i) to get row i of B^{-1} (coefficients for slack variables).
    
    Returns:
        tableau: numpy array of shape (num_rows, num_cols) where num_cols may be reduced
        col_to_var_idx: mapping from tableau column to original variable/slack index
        negated_rows: list of row indices that were negated (for handling negative pivots)
    """
    num_vars = h.numVariables
    num_rows = h.numConstrs
    num_cols = num_vars + num_rows  # variables + slacks
    
    tableau = np.zeros((num_rows, num_cols))
    
    # Use getReducedRow to get B^{-1}*A directly for each row
    # This is much more efficient than manually computing B^{-1} * A
    for row_idx in range(num_rows):
        # Get row i of B^{-1}*A (coefficients for original variables)
        status, reduced_row = h.getReducedRow(row_idx)
        assert status == hp.HighsStatus.kOk
        
        # Copy the reduced row (B^{-1}*A) into the tableau
        tableau[row_idx, :num_vars] = reduced_row
        
        # For slack columns, get B^{-1} directly (B^{-1} * I = B^{-1})
        status, binv_row = h.getBasisInverseRow(row_idx)
        assert status == hp.HighsStatus.kOk
        
        tableau[row_idx, num_vars:] = binv_row
    
    # Identify negated rows (where the basic variable has negative coefficient)
    for i, basic_var in enumerate(basis):
        if tableau[i, basic_var] < -0.1:
            print("NEGATED ROW at", i, basic_var)
            # The diagonal should be +1, if it's -1, this row is negated
            tableau[i, :] = -tableau[i, :]
    
    col_to_var_idx = np.arange(num_cols)
    
    if remove_basis_cols:
        # Remove columns corresponding to basic variables
        col_to_var_idx = np.delete(col_to_var_idx, basis)
        tableau = np.delete(tableau, basis, axis=1)
    
    return tableau, col_to_var_idx


def is_integer_constraint(constraint_idx: int, h: hp.Highs, int_var_set, tol=1e-6):
    """
    Check if a constraint has all integer coefficients and an integer RHS.
    This is necessary to determine if a slack variable can generate valid GMI cuts.
    """
    status, lb, ub, nnz = hp.Highs.getRow(h, constraint_idx)
    if status != hp.HighsStatus.kOk:
        return False
    
    # Check if RHS is integer (for <= or >= constraints)
    if ub < hp.kHighsInf and abs(ub - round(ub)) > tol:
        return False
    if lb > -hp.kHighsInf and abs(lb - round(lb)) > tol:
        return False
    
    # Check if all coefficients are integer and all variables are integer
    status, indices, values = h.getRowEntries(constraint_idx)
    if status != hp.HighsStatus.kOk:
        return False
    
    for idx, coeff in zip(indices, values):
        if abs(coeff - round(coeff)) > tol:
            return False
        if idx not in int_var_set:
            return False
    
    return True


def get_variable_basis_status(h: hp.Highs, var_idx: int):
    """
    Get the basis status of a variable.
    Returns the HighsBasisStatus value.
    """
    basis = h.getBasis()
    if var_idx < len(basis.col_status):
        return basis.col_status[var_idx]
    return None


def is_manual_slack_variable(h: hp.Highs, var_idx: int, int_var_set, tol=1e-5):
    """
    Determine if a variable is likely a manually-added slack variable.
    
    Manual slacks are continuous variables that:
    1. Are not integer variables
    2. Appear in exactly ONE constraint
    3. That constraint is an equality (lb == ub)
    4. Have coefficient ±1 in that constraint
    
    These are typically added by users to convert inequalities to equalities,
    and including them in GMI cuts creates redundant/weak cuts.
    
    Returns True if variable appears to be a manual slack.
    """
    # Must be continuous (not in integer variable set)
    if var_idx in int_var_set:
        return False
    
    # Check how many constraints this variable appears in
    status, cost, lb, ub, nnz = h.getCol(var_idx)
    if status != hp.HighsStatus.kOk:
        return False
    
    # Must appear in exactly 1 constraint
    if nnz != 1:
        return False
    
    # Get the constraint it appears in
    col_status, col_indices, col_values = h.getColEntries(var_idx)
    if col_status != hp.HighsStatus.kOk or len(col_indices) != 1:
        return False
    
    constraint_idx = col_indices[0]
    coefficient = col_values[0]
    
    # That constraint must be an equality
    status_c, lb_c, ub_c, nnz_c = hp.Highs.getRow(h, constraint_idx)
    if status_c != hp.HighsStatus.kOk:
        return False
    
    is_equality = abs(lb_c - ub_c) < tol
    if not is_equality:
        return False
    
    # Coefficient should be ±1 (typical for slack variables)
    has_unit_coefficient = abs(abs(coefficient) - 1.0) < tol
    
    return has_unit_coefficient


def make_gmi_cuts_highs(basis, var_status, con_status, tableau, col_to_var_idx, x, slack, int_var_set, h: hp.Highs, 
                        tol=1e-5):
    """
    Generate Gomory Mixed Integer (GMI) cuts from the tableau of a HiGHS model.
    
    Args:
        basis: List of basic variable indices for each row
        tableau: The simplex tableau (after removing basic columns)
        col_to_var_idx: Mapping from tableau columns to variable/slack indices
        x: Current solution vector (col_value from getSolution)
        slack: Slack values for constraints (row_value from getSolution)
        int_var_set: Set of integer variable indices
        h: The HiGHS model object
        tol: Tolerance for numerical comparisons
        
    Yields:
        Tuples of (cut_indices, cut_coeffs, cut_rhs) representing cuts of the form:
        sum(cut_coeffs[i] * x[cut_indices[i]]) >= cut_rhs
    """
    num_vars = h.numVariables
    frac = lambda a: a - np.floor(a)
    
    for row_idx, row in enumerate(tableau):
        basis_var_idx = basis[row_idx]
        assert basis_var_idx >= 0
        
        # Skip if basic variable is continuous
        if basis_var_idx < num_vars and basis_var_idx not in int_var_set:
            continue
        
        # Skip if basic slack corresponds to non-integer constraint
        if basis_var_idx >= num_vars:
            constraint_idx = basis_var_idx - num_vars
            if not is_integer_constraint(constraint_idx, h, int_var_set, tol):
                continue
        
        # Start with the original RHS (basic variable value)
        if basis_var_idx < num_vars:
            beta = x[basis_var_idx]
        else:
            constraint_idx = basis_var_idx - num_vars
            assert con_status[constraint_idx] == hp.HighsBasisStatus.kBasic
            status, lb, ub, nnz = hp.Highs.getRow(h, constraint_idx)
            activity = slack[constraint_idx]
            if lb > -hp.kHighsInf:
                beta = activity - lb
            elif ub < hp.kHighsInf:
                beta = ub - activity
            else:
                beta = 0.0

        # Now compute f0 from the TRANSFORMED beta
        f0 = frac(beta)
        
        # Skip if nearly integer
        if f0 < tol or f0 > 1 - tol:
            continue
        
        # Build the cut expression using a dictionary to accumulate coefficients
        cut_dict = {}
        cut_const = 0.0  # Constant term adjustments
        
        for col_idx, aij in enumerate(row):
            var_idx = col_to_var_idx[col_idx].item()
            # assert var_idx not in basis
            
            # CRITICAL: Transform tableau coefficient for non-basic variables at bounds
            # The GMI formula assumes all non-basic variables are at their LOWER bound (≥ 0).
            # We need to transform the tableau BEFORE computing GMI coefficients.
            
            aij_gmi = aij
            
            # For regular variables: check if at upper bound
            if var_idx < num_vars:
                status_col, cost, lb, ub, nnz = h.getCol(var_idx)
                assert status_col == hp.HighsStatus.kOk
                if var_status[var_idx] == hp.HighsBasisStatus.kUpper:
                    # Variable y is at upper bound u, so substitute y' = u - y
                    # Coefficient of y' becomes -a (negated)
                    aij_gmi = -aij
            
            # For slack/surplus variables: adjust for >= constraints
            elif var_idx >= num_vars:
                # This is a >= constraint with surplus
                constraint_idx = var_idx - num_vars
                if con_status[constraint_idx] == hp.HighsBasisStatus.kLower:
                    # Constraint is tight at lower bound, surplus is non-basic at 0
                    # Negate the coefficient to match GMI convention
                    aij_gmi = -aij
            
            # Now compute GMI coefficient from the TRANSFORMED tableau coefficient
            # Standard GMI with RHS = f0 * (1 - f0)
            if var_idx in int_var_set:
                fj = frac(aij_gmi)
                if fj <= f0:
                    fj = (1 - f0) * fj
                else:
                    fj = f0 * (1 - fj)
            else:  # continuous variable
                if aij_gmi >= 0:
                    fj = (1 - f0) * aij_gmi
                else:
                    fj = -f0 * aij_gmi

            if abs(fj) < tol or abs(fj - 1) < tol:
                continue

            if var_idx < num_vars:
                coeff = fj
                if var_status[var_idx] == hp.HighsBasisStatus.kLower:
                    # Account for variables that sit at a non-zero lower bound.
                    # The tableau coefficient is built around x_j = lb_j, but the GMI
                    # derivation assumes the shifted variable y_j = x_j - lb_j.
                    # Adjust the accumulated constant so the final cut is written in
                    # terms of the original x_j without cutting off the true optimum.
                    if lb > -hp.kHighsInf and abs(lb) > tol:
                        cut_const -= fj * lb
                elif var_status[var_idx] == hp.HighsBasisStatus.kUpper:
                    # Symmetric adjustment for variables at a finite upper bound when using
                    # the substitution y_j = u_j - x_j. Converting back to x_j introduces
                    # a sign flip on the coefficient while contributing a constant term.
                    if ub < hp.kHighsInf and abs(ub) > tol:
                        cut_const += fj * ub
                    coeff = -fj

                # For variables: add the (possibly sign-adjusted) coefficient.
                cut_dict[var_idx] = cut_dict.get(var_idx, 0.0) + coeff
            else:
                # Slack variable - check if equality or inequality
                constraint_idx = var_idx - num_vars
                status_c, lb_c, ub_c, nnz_c = hp.Highs.getRow(h, constraint_idx)
                assert status_c == hp.HighsStatus.kOk
                
                if abs(ub_c - lb_c) < tol:
                    # Equality constraint: skip the slack term entirely
                    # Equality constraints have no meaningful slack - they're always tight.
                    # Including the "slack" (which is just the activity) creates weak cuts.
                    continue
                else:
                    # Inequality constraint: expand implicit slack
                    status_r, indices_r, values_r = h.getRowEntries(constraint_idx)
                    assert status_r == hp.HighsStatus.kOk
                    
                    # Determine orientation using row basis status. HiGHS may mark
                    # equality rows as either upper- or lower-active (or even zero),
                    # so fall back to the finite bound if the status is inconclusive.
                    row_status = con_status[constraint_idx]
                    treat_as_upper = row_status == hp.HighsBasisStatus.kUpper
                    treat_as_lower = row_status == hp.HighsBasisStatus.kLower
                    assert treat_as_upper or treat_as_lower

                    if treat_as_upper:
                        # slack = ub - Ax
                        cut_const += fj * ub_c
                        for idx, coeff in zip(indices_r, values_r):
                            cut_dict[idx] = cut_dict.get(idx, 0.0) - (fj * coeff)
                    elif treat_as_lower:
                        # slack = Ax - lb
                        cut_const -= fj * lb_c
                        for idx, coeff in zip(indices_r, values_r):
                            cut_dict[idx] = cut_dict.get(idx, 0.0) + (fj * coeff)
        
        # Compute RHS: f0 * (1 - f0) - cut_const
        # The cut is: sum(cut_dict[i] * x[i]) >= f0*(1-f0) - cut_const
        # But HiGHS uses: sum >= rhs form, so rhs = f0*(1-f0) - cut_const
        cut_rhs = f0 * (1 - f0) - cut_const
        
        # Convert dictionary to lists
        cut_indices = list(k for k, v in cut_dict.items() if abs(v) >= tol)
        cut_coeffs = [cut_dict[idx] for idx in cut_indices]
        
        if len(cut_indices) > 0:
            yield (cut_indices, cut_coeffs, cut_rhs)


def run_gmi_cuts_highs(h: hp.Highs, int_var_set=None, rounds=1, verbose=False, callback=None):
    """
    Run multiple rounds of GMI cut generation on a HiGHS model.
    
    IMPORTANT: The model should already be an LP relaxation (all variables continuous).
    The int_var_set parameter indicates which variables SHOULD be integer in the original problem.
    
    Args:
        h: HiGHS model (should be LP relaxation, already solved to optimality)
        int_var_set: Set of indices for variables that should be integer (required)
        rounds: Number of cutting plane rounds
        verbose: Print progress information
        
    Returns:
        (starting_obj, final_obj, num_cuts): Tuple with initial objective, 
        final objective after cuts, and total number of cuts added
    """
    if int_var_set is None:
        raise ValueError("int_var_set is required - specify which variables should be integer")
    
    assert all(h.getColIntegrality(i)[1] == hp.HighsVarType.kContinuous for i in range(h.numVariables))
    
    h.setOptionValue("output_flag", False)
    if h.getModelStatus() != hp.HighsModelStatus.kOptimal:
        h.run()
    
    assert h.getModelStatus() == hp.HighsModelStatus.kOptimal, "Model must be solved to optimality before running GMI cuts"

    if callback is not None:
        callback(h)

    # Get starting objective
    starting_obj = h.getInfo().objective_function_value
    sense = h.getObjectiveSense()
    total_cuts = 0
    
    if verbose:
        print(f"  GMI initial: obj={starting_obj:.6f}, constraints={h.numConstrs}, variables={h.numVariables}")
    
    for r in range(rounds):
        # Read basis and tableau
        basis, var_status, con_status = read_basis(h)
        tableau, col_to_var_idx = read_tableau(h, basis, remove_basis_cols=True)
        
        # Get current solution
        solution = h.getSolution()
        x = np.array(solution.col_value)
        slack = np.array(solution.row_value)
        
        # Check if all integer variables are integral
        int_vars_fractional = False
        for var_idx in int_var_set:
            if abs(x[var_idx] - np.round(x[var_idx])) > 1e-6:
                int_vars_fractional = True
                break
        
        if not int_vars_fractional:
            if verbose:
                print(f"  All integer variables are integral; stopping at round {r}")
            break
        
        # Generate cuts
        cuts = list(make_gmi_cuts_highs(basis, var_status, con_status, tableau, col_to_var_idx, x, slack, int_var_set, h,
                                        tol=1e-6))
        
        if len(cuts) == 0:
            if verbose:
                print(f"  No cuts generated at round {r}; stopping")
            break
        
        # Add cuts to model
        for cut_indices, cut_coeffs, cut_rhs in cuts:
            h.addRow(cut_rhs, hp.kHighsInf, len(cut_indices), cut_indices, cut_coeffs)
            total_cuts += 1
        
        # Re-solve
        status = h.run()
        if status != hp.HighsStatus.kOk or h.getModelStatus() != hp.HighsModelStatus.kOptimal:
            if verbose:
                print(f"  Solve failed at round {r}: {h.getModelStatus()}")
            break

        if callback is not None:
            callback(h)
        
        final_obj = h.getInfo().objective_function_value
        if verbose:
            print(f"  GMI round {r}: obj={final_obj:.6f}, constraints={h.numConstrs}, cuts_added={len(cuts)}")
    
    final_obj = h.getInfo().objective_function_value
    return starting_obj, final_obj, total_cuts
