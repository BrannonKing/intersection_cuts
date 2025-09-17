import dikin_utils as du
import io
import numpy as np
import cplex as cp
import cplex_utils as cu
import linetimer as lt
import ntl_wrapper as ntl
import knapsack_loader as kl
import hsnf
import sympy as sp

# Experiment 7b: 
# Generate inequality knapsack instances.
# Measure the solve time in CPLEX.
# LLL(A|b; I|l; -I;u).
# Invert U and use that on objective only.
# Use sympy for c @ U.
# Compare the cuts.

def add_cuts_to_model(model: cp.Cplex, cuts):
    """Add GMI cuts to a CPLEX model."""
    if not cuts:
        return

    lin_exprs = []
    senses = []
    rhs = []
    names = []

    for i, cut in enumerate(cuts):
        lin_exprs.append(cut['lin_expr'])
        senses.append(cut['sense'])
        rhs.append(cut['rhs'])
        names.append(f"gmi_cut_{i}")

    model.linear_constraints.add(
        lin_expr=lin_exprs,
        senses=senses,
        rhs=rhs,
        names=names
    )

def transform(model: cp.Cplex, A: np.ndarray, U: np.ndarray):
    # Get model information using CPLEX APIs
    num_vars = model.variables.get_num()
    num_int_vars = len([t for t in model.variables.get_types() if t in [model.variables.type.integer, model.variables.type.binary]])
    assert num_vars == num_int_vars
    assert U.shape[0] == U.shape[1] and U.shape[1] == num_vars + 1

    # Get objective coefficients
    c = np.array(model.objective.get_linear())
    c = sp.Matrix(c)
    Us = sp.Matrix(U[0:-1, :])
    cUs = c.T @ Us
    # get the gcd of the vector cUs -- gcd was always 1
    cUsf = np.array(cUs, dtype=np.int64).reshape((-1,))

    # Check constraint senses
    senses = model.linear_constraints.get_senses()
    assert all(s == 'L' for s in senses)

    # Create new CPLEX model
    model2 = cp.Cplex()
    model2.set_problem_name("Transformed " + model.get_problem_name())

    # Add variables y
    num_new_vars = U.shape[0]
    var_names = [f"y_{i}" for i in range(num_new_vars)]
    model2.variables.add(
        obj=cUsf.tolist(),
        lb=[-cp.infinity] * num_new_vars,
        ub=[cp.infinity] * num_new_vars,
        types=[model2.variables.type.integer] * num_new_vars,
        names=var_names
    )

    # Set objective sense
    model2.objective.set_sense(model.objective.get_sense())

    # Add constraint: A @ y <= 0
    for i in range(A.shape[0]):
        non_zero_indices = np.nonzero(A[i, :])[0]
        if len(non_zero_indices) > 0:
            model2.linear_constraints.add(
                lin_expr=[cp.SparsePair(non_zero_indices.tolist(), A[i, non_zero_indices].tolist())],
                senses=['L'],
                rhs=[0.0],
                names=[f"transform_constraint_{i}"]
            )

    # Add constraint: U[-1, :] @ y = -1
    last_row = U[-1, :]
    non_zero_indices = np.nonzero(last_row)[0]
    if len(non_zero_indices) > 0:
        model2.linear_constraints.add(
            lin_expr=[cp.SparsePair(non_zero_indices.tolist(), last_row[non_zero_indices].tolist())],
            senses=['E'],
            rhs=[-1.0],
            names=['fix_constraint']
        )

    return model2

def silence(model: cp.Cplex):
    model.parameters.lpmethod.set(model.parameters.lpmethod.values.primal)
    model.parameters.threads.set(1)

    model.parameters.mip.display.set(0)
    model.parameters.simplex.display.set(0)
    model.parameters.barrier.display.set(0)
    model.parameters.network.display.set(0)
    model.parameters.sifting.display.set(0)
    model.parameters.conflict.display.set(0)
    model.parameters.tune.display.set(0)

    null_stream = None # io.StringIO()
    model.set_log_stream(null_stream)
    model.set_error_stream(null_stream)
    model.set_warning_stream(null_stream)
    model.set_results_stream(null_stream)

def main():
    np.random.seed(42)
    for con_count in [2]:
        for var_count in [20]:
            print(f"Generating instances with {con_count} constraints and {var_count} variables")
            runs = 5
            instances = kl.generate_cplex(runs, con_count, var_count, 5, 10, 1000, equality=False)
            before_gaps = []
            after_gaps = []
            for model in instances:
                print("Starting instance!")
                silence(model)
                model.solve()
                relaxed = cu.relaxed_copy(model)
                silence(relaxed)
                relaxed.solve()
                before = relaxed.solution.get_objective_value()
                cuts = cu.generate_gmi_cuts(model, relaxed.solution)
                add_cuts_to_model(relaxed, cuts)
                relaxed.solve()
                after = relaxed.solution.get_objective_value()
                print(f"  Cuts: {len(cuts)}, Before: {before}, After: {after}, Opt: {model.solution.get_objective_value()}")

                before_gaps.append(100 * (before - after) / (before - model.solution.get_objective_value()))

                A, b, c, l, u = cu.get_A_b_c_l_u(model, False)
                block = np.block([
                    [A, b], 
                    [-np.eye(A.shape[1]), -l],
                    [np.eye(A.shape[1]), u]
                ]).astype(np.int64)

                print("  Before max column norm:", np.linalg.norm(block, axis=0).max())
                with lt.CodeTimer("  LLL time", silent=True) as c2:
                    rank, det, U = ntl.lll(block, 9, 10)
                print("  After max column norm:", np.linalg.norm(block, axis=0).max())
                print(f"  LLL took: {c2.took:.2f} ms")

                mdl2 = transform(model, block, U)
                relaxed2 = cu.relaxed_copy(mdl2)
                silence(relaxed2)
                relaxed2.solve()
                cuts = cu.generate_gmi_cuts(mdl2, relaxed2.solution)
                add_cuts_to_model(relaxed2, cuts)
                relaxed2.solve()
                after = relaxed2.solution.get_objective_value()
                print(f"  After LLL cuts: {len(cuts)}, After: {after}")
                after_gaps.append(100 * (before - after) / (before - model.solution.get_objective_value()))

            print(f" Average gap closed by GMI cuts before LLL: {np.mean(before_gaps):.3f}%")
            print(f" Average gap closed by GMI cuts after LLL:  {np.mean(after_gaps):.3f}%")
            print()

if __name__ == "__main__":
    main()