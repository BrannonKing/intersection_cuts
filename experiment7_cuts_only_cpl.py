import dikin_utils as du
import numpy as np
import cplex_utils as cu
import cplex as cp
import linetimer as lt
import ntl_wrapper as ntl
import knapsack_loader as kl
import hsnf
import sympy as sp
import cypari2 as cyp
pari = cyp.Pari()

# Experiment 7b: 
# Generate inequality knapsack instances.
# Measure the solve time in CPLEX.
# LLL(A|b; I|l; -I;u).
# Invert U and use that on objective only.
# Use sympy for c @ U.
# Compare the cuts.

def transform(model: cp.Cplex, A: np.ndarray, U: np.ndarray):
    # Get model information using CPLEX API
    num_vars = model.variables.get_num()
    assert U.shape[0] == U.shape[1] and U.shape[1] == num_vars + 1

    # Get objective coefficients
    c_vals = model.objective.get_linear()
    c = sp.Matrix(c_vals)
    Us = sp.Matrix(U[0:-1, :])
    cUs = c.T @ Us
    # get the gcd of the vector cUs -- gcd was always 1
    cUsf = np.array(cUs, dtype=np.int64).reshape((-1, 1))

    # Get constraint senses and verify they are all <=
    senses = model.linear_constraints.get_senses()
    if not all(sense == 'L' for sense in senses):
        raise ValueError("All constraints must be <= (less than or equal)")

    # Create new CPLEX model
    model2 = cp.Cplex()
    model2.set_problem_name("Transformed " + model.get_problem_name())

    # Add variables with no bounds (will be constrained by U matrix)
    y_names = [f"y_{i}" for i in range(U.shape[0])]
    model2.variables.add(
        obj=cUsf.flatten().tolist(),
        lb=[-cp.infinity] * U.shape[0],
        ub=[cp.infinity] * U.shape[0],
        types="I" * U.shape[0],  # Integer variables
        names=y_names
    )

    # Add objective constant
    obj_const = model.objective.get_offset() if hasattr(model.objective, 'get_offset') else 0

    # Set objective sense (maximize or minimize)
    if model.objective.get_sense() == model.objective.sense.maximize:
        model2.objective.set_sense(model2.objective.sense.maximize)
    else:
        model2.objective.set_sense(model2.objective.sense.minimize)

    # Add constraints: A @ y <= 0
    for i in range(A.shape[0]):
        row_vars = list(range(A.shape[1]))
        row_coeffs = A[i, :].tolist()
        model2.linear_constraints.add(
            lin_expr=[cp.SparsePair(row_vars, row_coeffs)],
            senses="L",
            rhs=[0.0],
            names=[f"constraint_{i}"]
        )

    # Add constraint: -1 == U[-1, :] @ y (generally fixes a single variable to -1)
    last_row_vars = list(range(U.shape[1]))
    last_row_coeffs = U[-1, :].tolist()
    model2.linear_constraints.add(
        lin_expr=[cp.SparsePair(last_row_vars, last_row_coeffs)],
        senses="E",
        rhs=[-1.0],
        names=["u_constraint"]
    )

    return model2

def make_cuts_only(cpx: cp.Cplex):
    # Suppress console output
    cpx.parameters.mip.display.set(2)  # 0 = no display, 1 = display integer solutions, 2 = display nodes, etc.

    cpx.set_log_stream(None)
    cpx.set_error_stream(None)
    cpx.set_warning_stream(None)
    # cpx.set_results_stream(None)

    # No branching
    cpx.parameters.mip.limits.nodes.set(1)

    # Disable all heuristics
    cpx.parameters.mip.strategy.heuristicfreq.set(-1)  # Disable all heuristics
    cpx.parameters.mip.strategy.rinsheur.set(-1)       # Disable RINS heuristic
    cpx.parameters.mip.strategy.fpheur.set(-1)         # Disable feasibility pump heuristic
    cpx.parameters.mip.strategy.lbheur.set(0)          # Disable local branching heuristic
    cpx.parameters.mip.strategy.probe.set(-1)          # Disable probing
    cpx.parameters.mip.strategy.presolvenode.set(-1)   # Disable node presolve

    # Disable all other cuts
    cpx.parameters.mip.cuts.cliques.set(-1)
    cpx.parameters.mip.cuts.covers.set(-1)
    cpx.parameters.mip.cuts.flowcovers.set(-1)
    cpx.parameters.mip.cuts.gubcovers.set(-1)
    cpx.parameters.mip.cuts.implied.set(-1)
    cpx.parameters.mip.cuts.pathcut.set(-1)
    cpx.parameters.mip.cuts.zerohalfcut.set(-1)
    cpx.parameters.mip.cuts.mcfcut.set(-1)
    cpx.parameters.mip.cuts.liftproj.set(-1)
    cpx.parameters.mip.cuts.disjunctive.set(-1)
    cpx.parameters.mip.cuts.bqp.set(-1)

    # Enable Gomory only
    cpx.parameters.mip.cuts.mircut.set(2)
    cpx.parameters.mip.cuts.gomory.set(2)
    cpx.parameters.mip.cuts.localimplied.set(2)

    # Note: CPLEX doesn't have a direct "passes" parameter like Gurobi
    # The number of cut passes is controlled by other parameters or defaults

def main():
    np.random.seed(42)
    for con_count in [2]:
        for var_count in [20]:
            print(f"Generating instances with {con_count} constraints and {var_count} variables")
            runs = 5
            before_times = []
            after_times = []
            instances = kl.generate_cplex(runs, con_count, var_count, 5, 10, 1000, equality=False)
            for model in instances:
                # assumptions on the model: all equality constraints, fully linear objective & constraints, all vars >= 0, maximizing

                make_cuts_only(model)
                with lt.CodeTimer("Original optimization time", silent=True) as c1:
                    model.solve()
                before_times.append(c1.took)

                # Get cut statistics for original model
                # orig_gomory_cuts = model.solution.MIP.get_num_cuts(model.solution.MIP.cut_type.fractional)
                # orig_mir_cuts = model.solution.MIP.get_num_cuts(model.solution.MIP.cut_type.MIR)

                # can I also try it with the rift here? What kind of problems can I solve with the rift?
                # the transform from it won't do anything unless it better aligns the constraints.
                # can i measure the alignment of the starting constraints?!!
                # Then find a way to make them more aligned?
                # then convert that transform to unimodular form?

                # the rounding below doesn't work: x0 isn't feasible for the original model.
                # the cuts that apply to the equality model gain nothing with the slenderizer. It's only for LEQ.
                # because of that, my transform is irrelevant.
                A, b, c, l, u = cu.get_A_b_c_l_u(model, False)
                block = np.block([
                    [A, b], 
                    [-np.eye(A.shape[1]), -l],
                    [np.eye(A.shape[1]), u]
                ]).astype(np.int64)

                # H1, U1 = hsnf.column_style_hermite_normal_form(Ab)
                # np.savetxt("H1.csv", H1, fmt='%d')
                # np.savetxt("U1.csv", U1, fmt='%d')
                # np.savetxt("dumps/Ab.csv", block, fmt='%d')
                print("  Before max column norm:", np.linalg.norm(block, axis=0).max())
                with lt.CodeTimer("  LLL time", silent=True) as c2:
                    rank, det, U = ntl.lll(block, 9, 10)
                    # pri = pari.Mat(Ab)
                    # U = pri.qflll()
                    # U = du.lll_fpylll_cols(Ab, 0.9, verbose=0)
                print("  After max column norm:", np.linalg.norm(block, axis=0).max())
                print(f"  LLL took: {c2.took:.2f} ms")
                # xp, N = solve_via_snf(A, b)
                # now I have an integer null space and an integer starting solution (that may violate bounds)
                # np.savetxt("dumps/Abu.csv", block, fmt='%d')
                # np.savetxt("dumps/U.csv", U, fmt='%d')

                mdl2 = transform(model, block, U)
                # Set CPLEX parameters instead of Gurobi params
                # mdl2.parameters.emphasis.numerical.set(1)  # Equivalent to NumericFocus
                # mdl2.parameters.preprocessing.dual.set(-1)  # Equivalent to DualReductions = 0
                mdl2.set_log_stream(None)  # Disable logging instead of LogToConsole = 1
                make_cuts_only(mdl2)
                with lt.CodeTimer("   Transformed optimization time", silent=True) as c1:
                    mdl2.solve()
                after_times.append(c1.took)

                # Get cut statistics for transformed model
                # trans_gomory_cuts = mdl2.solution.MIP.get_num_cuts(mdl2.solution.MIP.cut_type.fractional)
                # trans_mir_cuts = mdl2.solution.MIP.get_num_cuts(mdl2.solution.MIP.cut_type.MIR)
                #
                # # Get MIP gaps
                # orig_gap = model.solution.MIP.get_mip_relative_gap()
                # trans_gap = mdl2.solution.MIP.get_mip_relative_gap()
                #
                # print(f"Resulting gaps: {orig_gap} vs {trans_gap}")
                # print(f"Gomory cuts: {orig_gomory_cuts} vs {trans_gomory_cuts}")
                # print(f"MIR cuts: {orig_mir_cuts} vs {trans_mir_cuts}")
            print(f" Average original time: {np.mean(before_times):.8f} ms")
            #     averages[(con_count, var_count)] = (np.mean(before_times), np.mean(after_times))
            if after_times:
                print(f" Average transformed time: {np.mean(after_times):.8f} ms")
            print()

if __name__ == "__main__":
    main()

