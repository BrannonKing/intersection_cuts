import cplex as cp
import cplex_utils as cu

def example63():
    # taken from Conforti book, chapter 6

    m = cp.Cplex()
    m.set_problem_name("Book example 6.3")

    # Add variables: x1, x2, x3 (all integer)
    m.variables.add(
        names=['x1', 'x2', 'x3'],
        types="III",  # I for integer
        lb=[0.0, 0.0, 0.0],  # Default lower bounds
        ub=[cp.infinity, cp.infinity, cp.infinity]  # No upper bounds
    )
    
    # Set objective: 0.5*x2 + x3 (maximize)
    m.objective.set_linear([(1, 0.5), (2, 1.0)])  # (variable_index, coefficient)
    m.objective.set_sense(m.objective.sense.maximize)

    # Add all constraints in a single call
    m.linear_constraints.add(
        lin_expr=[
            cp.SparsePair([0, 1, 2], [1.0, 1.0, 1.0]),      # x1 + x2 + x3 <= 2
            cp.SparsePair([0, 2], [1.0, -0.5]),             # x1 - 0.5*x3 >= 0
            cp.SparsePair([1, 2], [1.0, -0.5]),             # x2 - 0.5*x3 >= 0
            cp.SparsePair([0, 2], [1.0, 0.5]),              # x1 + 0.5*x3 <= 1
            cp.SparsePair([0, 1, 2], [-1.0, 1.0, 1.0])      # -x1 + x2 + x3 <= 1
        ],
        senses="LGGLL",  # String format: L=<=, G=>=
        rhs=[2.0, 0.0, 0.0, 1.0, 1.0],
        names=["r0", "r1", "r2", "r3", "r4"]
    )

    return m

model = example63()

relaxed = cu.relaxed_copy(model)
# relaxed.parameters.preprocessing.presolve.set(0)

relaxed.solve()
print("Solution status:", relaxed.solution.get_status_string())
print("Objective value:", relaxed.solution.get_objective_value())

basis = cu.read_basis(relaxed)
print("Basis:", basis)

tableau = cu.read_tableau(relaxed, basis, 0, True)
print("Tableau:\n", tableau)