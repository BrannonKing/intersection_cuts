import gurobipy as gp
import numpy as np

def generate(num_instances, num_constrs, num_vars, high_lb=0, high_ub=1, high_weight=1000, equality=True, seed=None, env=None):
    if seed is not None:
        np.random.seed(seed)

    # from the Lattice Reformulation Cuts paper

    for i in range(num_instances):
        # Create the model
        model = gp.Model(f"knapsack_{num_constrs}_{num_vars}_{i}", env=env)
        
        # Generate random weights, values, and capacities
        weights = np.random.randint(1, high_weight + 1, size=(num_constrs, num_vars))
        values = np.random.randint(1, high_weight + 1, size=num_vars)
        if high_ub <= 1:  # high_ub <= 0 for unbounded, 1 for binary
            ub = 1 if high_ub == 1 else gp.GRB.INFINITY
            capacities = np.sum(weights, axis=1) // 2  # axis=0 sums down the columns
        else:
            ub = np.random.randint(high_lb, high_ub + 1, size=num_vars)
            capacities = np.sum(ub * weights, axis=1) // 2
        
        # Create variables
        x = model.addMVar(num_vars, ub=ub, name="x",
                          vtype=gp.GRB.BINARY if high_ub == 1 else gp.GRB.INTEGER)
        
        # Set the objective function
        model.setObjective(values @ x, gp.GRB.MAXIMIZE)
        
        # Add capacity constraints (Aardal's paper uses equality constraints, which make it hard)
        for j in range(num_constrs):
            if equality:
                model.addConstr(x @ weights[j] == capacities[j], name=f"capacity_{j}")
            else:
                model.addConstr(x @ weights[j] <= capacities[j], name=f"capacity_{j}")
        
        model.update()
        yield model


def generate_cplex(num_instances, num_constrs, num_vars, high_lb=0, high_ub=1, high_weight=1000, equality=True, seed=None):
    """Generate knapsack instances using CPLEX instead of Gurobi.
    
    Args:
        num_instances: Number of instances to generate
        num_constrs: Number of constraints per instance
        num_vars: Number of variables per instance
        high_lb: Lower bound for variable bounds (when high_ub > 1)
        high_ub: Upper bound for variables (1 for binary, >1 for integer)
        high_weight: Maximum weight value
        equality: If True, use equality constraints; if False, use <= constraints
        seed: Random seed for reproducibility
    
    Yields:
        CPLEX model instances
    """
    import cplex as cp
    if seed is not None:
        np.random.seed(seed)

    # from the Lattice Reformulation Cuts paper

    for i in range(num_instances):
        # Create the CPLEX model
        model = cp.Cplex()
        model.set_problem_name(f"knapsack_{num_constrs}_{num_vars}_{i}")
        
        # Generate random weights, values, and capacities
        weights = np.random.randint(1, high_weight + 1, size=(num_constrs, num_vars))
        values = np.random.randint(1, high_weight + 1, size=num_vars)
        if high_ub <= 1:  # high_ub <= 0 for unbounded, 1 for binary
            ub = 1.0 if high_ub == 1 else cp.infinity
            capacities = np.sum(weights, axis=1) // 2  # axis=0 sums down the columns
        else:
            ub = np.random.randint(high_lb, high_ub + 1, size=num_vars)
            capacities = np.sum(ub * weights, axis=1) // 2
        
        # Create variables
        var_names = [f"x_{j}" for j in range(num_vars)]
        if high_ub == 1:
            # Binary variables
            model.variables.add(
                obj=values.tolist(),
                lb=[0.0] * num_vars,
                ub=[1.0] * num_vars,
                types=[model.variables.type.binary] * num_vars,
                names=var_names
            )
        else:
            # Integer variables
            if isinstance(ub, np.ndarray):
                ub_list = ub.flatten().tolist()
            elif isinstance(ub, (list, tuple)):
                ub_list = list(ub)
            else:
                ub_list = [float(ub)] * num_vars
            model.variables.add(
                obj=values.tolist(),
                lb=[0.0] * num_vars,
                ub=ub_list,
                types=[model.variables.type.integer] * num_vars,
                names=var_names
            )
        
        # Set objective to maximize (CPLEX minimizes by default, so we negate)
        model.objective.set_sense(model.objective.sense.maximize)
        
        # Add capacity constraints (vectorized version without loop)
        constraint_vars = list(range(num_vars))
        lin_exprs = [cp.SparsePair(constraint_vars, weights[j].tolist()) for j in range(num_constrs)]
        senses = ['E' if equality else 'L'] * num_constrs
        rhs = [float(capacities[j]) for j in range(num_constrs)]
        names = [f"capacity_{j}" for j in range(num_constrs)]
        
        model.linear_constraints.add(
            lin_expr=lin_exprs,
            senses=senses,
            rhs=rhs,
            names=names
        )
        
        yield model


if __name__ == "__main__":
    # Test Gurobi version
    print("Testing Gurobi version:")
    instances = generate(1, 50, 1000, 0, 1, equality=False, seed=42)
    for instance in instances:
        instance.optimize()
        if instance.Status != gp.GRB.OPTIMAL:
            print(f"Model {instance.ModelName} not solved to optimality.")
            continue
        print(f"Objective value: {instance.ObjVal}")
        # for v in instance.getVars():
        #     print(f"{v.VarName}: {v.X}")
    
    # Test CPLEX version
    print("\nTesting CPLEX version:")
    try:
        cplex_instances = generate_cplex(1, 50, 1000, 0, 1, equality=False, seed=42)
        for instance in cplex_instances:
            instance.solve()
            if instance.solution.get_status() != instance.solution.status.optimal:
                print(f"Model {instance.get_problem_name()} not solved to optimality.")
                continue
            print(f"Objective value: {instance.solution.get_objective_value()}")
            # var_names = instance.variables.get_names()
            # var_values = instance.solution.get_values()
            # for name, value in zip(var_names, var_values):
            #     print(f"{name}: {value}")
    except ImportError as e:
        print(f"CPLEX test skipped: {e}")