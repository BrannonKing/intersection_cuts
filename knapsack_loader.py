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


if __name__ == "__main__":
    instances = generate(1, 150, 5000, 0, 1, equality=False, seed=42)
    for instance in instances:
        instance.optimize()
        if instance.Status != gp.GRB.OPTIMAL:
            print(f"Model {instance.ModelName} not solved to optimality.")
            continue
        print(f"Objective value: {instance.ObjVal}")
        # for v in instance.getVars():
        #     print(f"{v.VarName}: {v.X}")