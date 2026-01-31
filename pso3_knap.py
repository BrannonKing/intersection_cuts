import pso3 as pso

def main():
    import gurobipy as gp
    gp.setParam('OutputFlag', 0)
    import numpy as np
    import knapsack_loader as kl
    import gurobi_utils as gu

    use_equality = False

    instances = kl.generate(3, 2, 20, 5, 10, 1000, equality=use_equality, seed=42)
    for model in instances:
        A, b, c, l, u = gu.get_A_b_c_l_u(model, keep_sparse=False)
        variables = model.getVars()
        integers = []
        for v in variables:
            if v.VType in (gp.GRB.BINARY, gp.GRB.INTEGER):
                integers.append(v.index)
        relaxed_x, relaxed_obj = gu.relaxed_optimum(model)
        print(f"Optimizing knapsack instance {model.ModelName} with relaxed optimum {relaxed_obj}")
        if use_equality:
            is_feasible = lambda x: np.abs(A @ x - b).max() <= 1e-6 and (x >= l - 1e-6).all() and (x <= u + 1e-6).all()
        else:
            is_feasible = lambda x: (A @ x <= b + 1e-6).all() and (x >= l - 1e-6).all() and (x <= u + 1e-6).all()

        best_position, best_value = pso.minimize_mip_pso(
            objective_func=lambda x: (-c.T @ x).item(),
            is_feasible_func=is_feasible,
            relaxed_x=relaxed_x,
            lb=l,
            ub=u,
            integers=integers,
            num_particles=50,
            max_iterations=500,
            seed=42
        )

        if best_position is None:
            print("No feasible solution found!")
        elif not is_feasible(best_position):
            print("Warning: Best position found is not feasible!")

        if use_equality:
            mdl_lll = gu.transform_via_LLL(model, verify=False)
            mdl_lll.optimize()
            actual_obj = mdl_lll.ObjVal
        else:
            model.optimize()
            actual_obj = model.ObjVal

        print(f"Best value found: {-best_value}, Relaxed optimum: {relaxed_obj}, Actual optimum: {actual_obj}")

if __name__ == "__main__":
    main()