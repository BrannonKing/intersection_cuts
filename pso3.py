import numpy as np

# what we need:
# 1. You have to be attracted to the nearest integer point (via rounding) unless it's not feasible. 
# 2. We always ignore infeasible integer points.
# 3. You start the swarm at the LP solution and send it in all directions.
# 4. If the particle goes into the infeasible region, we have some options:
# 5. First, we could just reflect it back towards the feasible region.
# 6. Second, we could j
# let's try this out for an update rule:
# w * vk + c1 * r1 * (pbest - xk) + c2 * r2 * (gbest - xk) + c3 * r3 * (nearest - xk)
# c1 is 0 if pbest is infeasible, c2 is 0 if gbest is infeasible or pbest=gbest, c3 is 0 if nearest is infeasible
# c4 is 0 if xk is feasible, otherwise we need c4 * r4 * (last feasible - xk)

class Particle:
    def __init__(self, x_feas):
        self.x_best = None
        self.x_best_value = np.inf
        self.x_feas = x_feas
        self.x = x_feas.copy()
        self.v = np.random.random(self.x_feas.shape) - 0.5
        self.nearest = None
        self.x_is_feasible = True
        self.nearest_is_feasible = False

    def update_velocity(self, x_best_global):
        # we don't want to lose inertia here.
        # in fact, if our nearest isn't feasible, we probably want to accelerate.
        # we want to accelerate if we are far from the bests too.
        # when do we want to slow down? Only if we're passing multiple nearby integer points per hop.

        c = 1.33 if self.nearest_is_feasible else 2.0
        v = self.v * 0.75
        
        if x_best_global is not None:
            r2 = np.random.random(self.x_feas.shape)
            v += c * r2 * (x_best_global - self.x)

        if self.x_best is not None:
            r1 = np.random.random(self.x_feas.shape)
            v += c * r1 * (self.x_best - self.x)

        if not self.x_is_feasible:
            r4 = np.random.random(self.x_feas.shape)
            v += c * r4 * (self.x_feas - self.x)

        nrm = np.linalg.norm(v)
        if nrm < 0.5:
            v /= nrm
        elif nrm > np.sqrt(self.x_feas.shape[0]): # not sure that we want this
            v *= np.sqrt(self.x_feas.shape[0]) / nrm
        
        self.v = v

    def update_position(self):
        self.x += self.v

def nearest_integer(x, integers, lb, ub):
    x_nearest = x.copy()
    for i in integers:
        x_nearest[i, 0] = round(x[i, 0])
        # Project back into bounds
        x_nearest[i, 0] = max(x_nearest[i, 0], lb[i, 0])
        x_nearest[i, 0] = min(x_nearest[i, 0], ub[i, 0])
    return x_nearest

def minimize_mip_pso(
    objective_func,
    is_feasible_func,
    relaxed_x,
    lb=None,
    ub=None,
    integers=None,
    num_particles=50,
    max_iterations=200,
    seed=42
):
    np.random.seed(seed)    
    particles = [Particle(relaxed_x) for _ in range(num_particles)]
    global_best_value = np.inf
    global_best_x = None

    for i in range(max_iterations):        
        for particle in particles:            
            # Evaluate in full space
            feasible_particle = is_feasible_func(particle.x)  # they don't become a best if they aren't feasible
            nearest = nearest_integer(particle.x, integers, lb, ub)
            feasible_nearest = is_feasible_func(nearest)
            nearest_value = objective_func(nearest) if feasible_nearest else np.inf

            if feasible_nearest and nearest_value < particle.x_best_value:
                particle.x_best_value = nearest_value
                particle.x_best = nearest.copy()
            particle.nearest_is_feasible = feasible_nearest
            if feasible_particle:
                particle.x_feas = particle.x.copy()
                particle.x_is_feasible = True
            else:
                particle.x_is_feasible = False            

            if nearest_value < global_best_value and feasible_particle and feasible_nearest:
                global_best_value = nearest_value
                global_best_x = nearest
                print(f"Iteration {i}: New global best value {global_best_value}")

        for particle in particles:
            particle.update_velocity(global_best_x)
            particle.update_position()

    return global_best_x, global_best_value
        
def main():
    import example_loader as el
    import gurobipy as gp
    import gurobi_utils as gu
    gp.setParam('OutputFlag', 0)

    examples = el.get_instances()
    instances = [examples['2Dbelow']] #, examples['2Dnoeasy'], examples['2Dsteep'], examples['2Dabove']]
    for instance in instances:
        model = instance.as_gurobi_model()
        A, b, c, l, u = gu.get_A_b_c_l_u(model, keep_sparse=False)
        senses = [con.Sense for con in model.getConstrs()]
        integers = []
        variables = model.getVars()
        for v in variables:
            if v.VType in (gp.GRB.BINARY, gp.GRB.INTEGER):
                integers.append(v.index)

        lp_solution, lp_obj_val = gu.relaxed_optimum(model)
        print(f"Optimizing instance {model.ModelName} with relaxed optimum {lp_obj_val}")

        if model.ModelSense == gp.GRB.MINIMIZE:
            objective = lambda x: (c.T @ x).item()
            print(f"Minimization problem detected.")
        else:
            objective = lambda x: -(c.T @ x).item()
            lp_obj_val = -lp_obj_val
            print(f"Maximization problem detected; converting to minimization.")
        
        def is_feasible(x):
            for i in range(len(variables)):
                if x[i, 0] < l[i, 0] - 1e-5 or x[i, 0] > u[i, 0] + 1e-5:
                    return False
            Ax = A @ x
            for i in range(b.shape[0]):
                if senses[i] == gp.GRB.LESS_EQUAL:
                    if Ax[i] > b[i, 0] + 1e-5:
                        return False
                elif senses[i] == gp.GRB.GREATER_EQUAL:
                    if Ax[i] < b[i, 0] - 1e-5:
                        return False
                elif senses[i] == gp.GRB.EQUAL:
                    if abs(Ax[i] - b[i, 0]) > 1e-5:
                        return False
            return True

        best_x, best_value = minimize_mip_pso(
            objective,
            is_feasible,
            lp_solution,
            lb=l,
            ub=u,
            integers=integers,
            num_particles=5,
            max_iterations=100,
            seed=42
        )

        print(f"Best integer solution found has value {best_value}, variables:\n{best_x.flatten()}")



if __name__ == "__main__":
    main()