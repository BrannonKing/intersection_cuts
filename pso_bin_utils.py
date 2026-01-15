import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class Particle:
    def __init__(self, n, v_max=4.0, rng=None, initial_x=None):
        self.n = n
        self.v_max = v_max
        self.rng = rng if rng is not None else np.random.default_rng()

        # Position: binary
        if initial_x is not None:
            self.x = initial_x.copy()
        else:
            self.x = self.rng.integers(0, 2, size=n, dtype=np.int8)

        # Velocity: real-valued log-odds
        self.v = self.rng.normal(0.0, 0.1, size=n)

        # Personal best
        self.best_x = self.x.copy()
        self.best_value = np.inf
        self.best_feasible = False

    def evaluate(self, obj_fn):
        val, feasible = obj_fn(self.x)
        if (feasible and not self.best_feasible) or \
           (feasible == self.best_feasible and val < self.best_value):
            self.best_value = val
            self.best_feasible = feasible
            self.best_x = self.x.copy()
        return val, feasible

    @staticmethod
    def bit_diff(a, b):
        """
        Signed bit difference: a ⊖ b
        +1 if a=1, b=0
        -1 if a=0, b=1
         0 otherwise
        """
        return a.astype(np.int8) - b.astype(np.int8)

    def update_velocity(self, global_best_x, w, c1, c2):
        r1 = self.rng.random(self.n)
        r2 = self.rng.random(self.n)

        cognitive = c1 * r1 * self.bit_diff(self.best_x, self.x)
        social    = c2 * r2 * self.bit_diff(global_best_x, self.x)

        self.v = w * self.v + cognitive + social

        # Clamp to avoid sigmoid saturation
        np.clip(self.v, -self.v_max, self.v_max, out=self.v)

    def update_position(self, mutation_prob=0.01):
        probs = sigmoid(self.v)
        self.x = (self.rng.random(self.n) < probs).astype(np.int8)
        
        if mutation_prob > 0:
            mask = self.rng.random(self.n) < mutation_prob
            self.x[mask] = 1 - self.x[mask]


def binary_pso(
    obj_fn,
    n_vars,
    n_particles=30,
    max_iters=200,
    w=0.7,
    c1=1.5,
    c2=1.5,
    v_max=4.0,
    mutation_prob=0.01,
    seed=None,
    init_fn=None,
    repair_fn=None,
    repair_prob=0.1,
):
    """
    Binary Particle Swarm Optimization.
    
    Args:
        obj_fn: Objective function returning (value, is_feasible)
        n_vars: Number of binary variables
        n_particles: Number of particles in the swarm
        max_iters: Maximum iterations
        w: Inertia weight
        c1: Cognitive coefficient
        c2: Social coefficient
        v_max: Maximum velocity
        mutation_prob: Probability of random bit flip
        seed: Random seed
        init_fn: Optional function(rng) -> np.array to generate initial positions
        repair_fn: Optional function(x) -> x to repair infeasible solutions
        repair_prob: Probability of repairing each infeasible particle per iteration
    """
    rng = np.random.default_rng(seed)

    # Initialize particles, optionally with custom initialization
    particles = []
    for _ in range(n_particles):
        if init_fn is not None:
            initial_x = init_fn(rng)
            p = Particle(n_vars, v_max=v_max, rng=rng, initial_x=initial_x)
        else:
            p = Particle(n_vars, v_max=v_max, rng=rng)
        particles.append(p)

    global_best_x = None
    global_best_val = np.inf
    global_best_feasible = False

    for it in range(max_iters):
        print(f"Iteration {it+1}/{max_iters} at {global_best_val}", end='\r')

        # Evaluate particles
        for p in particles:
            val, feasible = p.evaluate(obj_fn)
            
            # Stochastically repair infeasible particles
            if not feasible and repair_fn is not None and rng.random() < repair_prob:
                p.x = repair_fn(p.x)
                val, feasible = p.evaluate(obj_fn)
            
            if global_best_x is None or \
               (feasible and not global_best_feasible) or \
               (feasible == global_best_feasible and val < global_best_val):
                global_best_val = val
                global_best_feasible = feasible
                global_best_x = p.x.copy()

        # Update swarm
        for p in particles:
            p.update_velocity(global_best_x, w, c1, c2)
            p.update_position(mutation_prob=mutation_prob)

    return global_best_x, global_best_val, global_best_feasible
