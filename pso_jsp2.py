import pso_bin_utils as pbu
import jsplib_loader as jl
import rustworkx as rx
import numpy as np


def build_disjunctive_model(instance: jl.JspInstance):
    g = rx.PyDiGraph()
    start = g.add_node((-1, -1, 0))
    assert start == 0
    end = g.add_node((-2, -2, 0))
    mach_to_nodes = [[] for _ in range(instance.machines)]
    instance._ensure_work_loaded()
    for r, row in enumerate(instance.work):
        prev = start
        prev_d = 0
        for c, (m, d) in enumerate(row):
            node = g.add_node((r, m, d))
            g.add_edge(prev, node, prev_d)
            mach_to_nodes[m].append(node)
            prev = node
            prev_d = d
        g.add_edge(prev, end, prev_d)
    
    disjunctives = {}
    k = 0
    for m, nodes in enumerate(mach_to_nodes):
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                n1 = nodes[i]
                n2 = nodes[j]
                disjunctives[k] = (n1, n2)
                k += 1
    return g, disjunctives


def repair_solution(graph: rx.PyDiGraph, disjunctives: dict, x, max_fixes=5):
    """
    Repair an infeasible solution by flipping disjunctive edges that cause cycles.
    Returns the repaired solution. max_fixes limits how many edges to flip.
    """
    x = x.copy()
    g = graph.copy()
    
    # Add all disjunctive edges based on current x
    for k, (a, b) in disjunctives.items():
        if x[k]:
            g.add_edge(a, b, -k - 1)
        else:
            g.add_edge(b, a, -k - 1)
    
    # Iteratively fix cycles (limited by max_fixes)
    for _ in range(max_fixes):
        try:
            rx.topological_sort(g)
            break
        except rx.DAGHasCycle:
            cycle = rx.digraph_find_cycle(g, 0)
            for u, v in cycle:
                data = g.get_edge_data(u, v)
                if isinstance(data, int) and data < 0:
                    k = -data - 1
                    x[k] = 1 - x[k]
                    g.remove_edge(u, v)
                    g.add_edge(v, u, -k - 1)
                    break
    
    return x


def generate_feasible_solution(graph: rx.PyDiGraph, disjunctives: dict, rng):
    """Generate a feasible solution by random init + full repair."""
    n = len(disjunctives)
    x = rng.integers(0, 2, size=n, dtype=np.int8)
    return repair_solution(graph, disjunctives, x, max_fixes=n)  # Full repair


def count_cycle_edges(graph: rx.PyDiGraph, disjunctives: dict, x):
    """Count how many disjunctive edges need to be flipped to achieve feasibility."""
    g = graph.copy()
    for k, (a, b) in disjunctives.items():
        if x[k]:
            g.add_edge(a, b, -k - 1)
        else:
            g.add_edge(b, a, -k - 1)
    
    count = 0
    for _ in range(len(disjunctives)):
        try:
            rx.topological_sort(g)
            break
        except rx.DAGHasCycle:
            cycle = rx.digraph_find_cycle(g, 0)
            for u, v in cycle:
                data = g.get_edge_data(u, v)
                if isinstance(data, int) and data < 0:
                    g.remove_edge(u, v)
                    count += 1
                    break
    return count


def solve_disjunctive_model(graph: rx.PyDiGraph, disjunctives: dict, x):
    """Evaluate makespan. Returns (makespan, True) if feasible, else (penalty, False)."""
    g = graph.copy()
    for k, (a, b) in disjunctives.items():
        if x[k]:
            g.add_edge(a, b, None)
        else:
            g.add_edge(b, a, None)
    try:
        length = rx.dag_longest_path_length(g, weight_fn=lambda u, v, _: g.get_node_data(u)[2])
        return length, True
    except rx.DAGHasCycle:
        # Penalize based on number of cycle-causing edges
        n_violations = count_cycle_edges(graph, disjunctives, x)
        return 1e6 + n_violations * 1000, False


def main():
    instances = jl.get_instances()
    problems = [instances['abz4']]  # , instances['abz4'], instances['abz5']]
    num_problems = len(problems)

    for i in range(num_problems):
        problem = problems[i]
        print(f"Solving problem {i+1}/{num_problems}: {problem.name}")
        model, disjunctives = build_disjunctive_model(problem)
        objective = lambda x: solve_disjunctive_model(model, disjunctives, x)
        dims = len(disjunctives)
        
        # Create initialization function that generates feasible solutions
        init_fn = lambda rng: generate_feasible_solution(model, disjunctives, rng)
        
        # Create repair function - full repair
        n_disj = dims
        repair_fn = lambda x: repair_solution(model, disjunctives, x, max_fixes=n_disj)

        # Run optimization with stochastic repair for exploration
        best_position, best_value, is_feasible = pbu.binary_pso(
            objective,
            n_vars=dims,
            n_particles=100,
            max_iters=1000,
            w=0.7,
            c1=1.5,
            c2=2.5,
            v_max=4.0,            
            mutation_prob=0.01,
            seed=42 + i,
            init_fn=init_fn,
            repair_fn=repair_fn,
            repair_prob=0.2,
        )

        print(f"\nProblem {i+1}/{num_problems}: Best Value = {best_value}, Feasible = {is_feasible}")


if __name__ == "__main__":
    main()