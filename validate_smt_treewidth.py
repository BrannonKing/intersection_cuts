from __future__ import annotations

import argparse
import heapq
import math
from dataclasses import dataclass
from typing import Any, cast

import gurobipy as gp
import networkx as nx
import numpy as np
from pysat.formula import IDPool

import jsplib_loader as jl
from smt_integer_search import _add_pb_constraint


@dataclass
class FormulaStats:
    primary_vars: int
    auxiliary_vars: int
    clauses: list[list[int]]


class ClauseCollector:
    def __init__(self) -> None:
        self.clauses: list[list[int]] = []

    def append_formula(self, clauses: list[list[int]]) -> None:
        self.clauses.extend([list(clause) for clause in clauses])


def _encode_pure_discrete_sat_formula(model: gp.Model) -> FormulaStats:
    model.update()

    discrete_encoding_by_col: dict[int, tuple[int, list[int], list[int]]] = {}
    next_lit = 1
    for var in model.getVars():
        if var.VType not in (gp.GRB.BINARY, gp.GRB.INTEGER):
            continue

        lower = int(var.LB)
        upper = int(var.UB)
        if lower > upper:
            msg = f"Infeasible bounds for {var.VarName}: [{var.LB}, {var.UB}]"
            raise ValueError(msg)

        if lower == upper:
            discrete_encoding_by_col[var.index] = (lower, [], [])
            continue

        diff = upper - lower
        bit_count = diff.bit_length()
        lits = []
        weights = []
        for bit in range(bit_count):
            lits.append(next_lit)
            weights.append(1 << bit)
            next_lit += 1
        discrete_encoding_by_col[var.index] = (lower, lits, weights)

    collector = ClauseCollector()
    vpool = IDPool(start_from=next_lit)

    for col, (lower, lits, weights) in discrete_encoding_by_col.items():
        if not lits:
            continue
        var = model.getVars()[col]
        upper = int(math.floor(var.UB))
        diff = upper - lower
        if sum(weights) <= diff:
            continue
        _add_pb_constraint(cast(Any, collector), vpool, lits, weights, diff, "<")

    for constraint in model.getConstrs():
        row = model.getRow(constraint)
        has_continuous = any(
            row.getVar(i).VType not in (gp.GRB.BINARY, gp.GRB.INTEGER)
            for i in range(row.size())
        )
        if has_continuous:
            continue

        discrete_row_indices = [
            i for i in range(row.size()) if row.getVar(i).VType in (gp.GRB.BINARY, gp.GRB.INTEGER)
        ]
        if not discrete_row_indices:
            continue

        lits: list[int] = []
        weights: list[int] = []
        adjusted_rhs = constraint.RHS
        for i in discrete_row_indices:
            var = row.getVar(i)
            coeff = int(row.getCoeff(i))
            lower, var_lits, var_weights = discrete_encoding_by_col[var.index]
            adjusted_rhs -= coeff * lower
            for lit, weight in zip(var_lits, var_weights, strict=True):
                lits.append(lit)
                weights.append(coeff * weight)

        adjusted_rhs = int(round(adjusted_rhs))

        if not lits:
            if constraint.Sense == "<" and 0 > adjusted_rhs:
                msg = f"Infeasible fixed constraint: 0 <= {adjusted_rhs}"
                raise ValueError(msg)
            if constraint.Sense == ">" and 0 < adjusted_rhs:
                msg = f"Infeasible fixed constraint: 0 >= {adjusted_rhs}"
                raise ValueError(msg)
            if constraint.Sense == "=" and 0 != adjusted_rhs:
                msg = f"Infeasible fixed constraint: 0 == {adjusted_rhs}"
                raise ValueError(msg)
            continue

        _add_pb_constraint(cast(Any, collector), vpool, lits, weights, adjusted_rhs, constraint.Sense)

    max_var = max((abs(lit) for clause in collector.clauses for lit in clause), default=0)
    primary_vars = next_lit - 1
    auxiliary_vars = max(0, max_var - primary_vars)
    return FormulaStats(
        primary_vars=primary_vars,
        auxiliary_vars=auxiliary_vars,
        clauses=collector.clauses,
    )


def _build_primal_graph(clauses: list[list[int]]) -> nx.Graph:
    graph = nx.Graph()
    for clause in clauses:
        variables = sorted({abs(lit) for lit in clause})
        graph.add_nodes_from(variables)
        for idx, left in enumerate(variables):
            for right in variables[idx + 1 :]:
                graph.add_edge(left, right)
    return graph


def _build_incidence_graph(clauses: list[list[int]]) -> nx.Graph:
    graph = nx.Graph()
    for clause_idx, clause in enumerate(clauses):
        clause_node = ("clause", clause_idx)
        graph.add_node(clause_node, bipartite=1)
        for lit in clause:
            var_node = ("var", abs(lit))
            graph.add_node(var_node, bipartite=0)
            graph.add_edge(var_node, clause_node)
    return graph


def _degeneracy_lower_bound(graph: nx.Graph) -> int:
    if graph.number_of_nodes() == 0:
        return 0

    return max(nx.core_number(graph).values(), default=0)


def _count_fill_edges(
    node: int | tuple[str, int],
    adjacency: dict[int | tuple[str, int], set[int | tuple[str, int]]],
    active: set[int | tuple[str, int]],
) -> tuple[int, list[int | tuple[str, int]]]:
    neighbors = list(adjacency[node] & active)
    fill_edges = 0
    for idx, left in enumerate(neighbors):
        left_adj = adjacency[left]
        for right in neighbors[idx + 1 :]:
            if right not in left_adj:
                fill_edges += 1
    return fill_edges, neighbors


def _k_lowest_min_fill_upper_bound(graph: nx.Graph, k: int) -> int:
    if graph.number_of_nodes() == 0:
        return 0

    adjacency = {node: set(graph.neighbors(node)) for node in graph.nodes}
    active = set(adjacency)
    heap = [(len(neighbors), node) for node, neighbors in adjacency.items()]
    heapq.heapify(heap)

    width = 0
    while active:
        candidates: list[tuple[int, int | tuple[str, int]]] = []
        seen: set[int | tuple[str, int]] = set()
        while heap and len(candidates) < k:
            degree, node = heapq.heappop(heap)
            if node not in active or node in seen:
                continue

            current_degree = len(adjacency[node] & active)
            if current_degree != degree:
                heapq.heappush(heap, (current_degree, node))
                continue

            candidates.append((current_degree, node))
            seen.add(node)

        if not candidates:
            break

        best_degree, best_node = candidates[0]
        best_fill, best_neighbors = _count_fill_edges(best_node, adjacency, active)
        for degree, node in candidates[1:]:
            fill_edges, neighbors = _count_fill_edges(node, adjacency, active)
            if (fill_edges, degree) < (best_fill, best_degree):
                best_degree = degree
                best_node = node
                best_fill = fill_edges
                best_neighbors = neighbors

        for degree, node in candidates:
            if node != best_node:
                heapq.heappush(heap, (degree, node))

        width = max(width, best_degree)
        neighbors = best_neighbors
        for idx, left in enumerate(neighbors):
            left_adj = adjacency[left]
            for right in neighbors[idx + 1 :]:
                if right not in left_adj:
                    left_adj.add(right)
                    adjacency[right].add(left)

        active.remove(best_node)
        adjacency[best_node].clear()
        for neighbor in neighbors:
            adjacency[neighbor].discard(best_node)
            heapq.heappush(heap, (len(adjacency[neighbor] & active), neighbor))

    return width


def _summarize_clause_lengths(clauses: list[list[int]]) -> dict[str, float]:
    if not clauses:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}

    lengths = np.array([len(clause) for clause in clauses], dtype=float)
    return {
        "min": float(lengths.min()),
        "mean": float(lengths.mean()),
        "max": float(lengths.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", default="abz5", help="JSPLIB instance name to load")
    parser.add_argument(
        "--k-lowest",
        type=int,
        default=16,
        help="Choose the elimination vertex by minimum fill among the k lowest-degree live nodes",
    )
    parser.add_argument(
        "--no-presolve",
        action="store_true",
        help="Skip presolve before extracting the SAT encoding",
    )
    args = parser.parse_args()

    gp.setParam("OutputFlag", 0)
    # instance = jl.get_instances()[args.instance]
    # model = instance.as_gurobi_balas_model(use_big_m=True, all_int=True)
    # if not args.no_presolve:
    #     model = model.presolve()

    import knapsack_loader as kl
    model = list(kl.generate(1, 2, 20, 5, 10, 1000, equality=False, seed=43))[0]
    for var in model.getVars():
        var.Obj = -var.Obj
    model.ModelSense = gp.GRB.MINIMIZE
    model.update()
    if not args.no_presolve:
        model = model.presolve()

    formula = _encode_pure_discrete_sat_formula(model)
    incidence_graph = _build_incidence_graph(formula.clauses)

    incidence_treewidth_lb = _degeneracy_lower_bound(incidence_graph)
    incidence_treewidth_ub = _k_lowest_min_fill_upper_bound(incidence_graph, max(1, args.k_lowest))
    clause_lengths = _summarize_clause_lengths(formula.clauses)

    # primal_graph = _build_primal_graph(formula.clauses)
    # primal_treewidth_lb = _degeneracy_lower_bound(primal_graph)
    # primal_treewidth_ub = _k_lowest_min_fill_upper_bound(primal_graph, max(1, args.k_lowest))

    print(f"Instance: {args.instance}")
    print(f"Primary SAT variables: {formula.primary_vars}")
    print(f"Auxiliary SAT variables: {formula.auxiliary_vars}")
    print(f"CNF clauses: {len(formula.clauses)}")
    print(
        "Clause lengths: "
        f"min={clause_lengths['min']:.0f}, "
        f"mean={clause_lengths['mean']:.2f}, "
        f"max={clause_lengths['max']:.0f}"
    )
    # print(
    #     "Primal graph: "
    #     f"nodes={primal_graph.number_of_nodes()}, edges={primal_graph.number_of_edges()}, "
    #     f"treewidth_lower_bound={primal_treewidth_lb}, "
    #     f"treewidth_upper_bound={primal_treewidth_ub}"
    # )
    print(
        "Incidence graph: "
        f"nodes={incidence_graph.number_of_nodes()}, edges={incidence_graph.number_of_edges()}, "
        f"treewidth_lower_bound={incidence_treewidth_lb}, "
        f"treewidth_upper_bound={incidence_treewidth_ub}"
    )


if __name__ == "__main__":
    main()