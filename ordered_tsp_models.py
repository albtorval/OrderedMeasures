"""
Ordered-aggregation TSP benchmark models.

This module contains the mixed-integer optimization models used in the
computational experiments for ordered aggregation objectives on TSP-like
instances.


Requirements
------------
gurobipy
numpy
networkx
matplotlib  # only needed for plot_tsp_solution
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple

import gurobipy as gp
from gurobipy import GRB
import networkx as nx
import numpy as np


Node = Hashable
Arc = Tuple[Node, Node]
ArcSolution = Dict[Arc, float]


@dataclass(frozen=True)
class TSPResult:
    """Container for the solution of a TSP model."""

    objective_value: float
    selected_arcs: ArcSolution
    runtime: float
    sorted_outgoing_costs: List[float]
    tour: List[Node]
    status: int
    mip_gap: Optional[float] = None


def _node_list(G: nx.DiGraph) -> List[Node]:
    """Return a fixed list of nodes."""
    return list(G.nodes())


def _arc_list(G: nx.DiGraph) -> List[Arc]:
    """Return a fixed list of directed arcs."""
    return list(G.edges())


def _arc_costs(G: nx.DiGraph, weight: str = "weight") -> Dict[Arc, float]:
    """Return arc costs from a NetworkX directed graph."""
    return {(i, j): float(G[i][j][weight]) for i, j in G.edges()}


def _ordered_differences(weights: Sequence[float]) -> List[float]:
    """Return Delta_k = lambda_k - lambda_{k-1}, with Delta_0 = lambda_0."""
    return [
        float(weights[k] - weights[k - 1]) if k >= 1 else float(weights[0])
        for k in range(len(weights))
    ]


def _add_tsp_assignment_constraints(
    model: gp.Model,
    x: gp.tupledict,
    nodes: Sequence[Node],
    arcs: Sequence[Arc],
) -> None:
    """Add in-degree and out-degree assignment constraints."""
    for v in nodes:
        model.addConstr(
            gp.quicksum(x[i, j] for i, j in arcs if i == v) == 1,
            name=f"out_degree[{v}]",
        )
        model.addConstr(
            gp.quicksum(x[i, j] for i, j in arcs if j == v) == 1,
            name=f"in_degree[{v}]",
        )


def _add_mtz_constraints(
    model: gp.Model,
    x: gp.tupledict,
    nodes: Sequence[Node],
    arcs: Sequence[Arc],
    depot: Optional[Node] = None,
) -> gp.tupledict:
    """
    Add Miller--Tucker--Zemlin subtour elimination constraints.

    Assumes that all arcs needed in the MTZ constraints are available in x.
    """
    if depot is None:
        depot = nodes[0]

    n = len(nodes)
    u = model.addVars(nodes, lb=0.0, ub=n - 1, name="mtz_order")
    u[depot].lb = 0.0
    u[depot].ub = 0.0

    arc_set = set(arcs)
    for i in nodes:
        for j in nodes:
            if i != j and j != depot and (i, j) in arc_set:
                model.addConstr(
                    u[i] + 1 <= u[j] + (n - 1) * (1 - x[i, j]),
                    name=f"mtz[{i},{j}]",
                )

    return u


def _add_no_two_cycle_constraints(
    model: gp.Model,
    x: gp.tupledict,
    arcs: Sequence[Arc],
) -> None:
    """Forbid selecting both directions of the same two-node cycle."""
    arc_set = set(arcs)
    added = set()
    for i, j in arcs:
        if (j, i) in arc_set and (j, i) not in added:
            model.addConstr(x[i, j] + x[j, i] <= 1, name=f"no_2_cycle[{i},{j}]")
        added.add((i, j))


def _outgoing_cost_expression(
    node: Node,
    x: gp.tupledict,
    arcs: Sequence[Arc],
    costs: Mapping[Arc, float],
) -> gp.LinExpr:
    """Return the outgoing selected-arc cost expression for a node."""
    return gp.quicksum(costs[i, j] * x[i, j] for i, j in arcs if i == node)


def _extract_solution(
    model: gp.Model,
    x: gp.tupledict,
    nodes: Sequence[Node],
    arcs: Sequence[Arc],
    costs: Mapping[Arc, float],
    depot: Optional[Node] = None,
) -> TSPResult:
    """Extract selected arcs, sorted outgoing costs, and tour."""
    if model.SolCount == 0:
        raise RuntimeError(f"No feasible solution found. Gurobi status code: {model.Status}")

    if depot is None:
        depot = nodes[0]

    selected_arcs = {(i, j): x[i, j].X for i, j in arcs if x[i, j].X > 0.5}

    sorted_outgoing_costs = sorted(
        float(sum(costs[i, j] * x[i, j].X for i, j in arcs if i == v))
        for v in nodes
    )

    successor = {i: j for (i, j) in selected_arcs}
    tour = [depot]
    current = depot
    while len(tour) < len(nodes):
        current = successor[current]
        tour.append(current)
    tour.append(depot)

    mip_gap = model.MIPGap if model.IsMIP else None

    return TSPResult(
        objective_value=round(float(model.ObjVal), 4),
        selected_arcs=selected_arcs,
        runtime=float(model.Runtime),
        sorted_outgoing_costs=sorted_outgoing_costs,
        tour=tour,
        status=int(model.Status),
        mip_gap=mip_gap,
    )


def solve_tsp_linear_ordered(
    G: nx.DiGraph,
    lamb: Sequence[float],
    *,
    weight: str = "weight",
    depot: Optional[Node] = None,
    time_limit: Optional[float] = None,
    mip_gap: Optional[float] = None,
    verbose: bool = False,
) -> TSPResult:
    """
    Solve the TSP with a linear ordered aggregation objective.

    Parameters
    ----------
    G:
        Directed graph. Each arc must have a numerical cost in attribute ``weight``.
    lamb:
        Ordered aggregation weights. Its length must be equal to the number of nodes.
    weight:
        Arc attribute used as cost.
    depot:
        Starting node for tour extraction. If omitted, the first node of ``G`` is used.
    time_limit:
        Optional Gurobi time limit in seconds.
    mip_gap:
        Optional relative MIP gap.
    verbose:
        If True, enable Gurobi output.
    """
    nodes = _node_list(G)
    arcs = _arc_list(G)
    costs = _arc_costs(G, weight)
    n = len(nodes)

    if len(lamb) != n:
        raise ValueError("The length of lamb must be equal to the number of nodes.")

    model = gp.Model("tsp_linear_ordered")
    model.Params.OutputFlag = int(verbose)
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    if mip_gap is not None:
        model.Params.MIPGap = mip_gap

    positions = range(n)
    delta = _ordered_differences(lamb)
    delta_plus = [k for k in positions if delta[k] > 1e-8]
    delta_minus = [k for k in positions if delta[k] < -1e-8]

    x = model.addVars(arcs, vtype=GRB.BINARY, name="x")

    u = model.addVars(nodes, delta_plus, lb=0.0, name="u")
    w = model.addVars(nodes, delta_plus, lb=0.0, name="w")
    gamma = model.addVars(delta_plus, lb=-GRB.INFINITY, name="gamma")
    alpha = model.addVars(nodes, delta_minus, name="alpha")

    obj_positive = gp.quicksum(
        delta[k] * ((k / n) * u[v, k] + (1 - k / n) * w[v, k])
        for k in delta_plus
        for v in nodes
    )
    obj_negative = gp.quicksum(
        delta[k] * alpha[i, k] * costs[i, j] * x[i, j]
        for k in delta_minus
        for i, j in arcs
    )
    obj_constant = gp.quicksum(
        delta[k] * (1 - k / n) * costs[i, j] * x[i, j]
        for k in positions
        for i, j in arcs
    )

    model.setObjective(obj_positive + obj_negative + obj_constant, GRB.MINIMIZE)

    for v in nodes:
        outgoing_cost = _outgoing_cost_expression(v, x, arcs, costs)
        for k in delta_plus:
            model.addConstr(u[v, k] - w[v, k] == outgoing_cost - gamma[k])

    for k in delta_minus:
        model.addConstr(gp.quicksum(alpha[v, k] for v in nodes) == 0)
        for v in nodes:
            alpha[v, k].lb = k / n - 1
            alpha[v, k].ub = k / n

    _add_tsp_assignment_constraints(model, x, nodes, arcs)
    _add_mtz_constraints(model, x, nodes, arcs, depot)
    _add_no_two_cycle_constraints(model, x, arcs)

    model.optimize()
    return _extract_solution(model, x, nodes, arcs, costs, depot)


def solve_tsp_linear_ordered_with_dispersion(
    G: nx.DiGraph,
    lamb: Sequence[float],
    mu: float,
    *,
    weight: str = "weight",
    depot: Optional[Node] = None,
    time_limit: Optional[float] = None,
    mip_gap: Optional[float] = None,
    verbose: bool = False,
) -> TSPResult:
    """
    Solve the TSP with a convex combination of ordered aggregation and mean absolute deviation.

    The objective is

        mu * MAD + (1 - mu) * theta,

    where theta is the ordered aggregation value.
    """
    if not 0 <= mu <= 1:
        raise ValueError("mu must belong to [0, 1].")

    nodes = _node_list(G)
    arcs = _arc_list(G)
    costs = _arc_costs(G, weight)
    n = len(nodes)

    if len(lamb) != n:
        raise ValueError("The length of lamb must be equal to the number of nodes.")

    model = gp.Model("tsp_ordered_with_dispersion")
    model.Params.OutputFlag = int(verbose)
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    if mip_gap is not None:
        model.Params.MIPGap = mip_gap

    positions = range(n)
    delta = _ordered_differences(lamb)
    delta_plus = [k for k in positions if delta[k] > 1e-8]
    delta_minus = [k for k in positions if delta[k] < -1e-8]

    x = model.addVars(arcs, vtype=GRB.BINARY, name="x")
    u = model.addVars(nodes, positions, lb=0.0, name="u")
    w = model.addVars(nodes, positions, lb=0.0, name="w")
    gamma = model.addVars(positions, lb=-GRB.INFINITY, name="gamma")
    alpha = model.addVars(nodes, positions, name="alpha")

    rp = model.addVars(nodes, lb=0.0, name="rp")
    rm = model.addVars(nodes, lb=0.0, name="rm")
    theta = model.addVar(lb=-GRB.INFINITY, name="theta")

    obj_positive_part = gp.quicksum(
        abs(delta[k]) * ((k / n) * u[v, k] + (1 - k / n) * w[v, k])
        for k in delta_plus
        for v in nodes
    )
    obj_negative_part = gp.quicksum(
        delta[k] * alpha[i, k] * costs[i, j] * x[i, j]
        for k in delta_minus
        for i, j in arcs
    )
    constant_negative = gp.quicksum(
        delta[k] * (1 - k / n) * costs[i, j] * x[i, j]
        for k in delta_minus
        for i, j in arcs
    )

    obj_negative_as_positive_part = gp.quicksum(
        abs(delta[k]) * ((k / n) * u[v, k] + (1 - k / n) * w[v, k])
        for k in delta_minus
        for v in nodes
    )
    obj_positive_as_negative_part = gp.quicksum(
        delta[k] * alpha[i, k] * costs[i, j] * x[i, j]
        for k in delta_plus
        for i, j in arcs
    )
    constant_positive = gp.quicksum(
        delta[k] * (1 - k / n) * costs[i, j] * x[i, j]
        for k in delta_plus
        for i, j in arcs
    )

    model.addConstr(theta == obj_positive_part + obj_negative_part + constant_negative)
    model.addConstr(theta == obj_negative_as_positive_part + obj_positive_as_negative_part + constant_positive)

    mean_abs_deviation = (1 / n) * gp.quicksum(rp[v] + rm[v] for v in nodes)
    model.setObjective(mu * mean_abs_deviation + (1 - mu) * theta, GRB.MINIMIZE)

    for v in nodes:
        outgoing_cost = _outgoing_cost_expression(v, x, arcs, costs)
        for k in delta_plus:
            model.addConstr(u[v, k] - w[v, k] == outgoing_cost - gamma[k])

    for k in delta_minus:
        model.addConstr(gp.quicksum(alpha[v, k] for v in nodes) == 0)
        for v in nodes:
            alpha[v, k].lb = k / n - 1
            alpha[v, k].ub = k / n

    _add_tsp_assignment_constraints(model, x, nodes, arcs)
    _add_mtz_constraints(model, x, nodes, arcs, depot)
    _add_no_two_cycle_constraints(model, x, arcs)

    for v in nodes:
        outgoing_cost = _outgoing_cost_expression(v, x, arcs, costs)
        model.addConstr(rp[v] - rm[v] == outgoing_cost - theta)
        model.addSOS(GRB.SOS_TYPE1, [rp[v], rm[v]], [1.0, 2.0])

    model.optimize()
    return _extract_solution(model, x, nodes, arcs, costs, depot)


def solve_tsp_quadratic_ordered(
    G: nx.DiGraph,
    matrix: np.ndarray,
    lamb: Sequence[float],
    alpha_weight: float,
    *,
    weight: str = "weight",
    depot: Optional[Node] = None,
    time_limit: float = 3600,
    mip_gap: float = 0.05,
    verbose: bool = False,
) -> TSPResult:
    """
    Solve the TSP with a quadratic ordered aggregation term.

    Parameters
    ----------
    matrix:
        Quadratic ordered-weight matrix of shape (n, n).
    lamb:
        Linear ordered weights of length n.
    alpha_weight:
        Convex-combination parameter between the quadratic and linear terms.
    """
    if not 0 <= alpha_weight <= 1:
        raise ValueError("alpha_weight must belong to [0, 1].")

    nodes = _node_list(G)
    arcs = _arc_list(G)
    costs = _arc_costs(G, weight)
    n = len(nodes)

    if len(lamb) != n:
        raise ValueError("The length of lamb must be equal to the number of nodes.")
    if matrix.shape != (n, n):
        raise ValueError("matrix must have shape (number_of_nodes, number_of_nodes).")

    model = gp.Model("tsp_quadratic_ordered")
    model.Params.OutputFlag = int(verbose)
    model.Params.TimeLimit = time_limit
    model.Params.MIPGap = mip_gap

    positions = range(n)
    max_outgoing_cost = {
        v: max(costs[i, j] for i, j in arcs if i == v)
        for v in nodes
    }

    x = model.addVars(arcs, vtype=GRB.BINARY, name="x")
    z = model.addVars(nodes, positions, vtype=GRB.BINARY, name="z")
    y = model.addVars(nodes, nodes, positions, positions, lb=0.0, name="y")

    selected_cost = {
        v: _outgoing_cost_expression(v, x, arcs, costs)
        for v in nodes
    }

    pairwise_cost = {}
    for v1 in nodes:
        for v2 in nodes:
            if v1 == v2:
                pairwise_cost[v1, v2] = gp.quicksum(
                    costs[i, j] ** 2 * x[i, j] for i, j in arcs if i == v1
                )
            else:
                pairwise_cost[v1, v2] = gp.quicksum(
                    costs[i, j] * costs[p, q] * x[i, j] * x[p, q]
                    for i, j in arcs
                    for p, q in arcs
                    if i == v1 and p == v2
                )

    quadratic_term = gp.quicksum(
        float(matrix[k, ell]) * y[v1, v2, k, ell]
        for v1 in nodes
        for v2 in nodes
        for k in positions
        for ell in positions
    )
    linear_term = gp.quicksum(
        float(lamb[k]) * selected_cost[v] * z[v, k]
        for v in nodes
        for k in positions
    )

    model.setObjective(alpha_weight * quadratic_term + (1 - alpha_weight) * linear_term, GRB.MINIMIZE)

    for v in nodes:
        model.addConstr(gp.quicksum(z[v, k] for k in positions) == 1, name=f"assign_node[{v}]")
    for k in positions:
        model.addConstr(gp.quicksum(z[v, k] for v in nodes) == 1, name=f"assign_position[{k}]")

    for k in range(n - 1):
        model.addConstr(
            gp.quicksum(z[v, k] * selected_cost[v] for v in nodes)
            <= gp.quicksum(z[v, k + 1] * selected_cost[v] for v in nodes),
            name=f"ordered_costs[{k}]",
        )

    for v1 in nodes:
        for v2 in nodes:
            big_m = (max_outgoing_cost[v1] + 1.0) * (max_outgoing_cost[v2] + 1.0)
            for k in positions:
                for ell in positions:
                    model.addConstr(
                        y[v1, v2, k, ell]
                        >= pairwise_cost[v1, v2] - big_m * (2 - z[v1, k] - z[v2, ell]),
                        name=f"y_lb[{v1},{v2},{k},{ell}]",
                    )
                    model.addConstr(
                        y[v1, v2, k, ell]
                        <= pairwise_cost[v1, v2] + big_m * (2 - z[v1, k] - z[v2, ell]),
                        name=f"y_ub1[{v1},{v2},{k},{ell}]",
                    )
                    model.addConstr(
                        y[v1, v2, k, ell] <= big_m * z[v1, k],
                        name=f"y_ub2[{v1},{v2},{k},{ell}]",
                    )
                    model.addConstr(
                        y[v1, v2, k, ell] <= big_m * z[v2, ell],
                        name=f"y_ub3[{v1},{v2},{k},{ell}]",
                    )

    _add_tsp_assignment_constraints(model, x, nodes, arcs)
    _add_mtz_constraints(model, x, nodes, arcs, depot)
    _add_no_two_cycle_constraints(model, x, arcs)

    model.optimize()
    return _extract_solution(model, x, nodes, arcs, costs, depot)


def plot_tsp_solution(
    G: nx.DiGraph,
    selected_arcs: Mapping[Arc, float],
    *,
    title: str = "TSP solution",
    pos_attribute: str = "pos",
) -> None:
    """Plot a TSP solution on top of the input graph."""
    import matplotlib.pyplot as plt

    pos = nx.get_node_attributes(G, pos_attribute)
    highlighted = list(selected_arcs.keys())

    plt.figure(figsize=(4, 4))
    nx.draw(
        G,
        pos,
        with_labels=True,
        edge_color="gray",
        node_color="lightblue",
        node_size=200,
        width=0.5,
    )
    nx.draw_networkx_edges(G, pos, edgelist=highlighted, edge_color="red", width=2)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def build_euclidean_complete_digraph(
    coordinates: np.ndarray,
    *,
    decimals: int = 2,
) -> nx.DiGraph:
    """Build a complete directed graph with Euclidean arc costs."""
    G = nx.DiGraph()
    n = coordinates.shape[0]

    for i in range(n):
        G.add_node(i, pos=(float(coordinates[i, 0]), float(coordinates[i, 1])))

    distances = np.sqrt(
        ((coordinates[:, None, :] - coordinates[None, :, :]) ** 2).sum(axis=2)
    )
    distances = np.round(distances, decimals=decimals)

    for i in range(n):
        for j in range(n):
            if i != j:
                G.add_edge(i, j, weight=float(distances[i, j]))

    return G


if __name__ == "__main__":
    rng = np.random.default_rng(123)
    coordinates = rng.uniform(0, 100, size=(8, 2))
    graph = build_euclidean_complete_digraph(coordinates)

    n_nodes = graph.number_of_nodes()
    lamb_mean = [1.0 / n_nodes] * n_nodes

    result = solve_tsp_linear_ordered(graph, lamb_mean, verbose=True)
    print(result)
