"""
Ordered-aggregation set covering benchmark models.

This module contains clean, reproducible implementations of the set-covering
models used in the computational experiments for ordered aggregation objectives.

The code is intended for a public repository:
    - no wildcard imports,
    - no notebook-specific state,
    - structured return objects,
    - explicit solver options,
    - optional LP/IIS writing for debugging only.

Requirements
------------
gurobipy
numpy
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import gurobipy as gp
from gurobipy import GRB
import numpy as np


@dataclass(frozen=True)
class SCPResult:
    """Container for the solution of a set-covering model."""

    objective_value: float
    selected_sets: List[int]
    runtime: float
    sorted_coverage_values: List[float]
    status: int
    mip_gap: Optional[float] = None
    covered_items: Optional[List[int]] = None


def _ordered_differences(weights: Sequence[float]) -> List[float]:
    """Return Delta_k = lambda_k - lambda_{k-1}, with Delta_0 = lambda_0."""
    return [
        float(weights[k] - weights[k - 1]) if k >= 1 else float(weights[0])
        for k in range(len(weights))
    ]


def _validate_input(
    n_sets: int,
    n_items: int,
    item_weights: Sequence[float],
    covering_sets: Sequence[Iterable[int]],
    lamb: Sequence[float],
    coverage_proportion: float,
) -> None:
    """Validate common input data."""
    if n_sets <= 0:
        raise ValueError("n_sets must be positive.")
    if n_items <= 0:
        raise ValueError("n_items must be positive.")
    if len(item_weights) != n_items:
        raise ValueError("item_weights must have length n_items.")
    if len(covering_sets) != n_sets:
        raise ValueError("covering_sets must have length n_sets.")
    if len(lamb) != n_items:
        raise ValueError("lamb must have length n_items.")
    if not 0 < coverage_proportion <= 1:
        raise ValueError("coverage_proportion must belong to (0, 1].")


def _coverage_expressions(
    x: gp.tupledict,
    n_sets: int,
    n_items: int,
    item_weights: Sequence[float],
    covering_sets: Sequence[Set[int]],
) -> Dict[int, gp.LinExpr]:
    """
    Build coverage-value expressions.

    X[j] is the weighted number of selected sets covering item j:
        X[j] = weight[j] * sum_{i : j in S_i} x_i.
    """
    return {
        j: gp.quicksum(float(item_weights[j]) * x[i] for i in range(n_sets) if j in covering_sets[i])
        for j in range(n_items)
    }


def _add_coverage_constraints(
    model: gp.Model,
    x: gp.tupledict,
    n_sets: int,
    n_items: int,
    covering_sets: Sequence[Set[int]],
    coverage_proportion: float,
) -> Optional[gp.tupledict]:
    """
    Add full or partial set-covering constraints.

    If coverage_proportion == 1, every item must be covered.
    Otherwise, at least ceil(coverage_proportion * n_items) items must be covered.
    """
    if coverage_proportion == 1:
        for j in range(n_items):
            model.addConstr(
                gp.quicksum(x[i] for i in range(n_sets) if j in covering_sets[i]) >= 1,
                name=f"cover_item[{j}]",
            )
        return None

    z = model.addVars(n_items, vtype=GRB.BINARY, name="covered")
    for j in range(n_items):
        covering_indices = [i for i in range(n_sets) if j in covering_sets[i]]

        model.addConstr(
            gp.quicksum(x[i] for i in covering_indices) >= z[j],
            name=f"activate_covered_item[{j}]",
        )

        # If a selected set covers j, then item j is marked as covered.
        for i in covering_indices:
            model.addConstr(z[j] >= x[i], name=f"covered_if_selected[{j},{i}]")

    model.addConstr(
        gp.quicksum(z[j] for j in range(n_items)) >= coverage_proportion * n_items,
        name="minimum_covered_items",
    )

    return z


def _extract_solution(
    model: gp.Model,
    x: gp.tupledict,
    X: Dict[int, gp.LinExpr],
    n_sets: int,
    n_items: int,
    z: Optional[gp.tupledict] = None,
) -> SCPResult:
    """Extract selected sets and coverage values."""
    if model.SolCount == 0:
        raise RuntimeError(f"No feasible solution found. Gurobi status code: {model.Status}")

    selected_sets = [i for i in range(n_sets) if x[i].X > 0.5]
    sorted_coverage_values = sorted(float(X[j].getValue()) for j in range(n_items))
    covered_items = None if z is None else [j for j in range(n_items) if z[j].X > 0.5]

    mip_gap = model.MIPGap if model.IsMIP else None

    return SCPResult(
        objective_value=round(float(model.ObjVal), 4),
        selected_sets=selected_sets,
        runtime=float(model.Runtime),
        sorted_coverage_values=sorted_coverage_values,
        status=int(model.Status),
        mip_gap=mip_gap,
        covered_items=covered_items,
    )


def _configure_model(
    model: gp.Model,
    *,
    verbose: bool,
    time_limit: Optional[float],
    mip_gap: Optional[float],
) -> None:
    """Apply common Gurobi parameters."""
    model.Params.OutputFlag = int(verbose)
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    if mip_gap is not None:
        model.Params.MIPGap = mip_gap


def solve_scp_linear_ordered(
    n_sets: int,
    n_items: int,
    item_weights: Sequence[float],
    covering_sets: Sequence[Iterable[int]],
    lamb: Sequence[float],
    *,
    coverage_proportion: float = 1.0,
    time_limit: Optional[float] = None,
    mip_gap: Optional[float] = None,
    verbose: bool = False,
    write_debug_files: bool = False,
) -> SCPResult:
    """
    Solve the set-covering problem with a linear ordered aggregation objective.

    Parameters
    ----------
    n_sets:
        Number of available sets.
    n_items:
        Number of items to be covered.
    item_weights:
        Nonnegative item weights. ``item_weights[j]`` weights the coverage value of item j.
    covering_sets:
        ``covering_sets[i]`` contains the items covered by set i.
    lamb:
        Ordered aggregation weights, with length ``n_items``.
    coverage_proportion:
        Required proportion of covered items. Use 1.0 for the classical set-covering model.
    """
    _validate_input(n_sets, n_items, item_weights, covering_sets, lamb, coverage_proportion)

    covering_sets = [set(S_i) for S_i in covering_sets]
    items = range(n_items)

    model = gp.Model("scp_linear_ordered")
    _configure_model(model, verbose=verbose, time_limit=time_limit, mip_gap=mip_gap)

    delta = _ordered_differences(lamb)
    delta_plus = [k for k in items if delta[k] > 1e-8]
    delta_minus = [k for k in items if delta[k] < -1e-8]

    x = model.addVars(n_sets, vtype=GRB.BINARY, name="x")
    X = _coverage_expressions(x, n_sets, n_items, item_weights, covering_sets)

    u = model.addVars(items, delta_plus, lb=0.0, name="u")
    w = model.addVars(items, delta_plus, lb=0.0, name="w")
    gamma = model.addVars(delta_plus, lb=-GRB.INFINITY, name="gamma")
    alpha = model.addVars(items, delta_minus, name="alpha")

    obj_positive = gp.quicksum(
        delta[k] * ((k / n_items) * u[j, k] + (1 - k / n_items) * w[j, k])
        for k in delta_plus
        for j in items
    )
    obj_negative = gp.quicksum(
        delta[k] * alpha[j, k] * X[j]
        for k in delta_minus
        for j in items
    )
    obj_constant = gp.quicksum(
        delta[k] * (1 - k / n_items) * X[j]
        for k in items
        for j in items
    )

    model.setObjective(obj_positive + obj_negative + obj_constant, GRB.MINIMIZE)

    for j in items:
        for k in delta_plus:
            model.addConstr(u[j, k] - w[j, k] == X[j] - gamma[k], name=f"ordered_abs[{j},{k}]")

    for k in delta_minus:
        model.addConstr(gp.quicksum(alpha[j, k] for j in items) == 0, name=f"alpha_sum[{k}]")
        for j in items:
            alpha[j, k].lb = k / n_items - 1
            alpha[j, k].ub = k / n_items

    z = _add_coverage_constraints(model, x, n_sets, n_items, covering_sets, coverage_proportion)

    model.optimize()

    if model.Status == GRB.INFEASIBLE and write_debug_files:
        model.computeIIS()
        model.write("iis_scp_linear_ordered.ilp")

    return _extract_solution(model, x, X, n_sets, n_items, z)


def solve_scp_ordered_with_dispersion(
    n_sets: int,
    n_items: int,
    item_weights: Sequence[float],
    covering_sets: Sequence[Iterable[int]],
    lamb: Sequence[float],
    *,
    coverage_proportion: float = 1.0,
    time_limit: Optional[float] = None,
    mip_gap: Optional[float] = None,
    verbose: bool = False,
    write_debug_files: bool = False,
) -> SCPResult:
    """
    Solve the nested/dispersion set-covering model.

    This model computes the ordered aggregation value ``theta`` and minimizes
    the sum of absolute deviations of coverage values from ``theta``.
    """
    _validate_input(n_sets, n_items, item_weights, covering_sets, lamb, coverage_proportion)

    covering_sets = [set(S_i) for S_i in covering_sets]
    items = range(n_items)

    model = gp.Model("scp_ordered_with_dispersion")
    _configure_model(model, verbose=verbose, time_limit=time_limit, mip_gap=mip_gap)

    delta = _ordered_differences(lamb)
    delta_plus = [k for k in items if delta[k] > 1e-8]
    delta_minus = [k for k in items if delta[k] < -1e-8]

    x = model.addVars(n_sets, vtype=GRB.BINARY, name="x")
    X = _coverage_expressions(x, n_sets, n_items, item_weights, covering_sets)

    u = model.addVars(items, items, lb=0.0, name="u")
    w = model.addVars(items, items, lb=0.0, name="w")
    gamma = model.addVars(items, lb=-GRB.INFINITY, name="gamma")
    beta = model.addVars(items, items, name="beta")

    rp = model.addVars(items, lb=0.0, name="rp")
    rm = model.addVars(items, lb=0.0, name="rm")
    theta = model.addVar(lb=-GRB.INFINITY, name="theta")

    positive_representation = (
        gp.quicksum(
            abs(delta[k]) * ((k / n_items) * u[j, k] + (1 - k / n_items) * w[j, k])
            for k in delta_plus
            for j in items
        )
        + gp.quicksum(
            delta[k] * beta[j, k] * X[j]
            for k in delta_minus
            for j in items
        )
        + gp.quicksum(
            delta[k] * (1 - k / n_items) * X[j]
            for k in delta_minus
            for j in items
        )
    )

    negative_representation = (
        gp.quicksum(
            abs(delta[k]) * ((k / n_items) * u[j, k] + (1 - k / n_items) * w[j, k])
            for k in delta_minus
            for j in items
        )
        + gp.quicksum(
            delta[k] * beta[j, k] * X[j]
            for k in delta_plus
            for j in items
        )
        + gp.quicksum(
            delta[k] * (1 - k / n_items) * X[j]
            for k in delta_plus
            for j in items
        )
    )

    model.addConstr(theta == positive_representation, name="theta_positive_representation")
    model.addConstr(theta == negative_representation, name="theta_negative_representation")

    model.setObjective(gp.quicksum(rp[j] + rm[j] for j in items), GRB.MINIMIZE)

    for j in items:
        for k in items:
            model.addConstr(u[j, k] - w[j, k] == X[j] - gamma[k], name=f"ordered_abs[{j},{k}]")

    for k in items:
        model.addConstr(gp.quicksum(beta[j, k] for j in items) == 0, name=f"beta_sum[{k}]")
        for j in items:
            beta[j, k].lb = k / n_items - 1
            beta[j, k].ub = k / n_items

    for j in items:
        model.addConstr(rp[j] - rm[j] == X[j] - theta, name=f"abs_deviation[{j}]")
        model.addSOS(GRB.SOS_TYPE1, [rp[j], rm[j]], [1.0, 2.0])

    z = _add_coverage_constraints(model, x, n_sets, n_items, covering_sets, coverage_proportion)

    model.optimize()

    if model.Status == GRB.INFEASIBLE and write_debug_files:
        model.computeIIS()
        model.write("iis_scp_ordered_with_dispersion.ilp")

    return _extract_solution(model, x, X, n_sets, n_items, z)


def solve_scp_quadratic_ordered(
    n_sets: int,
    n_items: int,
    item_weights: Sequence[float],
    covering_sets: Sequence[Iterable[int]],
    lamb: Sequence[float],
    matrix: np.ndarray,
    alpha_weight: float,
    *,
    coverage_proportion: float = 1.0,
    time_limit: Optional[float] = None,
    mip_gap: Optional[float] = None,
    verbose: bool = False,
    write_debug_files: bool = False,
) -> SCPResult:
    """
    Solve the set-covering problem with a quadratic ordered aggregation term.

    The objective is

        alpha_weight * quadratic_ordered_term
        + (1 - alpha_weight) * linear_ordered_term.

    The products x_i x_r in the original prototype are represented explicitly
    through binary auxiliary variables q[i, r].
    """
    _validate_input(n_sets, n_items, item_weights, covering_sets, lamb, coverage_proportion)

    if matrix.shape != (n_items, n_items):
        raise ValueError("matrix must have shape (n_items, n_items).")
    if not 0 <= alpha_weight <= 1:
        raise ValueError("alpha_weight must belong to [0, 1].")

    item_weights = np.asarray(item_weights, dtype=float)
    covering_sets = [set(S_i) for S_i in covering_sets]
    items = range(n_items)
    sets = range(n_sets)

    model = gp.Model("scp_quadratic_ordered")
    _configure_model(model, verbose=verbose, time_limit=time_limit, mip_gap=mip_gap)

    max_weight = float(np.max(item_weights))
    max_covering_sets = {
        j: len([i for i in sets if j in covering_sets[i]])
        for j in items
    }
    max_number_covering_sets = max(max_covering_sets.values()) if n_items > 0 else 0
    big_m = (1.0 + max_weight**2) * max_number_covering_sets**2

    x = model.addVars(n_sets, vtype=GRB.BINARY, name="x")
    z = model.addVars(items, items, vtype=GRB.BINARY, name="z")
    y = model.addVars(items, items, items, items, lb=0.0, ub=big_m, name="y")

    X = _coverage_expressions(x, n_sets, n_items, item_weights, covering_sets)

    # Explicit linearization of q[p, r] = x[p] x[r].
    q = model.addVars(n_sets, n_sets, vtype=GRB.BINARY, name="q")
    for p in sets:
        for r in sets:
            model.addConstr(q[p, r] <= x[p], name=f"q_ub1[{p},{r}]")
            model.addConstr(q[p, r] <= x[r], name=f"q_ub2[{p},{r}]")
            model.addConstr(q[p, r] >= x[p] + x[r] - 1, name=f"q_lb[{p},{r}]")

    pairwise_coverage = {}
    for i in items:
        for j in items:
            pairwise_coverage[i, j] = float(item_weights[i] * item_weights[j]) * gp.quicksum(
                q[p, r]
                for p in sets
                for r in sets
                if i in covering_sets[p] and j in covering_sets[r]
            )

    quadratic_term = gp.quicksum(
        float(matrix[k, ell]) * y[i, j, k, ell]
        for i in items
        for j in items
        for k in items
        for ell in items
    )

    # Auxiliary variables for the linear ordered term: w[j,k] = X[j] z[j,k].
    w_linear = model.addVars(items, items, lb=0.0, ub=big_m, name="linear_ordered_product")

    linear_term = gp.quicksum(
        float(lamb[k]) * w_linear[j, k]
        for j in items
        for k in items
    )

    model.setObjective(
        alpha_weight * quadratic_term + (1 - alpha_weight) * linear_term,
        GRB.MINIMIZE,
    )

    for j in items:
        model.addConstr(gp.quicksum(z[j, k] for k in items) == 1, name=f"assign_item[{j}]")
    for k in items:
        model.addConstr(gp.quicksum(z[j, k] for j in items) == 1, name=f"assign_position[{k}]")

    for k in range(n_items - 1):
        model.addConstr(
            gp.quicksum(z[j, k] * X[j] for j in items)
            <= gp.quicksum(z[j, k + 1] * X[j] for j in items),
            name=f"ordered_coverage[{k}]",
        )

    z_covered = _add_coverage_constraints(model, x, n_sets, n_items, covering_sets, coverage_proportion)

    for i in items:
        for j in items:
            local_big_m = (1.0 + max_weight**2) * max_covering_sets[i] * max_covering_sets[j]
            for k in items:
                for ell in items:
                    model.addConstr(
                        y[i, j, k, ell]
                        >= pairwise_coverage[i, j] - local_big_m * (2 - z[i, k] - z[j, ell]),
                        name=f"y_lb[{i},{j},{k},{ell}]",
                    )
                    model.addConstr(
                        y[i, j, k, ell]
                        <= pairwise_coverage[i, j] + local_big_m * (2 - z[i, k] - z[j, ell]),
                        name=f"y_ub1[{i},{j},{k},{ell}]",
                    )
                    model.addConstr(
                        y[i, j, k, ell] <= local_big_m * z[i, k],
                        name=f"y_ub2[{i},{j},{k},{ell}]",
                    )
                    model.addConstr(
                        y[i, j, k, ell] <= local_big_m * z[j, ell],
                        name=f"y_ub3[{i},{j},{k},{ell}]",
                    )

    if alpha_weight < 1:
        for j in items:
            local_big_m = max_weight * max_covering_sets[j]
            for k in items:
                model.addConstr(
                    w_linear[j, k] >= X[j] - local_big_m * (1 - z[j, k]),
                    name=f"linear_product_lb[{j},{k}]",
                )
                model.addConstr(
                    w_linear[j, k] <= X[j],
                    name=f"linear_product_ub1[{j},{k}]",
                )
                model.addConstr(
                    w_linear[j, k] <= local_big_m * z[j, k],
                    name=f"linear_product_ub2[{j},{k}]",
                )
    else:
        for j in items:
            for k in items:
                w_linear[j, k].ub = 0.0

    model.optimize()

    return _extract_solution(model, x, X, n_sets, n_items, z_covered)


if __name__ == "__main__":
    # Minimal reproducible example.
    n_sets = 4
    n_items = 5
    item_weights = np.ones(n_items)

    covering_sets = [
        {0, 1},
        {1, 2, 3},
        {3, 4},
        {0, 2, 4},
    ]

    lamb_mean = [1.0 / n_items] * n_items

    result = solve_scp_linear_ordered(
        n_sets,
        n_items,
        item_weights,
        covering_sets,
        lamb_mean,
        verbose=True,
    )

    print(result)
