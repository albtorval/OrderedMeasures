# Optimization-Based Computation and Integration of Ordered Measures

This repository contains the computational experiments and implementation code developed as part of the research project **"Optimization-Based Computation and Integration of Ordered Measures"**. It includes the scripts, datasets, and reproducible workflows used to validate the theoretical results presented in the associated research article.

## Contents

The repository includes clean Python implementations of the optimization models used in the computational experiments. In particular, it contains two main modules:

### `ordered_tsp_models.py`

This file contains mixed-integer optimization models for ordered-aggregation objectives on Traveling Salesman Problem (TSP) instances. The implemented models include:

- a linear ordered-aggregation TSP formulation;
- a linear ordered-aggregation formulation with an additional dispersion term;
- a quadratic ordered-aggregation TSP formulation;
- utilities for building Euclidean complete directed graphs;
- tools for extracting and plotting TSP solutions.

The module returns structured solution objects containing the objective value, selected arcs, runtime, sorted outgoing costs, tour, solver status, and MIP gap when available.

### `ordered_scp_models.py`

This file contains mixed-integer optimization models for ordered-aggregation objectives on Set Covering Problem (SCP) instances. The implemented models include:

- a linear ordered-aggregation set-covering formulation;
- a nested/dispersion ordered-aggregation formulation;
- a quadratic ordered-aggregation set-covering formulation;
- support for full or partial coverage requirements;

The module returns structured solution objects containing the objective value, selected sets, runtime, sorted coverage values, solver status, MIP gap, and covered items when partial coverage is used.

## Requirements

The main dependencies are:

```bash
gurobipy
numpy
networkx
matplotlib
