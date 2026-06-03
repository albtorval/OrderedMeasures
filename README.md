# Optimization-Based Computation and Integration of Ordered Measures

This repository contains the computational experiments and implementation code developed as part of the research project **"Optimization-Based Computation and Integration of Ordered Measures"**. It includes the scripts, datasets, and reproducible workflows used to validate the theoretical results presented in the associated research article.

---

## Repository Structure

```
OrderedMeasures/
├── data/                            # Datasets for Section 6 experiments
├── ordered_measures_linear.py       # Section 6 – Linear measures
├── ordered_measures_quadratic.py    # Section 6 – Quadratic measures
├── ordered_measures_nested.py       # Section 6 – Nested measures
├── ordered_scp_models.py            # Section 7 – SCP models
└── ordered_tsp_models.py            # Section 7 – TSP models
```

---

## Section 6 – Computational Experiments on Ordered Measures

The files `ordered_measures_linear.py`, `ordered_measures_quadratic.py`, and `ordered_measures_nested.py`, together with the dataset folder `data_1/`, correspond to the computational experiments of **Section 6** of the article. They implement and benchmark the optimization models used to compute ordered measures directly over data samples.

### `ordered_measures_linear.py` — Linear Models (3 formulations)

This file implements three alternative mixed-integer linear programming formulations for computing ordered measures with a linear weighting vector Λ:

- **Formulation 1** (`OM_linear_1`): Based on sorting constraints via binary assignment variables. 
- **Formulation 2** (`OM_linear_2`): Based on k-sum decomposition. 
- **Formulation 3** (`OM_linear_3`): Based on quantile loss. 

All three functions accept as input the size `n`, the weight vector `Lambda`, and the data vector `X`, and return structured result objects including objective value, bound, runtime, and variable assignments.

### `ordered_measures_quadratic.py` — Quadratic Model (1 formulation)

This file implements a single mixed-integer quadratic programming formulation for ordered measures defined by a matrix M of pairwise position weights:

- **Formulation 1** (`OM_quadratic_binary`): Based on sorting constraints with linearization of bilinear terms. The model introduces binary assignment variables and continuous auxiliary variables `y[i,j,k,l]` to linearize the product `z[i,k] · z[j,l]`, enabling exact resolution via Gurobi's non-convex solver (`NonConvex = 2`).

### `ordered_measures_nested.py` — Nested Models (2 formulations)

This file implements two formulations for **nested ordered measures**, where the outer objective combines an ordered aggregation with an inner dispersion functional ρ evaluated at the ordered-measure centroid θ:

- **Formulation 1** (`OM_nested_1`): Based on k-sum bilevel decomposition. 
- **Formulation 2** (`OM_nested_2`): Based on quantile bilevel decomposition. 

Both functions share the same interface, accepting `n`, `Lambda`, `X`, and a string `which_rho` that selects the dispersion functional.

### `data/` — Datasets for Section 6

This folder contains the data instances used in the Section 6 experiments. The datasets are structured to test the performance of the above models across different problem sizes and weight configurations.

---

## Section 7 – Application to Combinatorial Optimization Problems

The files `ordered_scp_models.py` and `ordered_tsp_models.py` correspond to **Section 7** of the article, which applies ordered-aggregation objectives to two classical combinatorial optimization problems: the Set Covering Problem (SCP) and the Traveling Salesman Problem (TSP).

### `ordered_tsp_models.py` — Ordered TSP Models

This file contains mixed-integer optimization models for ordered-aggregation objectives on Traveling Salesman Problem (TSP) instances. The implemented models include:

- a linear ordered-aggregation TSP formulation;
- a linear ordered-aggregation formulation with an additional dispersion term;
- a quadratic ordered-aggregation TSP formulation;
- utilities for building Euclidean complete directed graphs;
- tools for extracting and plotting TSP solutions.

The module returns structured solution objects containing the objective value, selected arcs, runtime, sorted outgoing costs, tour, solver status, and MIP gap when available.

### `ordered_scp_models.py` — Ordered SCP Models

This file contains mixed-integer optimization models for ordered-aggregation objectives on Set Covering Problem (SCP) instances. The implemented models include:

- a linear ordered-aggregation set-covering formulation;
- a nested/dispersion ordered-aggregation formulation;
- a quadratic ordered-aggregation set-covering formulation;
- support for full or partial coverage requirements.

The module returns structured solution objects containing the objective value, selected sets, runtime, sorted coverage values, solver status, MIP gap, and covered items when partial coverage is used.

---

## Requirements

The main dependencies are:

```
gurobipy
numpy
networkx
matplotlib
```

A valid [Gurobi license](https://www.gurobi.com/solutions/licensing/) is required to run the optimization models.

---

## Authors

The article was written by Víctor Blanco (Universidad de Granada), Miguel A. Pozo, Justo Puerto, and Alberto Torrejón (Universidad de Sevilla). All authors contributed equally to this work.

For questions or inquiries regarding this repository or the associated paper, please contact the corresponding author, Alberto Torrejón, at **atorrejon@us.es**.

---

## Acknowledgements

This work was supported by grants PID2020-114594GB-C21, PID2024-156594NB-C21, and RED2022-134149-T (Thematic Network on Location Science and Related Problems), funded by MICIU/AEI; the FEDER+Junta de Andalucía project C‐EXP‐139‐UGR23 (Mathematical Optimization and Complex Networks); and the IMAG–María de Maeztu grant CEX2020-001105-M and IMUS–María de Maeztu grant CEX2024-001517-M, also funded by MICIU/AEI.
