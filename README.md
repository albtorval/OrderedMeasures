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

# Authors
The article was written by Víctor Blanco (Universidad de Granada), Miguel A. Pozo, Justo Puerto, and Alberto Torrejón (Universidad de Sevilla). All authors contributed equally to this work. Correspondence regarding this repository or the associated paper may be directed to any of the authors via their institutional emails.

For any questions or inquiries regarding this repository, please contact the corresponding author, Alberto Torrejón, at *atorrejon@us.es*.

## Acknowledgements

This work was supported by grants PID2020-114594GB-C21, PID2024-156594NB-C21, and RED2022-134149-T (Thematic Network on Location Science and Related Problems), funded by MICIU/AEI; the FEDER+Junta de Andalucía project C‐EXP‐139‐UGR23 (Mathematical Optimization and Complex Networks); and the IMAG–María de Maeztu grant CEX2020-001105-M and IMUS–María de Maeztu grant CEX2024-001517-M, also funded by MICIU/AEI.
