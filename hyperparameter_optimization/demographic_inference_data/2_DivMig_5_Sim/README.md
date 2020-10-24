## 2_DivMig_5_Sim


| Number of populations | Number of parameters | Max log likelihood | Size of spectrum |
| --- | --- | --- | --- |
| 2 | 5 | -1310.931 | 20x20 |


### Model Description

Simple two populations model. Ancestral population of constant size splits into two subpopulations of constant size with asymetrical migrations.

### Plots

Schematic model plot:

<img src="model_plot.png" height="500" />

Simulated allele frequency spectrum:

<img src="fs_plot.png" height="200" />


### Optimal parameter values

| Parameter | Value | Description |
| --- | --- | --- |
| `nu1` | 1.000 | Size of subpopulation 1 after split. |
| `nu2` | 0.100 | Size of subpopulation 2 after split. |
| `m12` | 5.000 | Migration rate from subpopulation 2 to subpopulation 1. |
| `m21` | 2.500 | Migration rate from subpopulation 1 to subpopulation 2. |
| `T` | 0.050 | Time of split. |

