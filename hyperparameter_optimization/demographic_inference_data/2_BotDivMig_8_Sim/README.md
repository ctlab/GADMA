## 2_BotDivMig_8_Sim


| Number of populations | Number of parameters | Max log likelihood | Size of spectrum |
| --- | --- | --- | --- |
| 2 | 8 | -1035.905 | 20x20 |


### Model Description

Demographic history of two populations with bottleneck of ancestral population followed by split and growth of both new formed populations exponentially and linearly correspondingly.

### Plots

Schematic model plot:

<img src="model_plot.png" height="500" />

Simulated allele frequency spectrum:

<img src="fs_plot.png" height="200" />


### Optimal parameter values

| Parameter | Value | Description |
| --- | --- | --- |
| `nu` | 0.100 | Size of ancestral population after sudden decrease. |
| `f` | 0.300 | Fraction in which ancestral population splits. |
| `nu1` | 2.000 | Size of population 1 after exponential growth. |
| `nu2` | 3.000 | Size of population 2 after linear growth. |
| `m12` | 1.000 | Migration rate from subpopulation 2 to subpopulation 1. |
| `m21` | 0.100 | Migration rate from subpopulation 1 to subpopulation 2. |
| `T1` | 0.500 | Time between sudden growth of ancestral population and its split. |
| `T2` | 1.000 | Time of ancestral population split. |

