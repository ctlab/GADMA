# Data for demographic inference

This repo contains AFS data (simulated and real) for different kinds of
demographic inference, for example, it could be used for some algorithm
benchmark and so on.

Right now everything is for the python package
[moments](https://bitbucket.org/simongravel/moments/src/master).
# Available AFS data

- [Simulated data](#simulated-data)
	* [1_Bot_4_Sim](#1_Bot_4_Sim)
	* [2_ExpDivNoMig_5_Sim](#2_ExpDivNoMig_5_Sim)
	* [2_DivMig_5_Sim](#2_DivMig_5_Sim)
	* [2_BotDivMig_8_Sim](#2_BotDivMig_8_Sim)
	* [3_DivMig_8_Sim](#3_DivMig_8_Sim)
	* [4_DivMig_11_Sim](#4_DivMig_11_Sim)
- [Real data](#real-data)
	* [2_ButAllA_3_McC](#2_ButAllA_3_McC)
	* [2_ButSynB2_5_McC](#2_ButSynB2_5_McC)
	* [2_YRI_CEU_6_Gut](#2_YRI_CEU_6_Gut)
	* [3_YRI_CEU_CHB_13_Gut](#3_YRI_CEU_CHB_13_Gut)


## Structure of data folders

### Name of folder

All data is located in its folder each of which has special name:

`{N}_{Descr}_{M}_{Origin}`

where `N` is the **number of populations**, `Descr` stands for **simple
description** of the demographic model, `M` is the **number of parameters**
in model and `Origin` is three letters of data origin. For example:

`2_DivMig_5_Sim` means AFS data for 2 populations that was simulated (`Sim`)
under the demographic history with divergence and migration that has in total
5 parameters.

For real data `Origin` contains the first three letters of the first author of the
corresponding paper. For example:

`2_YRI_CEU_6_Gut` means AFS data and demographic model for YRI and CEU
populations from the **Gut**enkunst et al., 2009 paper.

### Folder structure

Each directory contains several files:

* demographic_model.py - file with demographic model code. Code is written
for [moments](https://bitbucket.org/simongravel/moments/src/master).

* `main_script.py` - script with all information about AFS data. If data is
simulated then during script run it will be saved to `fs_data.fs`.

* `fs_data.fs` - AFS data in fs dadi format.

* `model_plot.png` - schematic plot of the demographic model.

* `fs_plot.png` or `fs_plot_projections.png` - plots of AFS.

File `main.py` is a valid file with python code and contains following
information about data:

| Variable | Description |
| --- | --- |
| `n_pop` | Number of populations |
| `par_labels` | Labels of model parameters |
| `max_ll` | Maximum log composite likelihood |
| `popt` | Optimal values parameters |
| `Nanc` | Size of ancestral population |
| `lower_bound` | Lower bound for parameters |
| `upper_bound` | Upper bound for parameters |
| `ns` | Sample sizes of AFS |

### Units

Note that all parameters values are in genetic units (relative to size
`Nanc` of ancestral population). Size of populations are relative to `Nanc`,
time is in `2 * Nanc` generations abd migrations are in `1 / (2 * Nanc)`
units. For more information see [moments manual](https://bitbucket.org/
simongravel/moments/src/master/doc/manual/manual.pdf) section `5.2 Units`.

# Simulated data

## 1_Bot_4_Sim


| Number of populations | Number of parameters | Max log likelihood | Size of spectrum |
| --- | --- | --- | --- |
| 1 | 4 | -88.560 | 20 |


### Model Description

Classical one population bottleneck model.

### Plots

Schematic model plot:

<img src="1_Bot_4_Sim/model_plot.png" height="500" />

Simulated allele frequency spectrum:

<img src="1_Bot_4_Sim/fs_plot.png" height="200" />


### Optimal parameter values

| Parameter | Value | Description |
| --- | --- | --- |
| `nuB` | 0.010 | Size of population during bottleneck. |
| `nuF` | 1.000 | Size of population now. |
| `tB` | 0.005 | Time of bottleneck duration. |
| `tF` | 0.050 | Time after bottleneck finished. |

## 2_ExpDivNoMig_5_Sim


| Number of populations | Number of parameters | Max log likelihood | Size of spectrum |
| --- | --- | --- | --- |
| 2 | 5 | -1503.119 | 20x20 |


### Model Description

Demographic model of isolation for two populations with exponential growth of an ancestral population followed by split.

### Plots

Schematic model plot:

<img src="2_ExpDivNoMig_5_Sim/model_plot.png" height="500" />

Simulated allele frequency spectrum:

<img src="2_ExpDivNoMig_5_Sim/fs_plot.png" height="200" />


### Optimal parameter values

| Parameter | Value | Description |
| --- | --- | --- |
| `nu` | 5.000 | Size of ancestral population after exponential growth. |
| `nu1` | 2.000 | Size of population 1 after split. |
| `nu2` | 4.000 | Size of population 2 after split. |
| `T1` | 4.000 | Time between exponential growth of ancestral population and its split. |
| `T2` | 1.000 | Time of ancestral population split. |

## 2_DivMig_5_Sim


| Number of populations | Number of parameters | Max log likelihood | Size of spectrum |
| --- | --- | --- | --- |
| 2 | 5 | -1310.931 | 20x20 |


### Model Description

Simple two populations model. Ancestral population of constant size splits into two subpopulations of constant size with asymetrical migrations.

### Plots

Schematic model plot:

<img src="2_DivMig_5_Sim/model_plot.png" height="500" />

Simulated allele frequency spectrum:

<img src="2_DivMig_5_Sim/fs_plot.png" height="200" />


### Optimal parameter values

| Parameter | Value | Description |
| --- | --- | --- |
| `nu1` | 1.000 | Size of subpopulation 1 after split. |
| `nu2` | 0.100 | Size of subpopulation 2 after split. |
| `m12` | 5.000 | Migration rate from subpopulation 2 to subpopulation 1. |
| `m21` | 2.500 | Migration rate from subpopulation 1 to subpopulation 2. |
| `T` | 0.050 | Time of split. |

## 2_BotDivMig_8_Sim


| Number of populations | Number of parameters | Max log likelihood | Size of spectrum |
| --- | --- | --- | --- |
| 2 | 8 | -1035.905 | 20x20 |


### Model Description

Demographic history of two populations with bottleneck of ancestral population followed by split and growth of both new formed populations exponentially and linearly correspondingly.

### Plots

Schematic model plot:

<img src="2_BotDivMig_8_Sim/model_plot.png" height="500" />

Simulated allele frequency spectrum:

<img src="2_BotDivMig_8_Sim/fs_plot.png" height="200" />


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

## 3_DivMig_8_Sim


| Number of populations | Number of parameters | Max log likelihood | Size of spectrum |
| --- | --- | --- | --- |
| 3 | 8 | -11178.277 | 20x20x20 |


### Model Description

Three populations demographic history with small number of parameters. In the model ancestral population is split into population 1 and population 2, each of which had constant size till now days. Population 3 is formed by split from population 2 without change of its size and had constant size till now too. Migration rates are symmetrical.

### Plots

Schematic model plot:

<img src="3_DivMig_8_Sim/model_plot.png" height="500" />

Simulated allele frequency spectrum:

<img src="3_DivMig_8_Sim/fs_plot.png" height="200" />

Simulated allele frequency spectrum (projections):

<img src="3_DivMig_8_Sim/fs_plot_projections.png" height="200" />


### Optimal parameter values

| Parameter | Value | Description |
| --- | --- | --- |
| `nu1` | 1.500 | Size of population 1. |
| `nu2` | 0.500 | Size of population 2. |
| `nu3` | 1.000 | Size of population 3 after split from population 2. |
| `m12` | 0.500 | Migration rate between population 1 and population 2. |
| `m13` | 1.000 | Migration rate between population 1 and population 3. |
| `m23` | 3.000 | Migration rate between population 2 and population 3. |
| `T1` | 0.100 | Time between ancestral population split and divergence of population 3 from population 2. |
| `T2` | 0.050 | Time of population 3 divergence from population 2. |

## 4_DivMig_11_Sim


| Number of populations | Number of parameters | Max log likelihood | Size of spectrum |
| --- | --- | --- | --- |
| 4 | 11 | -16932.827 | 10x10x10x10 |


### Model Description

Four population demographic history with 11 parameters. Ancestral population of constant size was split (T1 + T2 + T3) time ago into two new populations. First (1) population has constant size till nowdays. After divergence there was migration to population 1 from another population. The second formed population turned out to be a common population to another three populations: (T2 + T3) time ago it splits and formed so-called population 2 and T3 time ago the second formed population divided into population 3 and population 4. There were almost no migrations between populations except symmetric one between populations 3 and 4.

### Plots

Schematic model plot:

<img src="4_DivMig_11_Sim/model_plot.png" height="500" />

Simulated allele frequency spectrum (projections):

<img src="4_DivMig_11_Sim/fs_plot_projections.png" />


### Optimal parameter values

| Parameter | Value | Description |
| --- | --- | --- |
| `nu1` | 1.500 | Size of population 1 after split of ancestral population. |
| `nu234` | 0.800 | Size of common ancestor population of populations 2, 3 and 4 after split of ancestral population. |
| `nu2` | 1.000 | Size of population 2 after split of common ancestor population of populations 2, 3 and 4. |
| `nu34` | 0.500 | Size of common ancestor population of populations 3 and 4 after division of population 2 from their common ancestor population. |
| `nu3` | 0.200 | Size of population 3. |
| `nu4` | 0.300 | Size of population 4. |
| `m12_anc` | 2.000 | Migration rate to population 1 from common ancestor population of populations 2, 3 and 4. |
| `m34_sym` | 3.000 | Symmetric migration rate between populations 3 and 4. |
| `T1` | 0.100 | Time between ancestral population split, population 1 formation and next split. |
| `T2` | 0.150 | Time between ancestral population of populations 2, 3 and 4 split, population 2 formation and next split. |
| `T3` | 0.050 | Time of ancestral population of populations 3 and 4 split and formations of population 3 and population 4. |

# Real data

## 2_ButAllA_3_McC


| Number of populations | Number of parameters | Max log likelihood | Size of spectrum |
| --- | --- | --- | --- |
| 2 | 3 | -283.599 | 12x12 |


### Model Description

Demographic model without migration for two populations of butterflies. Data and model are from McCoy et al. 2013. Model is very simple: ancestral population splits into two new populations of constant size.

### Plots

Schematic model plot:

<img src="2_ButAllA_3_McC/model_plot.png" height="500" />

Simulated allele frequency spectrum:

<img src="2_ButAllA_3_McC/fs_plot.png" height="200" />


### Optimal parameter values

| Parameter | Value | Description |
| --- | --- | --- |
| `nuW` | 1.320 | Size of first subpopulation. |
| `nuC` | 0.173 | Size of second subpopulation. |
| `T` | 0.117 | Time of ancestral population split. |

## 2_ButSynB2_5_McC


| Number of populations | Number of parameters | Max log likelihood | Size of spectrum |
| --- | --- | --- | --- |
| 2 | 5 | -210.769 | 12x12 |


### Model Description

Demographic model with asymmetric migrations for two populations of butterflies. Data and model are from McCoy et al. 2013. Ancestral population split into two new formed populations with following continuous migrations between them.

### Plots

Schematic model plot:

<img src="2_ButSynB2_5_McC/model_plot.png" height="500" />

Simulated allele frequency spectrum:

<img src="2_ButSynB2_5_McC/fs_plot.png" height="200" />


### Optimal parameter values

| Parameter | Value | Description |
| --- | --- | --- |
| `nuW` | 0.873 | Size of first new formed population. |
| `nuC` | 0.121 | Size of second new formed population. |
| `T` | 0.080 | Time of ancestral population split. |
| `m12` | 0.923 | Migration rate from second population to first one. |
| `m21` | 0.000 | Migration rate from first population to second one. |

## 2_YRI_CEU_6_Gut


| Number of populations | Number of parameters | Max log likelihood | Size of spectrum |
| --- | --- | --- | --- |
| 2 | 6 | -1066.823 | 20x20 |


### Model Description

Demographic model for two modern human populations: YRI and CEU. Data and model are from Gutenkunst et al., 2009. Model with sudden growth of ancestral population size, followed by split, bottleneck in second population (CEU) with exponential recovery and symmetric migration.

### Plots

Schematic model plot:

<img src="2_YRI_CEU_6_Gut/model_plot.png" height="500" />

Simulated allele frequency spectrum:

<img src="2_YRI_CEU_6_Gut/fs_plot.png" height="200" />


### Optimal parameter values

| Parameter | Value | Description |
| --- | --- | --- |
| `nu1F` | 1.881 | The ancestral population size after growth. |
| `nu2B` | 0.071 | The bottleneck size for second population (CEU). |
| `nu2F` | 1.845 | The final size for second population (CEU). |
| `m` | 0.911 | The scaled symmetric migration rate. |
| `Tp` | 0.355 | The scaled time between ancestral population growth and the split. |
| `T` | 0.111 | The time between the split and present. |

## 3_YRI_CEU_CHB_13_Gut


| Number of populations | Number of parameters | Max log likelihood | Size of spectrum |
| --- | --- | --- | --- |
| 3 | 13 | -6316.578 | 20x20x20 |


### Model Description

Demographic model for three modern human populations: YRI, CEU and CHB. Data and model are from Gutenkunst et al., 2009. Model with sudden growth of ancestral population size, followed by split into population YRI and common population of CEU and CHB, which experience bottleneck and split with exponential recovery of both populations. Migrations between populations are symmetrical.

### Plots

Schematic model plot:

<img src="3_YRI_CEU_CHB_13_Gut/model_plot.png" height="500" />

Simulated allele frequency spectrum:

<img src="3_YRI_CEU_CHB_13_Gut/fs_plot.png" height="200" />

Simulated allele frequency spectrum (projections):

<img src="3_YRI_CEU_CHB_13_Gut/fs_plot_projections.png" height="200" />


### Optimal parameter values

| Parameter | Value | Description |
| --- | --- | --- |
| `nuAf` | 1.680 | The ancestral population size after sudden growth and size of YRI population. |
| `nuB` | 0.287 | The bottleneck size of CEU+CHB common population. |
| `nuEu0` | 0.129 | The bottleneck size for CEU population. |
| `nuEu` | 3.740 | The final size of CEU population after exponential growth. |
| `nuAs0` | 0.070 | The bottleneck size for CHB population. |
| `nuAs` | 7.290 | The final size of CHB population after exponential growth. |
| `mAfB` | 3.650 | The scaled symmetric migration rate between YRI and CEU+CHB populations. |
| `mAfEu` | 0.440 | The scaled symmetric migration rate between YRI and CEU populations. |
| `mAfAs` | 0.280 | The scaled symmetric migration rate between YRI and CHB populations. |
| `mEuAs` | 1.400 | The scaled symmetric migration rate between CEU and CHB populations. |
| `TAf` | 0.211 | The scaled time between ancestral population growth and first split. |
| `TB` | 0.338 | The time between the first split and second. Time of CEU+CHB population existence. |
| `TEuAs` | 0.058 | The time between second split and present. |

