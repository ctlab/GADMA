# YRI CEU example

* YRI - Yoruba in Ibadan, Nigeria
* CEU - Utah Residents (CEPH) with Northern and Western European Ancestry

It is example from [∂a∂i](https://bitbucket.org/gutenkunstlab/dadi/) and [*moments*](https://bitbucket.org/simongravel/moments/).

There best parameters are pertrubed and then local search (optimize_log, optimize_powell etc.) is launched.

Here we use global search - Genetic Algorithm (optimize_ga) from [GADMA](https://github.com/ctlab/GADMA) software.

You can find python code:
* [for ∂a∂i](https://github.com/ctlab/GADMA/blob/master/examples/YRI_CEU/dadi/YRI_CEU.py)
* [for *moments*](https://github.com/ctlab/GADMA/blob/master/examples/YRI_CEU/moments/YRI_CEU.py)

## Usage

To run Genetic Algorithm with ∂a∂i as package for simulation of AFS from demographic model:

```console
$ cd dadi
$ python YRI_CEU.py
```

Similarly for *moments*:

```console
$ cd moments
$ python YRI_CEU.py
```

## Ready outputs

You can also look at the finished launches instead running them:

* [for ∂a∂i](https://github.com/ctlab/GADMA/blob/master/examples/YRI_CEU/dadi/cmd_output)
* [for *moments*](https://github.com/ctlab/GADMA/blob/master/examples/YRI_CEU/moments/cmd_output)

## Summary tables

**OR** look theese tables with result parameters:

### ∂a∂i

| Optimization | log likelihood | nu1F | nu2B | nu2F | m | Tp | T |
| --- | --- | --- | --- | --- | --- | --- | --- |
| optimize_log | -1066.31 | 1.87958 | 0.0707652 | 1.83607 | 0.914157 | 0.356525 | 0.110422 |
| optimize_ga | -1066.28 | 1.87939 | 0.0736213 | 1.71809 | 0.937448 | 0.361852 | 0.112965 |

### *moments*

| Optimization | log likelihood | nu1F | nu2B | nu2F | m | Tp | T |
| --- | --- | --- | --- | --- | --- | --- | --- |
| optimize_powell | -1066.50 | 1.87468 | 0.0721142 | 1.82631 | 0.921762 | 0.359479 | 0.110738 |
| optimize_ga | -1066.47 | 1.87015 | 0.0741365 | 1.72865 | 0.935654 | 0.353981 | 0.112022 |



