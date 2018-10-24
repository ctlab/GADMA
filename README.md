# GADMA

Welcome to GADMA!\\

GADMA implements methods for automatic inferring joint demographic history of multiple populations from the genetic data.

GADMA is based on two open source packages: the ∂a∂i developed by Ryan Gutenkunst [\url{https://bitbucket.org/gutenkunstlab/dadi/}] and the *moments* developed by Simon Gravel [\url{https://bitbucket.org/simongravel/moments/}].

In contrast, GADMA is a command-line tool. It presents a series of launches of the genetic algorithm and infer demographic history from Allele Frequency Spectrum of multiple populations (up to three).

GADMA is developed by Ekaterina Noskova (ekaterina.e.noskova@gmail.com)

## Getting help
GADMA manual can be found in `doc/manual/` directory.
Please don't be afraid to contact me for different problems and offers via email ekaterina.e.noskova@gmail.com. I will be glad to answer all questions. 

## Installation

Please see the corresponding section of GADMA manual.

## Hands on

###Test case

GADMA has test case for simple demographic model for 1 population: just constant size of 10000 individuals in population. To run test case just print:

```console
$ gadma --test
```


### Example 2

Suppose we have SNP data for two populations. Data is in ∂a∂i's SNP format in file `snp_data.txt`. Suppose we want to get all output in some `gadma_output` directory:

```console
$ gadma -i snp_data.txt -o gadma_output
```

### Example 3
We didn't specify AFS size or labels for populations, they are taken automaticaly from the input file. We can see parameters file of our run in `gadma_output/params` file. We see that there are the following:

```console
# gadma_output/params
...
Population labels : pop_name_1, pop_name_2
Projections: 18, 20
...
```

But we know that spectrum should be 20×20! To specify size of AFS we need to create parameters file and set `Projections`:

```console
# params_file
Projections : 20,20
```

Also we can change order of populations. We should add:

```console
# params_file
Projections : 20,20
Population labels : pop_name_2, pop_name_1
```

If we want to rename populations, we should change names in `snp_data.txt` file.

Now assume we want to get the simplest demographic model that we can as faster as we can. We will tell GADMA that we don't need no other dynamics of population sizes except sudden (constant) population size change and that we want to use *moments* library.

We add corresponding string to parameters file and now it looks like:

```console
# params_file
Projections : 20,20
Population labels : pop_name_2, pop_name_1
Only sudden : True
Use moments or dadi : moments
```

To run GADMA we need to specify `-p/--params` command-line option in cmd:
```console
$ gadma -i snp_data.txt -o gadma_output -p params_file
```

### Example 4

Consider some AFS file `fs_data.fs`. There is spectrum for three populations: YRI, CEU, CHB. However axes are mixed up: CHB, YRI, CEU. To run GADMA we should order them from most ancient to last formed:

```console
# params_file
Population labels : YRI, CEU, CHB
```

We want to allow exponential growth (it is default behaviour) and have some extra change in size of the ancient population. To do so we should specify `Initial structure`. It is list of three numbers: first --- number of time intervals before first split (we want here 2); second --- number of time periods betseen forst and second split events (at least 1); third --- number of time periods after second split.

```console
# params_file
Population labels : YRI, CEU, CHB
Initial structure : 2,1,1
```

Also we can put information about input file and output directory to our parameters file:

```console
# params_file
Input file : fs_data.fs
Output directory : gadma_output
Population labels : YRI, CEU, CHB
Initial structure : 2,1,1
```

Now we can run GADMA the following way:

```console
$ gadma -p params
```

### Example 5

We have our GADMA launch interrupted for some reason. We want to resume it:

```console
$ gadma --resume gadma_output
```

`gadma_output` is output directory of previous run. We can find resumed run in `gadma_output_resumed`

### Example 6

Our launch was finished, we used ∂a∂i with default grid size which GADMA determines automaticly if it is not specify by user. We found out that it would be better to find some models using greater number of grid points in ∂a∂i scheme, but we want to take final models from previous run:

```console
# params_file
Pts : 40, 50, 60 #Greater value of grid size than it was
```

And run GADMA:

```console
$ gadma --resume gadma_output --only_models -p params 
```

`--only_models` tell GADMA to take from `gadma_output` final models only.

There is another way to do the same:

```console
# params_file
Resume from : gadma_output
Only models : True
Pts : 40, 50, 60 #Greater value of grid size than it was
```

And run GADMA the following way:

```console
$ gadma -p params
```

### Example YRI, CEU
GADMA has example of full parameters file `example_params`, that can be found [here](https://github.com/ctlab/GADMA/blob/master/example_params). To run GADMA with this parameters one should just run from the GADMA's home directory:

```console
$ gadma -p example_params
```

