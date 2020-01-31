# GADMA ![](http://jb.gg/badges/research-flat-square.svg)

Welcome to GADMA!

GADMA implements methods for automatic inferring joint demographic history of multiple populations from the genetic data.

GADMA is based on two open source packages: the ∂a∂i developed by Ryan Gutenkunst [<https://bitbucket.org/gutenkunstlab/dadi/>] and the *moments* developed by Simon Gravel [<https://bitbucket.org/simongravel/moments/>].

In contrast, GADMA is a command-line tool. It presents a series of launches of the genetic algorithm and infer demographic history from Allele Frequency Spectrum of multiple populations (up to three).

GADMA is implemented by Ekaterina Noskova (ekaterina.e.noskova@gmail.com)

Contributors:

* Ekaterina Noskova

* Vladimir Ulyantsev

* Pavel Dobrynin

## Getting help
GADMA manual can be found in `doc/manual/` directory.
Please don't be afraid to contact me for different problems and offers via email ekaterina.e.noskova@gmail.com. I will be glad to answer all questions. 

## Installation

One can try to install GADMA and all its possible dependencies via shell script from GADMA home directory the following way (for Python3):
```console
$ ./install
```

or (for Python2):

```console
$ ./install --py2
```

If some problems occur please try to install GADMA by steps that are described in the corresponding section of GADMA manual.

## Hands on

### Test case

GADMA has a test case for a simple demographic model for one population: just a constant size of $10000$ individuals in population. To run a test case:

```console
$ gadma --test
```


### Example 2

Suppose we have SNP data for two populations. Data is in ∂a∂i's SNP file format in the `snp_data.txt`. Suppose we want to get all output in some `gadma_output` directory:

```console
$ gadma -i snp_data.txt -o gadma_output
```

### Example 3

We didn't specify AFS size or labels for populations, they are taken automatically from the input file. We can see a parameter file of our run in the `gadma_output/param_file`.

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

Order of populations can be changed as:

```console
# params_file
Projections : 20,20
Population labels : pop_name_2, pop_name_1
```

If we want to rename populations, we should change names on `snp_data.txt` file.

Now assume we want to get the simplest demographic model as fast as we can. We will tell GADMA that we need no other dynamics of population sizes except sudden (constant) population size change and that we want to use *moments* library.


```console
# param_file
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

We want to allow exponential growth (it is default behaviour) and have some extra change in size of the ancient population. To do so we should specify `Initial structure`. It is list of three numbers: first - number of time intervals before first split (we want here 2); second - number of time periods betseen forst and second split events (at least 1); third - number of time periods after second split.

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

where `gadma_output` is output directory of previous run. We can find resumed run in `gadma_output_resumed`

### Example 6

Our launch was finished, we used ∂a∂i with default grid size which GADMA determines automatically if it is not specify by user. We found out that it would be better to find some models using greater number of grid points in ∂a∂i scheme, but we want to take final models from previous run:

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


### Example 7

We can add a custom model using a parameter `Custom filename` in the parameter file:

```console
# param_file
Custom filename : YRI_CEU_demographic_model.py
```

Our custom file need to contain a function with a fixed name `model_func` (see Appendix A of GADMA manual). For example:

```console
# YRI_CEU_demographic_model.py
def model_func((nu1F, nu2B, nu2F, m, Tp, T), (n1,n2), pts)
    xx = yy = dadi.Numerics.default_grid(pts)

    phi = dadi.PhiManip.phi_1D(xx)
    phi = dadi.Integration.one_pop(phi, xx, Tp, nu=nu1F)

    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    nu2_func = lambda t: nu2B*(nu2F/nu2B)**(t/T)
    phi = dadi.Integration.two_pops(phi, xx, T, nu1=nu1F, 
                                    nu2=nu2_func, m12=m, m21=m)

    sfs = dadi.Spectrum.from_phi(phi, (n1,n2), (xx,yy))
    return sfs
```

In addition, we can easily specify values for lower and upper bounds through a parameter file. Let's set lower and upper bounds for the model we defined above:

```console
# param_file
Lower bounds : [1e-2, 1e-2, 1e-2, 0, 0, 0]
Upper bounds : [100, 100, 100, 10, 3, 3]
```

### Example 8}

Also, we can get the values of lower/upper bounds, both, or none of them in the parameter file automatically. For this, each identifier in the parameter file must be declared through a parameter `Parameter identifiers`. Below is an identifier list:

```console
# param_file
#   An identifier list:
#   T - time
#   N - size of population
#   m - migration
#   s - split event, proportion in which population size
#   is divided to form two new populations.
```

For example, we set a lower bound for the model we defined above (see Example 7) and we want to get an upper bound automatically.

```console
# param_file
Lower bound : [1e-2, 1e-2, 1e-2, 0, 0, 0]
Upper bound : None

Parameter identifiers : ['n', 'n', 'n', 'm', 't', 't']
```

### Example YRI, CEU
GADMA has example of full parameters file `example_params`, that can be found [here](https://github.com/ctlab/GADMA/blob/master/example_params). To run GADMA with this parameters one should just run from the GADMA's home directory:

```console
$ gadma -p example_params
```


## GADMA contributors

* Ekaterina Noskova

* Vladimir Ulyantzev

* Pavel Dobrynin
