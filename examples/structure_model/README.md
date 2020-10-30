# GADMA with structure model

This is example for base run of GADMA with structure demographic model. More information is in `run_gadma_with_structure_model.ipynb` notebook.

Files here are:

* `dadi_2pops_CVLN_CVLS_snps.txt` - Data for two populations of Gaboon forest frog.
* `params_file` - File with settings for GADMA run.
* `gadma_result` - Already run GADMA results.
* `model_from_GADMA.png` - Picture of best model from this run.

To rerun GADMA:

```console
    $ rm -rf gadma_result
    $ gadma -p params_file
```