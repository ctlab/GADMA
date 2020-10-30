# GADMA with custom model

This is example for run of GADMA with specified demographic model. One could generate detailed demographic history with ``dadi`` or ``moments`` and ask GADMA to find its parameters. More information is in `run_gadma_with_custom_model.ipynb` notebook.

Files here are:

* `2pop_e_gillettii_all_snp.txt` - Data for two populations of Gillettii butterfly.
* demographic_model.py - file with demographic model (for ``moments``).
* `params_file` - File with settings for GADMA run.
* `gadma_result` - Already run GADMA results.
* `model_from_GADMA.png` - Picture of best model from this run.

To rerun GADMA:

```console
    $ rm -rf gadma_result
    $ gadma -p params_file
```