# GADMA with custom model (momi)

This is example for GADMA run with specified demographic model and ``momi`` engine. More information is in `run_gadma_with_custom_model_momi.ipynb` notebook.

Files here are:

* `YRI_CEU.fs` - Data for two modern human populations (from Gutenkunst et al., 2009).
* `demographic_model.py` - file with demographic model (for ``momi``).
* `params_file` - File with settings for GADMA run.
* `gadma_result` - Already run GADMA results.

To rerun GADMA:

```console
    $ rm -rf gadma_result
    $ gadma -p params_file
```