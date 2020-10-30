# Minimal makefile for Sphinx documentation
#

# <noscode>
#cd source/examples/
#jupyter-nbconvert --to rst --output moments_YRI_CEU_optimize_ga ../../../examples/optimize_ga_for_YRI_CEU/YRI_CEU_with_optimize_ga_and_moments.ipynb --output-dir  .
#jupyter-nbconvert --to rst --output dadi_YRI_CEU_optimize_ga ../../../examples/optimize_ga_for_YRI_CEU/YRI_CEU_with_optimize_ga_and_dadi.ipynb --output-dir  .
#cd ../..
# </noscode>

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
