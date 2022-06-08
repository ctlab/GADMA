# GADMA ![](http://jb.gg/badges/research-flat-square.svg)

[![Docs](https://readthedocs.org/projects/gadma/badge/?version=latest)](https://gadma.readthedocs.io/en/latest/?badge=latest) [![Build status](https://github.com/ctlab/GADMA/workflows/build/badge.svg)](https://github.com/ctlab/GADMA/actions) [![codecov](https://codecov.io/gh/ctlab/GADMA/branch/master/graph/badge.svg?token=F303UDEWDJ)](https://codecov.io/gh/ctlab/GADMA) [![PyPI - Downloads](https://img.shields.io/pypi/dm/gadma)](https://pypistats.org/packages/gadma)

Welcome to GADMA v2!

GADMA implements methods for automatic inference of the joint demographic history of multiple populations from the genetic data.

**GADMA is a command-line tool**. Basic pipeline presents a series of launches of the genetic algorithm folowed by local search optimization and infers demographic history from the Allele Frequency Spectrum of multiple populations (up to three).<br/>
GADMA features variuos optimization methods ([global](https://gadma.readthedocs.io/en/latest/api/gadma.optimizers.html#global-optimizers-list) and [local](https://gadma.readthedocs.io/en/latest/api/gadma.optimizers.html#local-optimizers-list) search algorithms) which may be used for [any general optimization problem](https://gadma.readthedocs.io/en/latest/api_examples/optimization_example.html).

GADMA provides choice of several engines of demographic inference (this list will be extended in the future):

* [∂a∂i](https://bitbucket.org/gutenkunstlab/dadi/)
* [*moments*](https://bitbucket.org/simongravel/moments/)
* [*momi2*](https://github.com/popgenmethods/momi2/)
* [*momentsLD*](https://bitbucket.org/simongravel/moments/) - extenstion of *moments*

GADMA is developed in Computer Technologies laboratory at ITMO University under the supervision of [Vladimir Ulyantsev](https://ulyantsev.com/) and Pavel Dobrynin. The principal maintainer is [Ekaterina Noskova](http://enoskova.me/) (ekaterina.e.noskova@gmail.com)

**GADMA is now of version 2!** See [Changelog](https://gadma.readthedocs.io/en/latest/changelogs.html).

### Documentation

Please see [documentation](https://gadma.readthedocs.io) for more information including installation instructions, usage, examples and API.

## Getting help

[F.A.Q.](https://gadma.readthedocs.io/en/latest/faq.html)

Please don't be afraid to contact me for different problems and offers via email ekaterina.e.noskova@gmail.com. I will be glad to answer all questions.

Also you are always welcome to [create an issue](https://github.com/ctlab/GADMA/issues) on the GitHub page of GADMA with your question.

## Citations

Please see full list of citations in [documentation](https://gadma.readthedocs.io/en/latest/citations.html).

If you use GADMA in your research please cite:

Ekaterina Noskova, Vladimir Ulyantsev, Klaus-Peter Koepfli, Stephen J O’Brien, Pavel Dobrynin, GADMA: Genetic algorithm for inferring demographic history of multiple populations from allele frequency spectrum data, *GigaScience*, Volume 9, Issue 3, March 2020, giaa005, <https://doi.org/10.1093/gigascience/giaa005>
