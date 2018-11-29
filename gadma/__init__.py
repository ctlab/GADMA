#!/usr/bin/env python

############################################################################
# Copyright (c) 2018 Noskova Ekaterina
# All Rights Reserved
# See the LICENSE file for details
############################################################################

import sys
if 'matplotlib' in sys.modules:
    # First we make matplotlib backend as Agg
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')

from version import __version__
import Inference
