# First we make matplotlib backend as Agg
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from version import __version__
import Inference
