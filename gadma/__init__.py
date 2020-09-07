try:
    import matplotlib
    matplotlib.use("Agg")
except ImportError:
    pass

try:
    import PIL
    from PIL import Image
except ImportError:
    PIL = None
    Image = None
try:
    import matplotlib
except ImportError:
    matplotlib = None
try:
    import moments
except ImportError:
    moments = None
try:
    import dadi
except ImportError:
    dadi = None
try:
    import GPyOpt
except ImportError:
    GPyOpt = None

PIL_available = PIL is not None
matplotlib_available = matplotlib is not None
moments_available = moments is not None
dadi_available = dadi is not None
GPyOpt_available = GPyOpt is not None

from .data import *
from .engines import *
from .models import *
from .code_generator import id2printfunc
from .utils import *
from .optimizers import *
from .cli import *
from .core import *
from .Inference import *

import warnings
if moments_available:
    warnings.filterwarnings("default", 
                            ".*", 
                            UserWarning,
                            'moments.ModelPlot',
                            )
warnings.formatwarning = utils.warning_format
