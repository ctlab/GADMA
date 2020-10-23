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
    import GPy
except ImportError:
    GPy = None
try:
    import GPyOpt
except ImportError:
    GPyOpt = None
import warnings

PIL_available = PIL is not None
matplotlib_available = matplotlib is not None
moments_available = moments is not None
dadi_available = dadi is not None
GPy_available = GPy is not None
GPyOpt_available = GPyOpt is not None

from .data import *  # NOQA
from .engines import *  # NOQA
from .models import *  # NOQA
from .code_generator import id2printfunc  # NOQA
from .utils import *  # NOQA
from .optimizers import *  # NOQA
from .cli import *  # NOQA
from .core import *  # NOQA
from .Inference import *  # NOQA
from .run_ls_on_boot_data import *  # NOQA
from .get_confidence_intervals import *  # NOQA

warnings.simplefilter('always', UserWarning)
warnings.filterwarnings(action='ignore', category=UserWarning,
                        module='matplotlib')
warnings.filterwarnings(action='ignore', category=UserWarning,
                        module='moments')
warnings.filterwarnings(action='ignore', category=UserWarning,
                        module='dadi')
warnings.formatwarning = utils.warning_format
