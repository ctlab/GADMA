try:
    import matplotlib
    matplotlib.use("Agg")
except ImportError:
    pass

__version__ = "unknown"
try:
    from . import version
    __version__ = version.version
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
    import demes
except ImportError:
    demes = None
try:
    import demesdraw
except ImportError:
    demesdraw = None

try:
    import GPy
except ImportError:
    GPy = None
try:
    import GPyOpt
except ImportError:
    GPyOpt = None
try:
    import smac  # NOQA
    import ConfigSpace  # NOQA
    smac_available = True
except ImportError:
    smac = None
    ConfigSpace = None
    smac_available = False

try:
    import bayesmark
except ImportError:
    bayesmark = None

import warnings

PIL_available = PIL is not None
matplotlib_available = matplotlib is not None
moments_available = moments is not None
dadi_available = dadi is not None
demes_available = demes is not None
demesdraw_available = demesdraw is not None

GPy_available = GPy is not None
GPyOpt_available = GPyOpt is not None
bayesmark_available = bayesmark is not None

from .data import DataHolder, SFSDataHolder, VCFDataHolder  # NOQA
from .engines import get_engine, all_engines  # NOQA
from .engines import all_available_engines, all_simulation_engines  # NOQA
from .engines import all_drawing_engines  # NOQA

from .models import DemographicModel, EpochDemographicModel  # NOQA
from .models import CustomDemographicModel, StructureDemographicModel  # NOQA
from .models import Addition, Subtraction, Multiplication, Division  # NOQA

from .code_generator import id2printfunc  # NOQA

from .utils import warning_format, get_aic_score  # NOQA
from .utils import Variable, ContinuousVariable, DiscreteVariable  # NOQA
from .utils import TimeVariable, PopulationSizeVariable, MigrationVariable  # NOQA
from .utils import SelectionVariable, DynamicVariable, FractionVariable  # NOQA
from .utils import DemographicVariable, VariablePool  # NOQA
from .utils import abspath, check_file_existence, check_dir_existence  # NOQA
from .utils import ensure_file_existence, ensure_dir_existence  # NOQA

from .optimizers import get_local_optimizer, all_local_optimizers  # NOQA
from .optimizers import get_global_optimizer, all_global_optimizers  # NOQA
from .cli import SettingsStorage, version, usage, get_settings  # NOQA
from .cli import check_required_settings, settings  # NOQA
from .core import CoreRun, shared_dict  # NOQA
from .Inference import load_data_from_dir, get_claic_score, optimize_ga  # NOQA
from .run_ls_on_boot_data import load_parameters_from_python_file  # NOQA
from . import get_confidence_intervals  # NOQA

warnings.simplefilter('always', UserWarning)
warnings.filterwarnings(action='ignore', category=UserWarning,
                        module='matplotlib')
warnings.filterwarnings(action='ignore', category=UserWarning,
                        module='moments')
warnings.filterwarnings(action='ignore', category=UserWarning,
                        module='dadi')
warnings.formatwarning = warning_format
