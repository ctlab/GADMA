from .dadi_generator import print_dadi_code
from .moments_generator import print_moments_code
from .momentsLD_generator import print_momentsLD_code
id2printfunc = {
    'dadi': print_dadi_code,
    'moments': print_moments_code,
    'momentsLD': print_momentsLD_code
}
