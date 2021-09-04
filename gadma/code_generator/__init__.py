from .dadi_generator import print_dadi_code
from .moments_generator import print_moments_code
from .moments_ld_generator import print_moments_ld_code
id2printfunc = {
    'dadi': print_dadi_code,
    'moments': print_moments_code,
    'momentsLD': print_moments_ld_code
}
