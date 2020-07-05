import tempfile
from ..version import __version__
from .settings_storage import HOME_DIR
from . import SettingsStorage
import argparse
import sys
import os

TEST_SETTINGS = os.path.join(HOME_DIR, "test_settings")

def version():
    '''
    Returns string with current version.
    '''
    return "GADMA version " + str(
        __version__
    ) + "\tby Ekaterina Noskova (ekaterina.e.noskova@gmail.com)" + "\n"


def usage():
    '''
    Returns usage of tool.
    '''
    return version() + "" \
        "Usage: \n\tgadma -p/--params <params_file> -e/--extra <extra_params_file>\n"\
        "\n\n"\
        "Instead/With -p/--params and -e/--extra option you can set:\n"\
        "\t-o/--output <output_dir>\toutput directory.\n"\
        "\t-i/--input <in.fs>/<in.txt>\tinput file with AFS or in dadi format.\n"\
        "\t--resume <resume_dir>\t\tresume another launch from <resume_dir>.\n"\
        "\t--only_models\t\t\tflag to take models only from another launch (--resume option).\n"\
        "\n\n"\
        "\t-h/--help\t\tshow this help message and exit.\n"\
        "\t-v/--version\t\tshow version and exit.\n"\
        "\t--test\t\t\trun test case.\n" #+ support.SUPPORT_STRING


class ArgParser(argparse.ArgumentParser):

    def format_help(self):
        return usage()

    def error(self, message):
        support.error(message)


def test_args():
    '''
    Put default args for test case.
    '''
    # Create storage from test settings
    settings_storage = SettingsStorage.from_file(TEST_SETTINGS)
    # Input test file
    curent_dir = os.path.dirname(os.path.abspath(__file__))
    settings_storage.input_file = os.path.join(curent_dir,
                                               "../../fs_examples/test.fs")
    # There is no output_directory, we put it to temporary directory
    settings_storage.output_directory = tempfile.mkdtemp("gadma_test_dir")
    # And put path to input file
    settings_storage.input_file = os.path.join(HOME_DIR,
                                               settings_storage.input_file)
    return settings_storage


def get_settings():
    '''
    Parse args from command line and store them in options_storage.
    '''
    usage = str(sys.argv[0]) + " -p <params_file>"

    # Create arguments parser
    parser = ArgParser(add_help=False)
    parser.add_argument('-p', '--params', metavar="<params_file>",
                        required=False, default=None)
    parser.add_argument('-e', '--extra', metavar="<extra_params_file>",
                        required=False, default=None)
    parser.add_argument('--resume', metavar="<dir>",
                        required=False, default=None)
    parser.add_argument('--only_models', action='store_true')
    parser.add_argument('-o', '--output', metavar="<dir>",
                        required=False, default=None)
    parser.add_argument('-i', '--input',
                        metavar="<filename.fs>/<filename.txt>",
                        required=False, default=None)
    parser.add_argument('-v', '--version', action='version',
                        version=version(), default=argparse.SUPPRESS)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('-h', '--help', action='help',
                        default=argparse.SUPPRESS)

    # If not enough arguments
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    # Parse arguments
    args = parser.parse_args()

    # Create Settings storage
    if args.test:
        print("--Running test case--")
        settings_storage = test_args()
        # We return as we do not need to check
        return settings_storage, args
    else:
        settings_storage = SettingsStorage.from_file(args.params, args.extra)
        
    if args.only_models and args.resume is None:
        support.error("Option --only_models  must be used with --resume option.")

    if args.output is not None:
        if (settings_storage.output_dir is not None and
                settings_storage.output_dir != args.output):
            Warning("Output directory in parameters file doesn't match to one"
                    " from -o/--output option. The last is taken.")
        settings_storage.output_dir = args.output
    if args.input is not None:
        if (settings_storage.input_file is not None and
                settings_storage.input_file != args.input):
            Warning("Input file in parameters file doesn't match to one"
                    " from -i/--input option. The last is taken.")
        settings_storage.input_file = args.input
    if args.resume is not None:
        Warning("Resume is not working")
#        settings_storage.resume_dir = args.resume
    if args.only_models:
        Warning("Resume is not working")
#        settings_storage.only_models = True

    # Checks that we have got all required
    # TODO: case of resume!!
    if settings_storage.input_file is None:
        raise AttributeError("Input file is requiered. It could be set by "
                             "-i/--input option or via parameters file.")
    if settings_storage.output_directory is None:
        raise AttributeError("Output directory is requiered. It could be set "
                             "by -o/--output option or via parameters file.")

    if settings_storage.custom_filename is not None:
        if (settings_storage.lower_bound is None or
                settings_storage.upper_bound is None):
            raise AttributeError("Please specify either `Lower bound` and "
                                 "`Upper bound` or `Parameter identifiers` "
                                 "for custom model")
    else:
        if (settings_storage.initial_structure is None and
                settings_storage.final_structure is None):
            raise AttributeError("Please specify either structure of "
                                 "demographic history or filename with custom"
                                 " model.")
#    if args.resume is not None and os.path.abspath(
#        os.path.expanduser(
#            options_storage.resume_dir)) != os.path.abspath(
#            os.path.expanduser(
#                args.resume)):
#        support.error(
#            "Resume directory in parameters file doesn't match to one from --resume option")

    return settings_storage, args
