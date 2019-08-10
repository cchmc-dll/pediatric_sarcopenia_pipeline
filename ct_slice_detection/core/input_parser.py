import argparse, configparser
from distutils.util import strtobool

"""Example

from InputParser import InputParser


def main(args):
    print(args)

if __name__ == '__main__':
    parser = InputParser()

    parser.add_parameter('sigma', type=float, default=2.5, help='sigma value')
    parser.add_parameter('beta', int, 1)
    parser.add_parameter('name', str, 'test')

    args = parser.parse()

    main(args)

"""


class Parameter():
    def __init__(self, name, type, default, nargs=1, section='main', help='', choices=None, option=None):
        self.name = name
        self.type = type
        self.default = default
        self.section = section
        self.help = help
        self.option = option
        self.nargs = nargs
        self.choices = choices


class InputParser():
    def __init__(self, description=''):
        self.params = []
        self.description = description

    def parse(self):
        conf_parser = argparse.ArgumentParser(
            # Turn off help, so we print all options in response to -h
            add_help=False
        )
        conf_parser.add_argument("-c", "--config",
                                 help="Specify config file", metavar="FILE")
        args, remaining_argv = conf_parser.parse_known_args()

        param_index = {}
        for param in self.params:
            param_index[param.name] = param

        defaults = self.get_default_parameters()

        # read from config file
        if args.config:
            config = configparser.ConfigParser()
            config.read([args.config])

            # disregard section info
            for section_name in config:
                for name, value in config.items(section_name):
                    if name in param_index:
                        if param_index[name].type == bool:
                            param_index[name].type = strtobool
                        # handle lists/nargs > 1
                        if param_index[name].type != str:
                            value = value.split(',')
                            value = list(map(param_index[name].type, value))
                            if len(value) == 1:
                                value = value[0]
                        # check if value is from choices
                        if param_index[name].choices is not None and value not in param_index[name].choices:
                            print('error: value ({}) for {} not from {}'.format(value, name, param_index[name].choices))
                            exit(0)

                        try:
                            defaults[name] = value
                        except:
                            print('error parsing argument "{}" with given value {} as type {}.'.format(
                                name, value, param_index[name].type))
                        print('  {} = {}'.format(name,value))
                    else:
                        print('ignoring non-default argument: {}.'.format(name))

        parser = argparse.ArgumentParser(
            parents=[conf_parser],
            description=self.description,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

        for name, param in param_index.items():
            if param.nargs == 1: # to avoid getting a list
                param.nargs = None
            parser.add_argument('--' + param.name, default=param.default, type=param.type, nargs=param.nargs,
                                help=param.help, choices=param.choices)

        parser.set_defaults(**defaults)

        parser.add_argument(
            "-v",
            "--verbose",
            help="increase output verbosity",
            action="store_true")
        args = parser.parse_args(remaining_argv)


        return args

    def get_default_parameters(self):
        defaultparams = {}
        for param in self.params:
            defaultparams[param.name] = param.default

        return defaultparams

    def add_parameter(self, name, type, default, nargs=1, section='main', help='', option=None, choices=None):

        self.params.append(Parameter(name, type, default, nargs, section, help, option=option, choices=choices))