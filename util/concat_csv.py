from argparse import ArgumentParser, FileType
from pathlib import Path
import sys


def main(argv):
    args = parse_args(argv)

    result = []
    result.extend(args.csv_files[0].readlines())
    for f in args.csv_files[1:]:
        lines = f.readlines()
        result.extend(lines[1:])

    args.output_file.writelines(result)


def parse_args(argv):
    parser = ArgumentParser()
    parser.add_argument('csv_files', type=FileType('r'), nargs='+')
    parser.add_argument('output_file', type=FileType('w'))

    return parser.parse_args(argv)

if __name__ == "__main__":
    main(sys.argv[1:])
