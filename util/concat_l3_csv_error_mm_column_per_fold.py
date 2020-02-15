from argparse import ArgumentParser, FileType
from pathlib import Path
import sys
import csv
from decimal import Decimal
from statistics import mean

from toolz import groupby


def main(argv):
    args = parse_args(argv)

    csv_rows = []
    for f in args.csv_files:
        dict_reader = csv.DictReader(f)
        rows = [r for r in dict_reader]
        csv_rows.extend(rows)

    fold_results = []
    for subject_id, rows in groupby('', csv_rows).items():
        new_row = [subject_id]
        errors = [r['error_mm'] for r in rows]
        new_row.extend(errors)
        error_avg = mean([Decimal(e) for e in errors])
        new_row.append("{0:.3f}".format(error_avg))
        fold_results.append(new_row)

    writer = csv.writer(args.output_file)
    writer.writerow(['subject_id', 'fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5', 'mean'])
    writer.writerows(fold_results)

    num_greater_than_10 = len(list(filter(lambda x: x > 10, [Decimal(r[-1]) for r in fold_results])))
    print('> 10:' , num_greater_than_10)

    # args.output_file.writelines(result)


def parse_args(argv):
    parser = ArgumentParser()
    parser.add_argument('csv_files', type=FileType('r'), nargs='+')
    parser.add_argument('output_file', type=FileType('w'))

    return parser.parse_args(argv)

if __name__ == "__main__":
    main(sys.argv[1:])
