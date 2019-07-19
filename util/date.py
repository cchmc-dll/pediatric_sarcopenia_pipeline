import pandas as pd

from argparse import ArgumentParser


def main():
    args = parse_args()
    df = pd.read_excel(args.input_file)

    df[args.new_col_name] = df.apply(convert_date, axis=1)

    df.to_excel(args.out_file_path)


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('input_file', help='Excel to convert date string for')
    parser.add_argument('sheet', help='Excel Sheet name where column found')
    parser.add_argument('column', help='Column to convert')
    parser.add_argument('out_file_path', help='Output file path')
    parser.add_argument(
        '--new_col_name',
        default='age_in_years',
        help='What to name the new column with the new date',
        required=False
    )

    return parser.parse_args()


def convert_date(row):
    age = 0
    for token in (s.strip() for s in row['age'].split(',')):
        mdy = token[-1]
        val = int(token[0:-1])
        if mdy == 'y':
            age += val
        elif mdy == 'm':
            age += val / 12
        elif mdy == 'w':
            age += val / 52
        elif mdy == 'd':
            age += val / 365

    return age

if __name__ == '__main__':
    main()
