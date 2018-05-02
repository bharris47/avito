import csv
from argparse import ArgumentParser
from random import shuffle

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('csv_path')
    parser.add_argument('--sample-size', '-s', type=float, default=0.05)
    args = parser.parse_args()

    with open(args.csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
        shuffle(rows)

    n_rows = int(len(rows) * args.sample_size)
    with open(args.csv_path + '.sample', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows[:n_rows])
