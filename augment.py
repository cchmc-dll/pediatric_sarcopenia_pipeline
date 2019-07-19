from argparse import ArgumentParser
from collections import namedtuple
from pathlib import Path

import tables
from Augmentor import DataPipeline

Samples = namedtuple('Samples', 'imdata truth')


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('input_file_path', help='.h5 file path to augment, output will be same place with _aug added')

    return parser.parse_args()


def main():
    args = parse_args()
    data = tables.open_file(args.input_file_path, mode='r')
    samples = augment_images(xs=data.root.imdata, ys=data.root.truth)

    print(save_samples(samples))


def augment_images(xs, ys):
    pipeline = DataPipeline(xs, ys)

    pipeline.random_distortion(probability=0.75, grid_height=16, grid_width=16, magnitude=4)
    pipeline.rotate(probability=0.9, max_left_rotation=15, max_right_rotation=15)
    pipeline.flip_top_bottom(probability=0.05)
    pipeline.flip_left_right(probability=0.5)
    pipeline.zoom(0.75, min_factor=0, max_factor=0.1)
    pipeline.random_brightness(probability=0.9, min_factor=0.9, max_factor=1.1)

    return pipeline.sample(len(xs) * 10)


def save_samples(samples, input_file_path):
    p = Path(input_file_path)
    output_file_path = Path(p.parent, f'{p.stem}_aug.h5')
    # tables.open_file(str(output_file_path), 'w')
    return output_file_path



if __name__ == '__main__':
    main()