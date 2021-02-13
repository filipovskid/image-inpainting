from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import pathlib
import shutil


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input-path', dest='input_path', help='Path to input data', type=pathlib.Path)
    parser.add_argument('-o', '--output-path', dest='output_path', help='Path to output data', type=pathlib.Path)
    parser.add_argument('--size', type=int, help='Number of files to be sampled')
    parser.add_argument('--seed', type=int, default=42, help='Seed')
    args = parser.parse_args()

    return args


def list_files(dir_path):
    p = dir_path.glob('**/*')
    return [x for x in p if x.is_file()]


def sample_list(item_list, size, replace=False, seed=42):
    rs = RandomState(MT19937(SeedSequence(seed)))

    return list(rs.choice(item_list, size, replace))


def copy_files(file_paths, dest_path):
    for path in file_paths:
        new_path = dest_path.joinpath(path.name)
        shutil.copy(str(path), str(new_path))


def main():
    args = parse_args()
    files = list_files(args.input_path)
    sampled_files = sample_list(files, size=args.size, seed=args.seed)
    output_path = args.output_path

    output_path.mkdir(parents=True, exist_ok=True)
    copy_files(sampled_files, args.output_path)


if __name__ == "__main__":
    main()
