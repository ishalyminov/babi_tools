import argparse
from os import listdir, path, makedirs
from random import shuffle

from shutil import rmtree

import extract_turns


def main(in_root_folder, in_result_folder, in_trainset_ratio):
    dstc_files = [filename
                  for filename in listdir(in_root_folder)
                  if 'task6' in filename and 'candidates' not in filename]
    result = []
    for filename in dstc_files:
        result += extract_turns.main(path.join(in_root_folder, filename), 'user', True)
    save_dataset(result, in_result_folder, in_trainset_ratio)


def save_dataset(in_turns, in_result_folder, in_trainset_ratio):
    if path.exists(in_result_folder):
        rmtree(in_result_folder)
    makedirs(in_result_folder)

    shuffle(in_turns)
    trainset_size = int(in_trainset_ratio * len(in_turns))
    devset_size = int((len(in_turns) - trainset_size) / 2.0)
    trainset = in_turns[:trainset_size]
    devset = in_turns[trainset_size: trainset_size + devset_size]
    testset = in_turns[trainset_size + devset_size:]

    for dataset_name, dataset in zip(['train', 'dev', 'test'], [trainset, devset, testset]):
        makedirs(path.join(in_result_folder, dataset_name))
        with open(path.join(in_result_folder, dataset_name, 'encoder.txt'), 'w') as encoder_out:
            for line in dataset:
                print >>encoder_out, line
        with open(path.join(in_result_folder, dataset_name, 'decoder.txt'), 'w') as decoder_out:
            for line in dataset:
                print >>decoder_out, line


def build_argument_parser():
    result = argparse.ArgumentParser()
    result.add_argument('babi_folder', help='bAbI Dialog tasks folder')
    result.add_argument('result_folder')
    result.add_argument('--trainset_ratio', type=float, default=0.8)

    return result


if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()
    main(args.babi_folder, args.result_folder, args.trainset_ratio)
