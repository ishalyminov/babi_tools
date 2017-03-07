from os import path, makedirs
import random

import sys
from shutil import copyfile

from lib.babi import load_babble_dataset, save_csv

random.seed(273)

FOLD_NUMBER = 4
SAMPLE_SIZE = 100


def main(in_babi_root, in_result_root, in_shuffle_file):
    with open(in_shuffle_file) as shuffle_in:
        shuffle = map(int, shuffle_in.readline().strip().split(';'))
    filenames = load_babble_dataset(in_babi_root)

    if not path.exists(in_result_root):
        makedirs(in_result_root)
    for dialogue_index in shuffle[FOLD_NUMBER * SAMPLE_SIZE: SAMPLE_SIZE * (FOLD_NUMBER + 1)]:
        dialogue_name = filenames[dialogue_index]
        copyfile(
            path.join(in_babi_root, dialogue_name),
            path.join(in_result_root, dialogue_name)
        )


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print 'Usage: {} <babi root> <result root> <shuffle file>'.format(
            path.basename(__file__)
        )
        exit()
    babi_root, result_root, shuffle_filename = sys.argv[1:4]
    main(babi_root, result_root, shuffle_filename)
