from os import path
import random

import sys

from lib.babi import load_dataset, TASK_ID

random.seed(273)


def main(in_data_root):
    train_set, dev_set, test_set, testoov_set = load_dataset(in_data_root, TASK_ID)
    index = range(len(train_set) + len(dev_set) + len(test_set) + len(testoov_set))
    random.shuffle(index)
    return index


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'Usage: {} <babi root>'.format(path.basename(__file__))
        exit()
    print ' '.join(map(str, main(sys.argv[1])))
