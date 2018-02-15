from collections import defaultdict
import argparse

import numpy as np

from lib.babi import read_task


def main(in_root, in_agent, filter_outliers=False):
    turns = []
    turns_freq_dict = defaultdict(lambda: 0)
    for dialogue_name, dialogue in read_task(in_root):
        for turn in dialogue:
            if turn['agent'] == in_agent:
                turns.append(turn['text'])
                turns_freq_dict[turn['text']] += 1
    frequency_threshold = np.percentile(turns_freq_dict.values(), 95)
    result = [turn for turn in turns if turns_freq_dict[turn] < frequency_threshold]
    return result

def build_argument_parser():
    result = argparse.ArgumentParser()
    result.add_argument('input_file', help='bAbI Dialog tasks folder')
    result.add_argument('agent')
    result.add_argument('--filter_outliers', action='store_true')
    return result


if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()
    main(args.input_file, args.agent, filter_outliers=args.filter_outliers)
