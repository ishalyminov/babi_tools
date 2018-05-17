from argparse import ArgumentParser
from itertools import cycle

from babi_plus import (read_task,
                       init,
                       extract_slot_values,
                       print_stats,
                       make_dialogue_tsv,
                       augment_dialogue,
                       DEFAULT_CONFIG_FILE)


def configure_argument_parser():
    parser = ArgumentParser(description='augment a single file with bAbI+ modifications')
    parser.add_argument('src_file', help='input file')
    parser.add_argument('dst_file', help='output file')
    return parser


def process_file(in_file_name):
    babi_task = read_task(in_file_name)
    slots_map = extract_slot_values(babi_task)
    babi_plus = []
    for dialogue_index, dialogue in zip(xrange(len(babi_task)), cycle(babi_task)):
        babi_plus.append(augment_dialogue(dialogue, slots_map.values()))
    return babi_plus


def save_dataset(in_dialogues, in_dst_filename):
    with open(in_dst_filename, 'w') as task_out:
        for dialogue in in_dialogues:
            print >> task_out, make_dialogue_tsv(dialogue) + '\n\n'


if __name__ == '__main__':
    parser = configure_argument_parser()
    args = parser.parse_args()
    init(DEFAULT_CONFIG_FILE)
    babi_plus_dialogues = process_file(args.src_file)
    save_dataset(babi_plus_dialogues, args.dst_file)
    print_stats()
