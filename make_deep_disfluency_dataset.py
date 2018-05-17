import json
from argparse import ArgumentParser
from itertools import cycle
from os import path, makedirs
import random
from collections import defaultdict

import numpy as np
import pandas as pd

from lib.babi import extract_slot_values, get_files_list, read_task
from lib.babi_plus import sample_transformations, perform_action

random.seed(273)
np.random.seed(273)

DEFAULT_CONFIG_FILE = 'babi_plus.json'
CONFIG = None

ACTION_LIST = None
STATS = defaultdict(lambda: 0)


def fix_data(in_utterance):
    REPLACEMENTS = [
        # ('are looking', 'are you looking')
    ]
    for pattern, replacement in REPLACEMENTS:
        in_utterance = in_utterance.replace(pattern, replacement)
    return in_utterance


def init(in_config_file):
    global CONFIG, ACTION_LIST
    with open(in_config_file) as actions_in:
        CONFIG = json.load(actions_in)
    ACTION_LIST = sorted(CONFIG['action_templates'].keys())


def augment_dialogue(in_dialogue, in_slot_values):
    slot_values_flat = reduce(lambda x, y: x + list(y), in_slot_values, [])
    dialogue_name, dialogue = in_dialogue
    tokenized_dialogue = []
    for utterance in dialogue:
        tokenized_utterance = dict(utterance)
        tokenized_utterance['text'] = fix_data(utterance['text']).split()
        tokenized_utterance['tags'] = ['<f/>' for _ in xrange(len(tokenized_utterance['text']))]
        tokenized_dialogue.append(tokenized_utterance)

    dialogue_modified = False
    utterances_modified = 0
    action_stats = defaultdict(lambda: 0)
    for utterance_index in xrange(len(tokenized_dialogue) - 1, -1, -1):
        utterance = tokenized_dialogue[utterance_index]
        if utterance_index % 2 == 1 or utterance['text'] == [u'<SILENCE>']:
            continue
        transformations = sample_transformations(utterance, slot_values_flat, CONFIG)
        if set(transformations) != {'NULL'}:
            utterances_modified += 1
        for transformation in transformations:
            action_stats[transformation] += 1

        for reverse_token_index, action in enumerate(transformations[::-1]):
            if action != 'NULL':
                dialogue_modified = True
            token_index = len(transformations) - reverse_token_index - 1
            perform_action(action,
                           tokenized_dialogue,
                           [utterance_index, token_index],
                           set(reduce(lambda x, y: x + list(y),
                               [values_set
                                for values_set in in_slot_values
                                if utterance['text'][token_index] in values_set],
                               [])),
                           CONFIG['action_templates'])
    for utterance in tokenized_dialogue:
        utterance['text'] = ' '.join(utterance['text'])

    global STATS
    STATS['dialogues_modified'] += int(dialogue_modified)
    STATS['utterances_modified'] += utterances_modified
    for action, count in action_stats.iteritems():
        STATS[action] += count

    return tokenized_dialogue


def plus_dataset(in_src_root, in_result_size):
    dataset_files = get_files_list(in_src_root, 'task1-API-calls')
    babi_files = [(filename, read_task(filename)) for filename in dataset_files]
    full_babi = reduce(
        lambda x, y: x + y[1],
        babi_files,
        []
    )
    slots_map = extract_slot_values(full_babi)
    babi_plus = defaultdict(lambda: [])
    result_size = in_result_size if in_result_size else len(babi_files)
    for task_name, task in babi_files:
        for dialogue_index, dialogue in zip(xrange(result_size), cycle(task)):
            babi_plus[task_name].append(
                augment_dialogue(dialogue, slots_map.values())
            )
    return babi_plus


def plus_single_task(in_task, slot_values):
    slots_map = extract_slot_values(in_task) \
        if slot_values is None \
        else slot_values
    babi_plus = map(
        lambda dialogue: augment_dialogue(dialogue, slots_map.values()),
        in_task
    )
    return babi_plus


def make_dialogue_tsv(in_dialogue):
    assert len(in_dialogue) % 2 == 0
    return '\n'.join([
        '{} {}\t{}'.format(index + 1, usr['text'], sys['text'])
        for index, (usr, sys) in enumerate(zip(in_dialogue[::2], in_dialogue[1::2]))
    ])


def save_babble(in_dialogues, in_dst_root):
    if not path.exists(in_dst_root):
        makedirs(in_dst_root)

    for dialogue_index, dialogue in enumerate(in_dialogues):
        with open(path.join(in_dst_root, 'babi_plus_{}.txt'.format(dialogue_index)), 'w') as dialogue_out:
            print >>dialogue_out, '\n'.join([
                '{}:\t{}'.format(utterance['agent'], utterance['text'])
                for utterance in dialogue
            ])


def print_stats():
    print 'Data modification statistics:'
    for key, value in STATS.iteritems():
        print '{}\t{}'.format(key, value)


def save_babi(in_dialogues, in_dst_root):
    if not path.exists(in_dst_root):
        makedirs(in_dst_root)

    for task_name, task_dialogues in in_dialogues.iteritems():
        filename = path.join(in_dst_root, path.basename(task_name))
        with open(filename, 'w') as task_out:
            for dialogue in task_dialogues:
                print >>task_out, make_dialogue_tsv(dialogue) + '\n\n'


def configure_argument_parser():
    parser = ArgumentParser(description='make a dataset of bAbI+ utterances for disfluency tagging')
    parser.add_argument('babi_file', help='file with bAbI Dialogs')
    parser.add_argument('result_file')
    parser.add_argument(
        '--config',
        default=DEFAULT_CONFIG_FILE,
        help='dicustom disfluency config (json file)'
    )

    return parser


def main(in_config, in_babi_file, in_result_file):
    init(in_config)
    task = read_task(in_babi_file)
    slot_values = extract_slot_values(task)
    babi_plus_dialogues = plus_single_task(task, slot_values)
    utterances, tags, pos = [], [], []

    for dialogue in babi_plus_dialogues:
        for turn in dialogue:
            if turn['agent'] == 'user':
                utterances.append(turn['text'])
                tags.append(turn['tags'])
                pos.append(turn['pos'])
    result = pd.DataFrame({'utterance': utterances, 'tags': tags, 'pos': pos})
    result.to_json(in_result_file)
    print_stats()


if __name__ == '__main__':
    parser = configure_argument_parser()
    args = parser.parse_args()
    main(args.config, args.babi_file, args.result_file)
