import json
import sys
from os import path, makedirs
import random
from collections import defaultdict

import numpy as np
import nltk

from lib.babi import extract_slot_values, get_files_list, read_task

random.seed(273)

CONFIG_FILE = 'babi_plus.json'
with open(CONFIG_FILE) as actions_in:
    CONFIG = json.load(actions_in)

ACTIONS = CONFIG['actions']
TAGGER = nltk.UnigramTagger(nltk.corpus.brown.tagged_sents(categories='news'))


def apply_replacements(in_template, in_slots_map):
    result = in_template
    for slot_name, slot_value in in_slots_map.iteritems():
        result = result.replace(slot_name, slot_value)
    return result


def perform_action(in_action, in_dialog, in_token_coordinates, in_slot_values):
    utterance_index, token_index = in_token_coordinates
    word = in_dialog[utterance_index]['text'][token_index]
    templates = ACTIONS[in_action]['templates']
    action_outcome = None if not len(templates) else np.random.choice(templates)
    if in_action == 'correct':
        if word in in_slot_values:
            replacement_map = {
                '$incorrect_value': np.random.choice([
                    value
                    for value in in_slot_values
                    if value != word
                ]),
                '$correct_value': word
            }
            in_dialog[utterance_index]['text'][token_index:token_index + 1] = apply_replacements(
                action_outcome,
                replacement_map
            ).split()
    if in_action == 'multiturn_correct':
        if word in in_slot_values:
            replacement_map = {
                '$incorrect_value': np.random.choice([
                    value
                    for value in in_slot_values
                    if value != word
                ]),
                '$correct_value': word
            }
            in_dialog[utterance_index]['text'][token_index] = replacement_map['$incorrect_value']
            correction_turn = {
                'agent': 'usr',
                'text': apply_replacements(action_outcome, replacement_map).split()
            }
            in_dialog[utterance_index + 1: utterance_index + 2] = \
                [dict(in_dialog[utterance_index + 1]), correction_turn, dict(in_dialog[utterance_index + 1])]
    if in_action == 'selfcheck' and word in in_slot_values:
            replacement_map = {'$token' : word}
            in_dialog[utterance_index]['text'][token_index:token_index + 1] = apply_replacements(
                action_outcome,
                replacement_map
            ).split()
    if in_action == 'hesitate':
        replacement_map = {'$token': word}
        in_dialog[utterance_index]['text'][token_index:token_index + 1] = apply_replacements(
            action_outcome,
            replacement_map
        ).split()
    if in_action == 'restart':
        replacement_map = {
            '$token': word,
            '$utterance_from_beginning': ' '.join(in_dialog[utterance_index]['text'][:token_index + 1])
        }
        in_dialog[utterance_index]['text'][token_index:token_index + 1] = apply_replacements(
            action_outcome,
            replacement_map
        ).split()


def fix_data(in_utterance):
    REPLACEMENTS = [
        # ('are looking', 'are you looking')
    ]
    for pattern, replacement in REPLACEMENTS:
        in_utterance = in_utterance.replace(pattern, replacement)
    return in_utterance


def compute_probability_distribution(in_actions):
    weights = [ACTIONS[action]['weight'] for action in in_actions]
    probabilities = [weight / sum(weights) for weight in weights]
    return probabilities


def sample_transformations(in_utterance_length):
    token_indices = set(range(in_utterance_length))
    token_indices_to_transform = []
    for _ in xrange(min(CONFIG['max_modifications_per_utterance'], in_utterance_length)):
        index = np.random.choice(list(token_indices))
        token_indices_to_transform.append(index)
        token_indices.remove(index)
    token_indices_to_transform = sorted(token_indices_to_transform)
    actions = []
    available_actions = set(ACTIONS.keys())
    for _ in xrange(CONFIG['max_modifications_per_utterance']):
        action_list = list(available_actions)
        action_probabilities = compute_probability_distribution(action_list)
        action = np.random.choice(action_list, p=action_probabilities)
        actions.append(action)
        available_actions.remove(action)
    return [
        (index, action)
        for index, action in zip(token_indices_to_transform, actions)
    ]


def augment_dialogue(in_dialogue, in_slot_values):
    dialogue_name, dialogue = in_dialogue
    tokenized_dialogue = []
    for utterance in dialogue:
        tokenized_utterance = dict(utterance)
        tokenized_utterance['text'] = fix_data(utterance['text']).split()
        tokenized_dialogue.append(tokenized_utterance)

    for utterance_index in xrange(len(tokenized_dialogue) - 1, -1, -1):
        utterance = tokenized_dialogue[utterance_index]
        if utterance_index % 2 == 1 or utterance['text'] == '<SILENCE>':
            continue
        transformations = sample_transformations(len(utterance['text']))
        for token_index, action in transformations[::-1]:
            perform_action(
                action,
                tokenized_dialogue,
                [utterance_index, token_index],
                set(
                    reduce(
                        lambda x, y: x + y,
                        [
                            values_set
                            for values_set in in_slot_values
                            if utterance['text'][token_index] in values_set],
                        []
                    )
                )
            )
    for utterance in tokenized_dialogue:
        utterance['text'] = ' '.join(utterance['text'])
    return tokenized_dialogue


def plus_dataset(in_src_root):
    dataset_files = get_files_list(in_src_root, 'task1-API-calls')
    babi_files = [(filename, read_task(filename)) for filename in dataset_files]
    full_babi = reduce(
        lambda x, y: x + y[1],
        babi_files,
        []
    )
    slots_map = extract_slot_values(full_babi)
    babi_plus = defaultdict(lambda: [])
    for task_name, task in babi_files:
        for dialogue in task:
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


def save_babi(in_dialogues, in_dst_root):
    if not path.exists(in_dst_root):
        makedirs(in_dst_root)

    for task_name, task_dialogues in in_dialogues.iteritems():
        filename = path.join(in_dst_root, path.basename(task_name))
        with open(filename, 'w') as task_out:
            for dialogue in task_dialogues:
                print >>task_out, make_dialogue_tsv(dialogue) + '\n\n'


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print 'Usage: {} <original bAbI root> <result root> <output format=babi/babble>'.format(
            path.basename(__file__)
        )
        exit()
    source, destination, output_format = sys.argv[1:4]
    babi_plus_dialogues = plus_dataset(source)
    save_function = locals()['save_' + output_format]
    save_function(babi_plus_dialogues, destination)
