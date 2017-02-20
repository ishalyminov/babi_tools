import json
import sys
from os import path, makedirs
import random

import numpy as np

from lib.babi import load_dataset, extract_slot_values

ACTIONS_FILE = 'actions.json'
with open(ACTIONS_FILE) as actions_in:
    ACTIONS = json.load(actions_in)

ACTION_WEIGHTS = [ACTIONS[action]['weight'] for action in ACTIONS]
ACTION_PROBABILITIES = [
    weight / sum(ACTION_WEIGHTS)
    for weight in ACTION_WEIGHTS
]

random.seed(273)


def perform_action(in_action, in_word, in_slot_values):
    result = [in_word]
    templates = ACTIONS[in_action]['templates']
    action_outcome = None if not len(templates) else np.random.choice(templates)
    if in_action == 'correct':
        if in_word in in_slot_values:
            incorrect_value = np.random.choice([
                value
                for value in in_slot_values
                if value != in_word
            ])
            result = [incorrect_value, action_outcome] + result
    if in_action == 'hesitate':
        result = [action_outcome] + result
    return result


def augment_dialogue(in_dialogue, in_slot_values):
    result = []
    for utterance in in_dialogue:
        augmented_utterance = []
        for word in utterance['text'].split():
            action = np.random.choice(list(ACTIONS.keys()), p=ACTION_PROBABILITIES)
            augmented_utterance += perform_action(
                action,
                word,
                reduce(lambda x, y: x + y, [values_set for values_set in in_slot_values if word in values_set], [])
            )
        result.append(' '.join(augmented_utterance))
    return result


def make_babi_plus(in_src_root):
    babi = reduce(lambda x, y: x + y, load_dataset(in_src_root, 'task1-API-calls'), [])
    slots_map = extract_slot_values(babi)
    babi_plus = []
    for dialogue in babi:
        babi_plus.append(augment_dialogue(dialogue, slots_map.values()))
    return babi_plus


def save_plaintext(in_dialogues, in_dst_root):
    if not path.exists(in_dst_root):
        makedirs(in_dst_root)
    for dialogue_index, dialogue in enumerate(in_dialogues):
        with open(path.join(in_dst_root, 'babi_plus_{}.txt'.format(dialogue_index)), 'w') as dialogue_out:
            print >>dialogue_out, '\n'.join(dialogue)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print 'Usage: {} <original bAbI root> <result root>'.format(
            path.basename(__file__)
        )
        exit()
    source, destination = sys.argv[1:3]
    babi_plus_dialogues = make_babi_plus(source)
    save_plaintext(babi_plus_dialogues, destination)
