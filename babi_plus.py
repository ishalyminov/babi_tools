import json
import sys
from operator import itemgetter
from os import path, makedirs
import random

import numpy as np
import nltk

from lib.babi import load_dataset, extract_slot_values

random.seed(273)

CONFIG_FILE = 'babi_plus.json'
with open(CONFIG_FILE) as actions_in:
    CONFIG = json.load(actions_in)

ACTIONS = CONFIG['actions']
TAGGER = nltk.UnigramTagger(nltk.corpus.brown.tagged_sents(categories='news'))


def perform_action(in_action, in_tagged_tokens, in_token_index, in_slot_values):
    word, pos_tag = in_tagged_tokens[in_token_index]
    templates = ACTIONS[in_action]['templates']
    action_outcome = None if not len(templates) else np.random.choice(templates)
    if in_action == 'correct':
        if word in in_slot_values:
            incorrect_value = np.random.choice([
                value
                for value in in_slot_values
                if value != word
            ])
            in_tagged_tokens[in_token_index][0] = ' '.join(
                [incorrect_value, action_outcome, word]
            )
    if in_action == 'hesitate':
        in_tagged_tokens[in_token_index][0] = ' '.join([action_outcome, word])
    if in_action == 'restart':
        in_tagged_tokens[in_token_index][0] = ' '.join(
            [word] + [action_outcome] + map(itemgetter(0), in_tagged_tokens[:in_token_index + 1])
        )


def apply_replacements(in_utterance):
    REPLACEMENTS = [
        ('are looking', 'are you looking')
    ]
    for pattern, replacement in REPLACEMENTS:
        in_utterance = in_utterance.replace(pattern, replacement)
    return in_utterance


def compute_probability_distribution(in_actions):
    weights = [ACTIONS[action]['weight'] for action in in_actions]
    probabilities = [weight / sum(weights) for weight in weights]
    return probabilities


def sample_transformations(in_utterance_length):
    transformed_token_indices = [
        np.random.choice(range(in_utterance_length))
        for _ in xrange(CONFIG['max_modifications_per_utterance'])
    ]
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
        for index, action in zip(transformed_token_indices, actions)
    ]


def augment_dialogue(in_dialogue, in_slot_values):
    result = []
    dialogue_name, dialogue = in_dialogue
    for utterance_index, utterance in enumerate(dialogue):
        utterance['text'] = apply_replacements(utterance['text'])
        if utterance_index % 2 == 1:
            result.append(utterance['text'])
            continue
        tagged_tokens = map(list, TAGGER.tag(utterance['text'].split()))
        transformations = sample_transformations(len(tagged_tokens))
        for token_index, action in transformations[::-1]:
            word, tag = tagged_tokens[token_index]
            perform_action(
                action,
                tagged_tokens,
                token_index,
                reduce(
                    lambda x, y: x + y,
                    [
                        values_set
                        for values_set in in_slot_values
                        if word in values_set
                    ],
                    []
                )
            )
        result.append(' '.join(map(itemgetter(0), tagged_tokens)))
    return result


def make_babi_plus(in_src_root):
    babi = reduce(
        lambda x, y: x + y,
        load_dataset(in_src_root, 'task1-API-calls'),
        []
    )
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
