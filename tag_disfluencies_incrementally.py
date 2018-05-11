from argparse import ArgumentParser

import pandas as pd
import json

from lib.babi import get_files_list, read_task, extract_slot_values


def tag_utterance(in_utterance, in_slot_values, in_action_templates):
    tags = []
    fluent_phrase_buffer = []
    disfluent_phrase_buffer = []
    repair = []
    repair_tags = []
    disfluent_tokens_original = in_utterance.split()
    disfluent_tokens = ['<value>' if token in in_slot_values else token
                        for token in disfluent_tokens_original]

    for token in disfluent_tokens:
        if len(repair):
            if ' '.join(repair + [token]) in ' '.join(fluent_phrase_buffer):
                repair.append(token)
                repair_tags.append('rp')
                continue
            else:
                repair = []
                tags += repair_tags
                repair_tags = []
                fluent_phrase_buffer = []
        disfluent_phrase_buffer_extended = disfluent_phrase_buffer + [token]
        if full_match(disfluent_phrase_buffer, in_action_templates) and not partial_match(
                disfluent_phrase_buffer_extended, in_action_templates):
            tags += ['<f>'] * len(fluent_phrase_buffer)
            tags += ['<e>'] * len(disfluent_phrase_buffer)
            if full_match(disfluent_phrase_buffer, in_action_templates) in ['correct', 'restart',
                                                                         'correct_long_distance']:
                if token in fluent_phrase_buffer:
                    repair_tokens_number = len(fluent_phrase_buffer) - fluent_phrase_buffer.index(
                        token) + len(disfluent_phrase_buffer)
                    repair_tags.append('<rm-{}>'.format(repair_tokens_number))
                    repair.append(token)
                else:
                    tags.append('<f>')
            else:
                fluent_phrase_buffer = []
            disfluent_phrase_buffer = []
        elif partial_match(disfluent_phrase_buffer_extended, in_action_templates):
            disfluent_phrase_buffer = disfluent_phrase_buffer_extended
        else:
            fluent_phrase_buffer.append(token)

    if len(repair):
        tags += repair_tags
    else:
        tags += ['<f>'] * len(fluent_phrase_buffer)
    tags += ['<e>'] * len(disfluent_phrase_buffer)

    return disfluent_tokens_original, tags


def partial_match(in_buffer, in_templates):
    if not len(in_buffer):
        return False
    for key, templates in in_templates.iteritems():
        for template in templates:
            if template[:len(in_buffer)] == in_buffer:
                return key
    return False


def full_match(in_buffer, in_templates):
    if not len(in_buffer):
        return False
    for key, templates in in_templates.iteritems():
        for template in templates:
            if template == in_buffer:
                return key
    return False


def collect_babi_slot_values(in_babi_root):
    dataset_files = get_files_list(in_babi_root, 'task1-API-calls')
    babi_files = [(filename, read_task(filename)) for filename in dataset_files]
    full_babi = reduce(lambda x, y: x + y[1],
                       babi_files,
                       [])
    slots_map = extract_slot_values(full_babi)
    return reduce(lambda x, y: list(x) + list(y), slots_map.values(), [])


def get_action_templates(in_config):
    action_templates = dict(in_config['action_templates'])
    for key in action_templates.keys():
        value = action_templates[key]
        value = map(lambda x: x.split(), value)
        value = [filter(lambda x: not x.startswith('$'), tokens) for tokens in value]
        action_templates[key] = value
    del action_templates['NULL']
    return action_templates


def configure_argument_parser():
    parser = ArgumentParser(description='Tag disfluencies SWDA-style')
    parser.add_argument('parallel_file')
    parser.add_argument('result_file')
    parser.add_argument('config_file')
    parser.add_argument('babi_folder', help='original bAbI Dialog dataset root')

    return parser


def main():
    parser = configure_argument_parser()
    args = parser.parse_args()

    with open(args.config_file) as config_in:
        config = json.load(config_in)

    dataset = pd.read_csv(args.parallel_file, delimiter=';')
    slot_values = collect_babi_slot_values(args.babi_folder)
    action_templates = get_action_templates(config)

    result_utterances, result_tags = [], []
    for idx, (utterance_disfluent, utterance_fluent) in dataset.iterrows():
        tokens, tags = tag_utterance(utterance_disfluent, slot_values, action_templates)
        result_utterances.append(tokens)
        result_tags.append(tags)
    result = pd.DataFrame({'utterance': result_utterances, 'tags': result_tags})
    result.to_json(args.result_file)


if __name__ == '__main__':
    main()
