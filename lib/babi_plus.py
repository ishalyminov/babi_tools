from collections import defaultdict

import numpy as np


def perform_action(in_action, in_dialog, in_token_coordinates, in_slot_values, in_action_templates):
    utterance_index, token_index = in_token_coordinates
    word = in_dialog[utterance_index]['text'][token_index]
    templates = in_action_templates[in_action]
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
    # can only be taken at the PP start
    if in_action == 'pp_restart':
        pp = word
        pp_begin, pp_end = [token_index, token_index + 1]
        if in_dialog[utterance_index]['pos'][token_index + 1] == 'DT':
            pp += ' ' + in_dialog[utterance_index]['text'][token_index + 1]
            pp_end += 1
        replacement_map = {
            '$token': word,
            '$pp': pp
        }
        in_dialog[utterance_index]['text'][pp_begin: pp_end] = apply_replacements(
                action_outcome,
                replacement_map
        ).split()
        in_dialog[utterance_index]['tags'][pp_begin: pp_end] = get_tags_after_replacement(in_action, action_outcome, replacement_map)
    if in_action == 'correct_long_distance':
        phrase_begin, phrase_end = get_enclosing_phrase(
            in_dialog[utterance_index]['text'],
            token_index
        )
        if word in in_slot_values:
            correct_phrase = ' '.join(
                in_dialog[utterance_index]['text'][phrase_begin: phrase_end + 1]
            )
            incorrect_phrase = in_dialog[utterance_index]['text'][phrase_begin: phrase_end + 1]
            incorrect_phrase[token_index - phrase_begin] = np.random.choice([
                value
                for value in in_slot_values
                if value != word
            ])
            incorrect_phrase = ' '.join(incorrect_phrase)
            replacement_map = {
                '$incorrect_phrase': incorrect_phrase,
                '$correct_phrase': correct_phrase
            }
            in_dialog[utterance_index]['text'][phrase_begin:phrase_end + 1] = apply_replacements(
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
        in_dialog[utterance_index]['tags'][token_index: token_index + 1] = get_tags_after_replacement(in_action, action_outcome, replacement_map)
    if in_action == 'restart':
        replacement_map = {
            '$token': word,
            '$utterance_from_beginning': ' '.join(in_dialog[utterance_index]['text'][:token_index + 1])
        }
        in_dialog[utterance_index]['text'][token_index:token_index + 1] = apply_replacements(
            action_outcome,
            replacement_map
        ).split()
        in_dialog[utterance_index]['tags'][
        token_index:token_index + 1] = get_tags_after_replacement(in_action, action_outcome, replacement_map)


def get_tags_after_replacement(in_action_name, in_action, in_replacement_map):
    result = []
    action_tokens = in_action.split()
    if in_action_name == 'hesitate':
        for token in action_tokens:
            if token in in_replacement_map:
               result += ['<f/>' for _ in xrange(len(in_replacement_map[token].split()))]
            else:
                result.append('<e/>')
    elif in_action_name in ['pp_restart']:
        current_idx = 0
        for token in action_tokens:
            if token in in_replacement_map:
                replacement_tokens = in_replacement_map[token].split()
                if current_idx == 0:
                    result += ['<f/>'] * len(replacement_tokens)
                else:
                    if len(replacement_tokens) == 1:
                        result += ['<rm-{}/><rpEndSub/>'.format(current_idx)]
                    else:
                        result += ['<rm-{}/><rpMid/>'.format(current_idx)]
                        result += ['<f/>'] * (len(replacement_tokens) - 2)
                        result += ['<rpEndSub/>']
                current_idx += len(replacement_tokens)
            else:
                result.append('<e/>')
                current_idx += 1
    elif in_action_name in ['restart']:
        current_idx = 0
        for token in action_tokens:
            if token in in_replacement_map:
                replacement_tokens = in_replacement_map[token].split()
                if current_idx == 0:
                    result += ['<f/>']
                else:
                    if len(replacement_tokens) == 1:
                        result += ['<rm-{}/><rpEndSub/>'.format(len(replacement_tokens) * 2)]
                    else:
                        result += ['<rm-{}/><rpMid/>'.format(len(replacement_tokens) + 1)]
                        result += ['<f/>'] * (len(replacement_tokens) - 2)
                        result += ['<rpEndSub/>']
                current_idx += len(replacement_tokens)
            else:
                result.append('<e/>')
                current_idx += 1
    return result


def apply_replacements(in_template, in_slots_map):
    result = in_template
    for slot_name, slot_value in in_slots_map.iteritems():
        result = result.replace(slot_name, slot_value)
    return result


def calculate_action_probabilities(in_config, in_action_limits):
    limits = dict(in_action_limits)
    for action in limits:
        limits[action] = float(0.0 < limits[action])
    action_weights_masked = defaultdict(lambda: {})
    # action weight masks differ
    # for the cases of background words and slot values
    for case, mask_map in in_config['action_weight_mask'].iteritems():
        sum_masked_weight = 0.0
        for action, mask_value in mask_map.iteritems():
            masked_weight = in_config['action_weights'][action] * mask_value * limits[action]
            if action != 'NULL':
                masked_weight *= limits['GLOBAL']
            action_weights_masked[case][action] = masked_weight
            sum_masked_weight += masked_weight
        for action in action_weights_masked[case]:
            action_weights_masked[case][action] /= sum_masked_weight
        assert abs(sum(action_weights_masked[case].values()) - 1.0) < 1e-7
    return {
        case: [weight_map[action] for action in in_config['action_weights']]
        for case, weight_map in action_weights_masked.iteritems()
    }


def sample_transformations(in_utterance, in_slot_values, in_config):
    action_limits = dict(in_config['action_limits'])

    token_types = []
    for idx, (token, tag) in enumerate(zip(in_utterance['text'], in_utterance['pos'])):
        if token in in_slot_values:
            token_types.append('slot_value')
        elif tag == 'IN' and token != 'like':
            token_types.append('pp_start')
        else:
            token_types.append('background_word')

    per_token_actions = []
    for token_type in token_types:
        action_probs = calculate_action_probabilities(in_config, action_limits)
        action = np.random.choice(in_config['action_weights'].keys(), p=action_probs[token_type])
        per_token_actions.append(action)
        action_limits[action] -= 1
        action_limits['GLOBAL'] -= 1 * int(action != 'NULL')

    count_map = defaultdict(lambda: 0)
    for action in per_token_actions:
        count_map[action] += 1
    for action, count in count_map.iteritems():
        assert count <= in_config['action_limits'][action]
    return per_token_actions


def get_enclosing_phrase(in_tokens, in_token_index):
    phrase_begin, phrase_end = in_token_index, in_token_index

    while 0 < phrase_begin and in_tokens[phrase_begin - 1] in ['with', 'for', 'in', 'a']:
        phrase_begin -= 1
    while phrase_end < len(in_tokens) - 1 and in_tokens[phrase_end + 1] in ['cuisine',
                                                                            'food',
                                                                            'people',
                                                                            'price',
                                                                            'range']:
        phrase_end += 1
    return phrase_begin, phrase_end