from argparse import ArgumentParser

import pandas as pd
import json

from lib.babi import get_files_list, read_task, extract_slot_values, extract_slot_value_pps

UTTERANCES_TEST = [('i i um i want a a place with italian sorry spanish cuisine', ''),
                   ('i want a place with italian oh no spanish cuisine', ''),
                   ('i want a place with with let me check italian cuisine', ''),
                   ('i want a place with italian cuisine um sorry with spanish cuisine', ''),
                   ('i want um yeah i want a place with spanish cuisine', '')]


class SWDADisfluencyTagger(object):
    def __init__(self, in_slot_values, in_slot_value_phrases, in_action_templates):
        self.slot_values = in_slot_values
        self.slot_value_phrases = in_slot_value_phrases
        self.action_templates = []
        for key, templates in in_action_templates.iteritems():
            self.action_templates += map(tuple, templates)

    def reset_state(self):
        self.state = 'FLUENT'

    def delexicalize_slot_values(self, in_tokens):
        return ['<value>' if token in self.slot_values else token
                for token in in_tokens]

    def tag_utterance(self, in_utterance):
        self.reset_state()

        utterance_tokens_original = in_utterance.split()
        utterance_tokens = self.delexicalize_slot_values(utterance_tokens_original)

        fluent_buffer = []
        reparandum_buffer = []
        interregnum_buffer = []
        repair_buffer = []
        tags = []

        for token in utterance_tokens:
            if self.state == 'FLUENT':
                if matches_template_prefix([token], self.action_templates):
                    interregnum_buffer.append(token)
                    self.state = 'INSIDE_INTERREGNUM'
                else:
                    fluent_buffer.append(token)
                    if matches_template_prefix(reparandum_buffer + [token], self.slot_value_phrases):
                        reparandum_buffer.append(token)
                        self.state = 'INSIDE_REPARANDUM'
            elif self.state == 'INSIDE_INTERREGNUM':
                if matches_template_prefix(interregnum_buffer + [token], self.action_templates):
                    interregnum_buffer.append(token)
                else:
                    if matches_template_prefix(repair_buffer + [token], [tuple(reparandum_buffer)]):
                        repair_buffer.append(token)
                        self.state = 'INSIDE_REPAIR'
                    elif token == '<value>':
                        repair_buffer = [token]
                    else:
                        tags += self.flush_tags(fluent_buffer,
                                                reparandum_buffer,
                                                interregnum_buffer,
                                                repair_buffer)
                        fluent_buffer = [token]
                        reparandum_buffer = []
                        interregnum_buffer = []
                        repair_buffer = []
                        self.state = 'FLUENT'
            elif self.state == 'INSIDE_REPARANDUM':
                if matches_template_prefix(reparandum_buffer + [token], self.slot_value_phrases):
                    reparandum_buffer.append(token)
                    fluent_buffer.append(token)
                elif matches_template_prefix(interregnum_buffer + [token], self.action_templates):
                    interregnum_buffer.append(token)
                    self.state = 'INSIDE_INTERREGNUM'
                else:
                    self.state = 'FLUENT'
                    reparandum_buffer = []
                    fluent_buffer.append(token)
                    if matches_template_prefix(reparandum_buffer + [token],
                                               self.slot_value_phrases):
                        reparandum_buffer.append(token)
                        self.state = 'INSIDE_REPARANDUM'
            elif self.state == 'INSIDE_REPAIR':
                if matches_template_prefix(repair_buffer + [token], [tuple(reparandum_buffer)]):
                    repair_buffer.append(token)
                else:
                    tags += self.flush_tags(fluent_buffer,
                                            reparandum_buffer,
                                            interregnum_buffer,
                                            repair_buffer)
                    reparandum_buffer = []
                    interregnum_buffer = []
                    repair_buffer = []
                    fluent_buffer = []
                    if matches_template_prefix([token], self.action_templates):
                        interregnum_buffer.append(token)
                        self.state = 'INSIDE_INTERREGNUM'
                    else:
                        fluent_buffer = [token]
                        self.state = 'FLUENT'

            else:
                raise NotImplementedError
        tags += self.flush_tags(fluent_buffer,
                                reparandum_buffer,
                                interregnum_buffer,
                                repair_buffer)
        assert len(tags) == len(utterance_tokens)
        return tags

    def flush_tags(self,
                   in_fluent_buffer,
                   in_reparandum_buffer,
                   in_interregnum_buffer,
                   in_repair_buffer):
        result = []
        result += ['<f/>'] * len(in_fluent_buffer)
        if matches_template(in_interregnum_buffer, self.action_templates):
            result += ['<e/>'] * len(in_interregnum_buffer)
        else:
            result += ['<f/>'] * len(in_interregnum_buffer)
        if 1 == len(in_repair_buffer):
            result.append('<rm-{}/><rpEndSub/>'.format(len(in_interregnum_buffer) + 1))
        elif 1 < len(in_repair_buffer):
            result.append('<rm-{}/><rpMid/>'.format(len(in_interregnum_buffer) + 1))
            for _ in xrange(len(in_repair_buffer) - 2):
                result.append('<f/>')
            result.append('<rpEnd/>')
        return result


def matches_template_prefix(in_buffer, in_templates):
    if not len(in_buffer):
        return False
    buffer_tuple = tuple(in_buffer)
    for template in in_templates:
        if template[:len(in_buffer)] == buffer_tuple:
            return True
    return False


def matches_template(in_buffer, in_templates):
    buffer_tuple = tuple(in_buffer)
    if not len(in_buffer):
        return None
    for template in in_templates:
        if template == buffer_tuple:
            return True
    return None


def collect_babi_slot_values(in_babi_root):
    dataset_files = get_files_list(in_babi_root, 'task1-API-calls')
    babi_files = [(filename, read_task(filename)) for filename in dataset_files]
    full_babi = reduce(lambda x, y: x + y[1],
                       babi_files,
                       [])
    slots_map = extract_slot_values(full_babi)
    return reduce(lambda x, y: list(x) + list(y), slots_map.values(), [])


def collect_babi_slot_value_pps(in_babi_root, in_slot_values):
    dataset_files = get_files_list(in_babi_root, 'task1-API-calls')
    babi_files = [(filename, read_task(filename)) for filename in dataset_files]
    full_babi = reduce(lambda x, y: x + y[1],
                       babi_files,
                       [])
    return extract_slot_value_pps(full_babi, in_slot_values)


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
    parser.add_argument('babi_folder',
                        help='original bAbI Dialog dataset root',
                        default='dialog-bAbI-tasks')

    return parser


def main():
    parser = configure_argument_parser()
    args = parser.parse_args()

    with open(args.config_file) as config_in:
        config = json.load(config_in)

    dataset = pd.read_csv(args.parallel_file, delimiter=';')
    slot_values = collect_babi_slot_values(args.babi_folder)
    slot_value_pps = collect_babi_slot_value_pps(args.babi_folder, slot_values)
    action_templates = get_action_templates(config)

    result_utterances, result_tags = [], []
    tagger = SWDADisfluencyTagger(slot_values, slot_value_pps, action_templates)

    for idx, (utterance_disfluent, utterance_fluent) in enumerate(UTTERANCES_TEST):  # dataset.iterrows():
        tags = tagger.tag_utterance(utterance_disfluent)
        print utterance_disfluent
        print tags
        result_utterances.append(utterance_disfluent.split())
        result_tags.append(tags)
    result = pd.DataFrame({'utterance': result_utterances, 'tags': result_tags})
    result.to_json(args.result_file)


if __name__ == '__main__':
    main()
