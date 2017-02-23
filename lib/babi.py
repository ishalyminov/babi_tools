import re
from codecs import getreader
from collections import deque

from os import path

from collections import defaultdict

TASK_ID = 'task1-API-calls'


def read_task(in_file_name):
    result = []
    with getreader('utf-8')(open(in_file_name)) as task_in:
        task_content = task_in.read()
    dialogs = [
        filter(lambda line: len(line.strip()), dialog.split('\n'))
        for dialog in task_content.split('\n\n')
    ]
    dialogs = filter(len, dialogs)

    for dialog in dialogs:
        result.append([])
        for line in dialog:
            line = re.sub('^\d+\s', '', line)
            user_turn, system_turn = line.split('\t')
            result[-1].append({'agent': 'user', 'text': user_turn})
            result[-1].append({'agent': 'system', 'text': system_turn})
    return result


def preprocess_for_seq2seq(in_task_dialogs, in_config):
    encoder_input, decoder_input = [], []
    encoder_buffer = deque([], maxlen=in_config['encoder_context_size'])

    for dialog in in_task_dialogs:
        encoder_buffer.clear()
        for turn in dialog:
            user_turn, system_turn = turn['user'], turn['system']
            encoder_buffer.append(user_turn)
            encoder_input.append(' '.join(encoder_buffer))
            encoder_buffer.append(system_turn)
            decoder_input.append(system_turn)
    return encoder_input, decoder_input


def load_dataset(in_root, in_task_id):
    file_prefix = path.join(in_root, 'dialog-babi-{}'.format(in_task_id))
    train_file = file_prefix + '-trn.txt'
    dev_file = file_prefix + '-dev.txt'
    test_file = file_prefix + '-tst.txt'

    task_train = read_task(train_file)
    task_dev = read_task(dev_file)
    task_test = read_task(test_file)
    return task_train, task_dev, task_test


def extract_slot_values(in_dialogues):
    result = defaultdict(lambda: [])
    for dialogue in in_dialogues:
        slot_values = dialogue[-1]['text'].split()[1:]
        [result[index].append(value) for index, value in enumerate(slot_values)]
    return result
