import json
from collections import deque
from sys import argv
from os import path, makedirs
import re
from codecs import getreader, getwriter

CONFIG_FILE = 'config.json'
with getreader('utf-8')(open(CONFIG_FILE)) as config_in:
    CONFIG = json.load(config_in)


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
            result[-1].append({'user': user_turn, 'system': system_turn})
    return result


def preprocess_for_seq2seq(in_task_dialogs):
    encoder_input, decoder_input = [], []
    encoder_buffer = deque([], maxlen=CONFIG['encoder_context_size'])

    for dialog in in_task_dialogs:
        for turn in dialog:
            user_turn, system_turn = turn['user'], turn['system']
            encoder_buffer.append(user_turn)
            encoder_input.append(' '.join(encoder_buffer))
            decoder_input.append(system_turn)
    return encoder_input, decoder_input


def main(in_dataset_folder, in_task_id, in_output_folder):
    file_prefix = path.join(in_dataset_folder, 'dialog-babi-' + in_task_id)
    train_file = file_prefix + '-trn.txt'
    dev_file = file_prefix + '-dev.txt'
    test_file = file_prefix + '-tst.txt'

    task_train = read_task(train_file)
    task_dev = read_task(dev_file)
    task_test = read_task(test_file)

    if not path.exists(in_output_folder):
        makedirs(in_output_folder)
    dataset_names = ['train', 'dev', 'test']
    datasets = [task_train, task_dev, task_test]
    for dataset_name, task in zip(dataset_names, datasets):
        encoder_input, decoder_input = preprocess_for_seq2seq(task)
        enc_filename = path.join(in_output_folder, dataset_name + '.enc')
        with getwriter('utf-8')(open(enc_filename, 'w')) as encoder_out:
            for line in encoder_input:
                print >>encoder_out, line
        dec_filename = path.join(in_output_folder, dataset_name + '.dec')
        with getwriter('utf-8')(open(dec_filename, 'w')) as decoder_out:
            for line in decoder_input:
                print >> decoder_out, line


if __name__ == '__main__':
    if len(argv) < 4:
        print 'Usage: process_babi.py <dataset folder> <task id> <output_folder>'
        exit()
    dataset_folder, task_id, output_folder = argv[1:]
    main(dataset_folder, task_id, output_folder)
