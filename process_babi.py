import json
from sys import argv
from os import path, makedirs
from codecs import getreader, getwriter

from lib.babi import preprocess_for_seq2seq, load_dataset

CONFIG_FILE = 'config.json'
with getreader('utf-8')(open(CONFIG_FILE)) as config_in:
    CONFIG = json.load(config_in)


def main(in_dataset_folder, in_task_id, in_output_folder):
    train, dev, test = load_dataset(in_dataset_folder, in_task_id)

    if not path.exists(in_output_folder):
        makedirs(in_output_folder)
    dataset_names = ['train', 'dev', 'test']
    datasets = [train, dev, test]
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
        print 'Usage: {} <dataset folder> <task id> <output_folder>'.format(
            argv[0]
        )
        exit()
    dataset_folder, task_id, output_folder = argv[1:]
    main(dataset_folder, task_id, output_folder)
