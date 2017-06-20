import json
import os
import re
import subprocess
import sys

import shutil
from codecs import getreader
from collections import defaultdict
from multiprocessing import Pool
from operator import itemgetter

CONFIG_FILE = 'extract_ds_features.json'
with open(CONFIG_FILE) as config_in:
    CONFIG = json.load(config_in)


def process_single_file(in_params):
    in_src_file, in_dst_file = in_params
    subprocess.call([
        'java',
        '-jar',
        CONFIG['babble_binary_name'],
        in_src_file, in_dst_file
    ])


def load_babi_dialogues(in_file_name):
    with getreader('utf-8')(open(in_file_name)) as task_in:
        task_content = task_in.read()
    dialogs = [
        filter(lambda line: len(line.strip()), dialog.split('\n'))
        for dialog in task_content.split('\n\n')
    ]
    dialogs = filter(len, dialogs)
    return dialogs


def make_tasks(in_dialogues, in_dst_root, dataset_filename):
    dialogues_per_worker = max(
        len(in_dialogues) / CONFIG['workers_number'],
        len(in_dialogues) % CONFIG['workers_number']
    )
    tasks = []
    for start_dialogue_index in xrange(0, len(in_dialogues), dialogues_per_worker):
        worker_dialogues = in_dialogues[start_dialogue_index:start_dialogue_index + dialogues_per_worker]
        worker_filename = dataset_filename + str(start_dialogue_index)
        with open(os.path.join(CONFIG['tmp_folder'], worker_filename), 'w') as task_out:
            print >>task_out, '\n\n'.join([
                '\n'.join(dialogue_lines)
                for dialogue_lines in worker_dialogues
            ])
            print >>task_out, ''
        start_dialogue_index += dialogues_per_worker
        tasks.append((
            os.path.join(CONFIG['tmp_folder'], worker_filename),
            os.path.join(in_dst_root, worker_filename)
        ))
    return tasks


def process_corpus(in_src_root, in_dst_root):
    if not os.path.exists(CONFIG['tmp_folder']):
        os.makedirs(CONFIG['tmp_folder'])

    workers_pool = Pool(processes=CONFIG['workers_number'])
    for filename in os.listdir(in_src_root):
        if 'task1-API-calls' not in filename:
            continue
        dialogues = load_babi_dialogues(os.path.join(in_src_root, filename))
        tasks = make_tasks(dialogues, in_dst_root, filename)
        results = workers_pool.map(process_single_file, tasks)

    group_partial_results(in_dst_root)
    shutil.rmtree(CONFIG['tmp_folder'])


def group_partial_results(in_root):
    files_map = defaultdict(lambda: [])
    for filename in os.listdir(in_root):
        dataset, ext, start_dialogue = filename.partition('.txt')
        files_map[dataset + ext].append((int(start_dialogue), filename))
    for dataset, files in files_map.iteritems():
        with open(os.path.join(in_root, dataset), 'w') as dataset_out:
            for start_dialogue, filename in sorted(files, key=itemgetter(0)):
                with open(os.path.join(in_root, filename)) as dataset_in:
                    print >>dataset_out, dataset_in.read()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'Usage: {} <input root> <output root>'.format(
            os.path.basename(__file__)
        )
        exit()
    process_corpus(sys.argv[1], sys.argv[2])

