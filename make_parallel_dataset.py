import argparse
import os

from lib.babi import read_task, get_files_list


def build_argument_parser():
    result = argparse.ArgumentParser()
    result.add_argument('babi', help='bAbI Dialog tasks folder')
    result.add_argument('babi_plus', help='bAbI+ folder')
    result.add_argument('result', help='result folder')
    return result


def main(in_babi, in_babi_plus, in_result_folder):
    babi_files = get_files_list(in_babi, 'task1-API-calls')
    babi_plus_files = get_files_list(in_babi_plus, 'task1-API-calls')

    if not os.path.exists(in_result_folder):
        os.makedirs(in_result_folder)

    for babi_file, babi_plus_file in zip(babi_files, babi_plus_files):
        babi = read_task(babi_file)
        babi_plus = read_task(babi_plus_file)

        result = []
        for babi_dialogue, babi_plus_dialogue in zip(babi, babi_plus):
            for babi_turn, babi_plus_turn in zip(babi_dialogue[1], babi_plus_dialogue[1]):
                if babi_turn['agent'] == 'user':
                    result.append({
                        'babi': babi_turn['text'],
                        'babi_plus': babi_plus_turn['text']
                    })

        with open(os.path.join(in_result_folder, os.path.basename(babi_file)), 'w') as result_out:
            print >> result_out, ';'.join(['babi_plus', 'babi'])
            for turn_pair in result:
                print >> result_out, ';'.join(
                    [turn_pair['babi_plus'], turn_pair['babi']])


if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()
    main(args.babi, args.babi_plus, args.result)
