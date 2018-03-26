import argparse
import os

from lib.babi import read_task, get_files_list


def build_argument_parser():
    result = argparse.ArgumentParser()
    result.add_argument('babi', help='bAbI Dialog tasks folder')
    result.add_argument('babi_plus', help='bAbI+ folder')
    result.add_argument('result', help='result folder')
    result.add_argument('--output_format', help='output format file(seq2seq/csv)', default='seq2seq')
    return result


def save_csv(in_utterance_pairs, in_result_file):
    with open(in_result_file, 'w') as result_out:
            print >> result_out, ';'.join(['babi_plus', 'babi'])
            for turn_pair in in_utterance_pairs:
                print >>result_out, ';'.join(
                    [turn_pair['babi_plus'], turn_pair['babi']]
                )


def save_seq2seq(in_utterance_pairs, in_output_folder):
    with open(os.path.join(in_output_folder, 'encoder.txt'), 'w') as result_enc:
        with open(os.path.join(in_output_folder, 'decoder.txt'), 'w') as result_dec:
            for turn_pair in in_utterance_pairs:
                print >>result_enc, turn_pair['babi_plus']
                print >>result_dec, turn_pair['babi']


def main(in_babi, in_babi_plus, in_result_folder, in_output_format):
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
                if babi_turn['agent'] == 'user' and babi_turn['text'].lower() != '<silence>':
                    result.append({
                        'babi': babi_turn['text'],
                        'babi_plus': babi_plus_turn['text']
                    })
        if in_output_format == 'csv':
            save_csv(result, os.path.join(in_result_folder, os.path.basename(babi_file)))
        elif in_output_format == 'seq2seq':
            result_folder = os.path.join(in_result_folder, os.path.basename(babi_file))
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)
            save_seq2seq(result, result_folder)


if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()
    main(args.babi, args.babi_plus, args.result, args.output_format)

