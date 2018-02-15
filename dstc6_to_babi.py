import json
import os
import sys


def main(in_src_file, in_dst_file):
    with open(in_src_file) as dialogues_in:
        dialogues = json.load(dialogues_in)
    processed_dialogues = []
    for dialogue in dialogues:
        if not dialogue['answer']['utterance'].startswith('api_call'):
            continue
        utterances = dialogue['utterances'][:]
        utterances.append(dialogue['answer']['utterance'])
        processed_dialogues.append(utterances)
    with open(in_dst_file, 'w') as dialogues_out:
        for dialogue in processed_dialogues:
            for turn, index in enumerate(xrange(0, len(dialogue), 2)):
                usr, sys = dialogue[index: index + 2]
                print >>dialogues_out, '{} {}\t{}'.format(turn + 1, usr, sys)
            print >>dialogues_out, ''


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print 'Usage: {} <dstc6 dialogues file> <result file>'.format(
            os.path.basename(__file__)
        )
        exit()
    main(sys.argv[1], sys.argv[2])
