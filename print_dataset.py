import sys

import pandas as pd

def print_entry(in_entry):
    return ' '.join('{}|{}'.format(token, tag) for token, tag in zip(in_entry['utterance'], in_entry['tags']))


table = pd.read_json(sys.argv[1])
entries = set([print_entry(entry) for _, entry in table.iterrows()])
print >>sys.stdout, '\n'.join(entries)

