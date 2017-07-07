from os import path
import json


CONFIG_FILE = path.join(path.dirname(__file__), 'babi_plus.json')

with open(CONFIG_FILE) as config_in:
    CONFIG = json.load(config_in)
