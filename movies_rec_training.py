import os
import sys
sys.path.append('')
import os
from parse_config import ConfigParser
import argparse
import ast
from utils.util import *
import numpy as np
import random
import csv
from train_test import *
from utils.movies_util import *


parser = argparse.ArgumentParser(description='KRED')
parser.add_argument('-c', '--config', default="./config.yaml", type=str,
                    help='config file path (default: None)')
parser.add_argument('-r', '--resume', default=None, type=str,
                    help='path to latest checkpoint (default: None)')
parser.add_argument('-d', '--device', default=None, type=str,
                    help='indices of GPUs to enable (default: all)')

config = ConfigParser.from_args(parser) 
config['trainer']['movies_adaptation'] = 'True'

if not os.path.isfile("./data/mind_reader_dataset/movies.pkl"):
    data = load_movies_data_mind
    write_pickle(data, "./data/mind_reader_dataset/movies.pkl")
else: 
    data = read_pickle("./data/mind_reader_dataset/movies.pkl")

ids_to_not_remove = []
for el in data[-1]['item1']:
        if el in data[0].keys():
            ids_to_not_remove.append(el)

data[-1]['item1'] = ids_to_not_remove



single_task_training(config, data)