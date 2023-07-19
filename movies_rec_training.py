"""
########################################################################################################################
This file is for the execution of the Domain Adaptation to Movies Recommendations extension of the KRED model

The model is trained on the MindReader dataset to perform a user2item recommendation task on movies items

The macro-steps performed are the following:
1. Environment setup
    - Import libraries
    - Load config.yaml file
2. Data loading
    - Load dataset
    - Fix a "missing keys" problem of the dataset
3. Model training and testing
    - Selection between:
        * single model training and testing with hyper-parameters specified in config.yaml
        * grid search on a (small) grid of hyper-parameters (18 models trained, ~8h in total with Google Colab Pro)
########################################################################################################################
"""

"""
1. Environment setup
    - Import libraries
    - Load config.yaml file
"""

import os
import sys
sys.path.append('')
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

# The following parameters define which of the extensions are used,
#   by setting them to False the original KRED model is executed
config['trainer']['movies_adaptation'] = 'True'
config['trainer']['adressa_adaptation'] = 'False'

"""
2. Data loading
    - Load dataset
    - Fix a "missing keys" problem of the dataset
"""

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

"""
3. Model training and testing
    - Selection between:
        * single model training and testing with hyper-parameters specified in config.yaml
        * grid search on a (small) grid of hyper-parameters (18 models trained, ~8h in total with Google Colab Pro)
"""

ENABLE_GRID_SEARCH = True

if ENABLE_GRID_SEARCH:

    print("Starting grid search for hyper-parameters optimization:")

    num_epochs_values = [5, 7, 10]
    batch_sizes_values = [64, 128]
    learning_rates_values = [0.00001, 0.00002, 0.00005]

    grid_search_results = list()

    for e in num_epochs_values:
        for b in batch_sizes_values:
            for lr in learning_rates_values:
                print('\n')
                print("Testing the following configuration:")
                print(f"Number of epochs: {e}")
                print(f"Batch size: {b}")
                print(f"Learning rate: {lr}")
                print('\n')

                config["trainer"]["epochs"] = e
                config["data_loader"]["batch_size"] = b
                config["optimizer"]["lr"] = lr

                single_task_training(config, data)
                test_data = data[-1]
                auc_score, ndcg_score = testing_movies(test_data, config)

                res = dict()
                res['epochs'] = e
                res['batch_size'] = b
                res['learning_rate'] = lr
                res['auc_score'] = auc_score
                res['ndcg_score'] = ndcg_score

                grid_search_results.append(res)

    for r in grid_search_results:
        print(r)

else:
    single_task_training(config, data)
    test_data = data[-1]
    auc_score, ndcg_score = testing_movies(test_data, config)
