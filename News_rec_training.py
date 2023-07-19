"""
########################################################################################################################
This file is for the execution of the original KRED: Knowledge-Aware Document Representation for News Recommendations
    base model execution.

The base model is trained on the MIcrosoft News Dataset (MIND) small version to perform a user2item recommendation task

The macro-steps performed are the following:
1. Environment setup
    - Import libraries
    - Set input data paths
    - Load config.yaml file
2. Data loading
    - Load dataset
    - Limit validation dataset size for a faster validation phase
3. Model training and testing
    - Selection between:
        * single model training and testing with hyper-parameters specified in config.yaml
        * grid search on a (small) grid of hyper-parameters (4 models trained, ~10h in total with Google Colab Pro)
########################################################################################################################
"""

"""
1. Environment setup
    - Import libraries
    - Set input data paths
    - Load config.yaml file
"""
import os
import sys
sys.path.append('')

import argparse
from parse_config import ConfigParser
from utils.util import *
from train_test import *

# Options: demo, small, large
MIND_type = 'small'
data_path = "./data/"

train_news_file = os.path.join(data_path, 'train', r'news.tsv')
train_behaviors_file = os.path.join(data_path, 'train', r'behaviors.tsv')
valid_news_file = os.path.join(data_path, 'valid', r'news.tsv')
valid_behaviors_file = os.path.join(data_path, 'valid', r'behaviors.tsv')
knowledge_graph_file = os.path.join(data_path, 'kg/kg', r'wikidata-graph.tsv')
entity_embedding_file = os.path.join(data_path, 'kg/kg', r'entity2vecd100.vec')
relation_embedding_file = os.path.join(data_path, 'kg/kg', r'relation2vecd100.vec')

mind_url, mind_train_dataset, mind_dev_dataset, _ = get_mind_data_set(MIND_type)

kg_url = "https://kredkg.blob.core.windows.net/wikidatakg/"

if not os.path.exists(train_news_file):
    download_deeprec_resources(mind_url, os.path.join(data_path, 'train'), mind_train_dataset)
    
if not os.path.exists(valid_news_file):
    download_deeprec_resources(mind_url, \
                               os.path.join(data_path, 'valid'), mind_dev_dataset)

if not os.path.exists(knowledge_graph_file):
    download_deeprec_resources(kg_url, \
                               os.path.join(data_path, 'kg'), "kg")


parser = argparse.ArgumentParser(description='KRED')


parser.add_argument('-c', '--config', default="./config.yaml", type=str,
                    help='config file path (default: None)')
parser.add_argument('-r', '--resume', default=None, type=str,
                    help='path to latest checkpoint (default: None)')
parser.add_argument('-d', '--device', default=None, type=str,
                    help='indices of GPUs to enable (default: all)')

config = ConfigParser.from_args(parser)

# For debugging purposes:
# epochs = 1
# config['trainer']['epochs'] = epochs
# batch_size = 64
# config['data_loader']['batch_size'] = batch_size

train_type = "single_task"
task = "user2item"  # task should be within: user2item, item2item, vert_classify, pop_predict
epochs = config['trainer']['epochs']
config['trainer']['training_type'] = train_type
config['trainer']['task'] = task
config['trainer']['save_period'] = epochs/2
config['model']['document_embedding_dim'] = 768

# The following parameters define which of the extensions are used, 
# by setting them to False the original KRED model is executed
config['trainer']['movies_adaptation'] = 'False'
config['trainer']['adressa_adaptation'] = 'False'

if not os.path.isfile(f"{config['data']['sentence_embedding_folder']}/train_news_embeddings.pkl"):
  write_embedding_news("./data/train", config["data"]["sentence_embedding_folder"])

if not os.path.isfile(f"{config['data']['sentence_embedding_folder']}/valid_news_embeddings.pkl"):
  write_embedding_news("./data/valid", config["data"]["sentence_embedding_folder"])

"""
2. Data loading
    - Load dataset
    - Limit validation dataset size for a faster validation phase
"""

if not os.path.isfile(f"{data_path}/data_mind.pkl"):
  data = load_data_mind(config, config['data']['sentence_embedding_folder'])
  write_data_mind(config, data_path)
else:
  data = read_pickle(f"{data_path}/data_mind.pkl")

data = limit_user2item_validation_data(data, 10000)  # limit valid set size at valid phase due to exec time constraints
test_data = data[-1]
print("Data loaded, ready for training")

"""
3. Model training and testing
    - Selection between:
        * single model training and testing with hyper-parameters specified in config.yaml
        * grid search on a (small) grid of hyper-parameters (4 models trained, ~10h in total with Google Colab Pro)
"""

ENABLE_GRID_SEARCH = True

if ENABLE_GRID_SEARCH:

    print("Starting grid search for hyper-parameters optimization:")

    num_epochs_values = [5]
    batch_sizes_values = [64, 128]
    learning_rates_values = [0.00002, 0.00005]

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

                single_task_training(config, data)  # train a new model with selected hyper-params (user2item base KRED)
                auc_score, ndcg_score = testing_base_model(test_data, config)   # test model gets performance scores

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
    single_task_training(config, data)  # user2item
    auc_score, ndcg_score = testing_base_model(test_data, config)
