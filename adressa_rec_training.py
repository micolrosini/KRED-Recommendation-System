"""
########################################################################################################################
This file is for the execution of the Data Enrichment with News Reading Time extension of the KRED model

The model is trained on the Adressa dataset to perform a user2item recommendation task on norwegian news articles

The macro-steps performed are the following:
1. Environment setup
    - Import libraries
    - Load config.yaml file
    - Define input data paths
2. Data preparation
    - Load existing files
    - Extracting existing embeddings
    - Build user history and behaviors
    - Building adjacency matrix
    - Fix a "missing keys" problem of the dataset
3. Model training and testing
    - Selection between:
        * single model training and testing with hyper-parameters specified in config.yaml
        * grid search on a (small) grid of hyper-parameters
########################################################################################################################
"""
"""
1. Environment setup
    - Import libraries
    - Load config.yaml file
    - Define input data paths
"""
from parse_config import ConfigParser
import torch
import os
import numpy as np
import pandas as pd
import random
from time import time
from utils.adressa_util import *
import argparse
from train_test_adressa import *


print(f'\nIt is installed Pytorch V. {torch.__version__}')

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
config['trainer']['adressa_adaptation'] = 'True'
config['trainer']['movies_adaptation'] = 'False'

config['model']['document_embedding_dim'] = 768

# set folder and files path
path_entity = config['data']['entities_addressa']
path_relation = config['data']['relations']
entity_file= 'entities_embedding'
relation_file= 'relations_embedding'
data_path = './data/'
adj_path = data_path + 'addressa_adj_matrix.txt'

"""
2. Data preparation
    - Load existing files
    - Extracting existing embeddings
    - Build user history and behaviors
    - Building adjacency matrix
    - Fix a "missing keys" problem of the dataset
"""

# Creating a dictionary with key the wikidata id and value the corresponding id,
# the id is a number that goes from 1 to the length of "entities"\  \
print(config['data']['entity2id_adressa'])
if os.path.exists(config['data']['entity2id_adressa']):
    print('Dictionary mapping entities alredy exist.')
else:
    entities_dict = build_entity_id_dict_from_file(config)
    save_dict_to_txt(entities_dict, config['data']['entity2id_adressa'])

#This will be a dictionary with key the wikidata id and value the index in the 'addressa_entity_embedding' which corresponds to the relative embedding
ad_entity2embedd = {}

# function to obtain the embeddings vector and a dictionary that tells you the index of the embedding vectors for each wikidata id
ad_entity2embedd, ad_entity_embedding = get_addressa_entities_embedding(config, ad_entity2embedd)

# Read the TSV file with relationships
df_relations = pd.read_csv(path_relation, sep='\t', header=None)

# List of relationships (using the second column)
relations_adr = df_relations.iloc[:, 1].drop_duplicates().tolist()

# Obtaining a dictionary with wikidata id of all the relations as key and the id as values.
relations_dict = build_dictionary_from_list(relations_adr)

# Obtaining a dictionary with wikidata id of all the relations as key and the id as values.
relations = relation2id_addressa(config)

# List of all the relations that are in the Adressa dataset
adr_relations = get_addressa_relations(config)

# Dictionary with key the wikidata id of the relations and index the corresponding index of the news dataset
adr_relation2id = {}
for adr_r in adr_relations:
    adr_relation2id[adr_r] = relations[adr_r]

if os.path.exists(path_entity.split('/')[0] + '/' + path_entity.split('/')[1] + '/' + entity_file + '.vec'):
    print(f'The file {entity_file} exists in the folder {data_path}.')

if os.path.exists(path_entity.split('/')[0] + '/' + path_entity.split('/')[1] + '/' + relation_file + '.vec'):
    print(f'The file {relation_file} exists in the folder {data_path}.')

if os.path.exists(adj_path):
    print(f'The file adjacency matrix exists in the folder {data_path}.')
else:
    entity_adj, relation_adj = addressa_construct_adj(config, ad_entity2embedd, adr_relation2id)
    print('\nSaving adjacency matrix file in: ' + adj_path)
    with open(adj_path, 'w') as file:
        for element in entity_adj:
            if str(element) != '[]':
                file.write(str(element) + '\n')
        file.write('relation_adj\n')
        for element in relation_adj:
            if str(element) != '[]':
                file.write(str(element) + '\n')

# List of embedding vector for each relation
relation_embeddings = get_addressa_relations_embeddings(config)

entity_adj, relation_adj = addressa_construct_adj(config, ad_entity2embedd, adr_relation2id)

# open behaviours and extract urls
df_beahviours = pd.read_csv(config["data"]["train_adressa_behaviour"], delimiter='\t', header=None)

# Build test and train data
train_adressa_behaviour, test_adressa_behaviour = get_behavior_train_test(config)

# Build user histroy dictionary for both train and test user behaviours
user_history_dict = build_user_history_adressa(config, train_adressa_behaviour, test_adressa_behaviour)

train_data, dev_data = get_adressa_user2item_data(config, train_adressa_behaviour=train_adressa_behaviour, test_adressa_behaviour=test_adressa_behaviour)

# Select urls of news "clicked" by users
train_news = train_data['train_urls']
test_news = dev_data['dev_urls']

if os.path.exists(config['data']['valid_news_addressa']) and os.path.exists(config['data']['train_news_addressa']):
    print('\nNews Adressa test and train tsv already exist\n')
else:
    _, _ = create_train_test_adressa_datasets(config, train_urls=train_news, test_urls=test_news)

ad_features, max_entity_freq, max_entity_pos, max_entity_type = build_news_addressa_features(config,ad_entity2embedd)

ad_entity_embedding = torch.FloatTensor(np.array(ad_entity_embedding))
relation_embeddings = torch.FloatTensor(np.array(relation_embeddings))

data = [user_history_dict, ad_entity_embedding, relation_embeddings, entity_adj,relation_adj, ad_features,\
max_entity_freq, max_entity_pos, max_entity_type, train_data, dev_data]

ids_to_not_remove = []
for el in data[-1]['item1']:
    if el in data[0].keys():
        ids_to_not_remove.append(el)

data[-1]['item1'] = ids_to_not_remove

test_data = data[-1]
print("Data loaded - ready for training")

"""
3. Model training and testing
    - Selection between:
        * single model training and testing with hyper-parameters specified in config.yaml
        * grid search on a (small) grid of hyper-parameters
"""

ENABLE_GRID_SEARCH = True

if ENABLE_GRID_SEARCH:

    print("Starting grid search for hyper-parameters optimization:")

    num_epochs_values = [5]
    batch_sizes_values = [128, 64]
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

                single_task_training(config, data)  # train a new model with selected hyper-params (user2item)
                auc_score, ndcg_score = testing_adressa(test_data, config)   # test model gets performance scores

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
    # Test single task training for KRED model on Adressa dataset
    single_task_training(config, data)  # user2item
    # Test on validation data
    auc_score, ndcg_score = testing_adressa(test_data, config)
