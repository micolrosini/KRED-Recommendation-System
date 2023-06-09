from parse_config import ConfigParser
import os
import pandas as pd
import random
from time import time
from utils.addressa_util import *
import argparse

# set folder and files path
path_entity = './data/entities-week.csv'
path_relation = './data/knowledge_graph_addressa.tsv'
entity_file= 'entities_embedding'
relation_file= 'relations_embedding'
data_path = './data/'
adj_path = data_path + 'addressa_adj_matrix.txt'

parser = argparse.ArgumentParser(description='KRED')
parser.add_argument('-c', '--config', default="./config.yaml", type=str,
                    help='config file path (default: None)')
parser.add_argument('-r', '--resume', default=None, type=str,
                    help='path to latest checkpoint (default: None)')
parser.add_argument('-d', '--device', default=None, type=str,
                    help='indices of GPUs to enable (default: all)')

config = ConfigParser.from_args(parser)


#creating entities and relations dictionary to use while building adjacent matrix
# Read csv file with entities wikiid
df_entities = pd.read_csv(path_entity, index_col=False)
# List of entities
entities = df_entities['wikiid'].tolist()
# Take all the wikidata id of the entities

# Creating a dictionary with key the wikidata id and value the corresponding id,
# the id is a number that goes from 1 to the lenght of "entities"
entities_dict = build_entity_id_dict(entities)

# Creating the list of vector which will contain the list of the entities' embeddings
ad_entity_embedding = [np.zeros(config["model"]["entity_embedding_dim"])] # array with 100 zeros

#This will be a dictionary with key the wikidata id and value the index in the 'movies_entity_embedding' which corresponds to the relative embedding
ad_entity2embedd = {}

# function to obtain the embeddings vector and a dictionary that teels you the index of the embedding vectors for each wikidata id
ad_entity2embedd, ad_entity_embedding = get_addressa_entities_embedding(config,ad_entity_embedding,ad_entity2embedd)


# Read the TSV file with relationships
df_relations = pd.read_csv(path_relation, sep='\t', header=None)

# List of relationships (using the second column)
relations_adr = df_relations.iloc[:, 1].drop_duplicates().tolist()
# Obtaining a dictionary with wikidata id of all the relations as key and the id as values.
relations_dict = build_entity_id_dict(relations_adr)

# Obtaining a dictionary with wikidata id of all the relations as key and the id as values.
relations = relation2id_addressa(config)

# list of all the relations that are in the addressa dataset
adr_relations = get_addressa_relations(config)

# dictionary with key the wikidata id of the relations and index the corresponding index of the news dataset
adr_relation2id = {}
for adr_r in adr_relations:
    adr_relation2id[adr_r] = relations[adr_r]

if os.path.exists(path_entity.split('/')[0] + '/' + path_entity.split('/')[1] + '/' + entity_file + '.vec'):
    print(f'The file {entity_file} exists in the folder {data_path}.')
else:
    print(f'The file {entity_file} does not exist in the folder {data_path}.')
    addressa_entity_embedding(path_entity, 'entities_embedding')
    print('\nSaving entity embedding file in: ' + path_entity.split('/')[0] + '/' + path_entity.split('/')[1] + '/' + entity_file + '.vec')

if os.path.exists(path_entity.split('/')[0] + '/' + path_entity.split('/')[1] + '/' + relation_file + '.vec'):
    print(f'The file {relation_file} exists in the folder {data_path}.')
else:
    addressa_relation_embedding(path_relation, relation_file)
    print('\nSaving entity embedding file in: '+ path_entity.split('/')[0] + '/' + path_relation.split('/')[1] + '/' + relation_file + '.vec')

if os.path.exists(adj_path):
    print(f'The file adjacency matrix exists in the folder {data_path}.')
else:
    entity_adj, relation_adj = addressa_construct_adj_mind(config, ad_entity2embedd, adr_relation2id)
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
relation_embeddings =get_addressa_relations_embeddings(config)

print(relation_embeddings)

ad_features, max_entity_freq, max_entity_pos, max_entity_type  = build_news_features_mind(config,ad_entity2embedd)
entity_adj, relation_adj = addressa_construct_adj_mind(config, ad_entity2embedd, adr_relation2id)
print(relation_adj)

#adr_behaviours = construct_movies_behaviours(config)

#user_adr_behaviours = list(adr_behaviours.keys())
#train_size = 0.8
#index_train = int(len(user_adr_behaviours)*train_size)
#train_adr_behaviour = user_adr_behaviours[:index_train]
#for i,el in enumerate(train_adr_behaviour):

#    train_adr_behaviour[i] = el.strip()

#test_adr_behaviour = user_adr_behaviours[index_train:]
#for i,el in enumerate(test_adr_behaviour):
#    test_adr_behaviour[i] = el.strip()