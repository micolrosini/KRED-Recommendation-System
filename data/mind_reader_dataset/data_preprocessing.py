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




parser = argparse.ArgumentParser(description='KRED')
parser.add_argument('-c', '--config', default="./config.yaml", type=str,
                    help='config file path (default: None)')
parser.add_argument('-r', '--resume', default=None, type=str,
                    help='path to latest checkpoint (default: None)')
parser.add_argument('-d', '--device', default=None, type=str,
                    help='indices of GPUs to enable (default: all)')

config = ConfigParser.from_args(parser)

# Take all the wikidata id of the movies entities
movies_entities = entities_movies(config) 

# Creating a dictionary with key the wikidata id and value the corresponding id
movies_entity2id = entity_to_id_movies(config, movies_entities)

# Creating the entity embedding for each entity in the dataset 'movies'
movies_entity_embedding = [np.zeros(config["model"]["entity_embedding_dim"])] # array with 100 zeros

# Creating a dictionary with key the wikidata id and value a number from 1 to... which correspond to the index of teh embedding
movies_entity2embedd = {}

# Creating the entity embedding for each entity in the dataset 'movies'
movies_relation_embedding = [np.zeros(config['model']['entity_embedding_dim'])]

# Comparing the entities in the dataset movies with the ones in the dataset of news
entities = entities_news(config)  # get all wikidata IDs of each entities in the title and abstract in the training news data
entity2id = entity_to_id(config, entities)  # get dict with key entity WikidataId and value id
entity_embedding = [np.zeros(config["model"]["entity_embedding_dim"])] # array with 100 zeros
entity2embedd = {}
entity_embedding, relation_embedding, entity2embedd =\
        construct_embedding_mind(config, entity2id, entity_embedding, entity2embedd)

# Filter the movies entity whose wikidata id is in news entities
entity2id_movies_filtered = {}
for wikidataid in entity2id.keys():
    if wikidataid in entities:
        entity2id_movies_filtered[wikidataid] = entity2id[wikidataid]

# Obtaining a dictionary with wikidata id of the relation as key and the id as values.
relations = relation2id_movies(config)

# Construct an embedding mind with the entities of the dataset movie that are in common with the dataset of news
movies_entity_embedding, movies_relation_embedding, movies_entity2embedd =\
        construct_embedding_mind(config, entity2id_movies_filtered, movies_entity_embedding , movies_entity2embedd)





