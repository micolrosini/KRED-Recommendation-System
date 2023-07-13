import json
import torch
import random
import numpy as np
import math
import sys
import pandas as pd

sys.path.append('')
import pickle
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
from sentence_transformers import SentenceTransformer
import requests
import zipfile
# from logger.logger import *
from tqdm import tqdm
import os
import ast
from utils.util import *
import csv
from train_test import *

from sklearn.model_selection import train_test_split


def write_pickle(data, fname):
    with open(fname, 'wb') as file:
        pickle.dump(data, file)


def read_pickle(fname):
    with open(fname, 'rb') as file:
        data = pickle.load(file)
    return data


def addressa_entity_embedding(file_to_embed, output_file_name):
    # Read csv file with entities wikiid
    df_entities = pd.read_csv(file_to_embed, index_col=False)
    # List of entities
    entities = df_entities['wikiid']
    # Load the SentenceTransformer model
    model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

    print('\nStart entity embedding generation...\n')
    # Generate embeddings for the entities
    embeddings = []

    for i, entity in enumerate(tqdm(entities)):
        # Encode the entity
        embedding = model.encode(entity)
        embeddings.append(embedding)

    # Output .vec file path
    output_path = './data/' + output_file_name + '.vec'

    # Write embeddings to .vec file
    with open(output_path, 'w') as f:
        for entity, embedding in zip(entities, embeddings):
            embedding_str = '\t'.join(str(val) for val in embedding)
            f.write(f'{entity}\t{embedding_str}\n')

    print('\nEmbedding process finished.')


def addressa_relation_embedding(file_to_embed, output_file_name):
    # Read csv file with entities wikiid
    df_entities = pd.read_csv(file_to_embed, index_col=False)
    # List of entities
    entities = df_entities['wikiid']

    # build relationships dictionary
    # Read the TSV file with relationships
    df_relations = pd.read_csv(file_to_embed, sep='\t', header=None)
    # List of relationships (using the second column)
    relations = df_relations.iloc[:, 1].drop_duplicates()

    # Load the SentenceTransformer model
    model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

    print('\nStart relationship embedding generation...\n')

    # Generate embeddings for the relationships
    embeddings = []

    for i, relation in enumerate(tqdm(relations)):
        # Encode the relation
        embedding = model.encode(relation)
        embeddings.append(embedding)

    # Output .vec file path
    output_path = './data/' + output_file_name + '.vec'

    # Write embeddings to .vec file
    with open(output_path, 'w') as f:
        for relation, embedding in zip(relations, embeddings):
            embedding_str = '\t'.join(str(val) for val in embedding)
            f.write(f'{relation}\t{embedding_str}\n')

    print('\nEmbedding process finished.')


def entities_addressa(config):
    """
    Return all the wikidataIDs of each entities of the dataset 'mind_reader_dataset'

    """

    entities = set()

    # Read csv file with entities wikiid
    df_entities = pd.read_csv(config["data"]["entities_addressa"], index_col=False)
    # List of entities
    entities.append(entity for entity in df_entities['wikiid'])

    return entities


def entity_to_id_addressa(entities):
    """
    Return dictionary with entity `WikidataId' as key and entity id(numbers) as value. The entity id is the id of the entity in
    the file `entity2id.txt`.
    """
    entity2id = {}

    for entity_id, entity in enumerate(entities):
        entity2id[entity] = entity_id + 1  # increment the id by 1
    return entity2id


def relation2id_addressa(config):
    """
    Return a dictionary with the wikidataIDs of each property in the graph of the dataset movies as key and its corresponding id as value

    """
    relations = {}
    # Add the WikidataId from the train entities

    # build relationships dictionary
    # Read the TSV file with relationships
    df_relations = pd.read_csv(config["data"]["relations"], sep='\t', header=None)
    # List of relationships (using the second column)
    relations_list = df_relations.iloc[:, 1].drop_duplicates()
    for i, rel in enumerate(relations_list):
        wikidatid = rel
        relationid = i
        relations[wikidatid] = int(relationid) + 1
    return relations


def get_addressa_entities_embedding(config, addressa_entity_embedding, addressa_entity2embedd):
    """
    Return a dictionary with the wikidataIDs of the dataset addressa as key and its corresponding index in the entity embedding list as value , and it also return the list with the embeddings for each entity

    """

    embeddings = open(config['data']['addressa_entity_embedding'], 'r', encoding='utf-8')
    i = 0
    for line in embeddings:
        embedding = line.split()

        addressa_entity2embedd[embedding[0]] = i + 1
        addressa_entity_embedding.append(embedding[1:])
        i += 1
    return addressa_entity2embedd, addressa_entity_embedding


def get_addressa_relations(config):
    """
    Function to filter only the relations that are in the news dataset

    """

    relations = set()
    with open(config['data']['knowledge_graph_addressa'], 'r', encoding='utf-8') as triple_graph:
        for line in triple_graph:
            triple = line.split('\t')
            relation = triple[1]

            relations.add(relation)
        return relations


def get_addressa_relations_embeddings(config):
    """
    Return the embedding of all the relations of the news dataset

    """

    relation_embedding = [np.zeros(config['model']['entity_embedding_dim'])]
    movies_relation2id_corrected = {}
    with open(config['data']['addressa_relation_embedding'], 'r', encoding='utf-8') as fp_relation_embedding:
        for line in fp_relation_embedding:
            linesplit = line.strip().split('\t')
            linesplit = [float(i) for i in linesplit[1:]]
            relation_embedding.append(linesplit)
        return relation_embedding


def build_entity_id_dict(wikiid_list):
    entity_id_dict = {}
    for i, wikiid in enumerate(wikiid_list):
        entity_id_dict[wikiid] = i
    return entity_id_dict


def addressa_construct_adj_mind(config, entities_dict, relations_dict):
    print('constructing adjacency matrix ...')
    # ids = set(entities_dict.values()) # index of the corresponding embedding vector

    with open(config['data']['knowledge_graph_addressa'], 'r', encoding='utf-8') as graph_file_fp:
        kg = {}  # dictionary with entity(numbers from 1 to ...) as keys and relation+tail as values, also tails that are not already encoded(their head is not in the train ) become keys with values = head+relation
        for line in tqdm(graph_file_fp):  # check how values are encoded here
            linesplit = line.split('\t')
            head = linesplit[0]
            relation = linesplit[1]
            tail = linesplit[2]
            if head.strip() in list(entities_dict.keys()):
                if tail.strip() in list(entities_dict.keys()):

                    if head not in kg:  # if entity head is not in the 'kg' dictionary
                        kg[head] = []
                    kg[head].append((tail, relation))

            if tail.strip() in list(entities_dict.keys()):
                if head.strip() in list(entities_dict.keys()):

                    if tail not in kg:
                        kg[tail] = []
                    kg[tail].append((head, relation))

    entity_num = len(entities_dict)  # number of entities data in the training data
    entity_adj = []
    relation_adj = []
    # id2entity = {v: k for k, v in entities_dict.items()}
    for i in range(entity_num + 1):
        entity_adj.append([])  # list of a number of lists equal to the entities in the data train
        relation_adj.append([])
    for i in range(config['model']['entity_neighbor_num']):  # maximum number of neighbours
        entity_adj[0].append(0)
        relation_adj[0].append(0)

    for key in tqdm(kg.keys()):
        new_key = entities_dict[key.strip()]

        while len(entity_adj[int(new_key)]) < config['model']['entity_neighbor_num']:
            entity_adj.append([])
            relation_adj.append([])

        while len(entity_adj[new_key]) < config['model']['entity_neighbor_num']:
            # for _ in range(config['model']['entity_neighbor_num']):
            i = random.randint(0, len(kg[key]) - 1) # taking a random tail+relation from the list of values of the head entity
            # index of the corresponding embedding vector
            entity_adj[int(new_key)].append(entities_dict[kg[key][i][0].strip()])

            # relation [number from 1 to..] append relation ( is a number? or an embedding?)
            relation_adj[int(new_key)].append(relations_dict[kg[key][i][1]])

    return entity_adj, relation_adj


def obtain_train_test_movies(train_movies_behaviour, movies_behaviours ):

    """

    Return two different lists of users ids for the training and validation phase
    
    """


    train_movies = []
    test_movies = []
    for k,v in movies_behaviours.items():
        
        
        if k.strip() in train_movies_behaviour:
            
            for el in v:
                
                if el[1].strip() == 'True':
                    train_movies.append(el[0])
        else:
            for el in v:
                if el[1].strip() == 'True':
                    test_movies.append(el[0])
    return train_movies,test_movies


def build_news_addressa_features_mind(config, entity2embedd):
    # There are 4 features for each news: postion, freq, category, embeddings
    news_features = {}
    news_feature_dict = {}
    embedding_folder = config['data']['sentence_embedding_folder']
    # Load sentence embeddings from file if present
    with open(config['data']['train_news_addressa'], 'r', encoding='utf-8') as fp_train_news:
        if embedding_folder is not None:
            train_sentences_embedding = read_pickle(embedding_folder + "train_news_addressa_embeddings.pkl")
        for i, line in enumerate(fp_train_news):
            fields = line.strip().split('\t')
            # vert and subvert are the category and subcategory of the news
            url, entity_info, keywords, title, subvert, vert = fields
            entity_info_title = entity_info
            entity_info_abstract = []
            if embedding_folder is not None:
                news_feature_dict[url] = (train_sentences_embedding[i], entity_info_title, entity_info_abstract, vert, subvert)
            else:
                news_feature_dict[url] = (title, entity_info_title, entity_info_abstract, vert, subvert)
    # Load sentence embeddings from file if present
    with open(config['data']['valid_news_addressa'], 'r', encoding='utf-8') as fp_dev_news:
        if embedding_folder is not None:
            valid_sentences_embedding = read_pickle(embedding_folder + "valid_news_addressa_embeddings.pkl")
        for i, line in enumerate(fp_dev_news):
            fields = line.strip().split('\t')
            url, entity_info, keywords, title, subvert, vert = fields
            entity_info_title = entity_info
            entity_info_abstract = []
            if embedding_folder is not None:
                news_feature_dict[url] = (valid_sentences_embedding[i], entity_info_title, entity_info_abstract, vert, subvert)
            else:
                news_feature_dict[url] = (title, entity_info_title, entity_info_abstract, vert, subvert)

    # deal with doc feature
    entity_type_dict = {}
    entity_type_index = 1

    if embedding_folder is None:
        model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

    for i, news in enumerate(news_feature_dict):
        if embedding_folder is not None:
            sentence_embedding = news_feature_dict[news][0]  # Use the stored sentence embedding
        else:
            sentence_embedding = model.encode(news_feature_dict[news][0])
        news_entity_feature_list = []
        title_entity_json = json.loads(news_feature_dict[news][1])
        abstract_entity_json = json.loads(news_feature_dict[news][2])
        news_entity_feature = {}
        for item in title_entity_json:
            if item['Type'] not in entity_type_dict:
                entity_type_dict[item['Type']] = entity_type_index
                entity_type_index = entity_type_index + 1
            news_entity_feature[item['WikidataId']] =\
                (len(item['OccurrenceOffsets']), 1, entity_type_dict[item['Type']]) #entity_freq, entity_position, entity_type
        for item in abstract_entity_json:
            if item['WikidataId'] in news_entity_feature:
                news_entity_feature[item['WikidataId']] =\
                    (news_entity_feature[item['WikidataId']][0] + len(item['OccurrenceOffsets']), 1, entity_type_dict[item['Type']])
            else:
                if item['Type'] not in entity_type_dict:
                    entity_type_dict[item['Type']] = entity_type_index
                    entity_type_index = entity_type_index + 1
                news_entity_feature[item['WikidataId']] =\
                    (len(item['OccurrenceOffsets']), 2, entity_type_dict[item['Type']])  # entity_freq, entity_position, entity_type
        for entity in news_entity_feature:
            if entity in entity2embedd:
                news_entity_feature_list.append(
                    [entity2embedd[entity], news_entity_feature[entity][0], news_entity_feature[entity][1], news_entity_feature[entity][2]]
                )
        news_entity_feature_list.append([0, 0, 0, 0])
        if len(news_entity_feature_list) > config['model']['news_entity_num']:
            news_entity_feature_list = news_entity_feature_list[:config['model']['news_entity_num']]
        else:
            for i in range(len(news_entity_feature_list), config['model']['news_entity_num']):
                news_entity_feature_list.append([0, 0, 0, 0])
        news_feature_list_ins = [[],[],[],[],[]]
        for i in range(len(news_entity_feature_list)):
            for j in range(4):
                news_feature_list_ins[j].append(news_entity_feature_list[i][j])
        news_feature_list_ins[4] = sentence_embedding
        news_features[news] = news_feature_list_ins
    news_features["N0"] = [[],[],[],[],[]]
    for i in range(config['model']['news_entity_num']):
        for j in range(4):
            news_features["N0"][j].append(0)
    news_features["N0"][4] = np.zeros(config['model']['document_embedding_dim'])
    return news_features, 100, 10, 100


def build_user_history_adressa(config, train_adressa_behaviour, test_adressa_behaviour):
    """
    :param train_adressa_behaviour: list of strings, each row is a line of behaviors.tsv (train split)
    :param test_adressa_behaviour: list of strings, each row is a line of behaviors.tsv (test split)
    :return: a dictionary for each users with the value the films that are rated positively
    """
    user_history_dict = {}

    for line in train_adressa_behaviour:
        index, user_id, imp_time, history, behavior = line.strip().split('\t')
        if len(history.split(' ')) >= config['model']['user_his_num']:
            user_history_dict[user_id + "_train"] = history.split(' ')[:config['model']['user_his_num']]
        else:
            user_history_dict[user_id + "_train"] = history.split(' ')
            for i in range(config['model']['user_his_num'] - len(history.split(' '))):
                user_history_dict[user_id + "_train"].append("N0")
            if user_history_dict[user_id + "_train"][0] == '':
                user_history_dict[user_id + "_train"][0] = 'N0'

    for line in test_adressa_behaviour:
        index, user_id, imp_time, history, behavior = line.strip().split('\t')
        if len(history.split(' ')) >= config['model']['user_his_num']:
            user_history_dict[user_id + "_dev"] = history.split(' ')[:config['model']['user_his_num']]
        else:
            user_history_dict[user_id + "_dev"] = history.split(' ')
            for i in range(config['model']['user_his_num'] - len(history.split(' '))):
                user_history_dict[user_id + "_dev"].append("N0")
            if user_history_dict[user_id + "_dev"][0] == '':
                user_history_dict[user_id + "_dev"][0] = 'N0'

    return user_history_dict


def get_behavior_train_test(config, train_split_size=0.8):
    """
    :param config: Allows to access the input file
    :param train_split_size: Allows to specify different values of Train Split, must stay within [0.0 , 1,0]
    :return: Two lists of rows (strings) consisting of the train and test splits of the original file
    """

    if train_split_size < 0 or train_split_size > 1:
        train_split_size = 0.8

    with open(config["data"]["train_adressa_behaviour"]) as behaviour:

        rows = behaviour.read().splitlines()

        train_adressa_behaviour, test_adressa_behaviour = train_test_split(rows, train_size=train_split_size, random_state=42)

    return train_adressa_behaviour, test_adressa_behaviour


def get_adressa_user2item_data(config, train_adressa_behaviour, test_adressa_behaviour):
    """
    :param train_adressa_behaviour: list of strings, each row is a line of behaviors.tsv (train split)
    :param test_adressa_behaviour: list of strings, each row is a line of behaviors.tsv (test split)
    :return: two dictionaries with the needed positive lst, negative lst and remaining data
    """
    negative_num = config['trainer']['train_neg_num']
    train_data = {}
    user_id = []
    news_id = []
    label = []

    for line in train_adressa_behaviour:
        index, userid, imp_time, history, behavior = line.strip().split('\t')

        behavior = behavior.split(' ')
        positive_list = []
        negative_list = []
        for news in behavior:
            segments = news.split('-')
            news_label = segments[-1]
            segments.pop()
            newsid = ''
            for s in segments:
                newsid = newsid + s + '-'
            newsid = newsid[:-1]
            if news_label == "1":
                positive_list.append(newsid)
            else:
                negative_list.append(newsid)
        for pos_news in positive_list:
            user_id.append(userid + "_train")
            if len(negative_list) >= negative_num:
                neg_news = random.sample(negative_list, negative_num)
            else:
                neg_news = negative_list
                for i in range(negative_num - len(negative_list)):
                    neg_news.append("N0")
            all_news = neg_news
            all_news.append(pos_news)
            news_id.append(all_news)
            label.append([])
            for i in range(negative_num):
                label[-1].append(0)
            label[-1].append(1)

    train_data['item1'] = user_id
    train_data['item2'] = news_id
    train_data['label'] = label

    dev_data = {}
    session_id = []
    user_id = []
    news_id = []
    label = []

    for line in test_adressa_behaviour:
        index, userid, imp_time, history, behavior = line.strip().split('\t')
        behavior = behavior.split(' ')
        for news in behavior:
            segments = news.split('-')
            news_label = segments[-1]
            segments.pop()
            newsid = ''
            for s in segments:
                newsid = newsid + s + '-'
            newsid = newsid[:-1]
            session_id.append(index)
            user_id.append(userid + "_dev")
            if news_label == "1":
                news_id.append(newsid)
                label.append(1.0)
            else:
                news_id.append(newsid)
                label.append(0.0)

    dev_data['item1'] = user_id
    dev_data['session_id'] = session_id
    dev_data['item2'] = news_id
    dev_data['label'] = label

    return train_data, dev_data


def load_data_mind_adressa(config):
    """
    :return data: list with all the final data needed to run the model
    """
    # Take all the wikidata id of the news entities
    entities = entities_addressa(config)

    # Creating a dictionary with key the wikidata id and value the corresponding id, the id is a number that goes from 1 to the lenght of "movies_entities"
    entity2id = entity_to_id_addressa(entities)

    # Creating the list of vector which will contain the list of the entities' embeddings
    entity_embedding = [np.zeros(config["model"]["entity_embedding_dim"])]  # array with 100 zeros

    # This will be a dictionary with key the wikidata id and value the index in the 'movies_entity_embedding' which corresponds to the relative embedding
    entity2embedd = {}

    entity2embedd, entity_embedding = get_addressa_entities_embedding(config, entity_embedding,
                                                                                  entity2embedd)

    # Obtaining a dictionary with wikidata id of all the relations as key and the id as values.
    relations = relation2id_addressa(config)

    # list of all the relations that are in the mind_reader dataset
    news_relations = get_addressa_relations(config)

    # dictionary with key the wikidata id of the relations and index the corresponding index of the news dataset
    relation2id = {}
    for news_r in news_relations:
        relation2id[news_r] = relations[news_r]

        # List of embedding vector for each relation
    relation_embeddings = get_addressa_relations_embeddings(config)

    e_a, r_a = addressa_construct_adj_mind(config, entity2embedd, relation2id)

    # build news adessa features
    news_features, max_entity_freq, max_entity_pos, max_entity_type = build_news_addressa_features_mind(config, entity2embedd,entity_embedding)

    train_adressa_behaviour, test_adressa_behaviour = get_behavior_train_test(config)
    user_history_dict = build_user_history_adressa(config, train_adressa_behaviour, test_adressa_behaviour)

    train_data, dev_data = get_adressa_user2item_data(config, train_adressa_behaviour, test_adressa_behaviour)

    for i, v in enumerate(entity_embedding):
        emb_adr = []
        for index, number in enumerate(v):
            emb_adr.append(float(number))
        entity_embedding[i] = np.array(emb_adr)

    entity_embedding = torch.FloatTensor(np.array(entity_embedding))
    relation_embeddings = torch.FloatTensor(np.array(relation_embeddings))

    data = [user_history_dict, entity_embedding, relation_embeddings, e_a, r_a, news_features, \
            max_entity_freq, max_entity_pos, max_entity_type, train_data, dev_data]

    ids_to_not_remove = []
    for el in data[-1]['item1']:
        if el in data[0].keys():
            ids_to_not_remove.append(el)

    data[-1]['item1'] = ids_to_not_remove

    return data

