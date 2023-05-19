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




parser = argparse.ArgumentParser(description='KRED')
parser.add_argument('-c', '--config', default="./config.yaml", type=str,
                    help='config file path (default: None)')
parser.add_argument('-r', '--resume', default=None, type=str,
                    help='path to latest checkpoint (default: None)')
parser.add_argument('-d', '--device', default=None, type=str,
                    help='indices of GPUs to enable (default: all)')

config = ConfigParser.from_args(parser) 
#config = parser.parse_args(sys.argv)


# Take all the wikidata id of the movies entities
movies_entities = entities_movies(config) 

# Entities that I must not embedd
entities_to_NOT_embedd = ["Q16398403","Q20876309", "Q7152054","Q5967378","Q11330334","Q2996599","Q7928604","Q12579933","Q17345541","Q20726510","Q275473"]

# Final list of movies entities, all these wikidata id must be embedded
movies_entities= list(set(movies_entities)-set(entities_to_NOT_embedd))

# Creating a dictionary with key the wikidata id and value the corresponding id, the id is a number that goes from 1 to the lenght of "movies_entities"
movies_entity2id = entity_to_id_movies(movies_entities)

# Creating the loist of vector which will contain the list of the entities' embeddings
movies_entity_embedding = [np.zeros(config["model"]["entity_embedding_dim"])] # array with 100 zeros

# This will be a dictionary with key the wikidata id and value the index in the 'movies_entity_embedding' which corresponds to the relative embedding
movies_entity2embedd = {}

# function to obtain the embeddings vector and a dictionary that teels you the index of the embedding vectors for each wikidata id
def get_movies_entities_embedding(config,movies_entity_embedding,movies_entity2embedd):
    embeddings = open(config['data']['movies_entity_embedding'], 'r', encoding='utf-8')
    i = 0
    for line in embeddings:
        

            embedding = line.split()
            
            
            movies_entity2embedd[embedding[0]] = i +1
            movies_entity_embedding.append(embedding[1:])
            i += 1
    return movies_entity2embedd, movies_entity_embedding
    
movies_entity2embedd, movies_entity_embedding =  get_movies_entities_embedding(config,movies_entity_embedding,movies_entity2embedd)

# Obtaining a dictionary with wikidata id of all the relations as key and the id as values.
relations = relation2id_movies(config)


# Function to filter only the relations that are in the movies dataset

def get_movies_relations(config):
    relations = set()
    with open(config['data']['knowledge_graph_movies'] , 'r', encoding='utf-8') as triple_graph:
        for line in triple_graph:
            triple = line.split(',')
            relation = triple[1]
            
            relations.add(relation)
        return relations

# list of all the relations that are in the mind_reader dataset    
movies_relations = get_movies_relations(config)


# dictionary with key the wikidata id of the relations and index the corresponding index of the news dataset
movies_relation2id = {}
for movie_r in movies_relations:
    
    movies_relation2id[movie_r] = relations[movie_r] 
    



 # function to obtain the embedding of all the relations of the news dataset       

def get_movies_relations_embeddings(config) :

    relation_embedding = [np.zeros(config['model']['entity_embedding_dim'])]
    movies_relation2id_corrected = {}
    with open(config['data']['relation_embedding'], 'r', encoding='utf-8') as fp_relation_embedding:
        for line in fp_relation_embedding:
        
            linesplit = line.strip().split('\t')
            linesplit = [float(i) for i in linesplit]
            relation_embedding.append(linesplit)
        return relation_embedding

                    
            
 # List of embedding vector for each relation       
relation_embeddings =get_movies_relations_embeddings(config)

def movies_construct_adj_mind(config,movies_entity2embedd, movies_relation2id):
    print('constructing adjacency matrix ...')
    #ids = set(movies_entity2embedd.values()) # index of the corresponding embedding vector
    with open(config['data']['knowledge_graph_movies'], 'r', encoding='utf-8') as graph_file_fp:
        kg = {} # dictionary with entity(numbers from 1 to ...) as keys and relation+tail as values, also tails that are not already encoded(their head is not in the train ) become keys with values = head+relation
        for line in graph_file_fp: # check how values are encoded here
            linesplit = line.split(',')
            head = linesplit[0]
            relation = linesplit[1]
            tail = linesplit[2]
            if head.strip() in list(movies_entity2embedd.keys()):
                if tail.strip() in list(movies_entity2embedd.keys()):
                
                    if head not in kg: # if entity head is not in the 'kg' dictionary
                        kg[head] = []
                    kg[head].append((tail, relation))
            
            
            if tail.strip() in list(movies_entity2embedd.keys()): 
                if head.strip() in list(movies_entity2embedd.keys()):

                    if tail not in kg:
                        kg[tail] = []
                    kg[tail].append((head, relation))
           
    entity_num = len(movies_entity2embedd) # number of entities data in the training data
    entity_adj = []
    relation_adj = []
    #id2entity = {v: k for k, v in movies_entity2embedd.items()}
    for i in range(entity_num + 1):
        entity_adj.append([]) # list of a number of lists equal to the entities in the data train
        relation_adj.append([])
    for i in range(config['model']['entity_neighbor_num']): # maximum number of neighbours
        entity_adj[0].append(0)
        relation_adj[0].append(0)
    
    for key in kg.keys():
        new_key = movies_entity2embedd[key.strip()] 
        while len(entity_adj[int(new_key)]) < config['model']['entity_neighbor_num']:
        #for _ in range(config['model']['entity_neighbor_num']):
            i = random.randint(0, len(kg[key]) - 1) # taking a random tail+relation from the list of values of the head entity
            
            # index of the corresponding embedding vector 
            
            entity_adj[int(new_key)].append(movies_entity2embedd[(kg[key][i][0]).strip()]) # entity [number from 1 to..] append tail = number
            relation_adj[int(new_key)].append(movies_relation2id[(kg[key][i][1])]) # relation [number from 1 to..] append relation ( is a number? or an embedding?)
            
    return entity_adj, relation_adj 


          
e_a,r_a = movies_construct_adj_mind(config,movies_entity2embedd, movies_relation2id)
print(r_a)
# #print('r_a',len(r_a))
# for i,el in enumerate(r_a):
#      if len(el) > 20 :
#          print('index of ra', i)
#          print(len(el))
    
# for i,el in enumerate(e_a):
#      if len(el) > 20 :
#          print('index of ea', i)
#          print(len(el))
#print('e_a' ,    len(e_a))


'''
def save_adjacency_data_to_file(entity_adj, relation_adj, file_path= "data/mind_reader_dataset/adjacencies.txt"):
    with open(file_path, 'w') as file:
        for element in entity_adj:
            file.write(str(element) + '\n')
        file.write('relation_adj\n')
        for element in relation_adj:
            file.write(str(element) + '\n')

save_adjacency_data_to_file(e_a,r_a)
'''
'''
def read_adjacency_data_from_file(file_path):
    entitys_adj = []
    relations_adj = []
    with open(file_path, 'r') as file:
        for line in file:
            entity_adj = []
            relation_adj = []
            line = line.strip()
            
            if line == 'relation_adj':
                break
            for number in  ast.literal_eval(line):
                
                entity_adj.append(int(number))
            entitys_adj.append(entity_adj)
            relations_adj.append(relation_adj)
        for line in file:
            for number in  ast.literal_eval(line):
                relation_adj.append(int(number))
    return entitys_adj, relations_adj

e_a, r_a = read_adjacency_data_from_file(file_path= "data/mind_reader_dataset/adjacencies.txt")
'''


def construct_movies_behaviours(config):
    with open(config["data"]["train_movies_behaviour"]) as behaviour:
        behaviours= {}
        for line in behaviour:
            fields = line.split(",")
            fields[3] = fields[3][:-2]
            if fields[0] in behaviours.keys():
                behaviours[fields[0]].append([fields[1],fields[2],fields[3]])
            else:
                behaviours[fields[0]] = []
                behaviours[fields[0]].append([fields[1],fields[2],fields[3]])
        return behaviours
    
movies_behaviours = construct_movies_behaviours(config)

user_movies_behaviours = list(movies_behaviours.keys())
train_size = 0.8
index_train = int(len(user_movies_behaviours)*train_size)
train_movies_behaviour = user_movies_behaviours[:index_train]
for i,el in enumerate(train_movies_behaviour):
    train_movies_behaviour[i] = el.strip()

test_movies_behaviour = user_movies_behaviours[index_train:]
for i,el in enumerate(test_movies_behaviour):
    test_movies_behaviour[i] = el.strip()    

def obtain_train_test_movies(train_movies_behaviour, movies_behaviours ):
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


train_list_movies, test_list_movies = obtain_train_test_movies(train_movies_behaviour, movies_behaviours )


def build_movies_feature_mind(config,movies_entity2embedd, movies_entity_embedding):
    movies_features = {}

    movies_feature_dict = {}
    
    with open(config['data']['train_movies_linked_entities']) as fp_train_movies:
        
        
        for i,line in enumerate(fp_train_movies):
            variable1, variable2 = line.split(' ', 1)
            

# Rimuovi i caratteri non desiderati come ":" o spazi iniziali/finali
            variable1 = variable1.replace(':', '').strip() # movie's wikidata id
        
            variable2 = variable2.strip() # entities
            
            
            # vert and subvert are the category and subcategory of the news
            mov_emb = []
            for i in movies_entity_embedding[movies_entity2embedd[variable1]]:
                mov_emb.append(float(i))
            mov_emb = np.array(mov_emb, dtype=np.float32)
            
            movies_feature_dict[variable1] = (mov_emb, variable2)
       
    with open(config['data']['test_movies_linked_entities']) as fp_test_movies:
        for i,line in enumerate(fp_test_movies):
            variable1, variable2 = line.split(' ', 1)

# Rimuovi i caratteri non desiderati come ":" o spazi iniziali/finali
            variable1 = variable1.replace(':', '').strip() # movie's wikidata id
        
            variable2 = variable2.strip() # entities

            mov_emb = []
            for i in movies_entity_embedding[movies_entity2embedd[variable1]]:
                mov_emb.append(float(i))
            mov_emb = np.array(mov_emb)
            
            
            movies_feature_dict[variable1] = (mov_emb, variable2)
            
    entity_type_dict = {}
    entity_type_index = 1

    for k, v in movies_feature_dict.items():
        movie_entity_feature_list = []
        movie_entity_feature = {}
        
        
        sentence_embedding = v[0]
        
        entities_categories =  ast.literal_eval(v[1])
        

        
        for entity_category in entities_categories:    
            
            if entity_category[1] not in entity_type_dict.keys():
                entity_type_dict[entity_category[1]] = entity_type_index
                entity_type_index = entity_type_index + 1
                
            movie_entity_feature[entity_category[0].strip()] =\
                (1, 1, entity_type_dict[entity_category[1]])#entity_freq, entity_position, entity_type
        
        for entity,v in movie_entity_feature.items():
            if entity in movies_entity2embedd:
                
                movie_entity_feature_list.append(
                    [movies_entity2embedd[entity], movie_entity_feature[entity][0], movie_entity_feature[entity][1], movie_entity_feature[entity][2]])
             
        movie_entity_feature_list.append([0, 0, 0, 0])
        if len(movie_entity_feature_list) > config['model']['news_entity_num']:
            movie_entity_feature_list = movie_entity_feature_list[:config['model']['news_entity_num']]
        else:
            for i in range(len(movie_entity_feature_list), config['model']['news_entity_num']):
                movie_entity_feature_list.append([0, 0, 0, 0])
          
        
        movies_feature_list_ins = [[],[],[],[],[]] # first list is the entity embedding index, second list is the title position, third list is occurrence, fourth list is the category, last list is teh embedding
        for i in range(len(movie_entity_feature_list)):
            for j in range(4):
                movies_feature_list_ins[j].append(movie_entity_feature_list[i][j])
        
        movies_feature_list_ins[4] = sentence_embedding
        
        
        movies_features[k] = movies_feature_list_ins
        
        
    movies_features["N0"] = [[],[],[],[],[]]
    
    
    for i in range(config['model']['news_entity_num']):
        for j in range(4):
            movies_features["N0"][j].append(0)
    movies_features["N0"][4] = np.zeros(config['model']['document_embedding_dim'])
    
    return movies_features, 100, 10, 100




movies_features, max_entity_freq, max_entity_pos, max_entity_type  = build_movies_feature_mind(config,movies_entity2embedd, movies_entity_embedding)

    
                  
def build_movies_user_history(config, movies_behaviour, train_movies_behaviour, test_movies_behaviour): # adding
    user_history_dict = {}
    for user_id in train_movies_behaviour:
        entity_of_user = movies_behaviour[" "+user_id]
        for i in entity_of_user:
            
            if i[1].strip() == 'True' and i[2].strip() == '1':
                if user_id+'_train' not in user_history_dict.keys():
                    user_history_dict[user_id+'_train'] = []
                user_history_dict[user_id+'_train'].append(i[0])

    
    
    for user_id in test_movies_behaviour:
        entity_of_user = movies_behaviour[" "+user_id]
        for i in entity_of_user:
            
            if i[1].strip() == 'True' and i[2].strip() == '1':
                if user_id+'_dev' not in user_history_dict.keys():
                    user_history_dict[user_id+'_dev'] = []
                user_history_dict[user_id+'_dev'].append(i[0])

    for k,it in user_history_dict.items():
         if len(it) >= config['model']['user_his_num']:
             user_history_dict[k] = it[:config['model']['user_his_num']]
         else:
              for i in range(config['model']['user_his_num']-len(it)):
                user_history_dict[k].append("N0")
    
    return user_history_dict



movies_user_history_dict =build_movies_user_history(config, movies_behaviours, train_movies_behaviour, test_movies_behaviour)


def get_movies_user2item_data(config,movies_behaviours, train_movies_behaviour,test_movies_behaviour):
    negative_num = config['trainer']['train_neg_num']
    train_data = {}
    user_ids = []
    movies_id = []
    label = []
    
    for  user_id in train_movies_behaviour:
        
        positive_list = []
        negative_list = []
        v = movies_behaviours[" "+user_id]
        for element in v:
            if element[1].strip() == 'True':
                if element[2].strip() == '1':
                    positive_list.append(element[0])
                else:
                    negative_list.append(element[0])
        
        for pos_movies in positive_list:
            user_ids.append(user_id + "_train")
            if len(negative_list) >= negative_num:
                neg_movies = random.sample(negative_list, negative_num)
            else:
                neg_movies = negative_list
                for i in range(negative_num - len(negative_list)):
                    neg_movies.append("N0")
            all_movies = neg_movies
            all_movies.append(pos_movies)
            movies_id.append(all_movies)
            label.append([])
            for i in range(negative_num):
                label[-1].append(0)
            label[-1].append(1)

    train_data['item1'] = user_ids
    train_data['item2'] = movies_id
    train_data['label'] = label

            

    dev_data = {}
    session_id = [np.arange(1,len(test_movies_behaviour))]
    users_id = []
    movies_id = []
    label = []
    for  user_id in test_movies_behaviour:
        v = movies_behaviours[" "+user_id]
        for element in v:
            if element[1].strip() == 'True':
                if element[2].strip() == '1':
                    movies_id.append(element[0])
                    label.append(1.0)
                else:
                    movies_id.append(element[0])
                    label.append(0.0)
            
            user_ids.append(user_id+ "_dev")
        

    dev_data['item1'] = users_id
    dev_data['session_id'] = session_id
    dev_data['item2'] = movies_id
    dev_data['label'] = label

    return train_data, dev_data




movie_train_data, movie_dev_data =  get_movies_user2item_data(config,movies_behaviours, train_movies_behaviour,test_movies_behaviour)

for i,v in enumerate(movies_entity_embedding):
    emb_mov = []
    for index, number in enumerate(v):
        emb_mov.append(float(number))
    movies_entity_embedding[i]= np.array(emb_mov)




movies_entity_embedding = torch.FloatTensor(np.array(movies_entity_embedding))
relation_embeddings = torch.FloatTensor(np.array(relation_embeddings))

data = [movies_user_history_dict, movies_entity_embedding, relation_embeddings, e_a,r_a, movies_features,\
    max_entity_freq, max_entity_pos, max_entity_type, movie_train_data, movie_dev_data]


single_task_training(config, data)