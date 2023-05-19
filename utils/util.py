import json
import torch
import random
import numpy as np
import math
import os
import sys
import pickle
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
from sentence_transformers import SentenceTransformer
import requests
import math
import zipfile
#from logger.logger import *
from tqdm import tqdm


def write_pickle(data, fname):
    with open(fname, 'wb') as file:
        pickle.dump(data, file)

def read_pickle(fname):
    with open(fname, 'rb') as file:
        data = pickle.load(file)
    return data

def ensure_dir(dirname):
    if not os.path.isdir(dirname):
        os.makedirs(dirname, exist_ok=False)

def read_json(fname):
    with open(fname, 'rt') as file:
        return json.load(file, object_hook=OrderedDict)

def write_json(content, fname):
    with open(fname, 'wt') as file:
        json.dump(content, file, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def my_collate_fn(batch):
    return batch


def write_embedding_news(data_folder, save_folder):
     """
     Function to save the embedding of news in the training and valid data folders into the save folders. The embeddings
     are created with the dstil-bert-base-nli-stsb-mean-tokens model from sentence-transformers. The data_folder is as
     follows: `./data/train/` while the save_folder is `./data/train_embeddings/`.
     """
     if not os.path.isdir(save_folder):
         print(f"Creating folder {save_folder}")
         os.mkdir(save_folder)
     embeddings =[]
     model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
     with open(f"{data_folder}/news.tsv", 'r', encoding='utf-8') as news:
         for line in news:
             line = line.strip().split('\t')
             newsid, vert, subvert, title, abstract, url, entity_info_title, entity_info_abstract = line
             embeddings.append(model.encode(f"{title} {abstract}", show_progress_bar=False))
     write_pickle(embeddings, f"{save_folder}/{data_folder.split('/')[-1]}_news_embeddings.pkl")
     print(f"Saved news embeddings for {data_folder} to {save_folder}")

def write_data_mind(config, save_folder):
    if not os.path.isdir(save_folder):
        print(f"Creating folder {save_folder}")
        os.mkdir(save_folder)
    data_mind = load_data_mind(config)
    write_pickle(data_mind, f"{save_folder}/data_mind.pkl")
    print(f"Saved data_mind to {save_folder}")


def entities_news(config):
    """
    Return all the wikidataIDs of each entities in the title and abstract of the train and valid news data. The entities are extracted from the `WikidataId` of the `Title Entities`
    and `Abstract Entities` fields of the news in the train and valid news.tsv files (see description in: 
    https://github.com/msnews/msnews.github.io/blob/master/assets/doc/introduction.md).
    """
    entities = set()
    # Add the WikidataId from the train entities
    with open(config["data"]["train_news"]) as train_news:
        for line in train_news:
            _, _, _, _, _, _, entity_info_title, entity_info_abstract = line.strip().split('\t')  # only need last 2 columns
            for entity in eval(entity_info_title):  # see if title_entities is not an empty list 
                entities.add(entity["WikidataId"]) # taking the corresponding wikidataID of each entity in title and saving it in 'entities'
            for entity in eval(entity_info_abstract):  # see if abstract_entities is not an empty list
                entities.add(entity["WikidataId"])
    # Add the WikidataId from the valid entities
    with open(config["data"]["valid_news"]) as valid_news:
        for line in valid_news:
            _, _, _, _, _, _, entity_info_title, entity_info_abstract = line.strip().split('\t')
            for entity in eval(entity_info_title):
                entities.add(entity["WikidataId"])
            for entity in eval(entity_info_abstract):
                entities.add(entity["WikidataId"])
    return entities

def entity_to_id(config, entities):
    """
    Return dictionary with entity `WikidataId' as key and entity id(numbers) as value. The entity id is the id of the entity in
    the file `entity2id.txt`. Only entities found with the `entities_news` function are added to the dictionary. The ids
    are incremented by 1.
    """
    entity2id = {}
    with open(config["data"]["entity_index"]) as entity2id_file:  # File with wikidata ids + number from 0 to 100000
        next(entity2id_file)  # skip first line with number of entities
        for line in entity2id_file:
            entity, entity_id = line.strip().split('\t') # entity = wikidata ids, entity_id = number from 0 to 100000
            if entity in entities: # checking if the wikidata ids is in our list of wikidata ids of data training
                entity2id[entity] = int(entity_id) + 1  # increment the id by 1
    return entity2id

def ids_to_entity_id(config, ids):
    """
    We need this function in the `load_data_mind` function to get a dictionary with key the entity and value the id for
    the entities not embedded.
    """
    entity2id = {}
    with open(config["data"]["entity_index"]) as entity2id_file:
        next(entity2id_file)
        for line in entity2id_file:
            entity, entity_id = line.strip().split('\t')
            if int(entity_id) + 1 in ids:
                entity2id[entity] = int(entity_id) + 1
    return entity2id

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

def construct_adj(graph_file, entity2id_file, args): #graph is triple
    print('constructing adjacency matrix ...')
    graph_file_fp = open(graph_file, 'r', encoding='utf-8')
    graph = []
    for line in graph_file_fp:
        linesplit = line.split('\n')[0].split('\t')
        if len(linesplit) > 1:
            graph.append([linesplit[0], linesplit[2], linesplit[1]])

    kg = {}
    for triple in graph:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]
        # treat the KG as an undirected graph
        if head not in kg:
            kg[head] = []
        kg[head].append((tail, relation))
        if tail not in kg:
            kg[tail] = []
        kg[tail].append((head, relation))

    fp_entity2id = open(entity2id_file, 'r', encoding='utf-8')
    entity_num = int(fp_entity2id.readline().split('\n')[0])
    entity_adj = []
    relation_adj = []
    for i in range(entity_num):
        entity_adj.append([])
        relation_adj.append([])
    for key in kg.keys():
        for index in range(args.entity_neighbor_num):
            i = random.randint(0,len(kg[key])-1)
            entity_adj[int(key)].append(int(kg[key][i][0]))
            relation_adj[int(key)].append(int(kg[key][i][1]))

    return entity_adj, relation_adj

def construct_embedding(entity_embedding_file, relation_embedding_file):
    print('constructing embedding ...')
    entity_embedding = []
    relation_embedding = []
    fp_entity_embedding = open(entity_embedding_file, 'r', encoding='utf-8')
    fp_relation_embedding = open(relation_embedding_file, 'r', encoding='utf-8')
    for line in fp_entity_embedding:
        linesplit = line.strip().split('\t')
        linesplit = [float(i) for i in linesplit]
        entity_embedding.append(linesplit)
    for line in fp_relation_embedding:
        linesplit = line.strip().split('\t')
        linesplit = [float(i) for i in linesplit]
        relation_embedding.append(linesplit)
    return torch.FloatTensor(entity_embedding), torch.FloatTensor(relation_embedding)

def my_collate_fn(batch):
    return batch

def construct_entity_dict(entity_file):
    fp_entity2id = open(entity_file, 'r', encoding='utf-8')
    entity_dict = {}
    entity_num_all = int(fp_entity2id.readline().split('\n')[0])
    lines = fp_entity2id.readlines()
    for line in lines:
        entity, entityid = line.strip().split('\t')
        entity_dict[entity] = entityid
    return entity_dict

def real_batch(batch):
    data = {}
    data['item1'] = []
    data['item2'] = []
    data['label'] = []
    for item in batch:
        data['item1'].append(item['item1'])
        data['item2'].append(item['item2'])
        data['label'].append(item['label'])
    return data

def maybe_download(url, filename=None, work_directory=".", expected_bytes=None):
    """Download a file if it is not already downloaded.

    Args:
        filename (str): File name.
        work_directory (str): Working directory.
        url (str): URL of the file to download.
        expected_bytes (int): Expected file size in bytes.

    Returns:
        str: File path of the file downloaded.
    """
    if filename is None:
        filename = url.split("/")[-1]
    os.makedirs(work_directory, exist_ok=True)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):

        r = requests.get(url, stream=True)
        total_size = int(r.headers.get("content-length", 0))
        block_size = 1024
        num_iterables = math.ceil(total_size / block_size)

        with open(filepath, "wb") as file:
            for data in tqdm(
                    r.iter_content(block_size),
                    total=num_iterables,
                    unit="KB",
                    unit_scale=True,
            ):
                file.write(data)
    if expected_bytes is not None:
        statinfo = os.stat(filepath)
        if statinfo.st_size != expected_bytes:
            os.remove(filepath)
            raise IOError("Failed to verify {}".format(filepath))

    return filepath

def unzip_file(zip_src, dst_dir, clean_zip_file=True):
    """Unzip a file

    Args:
        zip_src (str): Zip file.
        dst_dir (str): Destination folder.
        clean_zip_file (bool): Whether or not to clean the zip file.
    """
    fz = zipfile.ZipFile(zip_src, "r")
    for file in fz.namelist():
        fz.extract(file, dst_dir)
    if clean_zip_file:
        os.remove(zip_src)

def get_mind_data_set(type):
    """ Get MIND dataset address

    Args:
        type (str): type of mind dataset, must be in ['large', 'small', 'demo']

    Returns:
        list: data url and train valid dataset name
    """
    assert type in ["large", "small", "demo"]

    if type == "large":
        return (
            "https://mind201910small.blob.core.windows.net/release/",
            "MINDlarge_train.zip",
            "MINDlarge_dev.zip",
            "MINDlarge_utils.zip",
        )

    elif type == "small":
        return (
            "https://mind201910small.blob.core.windows.net/release/",
            "MINDsmall_train.zip",
            "MINDsmall_dev.zip",
            "MINDsma_utils.zip",
        )

    elif type == "demo":
        return (
            "https://recodatasets.blob.core.windows.net/newsrec/",
            "MINDdemo_train.zip",
            "MINDdemo_dev.zip",
            "MINDdemo_utils.zip",
        )

def download_deeprec_resources(azure_container_url, data_path, remote_resource_name):
    """Download resources.

    Args:
        azure_container_url (str): URL of Azure container.
        data_path (str): Path to download the resources.
        remote_resource_name (str): Name of the resource.
    """
    os.makedirs(data_path, exist_ok=True)
    remote_path = azure_container_url + remote_resource_name
    #maybe_download(remote_path, remote_resource_name, data_path)
    zip_ref = zipfile.ZipFile(os.path.join(data_path, remote_resource_name), "r")
    zip_ref.extractall(data_path)
    zip_ref.close()
    os.remove(os.path.join(data_path, remote_resource_name))

def get_user2item_data(config):
    negative_num = config['trainer']['train_neg_num']
    train_data = {}
    user_id = []
    news_id = []
    label = []
    fp_train = open(config['data']['train_behavior'], 'r', encoding='utf-8')
    for line in fp_train:
        index, userid, imp_time, history, behavior = line.strip().split('\t')
        behavior = behavior.split(' ')
        positive_list = []
        negative_list = []
        for news in behavior:
            newsid, news_label = news.split('-')
            if news_label == "1":
                positive_list.append(newsid)
            else:
                negative_list.append(newsid)
        for pos_news in positive_list:
            user_id.append(userid+ "_train")
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
    fp_dev = open(config['data']['valid_behavior'], 'r', encoding='utf-8')
    for line in fp_dev:
        index, userid, imp_time, history, behavior = line.strip().split('\t')
        behavior = behavior.split(' ')
        for news in behavior:
            newsid, news_label = news.split('-')
            session_id.append(index)
            user_id.append(userid+ "_dev")
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

def build_user_history(config):
    user_history_dict = {}
    fp_train_behavior = open(config['data']['train_behavior'], 'r', encoding='utf-8')
    for line in fp_train_behavior:
        index, user_id, imp_time, history, behavior = line.strip().split('\t')
        if len(history.split(' ')) >= config['model']['user_his_num']:
            user_history_dict[user_id+"_train"] = history.split(' ')[:config['model']['user_his_num']]
        else:
            user_history_dict[user_id + "_train"] = history.split(' ')
            for i in range(config['model']['user_his_num']-len(history.split(' '))):
                user_history_dict[user_id + "_train"].append("N0")
            if user_history_dict[user_id + "_train"][0] == '':
                user_history_dict[user_id + "_train"][0] = 'N0'

    fp_dev_behavior = open(config['data']['valid_behavior'], 'r', encoding='utf-8')
    for line in fp_dev_behavior:
        index, user_id, imp_time, history, behavior = line.strip().split('\t')
        if len(history.split(' ')) >= config['model']['user_his_num']:
            user_history_dict[user_id+"_dev"] = history.split(' ')[:config['model']['user_his_num']]
        else:
            user_history_dict[user_id + "_dev"] = history.split(' ')
            for i in range(config['model']['user_his_num']-len(history.split(' '))):
                user_history_dict[user_id + "_dev"].append("N0")
            if user_history_dict[user_id + "_dev"][0] == '':
                user_history_dict[user_id + "_dev"][0] = 'N0'
    return user_history_dict

def build_news_features_mind(config, entity2embedd):
    # There are 4 features for each news: postion, freq, category, embeddings
    news_features = {}
    news_feature_dict = {}
    embedding_folder = config['data']['sentence_embedding_folder']
    # Load sentence embeddings from file if present
    with open(config['data']['train_news'], 'r', encoding='utf-8') as fp_train_news:
        if embedding_folder is not None:
            train_sentences_embedding = read_pickle(embedding_folder + "train_news_embeddings.pkl")
        for i, line in enumerate(fp_train_news):
            fields = line.strip().split('\t')
            # vert and subvert are the category and subcategory of the news
            newsid, vert, subvert, title, abstract, url, entity_info_title, entity_info_abstract = fields
            if embedding_folder is not None:
                news_feature_dict[newsid] = (train_sentences_embedding[i], entity_info_title, entity_info_abstract, vert, subvert)
            else:
                news_feature_dict[newsid] = (title+" "+abstract, entity_info_title, entity_info_abstract, vert, subvert)
        
        
            
    # Load sentence embeddings from file if present
    with open(config['data']['valid_news'], 'r', encoding='utf-8') as fp_dev_news:
        if embedding_folder is not None:
            valid_sentences_embedding = read_pickle(embedding_folder + "valid_news_embeddings.pkl")
        for i, line in enumerate(fp_dev_news):
            fields = line.strip().split('\t')
            newsid, vert, subvert, title, abstract, url, entity_info_title, entity_info_abstract = fields
            if embedding_folder is not None:
                news_feature_dict[newsid] = (valid_sentences_embedding[i], entity_info_title, entity_info_abstract, vert, subvert)
            else:
                news_feature_dict[newsid] = (title+" "+abstract, entity_info_title, entity_info_abstract, vert, subvert)

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

def construct_adj_mind(config, entity2id, entity2embedd):  # graph is triple
    print('constructing adjacency matrix ...')
    ids = set(entity2id.values()) # numbers from 1 to .....
    with open(config['data']['knowledge_graph'], 'r', encoding='utf-8') as graph_file_fp:
        kg = {} # dictionary with entity(numbers from 1 to ...) as keys and relation+tail as values, also tails that are not already encoded(their head is not in the train ) become keys with values = head+relation
        for line in graph_file_fp: # check how values are encoded here
            linesplit = line.split('\n')[0].split('\t')
            head = int(linesplit[0]) + 1
            relation = int(linesplit[2]) + 1
            tail = int(linesplit[1]) + 1
            # treat the KG as an undirected graph
            if head in ids: # check entity head is in train data 
                if head not in kg: # if entity head is not in the 'kg' dictionary
                    kg[head] = []
                kg[head].append((tail, relation))
            if tail in ids:
                if tail not in kg:
                    kg[tail] = []
                kg[tail].append((head, relation))

    entity_num = len(entity2embedd) # number of entities data in the training data
    entity_adj = []
    relation_adj = []
    id2entity = {v: k for k, v in entity2id.items()}
    for i in range(entity_num + 1):
        entity_adj.append([]) # list of a number of lists equal to the entities in the data train
        relation_adj.append([])
    for i in range(config['model']['entity_neighbor_num']): # maximum number of neighbours
        entity_adj[0].append(0)
        relation_adj[0].append(0)
    for key in kg.keys():
        for _ in range(config['model']['entity_neighbor_num']):
            i = random.randint(0, len(kg[key]) - 1) # taking a random tail+relation from the list of values of the head entity
            new_key = entity2embedd[id2entity[int(key)]] # = entity2embedd[wikidata ID's of the key] = number from 1 to ...
            entity_adj[int(new_key)].append(int(kg[key][i][0])) # entity [number from 1 to..] append tail = number
            relation_adj[int(new_key)].append(int(kg[key][i][1])) # relation [number from 1 to..] append relation ( is a number? or an embedding?)
    return entity_adj, relation_adj 

def construct_embedding_mind(config, entity2id, entity_embedding, entity2embedd):
    print('constructing embedding ...')
    relation_embedding = [np.zeros(config['model']['entity_embedding_dim'])]
    id2entity = {v:k for k, v in entity2id.items()} # dictionary with key the numbers and values wikidataIDS
    with open(config['data']['entity_embedding'], 'r', encoding='utf-8') as fp_entity_embedding:
        # The file has only the embeddings, not the entity names so we first need to get the entity ids (starting
        # from 1) and then get the corresponding entity names from the entity2id dictionary
        i = 1
        for line in fp_entity_embedding:
            if i in id2entity:
                linesplit = line.strip().split('\t')
                linesplit = [float(i) for i in linesplit]
                # Dictionary with key the wikidata ids and value the position of the embedding in the list
                entity2embedd[id2entity[i]] = len(entity_embedding)
          
                # The value is the current lenght of the list starting from 1
                entity_embedding.append(linesplit)
            i += 1
    with open(config['data']['relation_embedding'], 'r', encoding='utf-8') as fp_relation_embedding:
        i = 1
        for line in fp_relation_embedding:
            linesplit = line.strip().split('\t')
            linesplit = [float(i) for i in linesplit]
            relation_embedding.append(linesplit)
        i += 1
    return entity_embedding, relation_embedding, entity2embedd


def build_vert_data(config):
    random.seed(2020)
    vert_label_dict = {}
    label_index = 0
    all_news_data = []
    vert_train = {}
    vert_dev = {}
    item1_list_train = []
    item2_list_train = []
    label_list_train = []
    item1_list_dev = []
    item2_list_dev = []
    label_list_dev = []
    fp_train_news = open(config['data']['train_news'], 'r', encoding='utf-8')
    for line in fp_train_news:
        newsid, vert, subvert, title, abstract, url, entity_info_title, entity_info_abstract = line.strip().split('\t')
        if vert not in vert_label_dict:
            vert_label_dict[vert] = label_index
            label_index = label_index + 1
        all_news_data.append((newsid, vert_label_dict[vert]))
    print(vert_label_dict)
    for i in range(len(all_news_data)):
        if random.random()<0.8:
            item1_list_train.append("U0")
            item2_list_train.append(all_news_data[i][0])
            label_list_train.append(all_news_data[i][1])
        else:
            item1_list_dev.append("U0")
            item2_list_dev.append(all_news_data[i][0])
            label_list_dev.append(all_news_data[i][1])
    vert_train['item1'] = item1_list_train
    vert_train['item2'] = item2_list_train
    vert_train['label'] = label_list_train
    vert_dev['item1'] = item1_list_dev
    vert_dev['item2'] = item2_list_dev
    vert_dev['label'] = label_list_dev

    return vert_train, vert_dev

def build_pop_data(config):
    fp_train = open(config['data']['train_behavior'], 'r', encoding='utf-8')
    news_imp_dict = {}
    pop_train = {}
    pop_test = {}
    for line in fp_train:
        index, userid, imp_time, history, behavior = line.strip().split('\t')
        behavior = behavior.split(' ')
        for news in behavior:
            newsid, news_label = news.split('-')
            if news_label == "1":
                if newsid not in news_imp_dict:
                    news_imp_dict[newsid] = [1,1]
                else:
                    news_imp_dict[newsid][0] = news_imp_dict[newsid][0] + 1
                    news_imp_dict[newsid][1] = news_imp_dict[newsid][1] + 1
            else:
                if newsid not in news_imp_dict:
                    news_imp_dict[newsid] = [0,1]
                else:
                    news_imp_dict[newsid][1] = news_imp_dict[newsid][1] + 1
    return pop_train, pop_test

def build_item2item_data(config):
    fp_train = open(config['data']['train_behavior'], 'r', encoding='utf-8')
    item2item_train = {}
    item2item_test = {}
    item1_train = []
    item2_train = []
    label_train = []
    item1_dev = []
    item2_dev = []
    label_dev = []
    user_history_dict = {}
    news_click_dict = {}
    doc_doc_dict = {}
    all_news_set = set()
    for line in fp_train:
        index, userid, imp_time, history, behavior = line.strip().split('\t')
        behavior = behavior.split(' ')
        if userid not in user_history_dict:
            user_history_dict[userid] = set()
        for news in behavior:
            newsid, news_label = news.split('-')
            all_news_set.add(newsid)
            if news_label == "1":
                user_history_dict[userid].add(newsid)
                if newsid not in news_click_dict:
                    news_click_dict[newsid] = 1
                else:
                    news_click_dict[newsid] = news_click_dict[newsid] + 1
        news = history.split(' ')
        for newsid in news:
            user_history_dict[userid].add(newsid)
            if newsid not in news_click_dict:
                news_click_dict[newsid] = 1
            else:
                news_click_dict[newsid] = news_click_dict[newsid] + 1
    for user in user_history_dict:
        list_user_his = list(user_history_dict[user])
        for i in range(len(list_user_his) - 1):
            for j in range(i + 1, len(list_user_his)):
                doc1 = list_user_his[i]
                doc2 = list_user_his[j]
                if doc1 != doc2:
                    if (doc1, doc2) not in doc_doc_dict and (doc2, doc1) not in doc_doc_dict:
                        doc_doc_dict[(doc1, doc2)] = 1
                    elif (doc1, doc2) in doc_doc_dict and (doc2, doc1) not in doc_doc_dict:
                        doc_doc_dict[(doc1, doc2)] = doc_doc_dict[(doc1, doc2)] + 1
                    elif (doc2, doc1) in doc_doc_dict and (doc1, doc2) not in doc_doc_dict:
                        doc_doc_dict[(doc2, doc1)] = doc_doc_dict[(doc2, doc1)] + 1
    weight_doc_doc_dict = {}
    for item in doc_doc_dict:
        if item[0] in news_click_dict and item[1] in news_click_dict:
            weight_doc_doc_dict[item] = doc_doc_dict[item] / math.sqrt(
                news_click_dict[item[0]] * news_click_dict[item[1]])

    THRED_CLICK_TIME = 10
    freq_news_set = set()
    for news in news_click_dict:
        if news_click_dict[news] > THRED_CLICK_TIME:
            freq_news_set.add(news)
    news_pair_thred_w_dict = {}  # {(new1, news2): click_weight}
    for item in weight_doc_doc_dict:
        if item[0] in freq_news_set and item[1] in freq_news_set:
            news_pair_thred_w_dict[item] = weight_doc_doc_dict[item]

    news_positive_pairs = []
    for item in news_pair_thred_w_dict:
        if news_pair_thred_w_dict[item] > 0.05:
            news_positive_pairs.append(item)

    for item in news_positive_pairs:
        random_num = random.random()
        if random_num < 0.8:
            item1_train.append(item[0])
            item2_train.append(item[1])
            label_train.append(1)
            negative_list = random.sample(list(freq_news_set), 4)
            for negative in negative_list:
                item1_train.append(item[0])
                item2_train.append(negative)
                label_train.append(0)
        else:
            item1_dev.append(item[0])
            item2_dev.append(item[1])
            label_dev.append(1)
            negative_list = random.sample(list(freq_news_set), 4)
            for negative in negative_list:
                item1_train.append(item[0])
                item2_train.append(negative)
                label_dev.append(0)
    item2item_train["item1"] = item1_train
    item2item_train["item2"] = item2_train
    item2item_train["label"] = label_train
    item2item_test["item1"] = item1_dev
    item2item_test["item2"] = item2_dev
    item2item_test["label"] = label_dev
    return item2item_train, item2item_test


def entities_movies(config):

    """
    Return all the wikidataIDs of each entities of the dataset 'movies recommendation'

    """
    entities = set()
    with open(config["data"]["train_movies"]) as train_movies:
        for entity in train_movies:
            

            wikidataid, entity_label, entity_type = entity.strip().split(',')  # only need last 2 columns
            entities.add(wikidataid)
    """
    # Add the WikidataId from the valid entities
    with open(config["data"]["valid_news"]) as valid_news:
        for line in valid_news:
            _, _, _, _, _, _, entity_info_title, entity_info_abstract = line.strip().split('\t')
            for entity in eval(entity_info_title):
                entities.add(entity["WikidataId"])
            for entity in eval(entity_info_abstract):
                entities.add(entity["WikidataId"]) """
    return entities


def entity_to_id_movies(entities):
    """
    Return dictionary with entity `WikidataId' as key and entity id(numbers) as value. The entity id is the id of the entity in
    the file `entity2id.txt`. 
    """
    entity2id = {}
    
    for entity_id, entity in enumerate(entities):
        entity2id[entity] = entity_id + 1  # increment the id by 1
    return entity2id

def relation2id_movies(config):

    """
    Return a dictionary with the wikidataIDs of each property in the graph of the dataset movies as key and its corresponding id as value 

    """
    relations = {}
    # Add the WikidataId from the train entities
    with open(config["data"]["relation_index"]) as relation_movies:
        for relation in relation_movies:
            

            relation_all = relation.split() 
            wikidataid = relation_all[0]
            relation_id = relation_all[1]
            relations[wikidataid] = int(relation_id) +1
        return relations




def load_data_mind(config,embedding_folder=None):
    entities = entities_news(config)  # get all wikidata IDs of each entities in the title and abstract in the training news data
    entity2id = entity_to_id(config, entities)  # get dict with key entity WikidataId and value id
    entity_embedding = [np.zeros(config["model"]["entity_embedding_dim"])] # array with 100 zeros
    entity2embedd = {}
    # entity_embedding is a vector with the embedding values for each entity in id2entity,
    # relation_embedding is a vector with the embedding values for each relation 
    # entity2embedd is a dictionary with wikidataids as keys and numbers from 1  as values(=corresponding index in the 'entity_embedding' vector)
    entity_embedding, relation_embedding, entity2embedd =\
        construct_embedding_mind(config, entity2id, entity_embedding, entity2embedd)
    # entity_adj is a list of lists and for each entity it appends in its corresponding number ( utilizing as index of entity adj) 20 neighbours
    entity_adj, relation_adj = construct_adj_mind(config, entity2id, entity2embedd)

    # Get the entity ids that are in the neighorhood of each node in entity_adj but not in the dictionary of the ids
    # found in the train and valid title and abstract
    entities_ids_not_embedded = set([item for items in entity_adj for item in items]).difference(set(entity2id.values())) # taking the numbers connected to the wikidata ids for entities that are not in the training data 
    # Get the dictionary with key the entity and value the id for the entities not embedded
    entity2id_not_embedded = ids_to_entity_id(config, entities_ids_not_embedded) # creating a dictionary with wikidata ids as keys and numbers as values for each entity that is not in the training data but is a neighbour of the training data
    # Get the embedding for these new entities
    entity_embedding, relation_embedding, entity2embedd = construct_embedding_mind(config,
                                                                                   entity2id_not_embedded,
                                                                                   entity_embedding,
                                                                                   entity2embedd)
    # added now to check a thing remnove asap 
    news_feature, max_entity_freq, max_entity_pos, max_entity_type =\
        build_news_features_mind(config, entity2embedd)
    # Add the new entities to the dictionary
    entity2id.update(entity2id_not_embedded)
    # Invert the dictionary
    id2entity = {v: k for k, v in entity2id.items()}
    
    
    # The ids in entity_adj are the original ones, they need to be updated to the new ids given by entity2embedd, in entity adj there are index that are missing since they are not in the training data, with entity to embedd we can have all the numbers from 1 that are connected to an entity
    for i in range(1, len(entity_adj)):
        for j in range(0, len(entity_adj[i])):
            # print(entity_adj[i][j])
            entity_adj[i][j] = entity2embedd[id2entity[entity_adj[i][j]]]
            # print(entity_adj[i][j])
            # sys.exit()
    entity_embedding = torch.FloatTensor(np.array(entity_embedding))
    relation_embedding = torch.FloatTensor(np.array(relation_embedding))


    # Load the news
    news_feature, max_entity_freq, max_entity_pos, max_entity_type =\
        build_news_features_mind(config, entity2embedd)

    user_history = build_user_history(config)

    if config['trainer']['training_type'] == "multi-task":
        train_data, dev_data = get_user2item_data(config)
        vert_train, vert_test = build_vert_data(config)
        pop_train, pop_test = build_pop_data(config)
        item2item_train, item2item_test = build_item2item_data(config)
        return user_history, entity_embedding, relation_embedding, entity_adj, relation_adj, news_feature,\
            max_entity_freq, max_entity_pos, max_entity_type, train_data, dev_data, vert_train, vert_test,\
                pop_train, pop_test, item2item_train, item2item_test
    elif config['trainer']['task'] == "user2item":
        train_data, dev_data = get_user2item_data(config)
        return user_history, entity_embedding, relation_embedding, entity_adj, relation_adj, news_feature,\
            max_entity_freq, max_entity_pos, max_entity_type, train_data, dev_data
    elif config['trainer']['task'] == "item2item":
        item2item_train, item2item_test = build_item2item_data(config)
        return user_history, entity_embedding, relation_embedding, entity_adj, relation_adj, news_feature,\
            max_entity_freq, max_entity_pos, max_entity_type, item2item_train, item2item_test
    elif config['trainer']['task'] == "vert_classify":
        vert_train, vert_test = build_vert_data(config)
        return user_history, entity_embedding, relation_embedding, entity_adj, relation_adj, news_feature,\
            max_entity_freq, max_entity_pos, max_entity_type, vert_train, vert_test
    elif config['trainer']['task'] == "pop_predict":
        pop_train, pop_test = build_pop_data(config)
        return user_history, entity_embedding, relation_embedding, entity_adj, relation_adj, news_feature,\
            max_entity_freq, max_entity_pos, max_entity_type, pop_train, pop_test
    else:
        print("task error, please check config")
