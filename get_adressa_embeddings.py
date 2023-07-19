import numpy as np
import pandas as pd
import csv
import torch
import requests
from transformers import BertTokenizer, BertModel, set_seed
from tqdm import tqdm
#from pathlib import Path
#import os
import sys

sys.path.append('')

from parse_config import ConfigParser
import argparse
#import ast
from utils.adressa_util_prova import *
import random

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.manual_seed(seed)
set_seed(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model = model.to(device)


# Define a function to extract the embeddings for a given entity
def extract_embeddings(entity_id, model):
    # print('Entarto in extract embedding')
    # Get the text description for the entity
    entity_description = get_description(entity_id)
    if entity_description is None:
        return None

    # Encode the text description using the pre-trained tokenizer
    input_ids = tokenizer.encode(entity_description, return_tensors='pt', pad_to_max_length=True,
                                 max_length=100).long().to(device)
    # print('input_ids ', input_ids.shape) # len [1,100]
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(device)
    with torch.no_grad():
        embeddings = model(input_ids, attention_mask=attention_mask)  # last hidden state, pooler output

        # print('embedding lenght', len(embeddings), 'len embedd 0 ', len(embeddings[0])) #len(embedding[0]= batchsize)
        embeddings = embeddings[0]  # i take only the last hidden state (ha size [1,100,768])
        # print('embeddings0', embeddings.shape, embeddings) # torch.Size([1, 100, 768]) , (batch_size, seq_len (input ids is max 100), hidden_size)
        # print('seg lenght', embeddings[0,:,0])
        embeddings = embeddings[:, 0, :].cpu().numpy()  # why it takes only the first element of the sequence lenght
        # print('embedding, zero, shape ',embeddings.shape) # [1,768] take only the first seq.len = input ids
        # print('embedding ', embedding)
    if len(embeddings) < 1:
        return None

    return embeddings


def entities2vec(entity_ids: list):
    # print('Entrato in entity to vec')
    entity_embeddings = {}
    for entity_id in tqdm(entity_ids):
        entity_embeddings[entity_id] = extract_embeddings(entity_id, model)
    embedding_file = './content/KRED-Reccomendation-System/data/adr_entity_embeddings_100.vec'
    with open(embedding_file,
            'w') as entity_embedding_file:
        for key, values in entity_embeddings.items():
            if values is not None:
                values = np.array2string(values[0][:100], separator='\t',
                                         formatter={'float_kind': lambda x: "%.6f" % x}).strip("[]").replace("\n",
                                                                                                             "\t").replace(
                    " ", "")
                str_k_v = key + '\t' + values + '\n'
                entity_embedding_file.write(str_k_v)
            else:
                print('key ', key)
    return embedding_file


def get_description(id):
    # print('entrato in get description')
    query = f"""
    SELECT ?Label ?Description
    WHERE 
    {{
      wd:{id} rdfs:label ?Label .
      FILTER (LANG(?Label) = "en").
      OPTIONAL {{ wd:{id} schema:description ?Description . FILTER (LANG(?Description) = "en") }}
    }}
    """
    max_tries = 100
    for i in range(max_tries):

        try:
            response = requests.get("https://query.wikidata.org/sparql", params={'query': query, 'format': 'json'})
            response_json = response.json()
            # print('2:',response_json)
            label = response_json['results']['bindings'][0]['Label']['value']
            description = response_json['results']['bindings'][0].get('Description', {}).get('value', '')
            description = label + ' ' + description
            return description
        except:
            pass
    return None


parser = argparse.ArgumentParser(description='KRED')
parser.add_argument('-c', '--config', default="./config.yaml", type=str,
                    help='config file path (default: None)')
parser.add_argument('-r', '--resume', default=None, type=str,
                    help='path to latest checkpoint (default: None)')
parser.add_argument('-d', '--device', default=None, type=str,
                    help='indices of GPUs to enable (default: all)')

config = ConfigParser.from_args(parser)

# Read csv file with entities wikiid
df_entities = pd.read_csv(config["data"]["entities_addressa"], index_col=False)
# List of entities
entities_list = [entity for entity in df_entities['wikiid']]
entities_set = set(entities_list)
# Take all the wikidata id of the movies entities
adressa_entities = list(entities_set)
entities_embedded_adressa = entities2vec(adressa_entities)