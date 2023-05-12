import numpy as np
import pandas as pd
import csv
import torch
import requests
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
from pathlib import Path
import os
import sys
sys.path.append('')

from parse_config import ConfigParser
import argparse
import ast
from utils.util import *
import random



device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model = model.to(device)


# Define a function to extract the embeddings for a given entity
def extract_embeddings(entity_id, model):
    # Get the text description for the entity
    entity_description = get_description(entity_id)
    if entity_description is None:
        return None

    # Encode the text description using the pre-trained tokenizer
    input_ids = tokenizer.encode(entity_description, return_tensors='pt', pad_to_max_length=True,
                                 max_length=100).long().to('cuda:0')
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to('cuda:0')
    with torch.no_grad():
        embeddings = model(input_ids, attention_mask=attention_mask)[0][:, 0, :].cpu().numpy()
    if len(embeddings) < 1:
        return None

    return embeddings


def entities2vec(entity_ids: list):
    entity_embeddings = {}
    for entity_id in tqdm(entity_ids):
        entity_embeddings[entity_id] = extract_embeddings(entity_id, model)
    with open('entity_embedding.vec', 'w') as entity_embedding_file:
        for key, values in entity_embeddings.items():
            values = np.array2string(values[0][:100], separator='\t',
                                     formatter={'float_kind': lambda x: "%.6f" % x}).strip("[]").replace("\n",
                                                                                                         "\t").replace(
                " ", "")
            str_k_v = key + '\t' + values + '\n'
            entity_embedding_file.write(str_k_v)


def get_description(id):
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

# Take all the wikidata id of the movies entities
movies_entities = entities_movies(config) 
entities_embedded_movies = entities2vec(movies_entities)
