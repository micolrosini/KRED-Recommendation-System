import torch
import torch.nn as nn
from base.base_model import BaseModel


class KGAT(BaseModel):

    def __init__(self, config, doc_feature_dict, entity_embedding, relation_embedding, adj_entity, adj_relation):
        super(KGAT, self).__init__()
        self.config = config
        self.doc_feature_dict = doc_feature_dict
        self.adj_entity = adj_entity
        self.adj_relation = adj_relation
        self.entity_embedding = entity_embedding
        self.relation_embedding = relation_embedding
        self.attention_layer1 = nn.Linear(3 * self.config['model']['entity_embedding_dim'],
                                          self.config['model']['layer_dim'])
        self.attention_layer2 = nn.Linear(self.config['model']['layer_dim'], 1)
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU(inplace=True)
        self.convolve_layer = nn.Linear(2 * self.config['model']['entity_embedding_dim'],
                                        self.config['model']['entity_embedding_dim'])

    def get_neighbors(self, entities):
        number_of_entities = self.entity_embedding.shape[0]
        neighbor_entities = []
        neighbor_relations = []
        for entity_batch in entities:
            neighbor_entities.append([])
            neighbor_relations.append([])
            for entity in entity_batch:
                if type(entity) == int:
                    entity_adj_list = self.adj_entity[entity]
                    relation_adj_list = self.adj_relation[entity]
                    for index, (entity_i, relation_i) in enumerate(zip(entity_adj_list, relation_adj_list)):
                        if entity_i > number_of_entities:
                            entity_adj_list.pop(index)
                            relation_adj_list.pop(index)
                            entity_adj_list.insert(index, 0)
                            relation_adj_list.insert(index, 0)

                    neighbor_entities[-1].append(entity_adj_list)
                    neighbor_relations[-1].append(relation_adj_list)

                else:
                    neighbor_entities[-1].append([])
                    neighbor_relations[-1].append([])
                    for entity_i in entity:
                        entity_adj_list = self.adj_entity[entity_i]
                        relation_adj_list = self.adj_relation[entity_i]
                        # Some entities that are present in neighbor_entities does not exist in the embedding
                        for index, (entity_i, relation_i) in enumerate(zip(entity_adj_list, relation_adj_list)):
                            if entity_i > number_of_entities:
                                entity_adj_list.pop(index)
                                relation_adj_list.pop(index)
                                entity_adj_list.insert(index, 0)
                                relation_adj_list.insert(index, 0)

                        neighbor_entities[-1][-1].append(entity_adj_list)
                        neighbor_relations[-1][-1].append(relation_adj_list)

        return neighbor_entities, neighbor_relations

    def get_entity_embedding(self, neighbor_entities):
        entity_embedding_batch = []
        if type(neighbor_entities[0][0]) == int:
            neighbor_entities = torch.LongTensor(neighbor_entities)
            for i in range(len(neighbor_entities)):
                entity_embedding_batch.append([])
                for j in range(len(neighbor_entities[i])):
                    entity_embedding_batch[i].append([])
                    for entityid in neighbor_entities[i][j]:
                        entity_embedding_batch[i][j].append(self.entity_embedding[entityid])
        else:
            neighbor_entities = torch.LongTensor(neighbor_entities)
            for i in range(len(neighbor_entities)):
                entity_embedding_batch.append([])
                for j in range(len(neighbor_entities[i])):
                    entity_embedding_batch[i].append([])
                    for k in range(len(neighbor_entities[i][j])):
                        entity_embedding_batch[i][j].append([])
                        for entityid in neighbor_entities[i][j][k]:
                            entity_embedding_batch[i][j][k].append(self.entity_embedding[entityid])
        return torch.FloatTensor(torch.stack(entity_embedding_batch)).cuda()

    def get_relation_embedding(self, neighbor_relations):
        entity_embedding_batch = []
        if type(neighbor_relations[0][0]) == int:
            neighbor_relations = torch.LongTensor(neighbor_relations)
            for i in range(len(neighbor_relations)):
                entity_embedding_batch.append([])
                for j in range(len(neighbor_relations[i])):
                    entity_embedding_batch[i].append([])
                    for entityid in neighbor_relations[i][j]:
                        entity_embedding_batch[i][j].append(self.relation_embedding[entityid])
        else:
            neighbor_relations = torch.LongTensor(neighbor_relations)
            for i in range(len(neighbor_relations)):
                entity_embedding_batch.append([])
                for j in range(len(neighbor_relations[i])):
                    entity_embedding_batch[i].append([])
                    for k in range(len(neighbor_relations[i][j])):
                        entity_embedding_batch[i][j].append([])
                        for entityid in neighbor_relations[i][j][k]:
                            entity_embedding_batch[i][j][k].append(self.relation_embedding[entityid])
        return torch.FloatTensor(torch.stack(entity_embedding_batch)).cuda()

    def aggregate(self, entity_embedding, neighbor_embedding):
        concat_embedding = torch.cat([entity_embedding, neighbor_embedding], len(entity_embedding.shape) - 1)
        aggregate_embedding = self.relu(self.convolve_layer(concat_embedding))
        return aggregate_embedding

    def entity_ids_clearner(self, entity_ids):
        number_of_entities = self.entity_embedding.shape[0]
        if type(entity_ids[0][0]) == int:
            for i in entity_ids:
                for index, eny_id in enumerate(i):
                    if eny_id > number_of_entities:
                        i.pop(index)
                        i.insert(index, 0)
            return entity_ids

        # print('--- starting cleaning ---', number_of_entities)
        for batch in entity_ids:
            for i in batch:
                for index, eny_id in enumerate(i):
                    if eny_id > number_of_entities:
                        i.pop(index)
                        i.insert(index, 0)
        return entity_ids

    def forward(self, entity_ids):
        entity_ids = self.entity_ids_clearner(entity_ids)

        neighbor_entities, neighbor_relations = self.get_neighbors(entity_ids)

        entity_embedding_lookup = nn.Embedding.from_pretrained(self.entity_embedding.cuda())
        relation_embedding_lookup = nn.Embedding.from_pretrained(self.relation_embedding.cuda())

        # Convert neighbor_entities and neighbor_relations to tensors with appropriate shapes
        neighbor_entities_tensor = torch.tensor(neighbor_entities).cuda()
        neighbor_relations_tensor = torch.tensor(neighbor_relations).cuda()

        # Lookup embeddings for neighbor entities and relations
        neighbor_entity_embedding = entity_embedding_lookup(neighbor_entities_tensor)
        neighbor_relation_embedding = relation_embedding_lookup(neighbor_relations_tensor)

        entity_embedding = entity_embedding_lookup(torch.tensor(entity_ids).cuda())

        if len(entity_embedding.shape) == 3:
            entity_embedding_expand = torch.unsqueeze(entity_embedding, 2)
            entity_embedding_expand = entity_embedding_expand.expand(entity_embedding_expand.shape[0],
                                                                     entity_embedding_expand.shape[1],
                                                                     self.config['model']['entity_neighbor_num'],
                                                                     entity_embedding_expand.shape[3])
            embedding_concat = torch.cat(
                [entity_embedding_expand, neighbor_entity_embedding, neighbor_relation_embedding], 3)
            embedding_concat = embedding_concat.to(torch.float32).cuda()
            attention_value = self.softmax(self.attention_layer2(self.relu(self.attention_layer1(embedding_concat))))
            neighbor_att_embedding = torch.sum(attention_value * neighbor_entity_embedding, dim=2)
            kgat_embedding = self.aggregate(entity_embedding, neighbor_att_embedding)
        else:
            entity_embedding_expand = torch.unsqueeze(entity_embedding, 3)
            entity_embedding_expand = entity_embedding_expand.expand(entity_embedding_expand.shape[0],
                                                                     entity_embedding_expand.shape[1],
                                                                     entity_embedding_expand.shape[2],
                                                                     self.config['model']['entity_neighbor_num'],
                                                                     entity_embedding_expand.shape[4])
            embedding_concat = torch.cat(
                [entity_embedding_expand, neighbor_entity_embedding, neighbor_relation_embedding], 4)
            embedding_concat = embedding_concat.to(torch.float32).cuda()
            attention_value = self.softmax(self.attention_layer2(self.relu(self.attention_layer1(embedding_concat))))
            neighbor_att_embedding = torch.sum(attention_value * neighbor_entity_embedding, dim=3)
            kgat_embedding = self.aggregate(entity_embedding, neighbor_att_embedding)

        return kgat_embedding
