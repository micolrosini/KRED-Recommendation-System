""" 
#####################################################################################################################
# This file contains the code for the preprocessing of the MindReader dataset (Domain Adaptation extension of KRED) #
#                                                                                                                   #
# The code presents a set of functions used to transform the movies dataset in order to obtain suitable inputs      #
#      for the model of the Domain Adaptation to Movies Recommendation extension of KRED                            #
#                                                                                                                   #
# Remark: The code present in this file has been delivered as part of the project, but the results obtained by      #
#           this preprocessing have been saved on our team's Google Drive and can be used to directly run the       #
#           model. The purpose of this code, therefore, is to enhance the reproducibility of the project            #
#                                                                                                                   #
#####################################################################################################################
"""

import csv


# Use this function if you want to obtain from the file entities.csv a file with the entity id, the label and the description
def movies_entities_info():

      input_file = 'data/mind_reader_dataset/entities.csv'
      output_file = 'data/mind_reader_dataset/entity2label_movies.txt'

      with open(input_file, 'r') as csv_file, open(output_file, 'w') as txt_file:

            csv_reader = csv.reader(csv_file)
            row_text = {}

            for row in csv_reader:

                  fields = str(row).strip().split(",")
                  q_code = fields[0].split("/")[-1]
                  q_code = q_code[:-1]
                  field1 = fields[1]
                  field1 = field1[2:-1]
                  field2 = fields[2]
                  field2 = field2[2:-2]
                  row_text = q_code+"," + field1+"," +  field2
                  #row_text = str(row_text)
                  row_text = row_text

                  
                  

                  
                  txt_file.write(row_text)
                  txt_file.write("\n")


# Use this function to obtain from the file ratings a file of the behaviour of each user id
def creating_behaviours_movies():
    input_file = 'data/mind_reader_dataset/ratings-2.csv'
    output_file = 'data/mind_reader_dataset/behaviours_movies.txt'

    with open(input_file, 'r') as csv_file, open(output_file, 'w') as txt_file:

        csv_reader = csv.reader(csv_file)
        row_text = {}

        for row in csv_reader:
            fields = str(row).strip().split(",")
            field1 = fields[1]
            field1 = field1.replace("'","")
            field2 = fields[2]
            field2 = field2.split("/")
            field2  = field2[-1]
            field2 = field2[:-1]
            field3 = fields[3] # Boolean to indicate if it is a film
            field3= field3.replace("'","")
            field4 = fields[4] # label
            field4= field4.replace("'","")
            row_text =  field1 + "," +  field2 + "," + field3 + "," + field4

            txt_file.write(row_text)
            txt_file.write("\n")
     

# Use this function if you want to obtain a knowledge graph with the wikidata ids from the file triple.csv of the dataset_movies
def triple_to_id():
      input_file = 'data/mind_reader_dataset/triples.csv'
      output_file = 'data/mind_reader_dataset/triple2id1_movies.txt'

      with open(input_file, 'r') as csv_file, open(output_file, 'w') as txt_file:

            csv_reader = csv.reader(csv_file)
            row_text = {}

            for row in csv_reader:

                  fields = str(row).strip().split(",")
                  q_code = fields[1].split("/")[-1]
                  q_code = q_code[:-1]
                  field1 = fields[2]
                  field1 = field1[2:-1]
                  field2 = fields[3]
                  if field2.startswith(" 'D"):
                        
                        field2 = field2[2:-2]
                  else:
                        field2 = fields[3].split("/")[-1]
                        field2 = field2[:-2]
                  
                  row_text = q_code+"," + field1+"," +  field2
                  
                  row_text = row_text

                  
                  

                  
                  txt_file.write(row_text)
                  txt_file.write("\n")
        
      input_file_2 = 'data/mind_reader_dataset/triple2id1_movies.txt'
      output_file_2 = 'data/mind_reader_dataset/triple2id_movies.txt'

      with open(input_file_2, 'r') as csv_file,open(output_file_2, 'w') as txt_file:
            for entity in csv_file:
                        

                        head, relation, tail = entity.strip().split(',')  # only need last 2 columns
                        if relation == "FROM_DECADE":
                              relation = "P5444"
                        if relation == "PRODUCED_BY":
                              relation = "P272"
                        if relation == "STARRING":
                              relation = "P161"
                        if relation == "FOLLOWED_BY":
                              relation = "P179"
                        if relation == "HAS_GENRE":
                              relation = "P136"
                        if relation == "HAS_SUBJECT":
                              relation = "P921"
                        if relation == "DIRECTED_BY":
                              relation = "P57"
                        if tail == "Decade-1990":
                              tail = "Q34653"
                        if tail == "Decade-1980":
                              tail = "Q34644"     
                        if tail == "Decade-1970":
                              tail = "Q35014"
                        if tail == "Decade-1960":
                              tail = "Q35724"      
                        if tail == "Decade-1950":
                              tail = "Q36297"
                        if tail == "Decade-1940":
                              tail = "Q36561"
                        if tail == "Decade-1930":
                              tail = "Q35702"
                        if tail == "Decade-1920":
                              tail = "Q35736"
                        if tail == "Decade-2000":
                              tail = "Q35024"
                        if tail == "Decade-2010":
                              tail = "Q19022"

                        row_text = head+"," + relation+"," +  tail
                  
                        row_text = row_text

                  
                  

                  
                        txt_file.write(row_text)
                        txt_file.write("\n")


# Function to save in a txt file the movies wikidata and all the entities that are connected to the movies, with the corresponding relation
def link_entities_to_movies(train_list_movies, test_list_movies,movies_relation2id ,config):
    train_movies_with_features = {}
    test_movies_with_features = {}
    with open(config["data"]["knowledge_graph_movies"]) as kg, open('data/mind_reader_dataset/testmovieswith_entities.txt', 'w') as txt_file1, open('data/mind_reader_dataset/trainmovieswith_entities.txt', 'w') as txt_file2:
        for line in kg:
            fields = line.split(',')
            if fields[0] in train_list_movies:
                if fields[0] not in train_movies_with_features.keys():
                    train_movies_with_features[fields[0]] = []
                train_movies_with_features[fields[0]].append([fields[2], movies_relation2id[fields[1]]])
            if fields[2] in train_list_movies:
                if fields[2] not in train_movies_with_features.keys():
                    train_movies_with_features[fields[2]] = []
                train_movies_with_features[fields[2]].append([fields[0], movies_relation2id[fields[1]]])
            if fields[0] in test_list_movies:
                if fields[0] not in test_movies_with_features.keys():
                    test_movies_with_features[fields[0]] = []
                test_movies_with_features[fields[0]].append([fields[2],movies_relation2id[fields[1]]])
            if fields[2] in test_list_movies:
                if fields[2] not in test_movies_with_features.keys():
                    test_movies_with_features[fields[2]] = []
                test_movies_with_features[fields[2]].append([fields[0],movies_relation2id[fields[1]]])
        for key, value in test_movies_with_features.items():
            txt_file1.write(f"{key}: {value}\n")
        for key, value in train_movies_with_features.items():
            txt_file2.write(f"{key}: {value}\n")
