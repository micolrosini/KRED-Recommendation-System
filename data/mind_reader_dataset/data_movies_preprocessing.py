import csv

# Use this function if you want to obtain from the file entities.csv a file with the entity id, the label and the description

def movies_entities_info():

      input_file = 'data/dataset_movies/entities.csv'
      output_file = 'data/dataset_movies/entity2label_movies.txt'

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


