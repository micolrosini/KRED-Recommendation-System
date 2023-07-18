# -*- coding: utf-8 -*-
""" 
################################################################################################################
# This file contains the code for the preprocessing of the Adressa dataset (Data Enrichment extension of KRED) #
#                                                                                                              #
# The code has been developed on google Colab and is meant to be executed on the same platform thanks to the   #
#   offere computational resources. Nevertheless, thi code can be reproduced also locally if provided with     #
#   the correct input data and fixing the input/out paths whenever reading and saving files.                   #
#                                                                                                              #
# Original file is located at                                                                                  #
#     https://colab.research.google.com/drive/1uMYgMKydbwHmCvm_3DRDCh6nxc815Acn                                #
#                                                                                                              #
# Remark: The code present in this file has been delivered as part of the project, but the results obtained by #
#           this preprocessing have been saved on our team's Google Drive and can be used to directly run the  #
#           model. The purpose of this code therefore is to enhance the reproducibility of the project         #
#                                                                                                              #
#         Comments includind the word "REMARK" will guid the user to apply the needed changes to run locally   #
################################################################################################################

# **Deep NLP project - Recommendation Systems**

Python Notebook for the development and testing of the team project for the *Deep Natural Language Processing* course (a.a. 2022/2023) @ *Politecnico di Torino*.

The [corresponding GitHub repository](https://github.com/micolrosini/KRED-Reccomendation-System) consists of the following contents:
- Implementation of the code
- Problem statement & related works
- Dataset description
- Report paper about the whole activity
- Link to this notebook for the quick execution of the code

***Authors:*** \\
*@micolrosini - s302935@studenti.polito.it* \\
*@gaiasabbatini - s291532@studenti.polito.it* \\
*@matteogarbarino - s265386@studenti.polito.it*

"""


"""
#################################################
1. Import libraries & install the needed packages
#################################################
"""

# Required packages:
%pip install tensorboardX
%pip install -U sentence-transformers


# Import libraries & install packages
from google.colab import drive      # to mount a Drive containing the dataset
import os, sys, importlib           # to handle files and directories
from time import time               # to estimate performances
import tarfile                      # to unzip the dataset
import json                         # to read structured raw text
import pandas as pd                 # for dataset management
from sys import getsizeof           # to check size of a variable (use %whos to see loist of all variables)
import gc                           # garbage collector for memory management
import pickle                       # used to store pickled versions of the datasets
from tqdm import tqdm               # to show progress of operations
import glob                         # to use regular expressions & paths
import shutil                       # to easily copy files
from urllib.parse import urlparse   # to retrieve specified fields from url string
import ast                          # to convert the string representations to dictionaries
import requests                     # to contact wikidata server
import csv                          # to build kg for addressa news

import json

json_string = "[{'Label': 'bil', 'Type': 'category', 'WikidataId': 'Q598269', 'Confidence': 0.53515625, 'OccurrenceOffsets': 1, 'SurfaceForms': ['bil']}, {'Label': 'bil', 'Type': 'taxonomy', 'WikidataId': 'Q598269', 'Confidence': 0.53515625, 'OccurrenceOffsets': 1, 'SurfaceForms': ['bil']}, {'Label': 'biler', 'Type': 'concept', 'WikidataId': 'Q63860334', 'Confidence': 0.06640625, 'OccurrenceOffsets': 1, 'SurfaceForms': ['biler']}]"
updated_json_string = json_string.replace("'", "\"")
try:
    data = json.loads(updated_json_string)
    print(data)
except json.JSONDecodeError as e:
    print(f"Error decoding JSON: {str(e)}")

"""
##################################################################################################
2. Environment setup
- Clone the GitHub repository of the project to get the code
- Position the working directory in the appropriate folder
- Mount the Google Drive in order to access the dataset (contact the team to have access to it)
- Move the dataset in the appropriate folder to be accessed by the code
##################################################################################################
"""

# REMARK: IPython Magic command (%) might not work properly depending on your local system
# REMARK: By Cloning the GitHub repository it will be possible to access the rest of the code to luch the extension
#           (not stricly needed for the preprocessing)
# REMARK: Mounting the Goodle Drive allows access to the dataset, which is needed for the preprocessing
#           To run the preprocessing locally it's sufficient to:
#           - Download the Adressa Light version from https://reclab.idi.ntnu.no/dataset/
#           - Place the file AdressaSMALL.tar.gz into an arbitrary folder and read it from there
#             (when running tarfile.open(datapath+fname, "r:gz") in section 1.1)
# ===> therefore the next block of lines can be skipped:


t1 = time()

# Clone the GitHub repository
repo = "KRED-Reccomendation-System"
if os.path.isdir(repo):
  %rm -rf {repo}

%git clone https://github.com/micolrosini/KRED-Reccomendation-System.git

# Enter the correct directory (with main.py)
%cd 'KRED-Reccomendation-System'

# Mount Drive with Dataset
drive.mount('/content/drive')

# Move the dataset in the correct folder for the code to run
%cp -R /content/drive/MyDrive/KRED/KRED/data /content/KRED-Reccomendation-System
%cp -R /content/drive/MyDrive/AdressaProcessed /content/KRED-Reccomendation-System/data

t2 = time()
print(f"\n\nStage completed in {str(round(t2-t1, 2))} seconds")



"""
###################################################################################################################
# ** Data Enrichment Extension: Adressa Norwegian news dataset**

This experiment uses the Adressa (small: 1 week) dataset which is a Norwegian news dataset including also a field 
  representing the time spent by a useron an article. This information can be exploited to better infer the actual 
  interest the user had towards each article and therefore enrich the model.

**Requirements**:
In order to run this section it's necessary to execute before sections:
 - *1. Import libraries & install the needed packages*
 - Download and place the dataset manually into the desired directory (and pass the path to the next block of code)

###################################################################################################################

## 1.1 Unzip dataset

###################################################################################################################
"""

t1 = time()

# UNZIP THE DATASET

# File structured as:
#   AdressaSMALL.tar.gz --> one_week.tar --> one_week (folder) --> 7 files, one for each day of the week

fname = "AdressaSMALL.tar.gz"               # dataset compressed file name
# REMARK: fix the dataset path on the next line to run the code on your local machine
datapath = "./data/"                        # dataset location
tar = tarfile.open(datapath+fname, "r:gz")  # open the .tar.gz file
tar.extractall(path=datapath)               # extract one_week folder and place it in ./data
tar.close()                                 # close the file handle

t2 = time()
print(f"\n\nStage completed in {str(round(t2-t1, 2))} seconds")

"""## Insights about the dataset attributes:

The dataset is mostly, but not only, composed of 2 types of rows (namely "Long row" and "Short row")
 which are JSON formatted strings (read in memory as dictionaries).

The documentation of the dataset offers a description of most of the present fields.

The Short Row is (almost always) a subset of Long Row.



| Attribute | Type | Description | Long row | Short row |
|:--------------|:-----------|:------------|:------------|:------------|
| eventId | int | The identifier used to differentiate distinct events from the same user | True | True |
| activeTime | int | The active time on a page in seconds, if known | True | True |
| os | string | Operating system that the user used when log in. | True | True
| referrerUrl | sting | The URL of the referrer page. | True | False
| deviceType | string | The type of the device. | True | True
| sessionStart | boolean | Indicates whether the event is considered as the first event in session | True | True |
| sessionStop | boolean | Indicates whether the event is considered as the last event in session | True | True |
| userId | string | The cross-site user identifier which can be used to differentiate devices/browsers, or identify different subscription users by the user id | True | True |
| category | sting | The category of the news article. | False | False |
| city | string | The city name inferred from the IP address. | True | True |
| country | string | The country code inferred from the user’s IP address. | True | True |
| region | string | The region code inferred from the IP address. | True |  True |
| time | int | The time of event, measured in Unix time. | True | True |
| canonicalUrl | sting | Canonical URL as calculated based on incoming events and the fetched page content. | True | False |
| documentId (present as "id") | string | The document id. This will be the same for different URLs | True | False |
| title | string | The title of the article. | True | False |
| keywords | list | The keywords of the article. | True | False |
| namedEntities | list | The named entities of the article, including their types, counts and weights. | False | False |
| author | string | The author of the article. | True | False |
| publishTime | string | The publish time of the article. | True | False |
| profile | Array of object | A set of items which are extracted or generated from the page content. Usually a string or keywords or Named Entities from the page | True | False |
| item\* | sting | Item extracted or generated from the page content. Usually a string or keyword extracted from the page. | True* | False |
| referrerHostClass | string | NOT PRESENT IN THE DOCUMENTATION | True | True |
| url | string | NOT PRESENT IN THE DOCUMENTATION | True | True |
| referrerSearchEngine | string | NOT PRESENT IN THE DOCUMENTATION | True | False |
| query | string | NOT PRESENT IN THE DOCUMENTATION | ? | ? |
| referrerSocialNetwork | ? | NOT PRESENT IN THE DOCUMENTATION | ? | ? |
| category1 | ? | NOT PRESENT IN THE DOCUMENTATION | ? | ? |
| referrerQuery | ? | NOT PRESENT IN THE DOCUMENTATION | ? | ? |

(*) Item is not a field of the rows, but is a subfield of profile in the long rows

### Conclusions
Attributes to keep which are needed by the model:
- eventId --> primary key of the table
- time --> to get a ordering & needed by behaviours.tsv
- userId --> needed by behaviours.tsv
- url --> needed by both news.ts used as News ID
- profile --> needed by news.tsv to extract entities
- category1 --> needed by news.tsv to extract entities
- keywords --> needed by news.tsv to extract entities
- title --> needed by news.tsv to extract entities
- activeTime --> needed to recreate user behavior.tsv
- sessionStart --> needed to recreate user history (possibly removable)
- sessionStop --> needed to recreate user history (possibly removable)

Irregularity in the rows schema will be addressed later, for the moment those fields will be None

"""

"""
###################################################################################################################
## 1.2 Read dataset in a proper data structure

The dataset is composed of 7 files, one for each day of a week. Files are in .txt extension in which rows are strings with json formatting.

The 7 daily files are individually too big (w.r.t. Colab 12GB of RAM) to be directly loaded and processed in RAM within a single operation.

The procedure is the following:
Block 1.2.1
1. Read one daily file at the time
2. For each daily file, process it in batches of X rows at the time
3. The file is initially read as a list of strings (rows), then exploit JSON formatting to transition to a list of dictionaries and eventually obtain a pandas dataframe
4. In the process at point 3. many other transformations are applied in order to properly clean the data and reorganize the dataframe schema
5. Each dataframe obtained at point 4. (of the given batch size X) is saved as a segment of the given day
6. When all the segments of a day have been processed they are again loaded in memory and reassembled as a single day to be finally stored as a single .csv file

Block 1.2.2
7. All the 7 full daily .csv files are read and merged into a single weekly file
8. The weekly file is saved as a .csv file
9. All the daily and weekly files are also copied to the mounted Google Drive for future use

Remark: pickle has been abandoned due to lack of RAM, CSV format is in use, but
        the old variable and path names including "pickle" remain for sake of simplicity
        (could be refactored)

###################################################################################################################
"""

t_start = time()

days = [1,2,3,4,5,6,7]

for d in days:
  SELECTED_DAY = d                            # loop on all the days

  BATCH_SIZE = 350000                         # how many lines to process at the time

  if SELECTED_DAY not in range(1,8):          # check selection is in [1,2,3,4,5,6,7]
    exit(-1)

  VERBOSE = False                              # print some additional info

  SAVE_MODE = "csv"                           # use pickle only with more than 12 GB of ram
  if SAVE_MODE not in ['csv', 'pkl']:         # check selection is valid
    exit(-1)

  DELETE_FINAL_DATAFRAME = True               # clear final df (whole day)

  # REMARK: fix the input path according to where the original compressed data was extracted
  # REMARK: fix the output path where segments, days and final week file will be stored!
  datapath = "./data/"                                        # just a reminder
  adressa_datapath = datapath + "one_week/"                   # build the path of the unzipped Adressa dataset
  pickled_dataset_path = datapath + "pickled_dataset/"        # build the path where the csv datasets will be stored
  segments_dataset_path = pickled_dataset_path + "segments/"  # build the path where the daily csv segments will be stored

  filenames = os.listdir(adressa_datapath)    # get the list of the 7 daily filenames
  inputFileName = filenames[SELECTED_DAY-1]

  # Complete list of dataset attributes:
  """
  | eventId | city | activeTime | url | referrerHostClass | region | time
  | userId | sessionStart | deviceType | sessionStop | country | os | referrerUrl
  | profile | category1 | canonicalUrl | publishtime | keywords | id | title
  | author | referrerSearchEngine | referrerSocialNetwork | referrerQuery | query |
  """
  # Keep a subset of attributes
  attributes = [
                'eventId',        # primary key of the table
                'time',           # to get a ordering & needed by behaviours.tsv
                'userId',         # needed by behaviours.tsv

                #'id',            # newsId, needed by both news.tsv & behaviours.tsv BUT DROPPED since it not complete (url will work also as newsId, i.e. primary key for news)
                'url',            # needed by both news.ts (AND read line above)
                'profile',        # needed by both news.ts
                'category1',      # needed by both news.ts
                'keywords',       # needed by both news.ts
                'title',          # needed by both news.ts

                'activeTime',     # needed to classify into positive and negative list + useful to extract user history
                'sessionStart',   # needed to recreate user history (possibly removable, but it's just a Boolean, doesn't take much space)
                'sessionStop',    # needed to recreate user history (possibly removable, but it's just a Boolean, doesn't take much space)
                ]


  # =================================== START ===================================

  print(f"Preprocessing file {SELECTED_DAY} of {len(filenames)}")


  # 1. Read single file from disk
  # 2. split in rows
  # 3. free memory
  # 4. remove empty rows

  t1 = time()

  dataset = list()

  with open(adressa_datapath + inputFileName) as daily_file:  # open the selected file

    print('\t- Reading file from disk')
    fileContents = daily_file.read()                          # load the file into a single big string

    print('\t- Transforming file into a list of rows')
    dataset = dataset + fileContents.split("\n")              # split the whole file into lines & save the obtained rows into a list

    del fileContents                                          # free the memory
    gc.collect()                                              # call the garbage collector to force variables deletion

    dataset = list(filter(lambda a: a != '', dataset))        # lambda expression to keep only non-empty rows

  t2 = time()
  print(f"\t  Stage completed in {str(round(t2-t1, 2))} seconds")

  # 5. Loop on batches
  # 6. Restrict dataset size
  # 7. Cast to list of dictionaries
  # 8. Cast to datadrame
  # 9. Project on attributes
  # 10. Reorder & rename columns
  # 11. Free memory

  t1 = time()

  rowsOfTheDay = len(dataset)                             # total rows of selected day
  segmentsProcessed = 0                                   # how many batches have already been processed so far

  while segmentsProcessed*BATCH_SIZE < rowsOfTheDay:      # as long as there are still rows to process
    startIndex = segmentsProcessed*BATCH_SIZE             # reposition start index
    stopIndex = startIndex + BATCH_SIZE                   # reposition end index
    if stopIndex > rowsOfTheDay:                          # fix end index out of boundaries of dataframe
      stopIndex = rowsOfTheDay

    print(f'\n- Processing batch {segmentsProcessed+1} with indices [{startIndex},{stopIndex}] (total rows: {rowsOfTheDay})')

    datasetSmall = dataset[startIndex:stopIndex]          # restrict the dataset size to a batch
    gc.collect()                                          # free memory (nothing to free at first iteration)

    # Exploit JSON formatting of rows to get a list of dictionaries
    print('\t- Transforming the json-formatted list of rows into a list of dictionaries')
    c = 0                                                 # counter to track progression
    JSONDataset = list()                                  # new dataset handle
    for r in datasetSmall:                                # loop over each single row (string with json formatting)
      JSONDataset.append(json.loads(r))                   # each json row is transformed to a dictionary
      c += 1                                              # count the row as read
      print(f"\r\t- Casting JSON strings to dictionaries progress: {str(round(c/(stopIndex-startIndex)*100, 2))}%",end='')
    print(f"\r\t- Casting JSON strings to dictionaries progress: {str(round(c/(stopIndex-startIndex)*100, 2))}%")
    del datasetSmall                                      # free memory asap
    gc.collect()                                          # call the garbage collector to force variables deletion

    print('\t- Transforming the list of dictionaries into a pandas dataframe')
    df = pd.DataFrame(JSONDataset)                        # cast list of dictionaries to Pandas DataFrame
    del JSONDataset                                       # free memory asap
    gc.collect()                                          # call the garbage collector to force variables deletion

    print('\t- Projecting on the needed dataframe attributes')
    dfSmall = df[attributes]                              # project onto the needed attributes only
    dfSmall = dfSmall.reindex(columns=attributes)         # rearrange columns order

    if "category1" in dfSmall.columns:                    # rename column "category"
      dfSmall.rename(columns = {'category1':'category'}, inplace = True)

    del df                                                # free memory asap
    gc.collect()                                          # call the garbage collector to force variables deletion

    t2 = time()
    print(f"\t  Stage completed in {str(round(t2-t1, 2))} seconds\n")



    # 12. Save csv version of the segment
    # 13. Go back to 5. until all file has been processed

    t1 = time()
    outputSegmentName = f"day{SELECTED_DAY}segment{segmentsProcessed+1}.csv" # e.g. day1segment1.csv
    outputPath = segments_dataset_path + outputSegmentName                   # e.g. ./data/pickled_dataset/segments/day1segment1.csv

    if not os.path.exists(segments_dataset_path):          # create ./data/pickled_dataset/segments/ if not exists
      os.makedirs(segments_dataset_path)

    print('\t- Saving the segment on disk')
    dfSmall.to_csv(outputPath, index = False)              # ./data/pickled_dataset/segments/day1segment1.csv
    del dfSmall                                            # free the memory asap
    gc.collect()                                           # call the garbage collector to force variables deletion

    t2 = time()
    print(f"\t  Stage completed in {str(round(t2-t1, 2))} seconds\n")


    segmentsProcessed += 1
    print()

  print(f"- All batches have been processed and saved\n")

  # 14. re-load all .csv segments
  # 15. concatenate dataframe segments into a single dataframe
  # 16. save final pickled versionn of the day

  t1 = time()

  print(f"Reassembling segments of day {SELECTED_DAY}:")

  print('- Reading all the .CSV segments to build final dataset')

  segmentNames = glob.glob(segments_dataset_path + '*.csv')         # read all segments full paths

  c = 0                                                             # track the progression
  segments = list()                                                 # list of segments df that are going to be read
  for segmentName in segmentNames:                                  # for each csv segment
    print(f"\r\t- Unpickling segments: {str(round(c/len(segmentNames)*100, 2))}%",end='')
    segments.append(pd.read_csv(segmentName, index_col=False))      # read the segment
    c += 1
  print(f"\r\t- Unpickling segments: {str(round(c/len(segmentNames)*100, 2))}%")
  gc.collect()                                                      # free some memory in case of memory leaks

  print(f"\t- Concatenating segments into a single dataframe")
  finalDf = pd.concat(segments, ignore_index=True)                  # build final dataset concatenating segments (whole day)
  print(f"\t- Freeing memory")
  segments.clear()                                                  # free memory asap
  del segments                                                      # free memory asap
  gc.collect()                                                      # free memory asap

  if VERBOSE:                                                       # additional info on screen
    print(f"Dataframe length: {len(finalDf)} rows")
    print(f"Dataframe size: {str(round(getsizeof(finalDf) /1024 /1024,2))} MB")
    print(finalDf.info())

  if SAVE_MODE == 'pkl':                                            # DEPRECATED
    outputFileName = f"day{SELECTED_DAY}.pkl"
    outputPath = pickled_dataset_path + outputFileName
    print(f'- Saving final dataset as: {outputPath}')
    finalDf.to_pickle(outputPath)

  elif SAVE_MODE == 'csv':                                          # TO BE USED
    outputFileName = f"day{SELECTED_DAY}.csv"
    outputPath = pickled_dataset_path + outputFileName               # e.g. /content/KRED-Reccomendation-System/data/pickled_dataset/day1.csv
    print(f'- Saving final dataset as: {outputPath}')
    finalDf.to_csv(outputPath, index = False)

  else:
    # Should not happen
    print(f'- Failed to save final dataframe')

  print(f'- Saved successfully')

  if DELETE_FINAL_DATAFRAME:                                        # delete whole day df if not needed immediately
    del finalDf                                                     # free memory asap

  gc.collect()                                                      # free memory asap

  print(f'- Removing segments')
  if os.path.isdir(segments_dataset_path):                          # segments folder must be removed
    %rm -rf {segments_dataset_path}                                 #  to avoid interactions over different runs

  t2 = time()
  print(f"  Stage completed in {str(round(t2-t1, 2))} seconds\n")


t_end = time()
print(f"\n\nStage completed in {str(round(t_end-t_start, 2))} seconds")

"""1.2.2 Build final dataset:
- load all 7 daily files
- build weekly dataset
- store weekly dataset in the mounted Drive
"""

t1 = time()

DELETE = True                                               # cleanup at the end

dailyFileNames = glob.glob(pickled_dataset_path + '*.csv')  # get list of daily files paths

dayilyDfs = list()                                          # read all daily files into a list
progress = 0
for name in dailyFileNames:
  print(name)
  dayilyDfs.append(pd.read_csv(name, index_col=False))
  print(f"Dataframe length: {len(dayilyDfs[progress])} rows")
  print(f"Dataframe size: {str(round(getsizeof(dayilyDfs[progress]) /1024 /1024,2))} MB\n")
  progress += 1

weeklyDf = pd.concat(dayilyDfs, ignore_index=True)          # concatenate all daily files into a weekly file
print(f"Dataframe length: {len(weeklyDf)} rows")
print(f"Dataframe size: {str(round(getsizeof(weeklyDf) /1024 /1024,2))} MB\n")
print(weeklyDf.info())


if DELETE:                                                   # cleanup
  dayilyDfs.clear()
  del(dayilyDfs)
  gc.collect()


outputFileName = "week.csv"                         # Save weekly df on disk
outputPath = pickled_dataset_path + outputFileName  # e.g. /content/KRED-Reccomendation-System/data/pickled_dataset/week.csv
print(f'- Saving final dataset as: {outputPath}')
weeklyDf.to_csv(outputPath, index = False)          # WARNING: files is 9+GB --> takes a lot of time

# save on drive --> i.e. copy files in drive folder
# REMARK: no need to do this locally, but can as well backup in another folder for redundancy
shutil.copy("/content/KRED-Reccomendation-System/data/pickled_dataset/week.csv", "/content/drive/MyDrive/AdressaProcessed/week.csv")
shutil.copy("/content/KRED-Reccomendation-System/data/pickled_dataset/day1.csv", "/content/drive/MyDrive/AdressaProcessed/day1.csv")
shutil.copy("/content/KRED-Reccomendation-System/data/pickled_dataset/day2.csv", "/content/drive/MyDrive/AdressaProcessed/day2.csv")
shutil.copy("/content/KRED-Reccomendation-System/data/pickled_dataset/day3.csv", "/content/drive/MyDrive/AdressaProcessed/day3.csv")
shutil.copy("/content/KRED-Reccomendation-System/data/pickled_dataset/day4.csv", "/content/drive/MyDrive/AdressaProcessed/day4.csv")
shutil.copy("/content/KRED-Reccomendation-System/data/pickled_dataset/day5.csv", "/content/drive/MyDrive/AdressaProcessed/day5.csv")
shutil.copy("/content/KRED-Reccomendation-System/data/pickled_dataset/day6.csv", "/content/drive/MyDrive/AdressaProcessed/day6.csv")
shutil.copy("/content/KRED-Reccomendation-System/data/pickled_dataset/day7.csv", "/content/drive/MyDrive/AdressaProcessed/day7.csv")


t2 = time()
print(f"\nStage completed in {str(round(t2-t1, 2))} seconds\n")

"""
#######################################################################################################################
##1.3 Read & preprocess the structured Adressa dataset

1.3.1 Load the desired input

Can load one of the seven daily files or the week file (latter to be used for final model)
#######################################################################################################################
"""

t1 = time()

SELECTED_INPUT = 'week.csv'         # select the needed input file
VERBOSE = True                      # to print additional information

if SELECTED_INPUT not in ['week.csv', 'day1.csv', 'day2.csv', 'day3csv', 'day4.csv', 'day5.csv', 'day6.csv', 'day7.csv']:
  exit(-1)

# REMARK: fix the input path according to where the data was saved
inputFolder = '/content/KRED-Reccomendation-System/data/AdressaProcessed/'  # use this copy in local disk, never modify version in Drive folder
inputPath = inputFolder + SELECTED_INPUT                                    # e.g. /content/KRED-Reccomendation-System/data/AdressaProcessed/day1.csv

inputDf = pd.read_csv(inputPath, index_col=False)
print(f'Loaded file: {SELECTED_INPUT}')
if VERBOSE:
  print(f"Dataframe length: {len(inputDf)} rows")
  print(f"Dataframe size: {str(round(getsizeof(inputDf) /1024 /1024,2))} MB\n")
  print(inputDf.info())
  # print(inputDf[0:10])

t2 = time()
print(f"\nStage completed in {str(round(t2-t1, 2))} seconds\n")

"""1.3.2 Work towards behaviours.tsv format
- compute mean & standard deviation of reading time (for each user)
- classify each entry as element to be put into positive or negative list
- build dataset structured as behaviours.tsv
"""

# Build the columns of mean active time for each user
# --> obtain: userDf
t1 = time()

VERBOSE = True

userTimeDf = inputDf.groupby(['userId'])["activeTime"].aggregate(['min', 'max', 'median', 'mean', 'std'])     # for each userId obtain mean & std dev of activeTime (+ other aggregations useful for data exploration)
userTimeDf.rename(columns = {'mean':'userMeanActiveTime', 'std':'userStdDevActiveTime'}, inplace = True)      # rename useful columns
userTimeDf = userTimeDf[['userMeanActiveTime', 'userStdDevActiveTime']]                                       # select columns for the join (userId isn't a column, but the index of the table)

userDf = inputDf.join(userTimeDf, on='userId', validate='m:1')                                                # left-join over userId, validate n:1 multiplicity of key attributes

if VERBOSE:
  print(userTimeDf.info())
  print(userDf.info())

del userTimeDf                                                                  # free memory
gc.collect()                                                                    # free memory

t2 = time()
print(f"\nStage completed in {str(round(t2-t1, 2))} seconds\n")

"""
1. Filter out useless attributes
2. Filter out users whose activeTime is always NaN --> not informative (do not allow computation of userMeanActiveTime)
3. Set all remaining NaN activeTimes to -1 (this tell us that the user immediately closed the page, in less than a second)
4. Classify each row as element for a Positive List or Negative List
6. Build positive & negative lists and drop users with empty lists
7. Build a new datadrame with same schema of behaviors.tsv and save it
"""

t1 = time()

VERBOSE = True
DOUBLE_BOUNDARY = True

# 1. FILTER OUT USELESS ATTRIBUTES
#   Select only the columns needed by behaviours.tsv
#   ---> obtain: behaviorDf1

print("\n1. Projecting on columns subset")
attributes = ['userId', 'time', 'url', 'activeTime', 'userMeanActiveTime', 'userStdDevActiveTime']
behaviorDf1 = userDf[attributes]
if VERBOSE:
  print("\n\n=============== STEP 1 ===============\n")
  print(behaviorDf1.info())


# 2. FILTER OUT NON-INFORMATIVE USERS
#   Keep only a subset of rows where the users have at least one activeTime not NaN (i.e. discard users for which can't compute mean active time)
#   ---> obtain: behaviorDf2

print("\n2. Removing users without at least 1 activeTime!=NaN")
behaviorDf2 = behaviorDf1[behaviorDf1['userMeanActiveTime'].notna()]   # filter out users whose activeTime = NaN for all rows
behaviorDf2 = behaviorDf2[behaviorDf2['userStdDevActiveTime'].notna()]   # filter out users whose activeTime = NaN for all rows
if VERBOSE:
  print("\n\n=============== STEP 2 ===============\n")
  print(f'Dropped {len(behaviorDf1)-len(behaviorDf2)} rows of the initial {len(behaviorDf1)}')
  print(behaviorDf2.info())


# 3. FILL "MISSING" DATA
#   Whenever activeTime = NaN it means that activeTime < 1second therefore it can still be treated as a low active time and flagged with a -1 to make it numerical
#   ---> obtain: behaviorsDf3
#   should this approach have negative impact on the performances they can still be dropped instead (since are flagged with a unique value)
print("\n3. Filling remaining NaN activeTime with -1")
behaviorDf3 = behaviorDf2
behaviorDf3['activeTime'] = behaviorDf3['activeTime'].fillna(-1)
# behaviorDf3 = behaviorDf3[behaviorDf3['activeTime'].notna()]           # use this row instead to drop rows with -1
if VERBOSE:
  print("\n\n=============== STEP 3 ===============\n")
  filled = len(behaviorDf3[behaviorDf3['activeTime']==-1])
  print(f'Filled {filled} rows of the {len(behaviorDf3)} total rows')
  print(behaviorDf3.info())


# 4. CLASSIFY EACH ROW (i.e. each clicked news by each user) AS POSITIVE/NEGATIVE LIST ELEMENT
#   If activeTime <  meanUserActiveTime - 1 * userStdDevActiveTime --> Negative List
#   If activeTime >= meanUserActiveTime - 1 * userStdDevActiveTime --> Negative List
#   alternatively use a range: positive if inside range [mean - stdDev ; mean + stdDev] & negative otherwise
#   ---> obtain: behaviorDf4

print("\n4. Classifying clicks")
behaviorDf4 = behaviorDf3

if DOUBLE_BOUNDARY:
  behaviorDf4['condition1'] = (behaviorDf4['activeTime'] >= (behaviorDf4['userMeanActiveTime'] - behaviorDf4['userStdDevActiveTime']))
  behaviorDf4['condition2'] = (behaviorDf4['activeTime'] <= (behaviorDf4['userMeanActiveTime'] + behaviorDf4['userStdDevActiveTime']))
  behaviorDf4['positiveElement'] = (behaviorDf4['condition1'] == behaviorDf4['condition2'])
else:
  behaviorDf4['positiveElement'] = (behaviorDf4['activeTime'] >= (behaviorDf4['userMeanActiveTime'] - behaviorDf4['userStdDevActiveTime']))
attrib = ['userId', 'time', 'url', 'positiveElement']
behaviorDf4 = behaviorDf4[attrib]



#   (DEPRECATED approach below)
#   Add two columns with satisfaction indices as satisfactionIndexMean   = activeTime / userMeanActiveTime
#                                                satisfactionIndexMedian = activeTime / userMedianActiveTime
#   Normalize these indices in range [0,1] to make them comparable with each other --> not necessary since it's just used for ordering
# behaviorDf4['satisfactionIndexMean']   = userDf['activeTime'] / userDf['userMeanActiveTime']      # compute a statistic based on the mean active time of each user
# Indices normalization
# minSatisfactionIndexMean = behaviorDf4['satisfactionIndexMean'].min()
# maxSatisfactionIndexMean = behaviorDf4['satisfactionIndexMean'].max()
# behaviorDf4['satisfactionIndexMeanNormalized'] = (behaviorDf4['satisfactionIndexMean'] - minSatisfactionIndexMean) / (maxSatisfactionIndexMean - minSatisfactionIndexMean)


if VERBOSE:
  print("\n\n=============== STEP 4 ===============\n")
  posCount = len( behaviorDf4[behaviorDf4['positiveElement'] == True] )
  negCount = len( behaviorDf4[behaviorDf4['positiveElement'] == False] )
  print(f'{posCount} rows classified as positive clicks ({str(round(posCount/len(behaviorDf4)*100, 2))}%)')
  print(f'{negCount} rows classified as negative clicks ({str(round(negCount/len(behaviorDf4)*100, 2))}%)')
  print(behaviorDf4.info())



t2 = time()
print(f"\nStage completed in {str(round(t2-t1, 2))} seconds\n")

t1 = time()

updateFreq = 500
GC_FREQ = 1000
VERBOSE = True


behaviorDf5 = pd.DataFrame(columns=['userId', 'timestamp', 'positiveList', 'negativeList'])         # final dataframe initialization

uniqueUsersDf = behaviorDf4['userId'].unique()                                                      # list of unique userIds
amountOfUsers = len(uniqueUsersDf)                                                                  # count of unique userIds
totRowsProcessed = 0                             # count how many rows from behaviorDf4 have been processed
inputRowsKept = 0                                # count how many rows from behaviorDf4 have been kept and condensed into pos/neg lists
totRows = len(behaviorDf4)                       # tot rows to be processed from behaviorDf4
progress = 0                                     # track progress of behaviorDf5 creation (row by row) --> i.e. how many users have been processed and kept
usersDropped = 0                                 # count how many users dropped due to empty lists


print(f'(printing updates every: {updateFreq} users - tot users: {amountOfUsers})\n')

for usr in uniqueUsersDf:                                                                           # for each user
  if (progress+usersDropped) % updateFreq == 0:                                                     # print every X users processed (to not impact on performances)
    print(f"\rProcessing: {str(round((progress+usersDropped)/amountOfUsers*100, 2))}% - Elapsed time: {str(round(time()-t1, 2))}s",end='')
  userClicksDf = behaviorDf4[behaviorDf4['userId'] == usr]                                          # get the user's clicks
  tstamp = (userClicksDf.sort_values(by=['time'], ascending=True).head(1)).iloc[0]['time']          # get the earlier timestamp of that user's clicks

  positiveClicks = (userClicksDf[userClicksDf['positiveElement'] == True])['url'].unique().tolist() # get positive list
  negativeClicks = (userClicksDf[userClicksDf['positiveElement'] == False])['url'].unique().tolist()# get negative list
  commonElements = set(positiveClicks).intersection(negativeClicks)                                 # check if the lists have elements in common
  if len(commonElements) > 0:                                                                       # if they have, remove them from the longer list
    for elem in commonElements:
      if len(negativeClicks) >= len(positiveClicks):
        negativeClicks.remove(elem)
      else:
        positiveClicks.remove(elem)
  commonElements = set(positiveClicks).intersection(negativeClicks)                                 # double check if some common elements still exist (redundandt)
  if len(commonElements) > 0:
    print(f"\nWARNING: overlapping positive & negative lists at row {progress}")
    print(commonElements)

  totRowsProcessed += len(userClicksDf)                                                             # keep track of how many rows have been read from input

  if len(positiveClicks) == 0 or len(negativeClicks) == 0:                                          # if at least a list is empty --> discard user
    usersDropped += 1
  else:                                                                                             # else add new row and increase counter progress
    behaviorDf5.loc[progress] = [usr, tstamp, positiveClicks, negativeClicks]
    inputRowsKept += len(positiveClicks)                                                            # keep count of how many clicks have been kept
    inputRowsKept += len(negativeClicks)
    progress += 1

  del positiveClicks                                                                                # delete all temporary dfs
  del negativeClicks
  del userClicksDf
  del commonElements
  if (progress+usersDropped) % GC_FREQ == 0:
    gc.collect()                                                                                    # free memory every X users

print(f"\rProcessing: {str(round((progress+usersDropped)/amountOfUsers*100, 2))}%")

print(f'Tot input rows processed: {totRowsProcessed} wrt to total input rows: {len(behaviorDf4)}')
print(f'Dropped input rows: {totRowsProcessed-inputRowsKept} (due to empty lists)')
print(f'Users dropped: {usersDropped} wrt to processed users: {usersDropped+progress}(due to empty lists)')



# SAVE ON DISK & ON MOUNTED DRIVE:

DELETE = False                                                # cleanup dataframe at the end?

# REMARK: fix the output path according to where the data was saved
outputFileName = SELECTED_INPUT                               # usually "day1.csv" or "week.csv"
outputFolder = '/content/KRED-Reccomendation-System/data/AdressaProcessed/'
outputPath = outputFolder + 'behavior-' + outputFileName      # '/content/KRED-Reccomendation-System/data/AdressaProcessed/behavior-day1.csv'
print(f'\nSaving final dataset as: {outputPath}')
behaviorDf5.to_csv(outputPath, index = True)

if DELETE:                                                    # cleanup
  print('\n Deleting final dataframe')
  behaviorDf5.clear()
  del(behaviorDf5)
  gc.collect()

# save on drive --> i.e. copy files in drive folder
# REMARK: Unnecessary if run locally, might as well copy in another folder as backup
mountedDriveOutputPath = "/content/drive/MyDrive/AdressaProcessed/"+'behavior-' + outputFileName
print(f'Copying final dataset into mounted Drive at: {mountedDriveOutputPath}')
shutil.copy(outputPath, mountedDriveOutputPath)

if VERBOSE:
  print(behaviorDf5)

t2 = time()
print(f"\nStage completed in {str(round(t2-t1, 2))} seconds\n")

"""1.3.2.1  Build weekly dataset *behavior-week.tsv*

- Read the 7 daily behavior files: behavior-day1.csv, behavior-day2.csv, ...
- Reshape the columns properly (there was an error in the formatting of history and positive/negative lists)
- Concatenate the 7 files into one single file
"""

# LOAD THE DAILY FULLY PROCESSED DATAFRAMES
t1 = time()

SELECTED_INPUTS = ['behavior-day1.csv', 'behavior-day2.csv', 'behavior-day3.csv', 'behavior-day4.csv', 'behavior-day5.csv', 'behavior-day6.csv', 'behavior-day7.csv']
VERBOSE = False                                                                 # to print additional information

# REMARK: fix the input path according to where the data was saved
inputFolder = '/content/KRED-Reccomendation-System/data/AdressaProcessed/'      # source folder

dfsToReshape = list()                                                           # list where input DFs will be stored
counter = 0
for i in SELECTED_INPUTS:
 inputPath = inputFolder + i                                                    # e.g. /content/KRED-Reccomendation-System/data/AdressaProcessed/behavior-day1.csv
 dfsToReshape.append(pd.read_csv(inputPath, index_col=False))                   # read all daily behaviors and add them to the list
 print(f'Loaded file: {SELECTED_INPUTS[counter]}')
 if VERBOSE:
    print(f"Dataframe length: {len(dfsToReshape[counter])} rows")
    print(f"Dataframe size: {str(round(getsizeof(dfsToReshape[counter]) /1024 /1024,2))} MB\n")
    print(dfsToReshape[counter].info())
 counter += 1

print(f"The {len(dfsToReshape)} input dataframes have been loaded correctly")
if VERBOSE:
 print(dfsToReshape)

t2 = time()
print(f"\nStage completed in {str(round(t2-t1, 2))} seconds\n")

# RESHAPE THE COLUMNS PROPERLY
t1 = time()

reshapedDfs = list()                                                            # list to store the final version of the daily behavior dataframes
counter = 0

for df in dfsToReshape:                                                         # for each daily behavior DF

  newDf = df.copy(deep=True)                                                    # make a true copy, not just a link
  newDf['newPositiveList'] = ''                                                 # add two new columns
  newDf['newPosNegList'] = ''
  for index, row in newDf.iterrows():                                                                                           # process it row by row
    newPosListCol = row['positiveList'].replace("'", "").replace("[", "").replace("]", "").replace(",", "")                     # create a userHistory column from the previous positiveList (it's an approximation)
    oldPosList = row['positiveList'].replace("'", "").replace("[", "").replace("]", "-1").replace(",", "").replace(" ", "-1 ")  # append a -1 to all the elements of the positive list (as if they were indeed clicked in KRED)
    oldNegList = row['negativeList'].replace("'", "").replace("[", "").replace("]", "-0").replace(",", "").replace(" ", "-0 ")  # append a -0 to all the elements of the negative list (as if they were NOT clicked in KRED)
    newPosNegListCol = oldPosList + ' ' + oldNegList                                                                            # build one final list of positive and negative elements as required by KRED in input
    newDf.at[index, 'newPositiveList'] = newPosListCol                                                                          # store obtained results in the new columns
    newDf.at[index, 'newPosNegList'] = newPosNegListCol

  columnsToKeep = ['userId', 'timestamp', 'newPositiveList', 'newPosNegList']                                                   # project on the needed columns only
  newDf = newDf[columnsToKeep]
  newDf = newDf.rename(columns={'newPositiveList': 'userHistory', 'newPosNegList': 'positiveAndNegativeLists'})                 # give meaningful names

  counter += 1
  print(f"\n\nCompleted processing of day {counter} behaviors")
  print(newDf.info())


  reshapedDfs.append(newDf)                                                                                                     # store the final version of the daily behavior DF


t2 = time()
print(f"\nStage completed in {str(round(t2-t1, 2))} seconds\n")

t1 = time()
# CONCATENATE DAYS AND SAVE
behaviorWeek = pd.concat(reshapedDfs)                                                                       # build weekly behavior DF by concatenation
behaviorWeek['index'] = behaviorWeek.index + 1                                                              # build the index column (needed to match KRED input schema)
behaviorWeek = behaviorWeek[['index', 'userId', 'timestamp', 'userHistory', 'positiveAndNegativeLists']]

# REMARK: fix the output path according to where the data was saved
outputFileName = "behavior-week.tsv"                                                                        # Save weekly df on disk
outputPath = "/content/KRED-Reccomendation-System/data/AdressaProcessed/" + outputFileName
print(f'Saving final dataset as: {outputPath}')
# REMARK: fix the output path according to where the data was saved
if not os.path.exists("/content/KRED-Reccomendation-System/data/AdressaProcessed/"):                        # create directory to store data if doesnt exists
  os.makedirs("/content/KRED-Reccomendation-System/data/AdressaProcessed/")

behaviorWeek.to_csv(outputPath, sep="\t", index = False, header = False)                                    # save as .tsv (tab separated values)

# save on drive --> i.e. copy files in drive folder
# REMARK: unnecessary if run locally, might as well copy in another folder as backup
print(f'Saving backup on mounted Drive: /content/drive/MyDrive/AdressaProcessed/behavior-week.tsv')         # save backup on Google Drive
shutil.copy("/content/KRED-Reccomendation-System/data/AdressaProcessed/behavior-week.tsv", "/content/drive/MyDrive/AdressaProcessed/behavior-week.tsv")

t2 = time()
print(f"\nStage completed in {str(round(t2-t1, 2))} seconds\n")

# CHECKING FINAL RESULT
# REMARK: fix the input path according to where the data was saved
inputFolder = '/content/KRED-Reccomendation-System/data/AdressaProcessed/'
inputPath = inputFolder + 'behavior-week.tsv'
df = pd.read_csv(inputPath, header = None, sep='\t')
print(df)
print(df.info())

"""
The file behavior-week.tsv is the final version of the analogous of behavior.tsv needed as input by the KRED model
It will need to be renamed and placed into the directory specified by the file config.yaml
e.g. train_adressa_behaviour: "./data/behaviours_adressa.tsv"
"""


"""
########################################################################################################################
# 1.3.3 Work towards news.tsv format
*Now that all news have been extracted one can skip this section and go to 1.3.4 to directly build knowledge graph *
########################################################################################################################
"""

# Fill category and add subcategory column starting from url
def cat_subcat_from_url(url):
  """Given an URL as input, return title, category, subcategory as strings"""
  #Parse the URL
  parsed_url = urlparse(url)

  #Get path from parsed URL
  path = parsed_url.path

  #Split path into segments divided by '/'
  segments = path.split('/')

  #Extract desired parts

  #Category
  category = segments[1]

  #Subcategory
  if (len(segments) > 2):
    subcategory = segments[2]
  else:
    subcategory = None

  return category, subcategory

# get wikiid for each entity
# Function to retrieve Wikidata ID for an item
def get_wikidata_id(item):
    # Query Wikidata for the item using its label
    url = f'https://www.wikidata.org/w/api.php?action=wbsearchentities&format=json&language=en&search={item}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if 'search' in data:
            search_results = data['search']
            if search_results:
                # Get the first result (highest score)
                result = search_results[0]
                if 'id' in result:
                    return result['id']
    return None

# Transform 'profile' column
def df_get_wikidata_id(df):
  df['profile'] = df['profile'].apply(lambda x: [{'Label': item['item'], 'Type': item['group'], 'Confidence': item['weight'], 'WikidataId': get_wikidata_id(item['item'])} for item in x])
  return None

# Observe number of missing values per attribute
print(f'The Datasets contains the following null values in the fields:\n{inputDf.isnull().sum()}')
#create dataset with only profile rows
df_profile = inputDf.dropna(subset=['profile'])
df_profile = df_profile.drop(['eventId', 'time', 'userId', 'activeTime', 'sessionStart', 'sessionStop'], axis=1)
print(f'\nNew df containing only rows with profile:\n {df_profile.isnull().sum()}')

t1 = time()
print('\nStarting to extract info from news URL and update rows...\n')

# Apply the function to the 'url' column and store the results in new columns
df_profile[['new_category', 'subcategory']] = df_profile['url'].apply(lambda url: pd.Series(cat_subcat_from_url(url)))

# Fill NaN values in the 'category' column using values from 'title' column
df_profile['category'].fillna(df_profile['new_category'], inplace=True)
df_profile = df_profile.drop(['category'], axis=1)
df_profile.rename(columns={'new_category': 'category'}, inplace=True)
df_profile.head()# Apply the function to the 'url' column and store the results in new columns
df_profile[['new_category', 'subcategory']] = df_profile['url'].apply(lambda url: pd.Series(cat_subcat_from_url(url)))

# Fill NaN values in the 'category' column using values from 'title' column
df_profile['category'].fillna(df_profile['new_category'], inplace=True)
df_profile = df_profile.drop(['category'], axis=1)
df_profile.rename(columns={'new_category': 'category'}, inplace=True)
df_profile.head()

# Remove duplicate rows
df_profile = df_profile.drop_duplicates(subset='url', keep='first')

t2 = time()
print(f"\nExtraction completed in {str(round(t2-t1, 2))} seconds\n")

# Print the resulting DataFrame with unique URLs
print(f'Resulting DataFrame with unique URLs has {len(df_profile)} rows \n')

t1 = time()
print('Starting to modify profile column towards new.tsv format...\n')


# Update the 'profile' column to contain the desired list of dictionaries, remove items with 'group' : ['site', 'author', 'language', 'pageclass', 'adressa-access', 'adressa-tag']
#df_profile['profile'].to_dict()
try:
  df_profile['profile'] = df_profile['profile'].apply(ast.literal_eval)
except:
  print('profile column is already a python dictionary')
df_profile['profile'] = df_profile['profile'].apply(lambda x: [{'item': item['item'], **group} for item in x for group in item['groups']if item['item'] != '0' and group['group'] not in ['site', 'author', 'language', 'pageclass', 'adressa-access', 'adressa-tag']])# Update the 'profile' column to contain the desired list of dictionaries, remove items with 'group' : ['site', 'author', 'language', 'pageclass', 'adressa-access', 'adressa-tag']
#df_profile['profile'].to_dict()
try:
  df_profile['profile'] = df_profile['profile'].apply(ast.literal_eval)
  df_profile['profile'] = df_profile['profile'].apply(lambda x: [{'item': item['item'], **group} for item in x for group in item['groups']if item['item'] != '0' and group['group'] not in ['site', 'author', 'language', 'pageclass', 'adressa-access', 'adressa-tag']])
except:
  print('profile column is already a python dictionary')
  # Update the 'profile' column to contain the desired list of dictionaries
  df_profile['profile'] = df_profile['profile'].apply(lambda x: [item for item in x if item['item'] != '0' and item['group'] not in ['site', 'author', 'language', 'pageclass', 'adressa-access', 'adressa-tag']])
#df_profile['profile'] = df_profile['profile'].apply(lambda x: [{'item': item['item'], **group} for item in x for group in item['groups']if item['item'] != '0' and group['group'] not in ['site', 'author', 'language', 'pageclass', 'adressa-access', 'adressa-tag']])

# Update the 'profile' column to contain the desired list of dictionaries
#df_profile['profile'] = df_profile['profile'].apply(lambda x: [item for item in x if item['item'] != '0' and item['group'] not in ['site', 'author', 'language', 'pageclass', 'adressa-access', 'adressa-tag']])

t2 = time()
print(f"\nOperation completed in {str(round(t2-t1, 2))} seconds\n")
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
t1 = time()
print('Adding WikidataID to news entities and remove etities not found on wikidata server...\n')
# Load the datasets
inputPath_news = inputFolder + SELECTED_INPUT
inputPath_ent =  inputFolder + 'entities-week.csv'
df_entities = pd.read_csv(inputPath_ent, index_col=False)
df_news = df_profile

# Create a dictionary to map wikiid values to entities
wikiid_dict = dict(zip(df_entities['item'], df_entities['wikiid']))

# Iterate over each row in the news dataset
for index, row in tqdm(df_news.iterrows()):
    profile = row['profile']
    updated_profile = []

    # Iterate over each dictionary in the 'profile' list
    for dictionary in profile:
        label = dictionary['item']
        group = dictionary['group']
        weight = dictionary['weight']
        count = dictionary['count']

        # Get the corresponding wikiid from the entities dataset
        wikiid = wikiid_dict.get(label)

        # Only add dictionaries with a valid wikiid
        if wikiid is not None:
            updated_dict = {
                'Label': label,
                'Type': group,
                'WikidataId': wikiid,
                'Confidence': weight,
                'OccurrenceOffsets': count,
                'SurfaceForms': [label.split(',')[0]]  # Modify?
            }
            updated_profile.append(updated_dict)

    # Update the 'profile' column with the updated profile list
    df_news.at[index, 'profile'] = updated_profile

t2 = time()
print(f"\nOperation completed in {str(round(t2-t1, 2))} seconds\n")

# Update the output path construction
outputFileName = SELECTED_INPUT.split('.')[0] + '_updated_news_dataset.tsv'
# REMARK: update the output folder if running locally
outputPath = outputFolder + outputFileName

print(f'\nSaving final dataset as: {outputPath}')

# Save the updated dataset as TSV
df_news.to_csv(outputPath, sep='\t', index=False, quoting=csv.QUOTE_NONNUMERIC)

# Copy the file to the mounted drive folder
# REMARK: Unnecessary if running locally, might as well backup in another folder
mountedDriveOutputPath = "/content/drive/MyDrive/AdressaProcessed/" + outputFileName
print(f'Copying final dataset into mounted Drive at: {mountedDriveOutputPath}')
shutil.copy(outputPath, mountedDriveOutputPath)

df_news['profile'].iloc[0]

# REMARK: update the input folder if running locally
news_path = '/content/drive/MyDrive/AdressaProcessed/news_day2.csv'
if SELECTED_INPUT != 'day2.csv':
  # First dataset
  df1 = pd.read_csv(news_path, index_col=False, header= None)
  df1.columns = ['url']

  # Second dataset
  df2 = df_profile

  # Merge the datasets based on 'url' column
  merged = df2.merge(df1, on='url', how='left')

  # Filter the rows where the other columns are not null (not present in the first dataset)
  filtered_df2 = merged[merged['url'].isnull() | merged['profile'].isnull() | merged['keywords'].isnull() | merged['title'].isnull() | merged['category'].isnull() | merged['subcategory'].isnull()]
  df_profile = filtered_df2
  # Print the filtered dataframe
  print(f'new dataset length {len(filtered_df2)} and it has {filtered_df2.isnull().sum()} empty rows \n')

# Initialize an empty list
item_list = []

# Iterate over each row in the column
for row in tqdm(df_profile['profile']):
    # Iterate over each dictionary in the row
    for dictionary in row:
        # Extract the 'item' value from the dictionary
        item = dictionary['item']

        # Check if the 'item' value is already in the list
        if item not in item_list:
            # Append the 'item' value to the list
            item_list.append(item)

# Print the resulting list length
print(f'\n The resulting list of entities contains {len(item_list)} items')

# Create a list of dictionaries with 'item' and 'wikiid'
list_ent = []

# Iterate over the item_list
for el in tqdm(item_list):
    # Create a new dictionary for each item
    dict_ent = {}
    dict_ent['item'] = el
    dict_ent['wikiid'] = get_wikidata_id(el)

    # Append the dictionary to the list
    list_ent.append(dict_ent)

# save current day entities and wiki_id into a csv
#DELETE = False                                                # cleanup dataframe at the end?

# REMARK: update the output folder if running locally
outputFileName = SELECTED_INPUT                               # usually "day1.csv" or "week.csv"
outputFolder = '/content/KRED-Reccomendation-System/data/AdressaProcessed/'
outputPath = outputFolder + 'entities-' + outputFileName      # '/content/KRED-Reccomendation-System/data/AdressaProcessed/entities-day1.csv'
print(f'\nSaving final dataset as: {outputPath}')
df_entites = pd.DataFrame.from_dict(list_ent)
df_entites.to_csv(outputPath, index = True)


# save on drive --> i.e. copy files in drive folder
# REMARK: Unnecessary if running locally
mountedDriveOutputPath = "/content/drive/MyDrive/AdressaProcessed/"+'entities-' + outputFileName
print(f'Copying final dataset into mounted Drive at: {mountedDriveOutputPath}')
shutil.copy(outputPath, mountedDriveOutputPath)

# REMARK: update the output folder if running locally
if SELECTED_INPUT != 'day2.csv':
  # Append the DataFrame rows to the existing file
  outputPath = 'news_day2.csv'
  df_profile['url'].to_csv(outputPath, header=False, index=False, mode='a')
else:
  outputPath = 'news_'+ outputFileName +'.txt'
  file = open('news_'+ outputFileName +'.txt','w')
  for item in df_profile['url']:
    file.write(item+"\n")
  file.close()

# save on drive --> i.e. copy files in drive folder
# REMARK: Unnecessary if running locally
mountedDriveOutputPath = "/content/drive/MyDrive/AdressaProcessed/"+'news_' + outputPath
print(f'Copying final dataset into mounted Drive at: {mountedDriveOutputPath}')
shutil.copy(outputPath, mountedDriveOutputPath)

"""## 1.3.4 Build KG

"""

def get_entity_info(entity_id):
    response = requests.get(f"https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json")
    if response.status_code == 200:
        entity_data = response.json()
        entities = entity_data.get('entities', {})
        if entity_id in entities:
            return entities[entity_id]
    return None

# REMARK: update the inpuy folder if running locally (according to where these files where saved previously)
input_path = "/content/drive/MyDrive/AdressaProcessed/"
# Concatenate datasets vertically
# First dataset
df1 = pd.read_csv(input_path + 'entities-day1.csv', index_col=False) # entities extracted from day 1
df1.dropna(subset=['wikiid'], inplace=True) # Drop rows with NaN values in the 'wikiid' column
df2 = pd.read_csv(input_path + 'entities-day2.csv', index_col=False) # entities extracted from day 2
df2.dropna(subset=['wikiid'], inplace=True) # Drop rows with NaN values in the 'wikiid' column
df3 = pd.read_csv(input_path + 'entities-day3.csv', index_col=False) # entities extracted from day 3
df3.dropna(subset=['wikiid'], inplace=True) # Drop rows with NaN values in the 'wikiid' column
df4 = pd.read_csv(input_path + 'entities-day4.csv', index_col=False) # entities extracted from day 4
df4.dropna(subset=['wikiid'], inplace=True) # Drop rows with NaN values in the 'wikiid' column
df5 = pd.read_csv(input_path + 'entities-day5.csv', index_col=False) # entities extracted from day 5
df5.dropna(subset=['wikiid'], inplace=True) # Drop rows with NaN values in the 'wikiid' column
df6 = pd.read_csv(input_path + 'entities-day6.csv', index_col=False) # entities extracted from day 6
df6.dropna(subset=['wikiid'], inplace=True) # Drop rows with NaN values in the 'wikiid' column
df7 = pd.read_csv(input_path + 'entities-day7.csv', index_col=False) # entities extracted from day 7
df7.dropna(subset=['wikiid'], inplace=True) # Drop rows with NaN values in the 'wikiid' column

# The ignore_index=True option resets the index of the appended dataset
appended_df = pd.concat([df1, df2, df3, df4, df5, df6, df7], ignore_index=True)

# Remove duplicates from the DataFrame
appended_df = appended_df.drop_duplicates()
# Drop unamed column
appended_df = appended_df.drop('Unnamed: 0', axis=1)
# Print the appended dataset length
print(f'we have collected {len(appended_df)} entities')

outputFileName = 'week.csv'                            # usually "day1.csv" or "week.csv"
outputFolder = '/content/KRED-Reccomendation-System/data/AdressaProcessed/'
outputPath = outputFolder + 'entities-' + outputFileName      # '/content/KRED-Reccomendation-System/data/AdressaProcessed/entities-week.csv'
print(f'\nSaving final dataset as: {outputPath}')
appended_df.to_csv(outputPath, index = True)

# save on drive --> i.e. copy files in drive folder
# REMARK: Update input/output path if running locally
mountedDriveOutputPath = "/content/drive/MyDrive/AdressaProcessed/"+'entities-' + outputFileName
print(f'Copying final dataset into mounted Drive at: {mountedDriveOutputPath}')
shutil.copy(outputPath, mountedDriveOutputPath)

def get_entity_info(entity_id):
    response = requests.get(f"https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json")
    if response.status_code == 200:
        entity_data = response.json()
        entities = entity_data.get('entities', {})
        if entity_id in entities:
            return entities[entity_id]
    return None


entity_ids = appended_df['wikiid']  # Entity IDs to build to kg
entities = {}
for entity_id in tqdm(entity_ids):
    entity_info = get_entity_info(entity_id)
    if entity_info:
        entities[entity_id] = entity_info

knowledge_graph = {}
for entity_id, entity_info in tqdm(entities.items()):
    knowledge_graph[entity_id] = {}
    if 'claims' in entity_info:
        claims = entity_info['claims']
        for prop, values in claims.items():
            property_values = []
            for value in values:
                if 'mainsnak' in value and 'datavalue' in value['mainsnak']:
                    data_value = value['mainsnak']['datavalue']
                    if data_value.get('type') == 'wikibase-entityid' and 'value' in data_value:
                        property_values.append(data_value['value']['id'])
            if property_values:
                knowledge_graph[entity_id][prop] = property_values

# Specify the output file name
# REMARK: Update input/output path if running locally
output_file = 'knowledge_graph_addressa.tsv'
outputFolder = '/content/KRED-Reccomendation-System/data/AdressaProcessed/'
outputPath = output_file     # '/content/KRED-Reccomendation-System/data/AdressaProcessed/knowledge_graph_addressa.tsv'

# Write the knowledge graph to the .tsv file
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f, delimiter='\t')

    # Iterate over each entity and its properties in the knowledge graph
    for entity_id, properties in knowledge_graph.items():
        for prop, values in properties.items():
            # Write each relationship to a row in the .tsv file
            for value in values:
                writer.writerow([entity_id, prop, value])

print(f"Knowledge graph saved to '{outputFolder + output_file}'.")

# save on drive --> i.e. copy files in drive folder
# REMARK: unnecessary if running locally
mountedDriveOutputPath = "/content/drive/MyDrive/AdressaProcessed/"+ output_file
print(f'\n Copying final KG into mounted Drive at: {mountedDriveOutputPath}')
shutil.copy(outputPath, mountedDriveOutputPath)

"""
###########################################################################################
## 1.3.5 Execution of the Adressa extension
  (out of scope of this code, just an additional, but negligible block for testing
   the real execution is perfomed cloning the GitHub repo and running the proper .py file)
###########################################################################################
"""

import os
import sys
sys.path.append('')
import os

import argparse
from parse_config import ConfigParser
from utils.util import *
from train_test import *

# REMARK: Update input/output path if running locally
data_path = "/content/KRED-Reccomendation-System/data/AdressaProcessed"

train_news_file = os.path.join(data_path, 'train', r'news.tsv')
train_behaviors_file = os.path.join(data_path, 'train', r'behaviors.tsv')
valid_news_file = os.path.join(data_path, 'valid', r'news.tsv')
valid_behaviors_file = os.path.join(data_path, 'valid', r'behaviors.tsv')
knowledge_graph_file = os.path.join(data_path, 'kg/kg', r'wikidata-graph.tsv')
entity_embedding_file = os.path.join(data_path, 'kg/kg', r'entity2vecd100.vec')
relation_embedding_file = os.path.join(data_path, 'kg/kg', r'relation2vecd100.vec')

# mind_url, mind_train_dataset, mind_dev_dataset, _ = get_mind_data_set(MIND_type)

#kg_url = "https://kredkg.blob.core.windows.net/wikidatakg/"

#if not os.path.exists(train_news_file):
#    download_deeprec_resources(mind_url, os.path.join(data_path, 'train'), mind_train_dataset)

#if not os.path.exists(valid_news_file):
#    download_deeprec_resources(mind_url, \
#                               os.path.join(data_path, 'valid'), mind_dev_dataset)

#if not os.path.exists(knowledge_graph_file):
#    download_deeprec_resources(kg_url, \
#                               os.path.join(data_path, 'kg'), "kg")



parser = argparse.ArgumentParser(description='KRED')


parser.add_argument('-c', '--config', default="./config.yaml", type=str,
                    help='config file path (default: None)')
parser.add_argument('-r', '--resume', default=None, type=str,
                    help='path to latest checkpoint (default: None)')
parser.add_argument('-d', '--device', default=None, type=str,
                    help='indices of GPUs to enable (default: all)')

config = ConfigParser.from_args(parser)
# print(config['data'])

def limit_user2item_validation_data(data, size):
    test_data = data[-1]
    test_data_reduced = {key: test_data[key][:size] for key in test_data.keys()}
    # Concatenate the old tuple with the updated validation data
    return data[:-1] + (test_data_reduced,)


epochs = 1
batch_size = 64
train_type = "single_task"
task = "user2item" # task should be within: user2item, item2item, vert_classify, pop_predict

config['trainer']['epochs'] = epochs
config['data_loader']['batch_size'] = batch_size
config['trainer']['training_type'] = train_type
config['trainer']['task'] = task
config['trainer']['save_period'] = epochs/2
# The following parameters define which of the extensions are used,
# by setting them to False the original KRED model is executed
#if not os.path.isfile(f"{config['data']['sentence_embedding_folder']}/train_news_embeddings.pkl"):
#  write_embedding_news("./data/train", config["data"]["sentence_embedding_folder"])

#if not os.path.isfile(f"{config['data']['sentence_embedding_folder']}/valid_news_embeddings.pkl"):
#  write_embedding_news("./data/valid", config["data"]["sentence_embedding_folder"])

data = load_data_mind(config, config['data']['sentence_embedding_folder'])
print("Data loaded, ready for training")
#if not os.path.isfile(f"{data_path}/data_mind.pkl"):
#    write_data_mind(config, data_path)
#data = read_pickle(f"{data_path}/data_mind.pkl")

#test_data = data[-1]
#data = limit_user2item_validation_data(data, 10000)
print("Data loaded, ready for training")
#single_task_training(config, data)  # user2item

%python prova_adressa.py
