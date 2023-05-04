import random
import pandas as pd
from pydub import AudioSegment
from pydub.silence import split_on_silence
import os
from emotion_mapping import emotion_map
import yaml

    
################################################

training_set_percentage = 80
validation_set_percentage = 20

training_path = "Data/training.txt"
validation_path = "Data/validation.txt"
################################################

# Define stream 
dataframe = pd.read_csv("dataset/dataset.csv", sep=";")
dataframe = dataframe.sample(frac=1)
sad = dataframe[dataframe["emotion"] == "sad"]
neutral = dataframe[dataframe["emotion"] == "neutral"]
dataframe['actor_id'] = dataframe['actor_id'].astype(str)

conversion_file = open("dataset/data.txt","w")
# Create training file

for index in range(0, sad.shape[0]):
    sad_row = sad.iloc[index]
    try:
        neutral_row = neutral.iloc[index]
    except IndexError:
        neutral_row = neutral.iloc[random.randint(0, neutral.shape[0])]
    conversion_file.write(f"{neutral_row['path']}|{sad_row['path']}\n")
    
conversion_file.close()
data_path = "dataset/data.txt"
dataframe = pd.read_csv(data_path, sep="|").sample(frac=1)

training_dataframe = dataframe.iloc[0:int((dataframe.shape[0]*training_set_percentage)/100)].to_csv("Data/training.txt", index=False, header=None)
validation_dataframe = dataframe.iloc[dataframe.shape[0]-int((dataframe.shape[0]*validation_set_percentage)/100):].to_csv("Data/validation.txt", index=False, header=None)


    