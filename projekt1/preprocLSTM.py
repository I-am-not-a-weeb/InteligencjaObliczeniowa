import os
import shutil
import pandas as pd

import numpy as np
import tensorflow as tf

from sklearn.impute import SimpleImputer

valuable_columns  = [
    #"date",
    "serial_number",
    "model",
    #"capacity_bytes",
    "failure",
    "smart_1_normalized",#"smart_1_raw",          # Read Error Rate
    "smart_5_normalized",#"smart_5_raw",          # Reallocated Sectors Count/Retired Block Count
    #"smart_8_normalized",#"smart_8_raw",          # Seek Time Performance
    "smart_9_normalized",#"smart_9_raw",          # Power-On Hours
    "smart_12_normalized",#"smart_12_raw",        # Power Cycle Count
    "smart_173_normalized",#"smart_173_raw",      # Wear Leveling Count
    "smart_174_normalized",#"smart_174_raw",      # Unexpected power loss count
    #"smart_184_normalized",#"smart_184_raw",      # End-to-End error / IOEDC
    #"smart_187_normalized",#"smart_187_raw",      # Reported Uncorrectable Errors
    #"smart_194_normalized",#"smart_194_raw",      # Temperature
]

data_columns = [
    #"capacity_bytes",
    "smart_1_normalized",#"smart_1_raw",          # Read Error Rate
    "smart_5_normalized",#"smart_5_raw",          # Reallocated Sectors Count/Retired Block Count
    #"smart_8_normalized",#"smart_8_raw",          # Seek Time Performance
    "smart_9_normalized",#"smart_9_raw",          # Power-On Hours
    "smart_12_normalized",#"smart_12_raw",        # Power Cycle Count
    "smart_173_normalized",#"smart_173_raw",      # Wear Leveling Count
    "smart_174_normalized",#"smart_174_raw",      # Unexpected power loss count
    #"smart_184_normalized",#"smart_184_raw",      # End-to-End error / IOEDC
    #"smart_187_normalized",#"smart_187_raw",      # Reported Uncorrectable Errors
    #"smart_194_normalized",#"smart_194_raw",      # Temperature
]

input_folder = 'output'
output_folder = 'LSTM_ready'


failed_harddrives = []

dataframe = []
prevframe = pd.DataFrame()

days = 60



print("Preprocessing data for LSTM model...")
print("Reading data from: " + input_folder)
print("Writing data to: " + output_folder)
print
fordays = days
for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):  # Check for image files
        filepath = os.path.join(input_folder, filename)
        
        print(filename)
        df = pd.read_csv(filepath, usecols={'serial_number', 'failure'})
        
        df.dropna(inplace=True)
        
        if(fordays < 1):
            failed_harddrives.append(df[df['failure'] == 1]['serial_number'].tolist())
            print("Found failed hardrives")
    
        #prevframe = df
        #output_filename = os.path.splitext(filename)[0] + '_preprocessed_LSTM.csv'
        #df.to_csv(os.path.join(output_folder, output_filename), index=False)
        if(fordays == 0):
            break
        fordays = fordays - 1
        
failed_harddrives = [item for sublist in failed_harddrives for item in sublist]


dataset = {}

for serial in failed_harddrives:
    dataset[serial] = {}
    dataset[serial]['dataframe'] = pd.DataFrame()
    dataset[serial]['reached_1'] = False
    dataset[serial]['set_dataset'] = pd.DataFrame()

fordays = days
for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):  # Check for image files
        filepath = os.path.join(input_folder, filename)
        
        print(filename)
        df = pd.read_csv(filepath, usecols=valuable_columns)
        
        for i in failed_harddrives:
            print(i)

            try:#
                #if(dataset[i]['reached_1']==0 and df[df['serial_number']==i]['failure'].values[0] == 1):
                #    dataset[i]['dataframe'] = dataset[i]['dataframe'] + df[df['serial_number']==i]
                #    dataset[i]['reached_1'] = True
                #elif(dataset[i]['reached_1']):
                #    dataset[i]['dataframe'].concat(prevframe[i])  
                if(dataset[i]['reached_1'] == False):
                    dataset[i]['dataframe'] = pd.concat([dataset[i]['dataframe'], df[df['serial_number']==i]])
                    if(df[df['serial_number']==i]['failure'].values[0] == 1):
                        dataset[i]['reached_1'] = True
                        dataset[i]['set_dataset'] = df[df['serial_number']==i]
                elif(dataset[i]['reached_1'] == True):
                    dataset[i]['dataframe'] = pd.concat([dataset[i]['dataframe'],dataset[i]['set_dataset']])
            except:
                print("error")
      
        
        prevframe = df
        
        #output_filename = os.path.splitext(filename)[0] + '_preprocessed_LSTM.csv'
        #df.to_csv(os.path.join(output_folder, output_filename), index=False)
        if(fordays == 0):
            break
        fordays = fordays - 1


for data in dataset:
    dataset[data]['dataframe'].to_csv(os.path.join(output_folder, data + '_preprocessed_LSTM.csv'), index=False)
    print(data)
    print(dataset[data]['dataframe'])

#print(pd.concat([dataset['Z304KCM6']['dataframe'],dataset['Z304KCM6']['dataframe']]))