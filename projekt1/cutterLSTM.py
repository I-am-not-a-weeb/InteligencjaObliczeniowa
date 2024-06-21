import os
import shutil
import pandas as pd

import numpy as np
import tensorflow as tf

from sklearn.impute import SimpleImputer

valuable_columns  = [
    #"date",
    "serial_number",
    #"model",
    #"capacity_bytes",
    "failure",
    "smart_1_normalized",#"smart_1_raw",          # Read Error Rate
    "smart_5_normalized",#"smart_5_raw",          # Reallocated Sectors Count/Retired Block Count
    #"smart_8_normalized",#"smart_8_raw",          # Seek Time Performance
    "smart_9_normalized",#"smart_9_raw",          # Power-On Hours
    #"smart_12_normalized",#"smart_12_raw",        # Power Cycle Count
    #"smart_173_normalized",#"smart_173_raw",      # Wear Leveling Count
    #"smart_174_normalized",#"smart_174_raw",      # Unexpected power loss count
    #"smart_184_normalized",#"smart_184_raw",      # End-to-End error / IOEDC
    #"smart_187_normalized",#"smart_187_raw",      # Reported Uncorrectable Errors
    #"smart_194_normalized",#"smart_194_raw",      # Temperature
]

input_folder = 'data'
output_folder = 'data'

for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):  # Check for image files
        filepath = os.path.join(input_folder, filename)
        
        print(filename)
        df = pd.read_csv(filepath,usecols=valuable_columns)
        
        df.dropna(inplace=True)
        
        df.to_csv(os.path.join(output_folder, filename), index=False)