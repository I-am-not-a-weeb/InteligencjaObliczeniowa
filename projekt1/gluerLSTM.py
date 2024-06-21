import os
import shutil
import pandas as pd

import numpy as np

input_folder = 'LSTM_ready'
output_folder = 'LSTM_ready'

df = pd.DataFrame()

arrs = []
for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):  # Check for image files
        filepath = os.path.join(input_folder, filename)
        
        arrs.append(pd.read_csv(filepath))
        
df = pd.concat(arrs)

df.to_csv(os.path.join(output_folder, 'LSTM_ready.csv'), index=False)        