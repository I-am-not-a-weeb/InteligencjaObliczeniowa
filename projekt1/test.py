import pandas as pd
from keras.saving import load_model
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from sklearn.impute import SimpleImputer

import os

valuable_columns  = [
    #"date",
    #"serial_number",
    "model",
    #"capacity_bytes",
    "failure",
    "smart_1_normalized",#"smart_1_raw",          # Read Error Rate
    "smart_5_normalized",#"smart_5_raw",          # Reallocated Sectors Count/Retired Block Count
    #"smart_8_normalized",#"smart_8_raw",          # Seek Time Performance
    "smart_9_normalized",#"smart_9_raw",          # Power-On Hours
    "smart_12_normalized",#"smart_12_raw",        # Power Cycle Count
    #"smart_173_normalized",#"smart_173_raw",      # Wear Leveling Count
    #"smart_174_normalized",#"smart_174_raw",      # Unexpected power loss count
    #"smart_184_normalized",#"smart_184_raw",      # End-to-End error / IOEDC
    #"smart_187_normalized",#"smart_187_raw",      # Reported Uncorrectable Errors
    #"smart_194_normalized",#"smart_194_raw",      # Temperature
]

files = [
    '2023_Q4/2023-12-01.csv','2023_Q4/2023-12-02.csv','2023_Q4/2023-12-03.csv','2023_Q4/2023-12-04.csv',
    '2023_Q4/2023-12-05.csv','2023_Q4/2023-12-06.csv','2023_Q4/2023-12-07.csv','2023_Q4/2023-12-08.csv',
    '2023_Q4/2023-12-09.csv','2023_Q4/2023-12-10.csv','2023_Q4/2023-12-11.csv','2023_Q4/2023-12-12.csv',
    '2023_Q4/2023-12-13.csv','2023_Q4/2023-12-14.csv','2023_Q4/2023-12-15.csv','2023_Q4/2023-12-16.csv',
    '2023_Q4/2023-12-17.csv','2023_Q4/2023-12-18.csv','2023_Q4/2023-12-19.csv','2023_Q4/2023-12-20.csv',
    '2023_Q4/2023-12-21.csv','2023_Q4/2023-12-22.csv','2023_Q4/2023-12-23.csv','2023_Q4/2023-12-24.csv',
    '2023_Q4/2023-12-25.csv','2023_Q4/2023-12-26.csv','2023_Q4/2023-12-27.csv','2023_Q4/2023-12-28.csv',
    '2023_Q4/2023-12-29.csv','2023_Q4/2023-12-30.csv','2023_Q4/2023-12-31.csv',
]

dataframe = []

for i in range(len(files)):
    files[i] = 'output/' + files[i].split('/')[1].split('.')[0] + '_preprocessed.csv'
    dataframe.append(pd.read_csv(files[i], usecols=valuable_columns))




data_columns  = [
    "smart_1_normalized",#"smart_1_raw",          # Read Error Rate
    "smart_5_normalized",#"smart_5_raw",          # Reallocated Sectors Count/Retired Block Count
    #"smart_8_normalized",#"smart_8_raw",          # Seek Time Performance
    "smart_9_normalized",#"smart_9_raw",          # Power-On Hours
    "smart_12_normalized",#"smart_12_raw",        # Power Cycle Count
    #"smart_173_normalized",#"smart_173_raw",      # Wear Leveling Count
    #"smart_174_normalized",#"smart_174_raw",      # Unexpected power loss count
    #"smart_184_normalized",#"smart_184_raw",      # End-to-End error / IOEDC
    #"smart_187_normalized",#"smart_187_raw",      # Reported Uncorrectable Errors
    #"smart_194_normalized",#"smart_194_raw",      # Temperature
]

df = dataframe[0]

imputer = SimpleImputer(strategy='mean')
imputer.fit(df[data_columns])

df[data_columns] = imputer.transform(df[data_columns])


expected_0 = df[df['failure'] == 0]
X_0 = expected_0.drop(columns=['failure', 'model'])
X_0 = X_0.sample(n=10)

expected_1 = df[df['failure'] == 1]
X_1 = expected_1.drop(columns=['failure', 'model'])
X_1 = X_1.sample(n=10)

print("X_0:",X_0)
print("X_1:",X_1)

loaded_model = load_model('model.keras')

model = loaded_model
print(loaded_model.summary())

Y_1 = loaded_model.predict(X_1)
pred_Y1 = pd.Series(Y_1.flatten())
Y_0 = loaded_model.predict(X_0)
pred_Y0 = pd.Series(Y_0.flatten())

print("Y_1:",Y_1)
print("Y_0:",Y_0)

global_mean = (pred_Y1.mean() + pred_Y0.mean())/2

summary_1 = {
    'total_predictions': len(pred_Y1),
    'good_predictions': (pred_Y1 > global_mean).sum(),
    'percent_good_predictions': (pred_Y1 > global_mean).sum() / len(pred_Y1) * 100,
    'mean': pred_Y1.mean(),
    'median': pred_Y1.median(),
    'max': pred_Y1.max(),
    'min': pred_Y1.min(),
    'std_dev': pred_Y1.std()
}


summary_0 = {
    'total_predictions': len(pred_Y0),
    'good_predictions': (pred_Y0 < global_mean).sum(),
    'percent_good_predictions': (pred_Y0 < global_mean).sum() / len(pred_Y0) * 100,
    'mean': pred_Y0.mean(),
    'median': pred_Y0.median(),
    'max': pred_Y0.max(),
    'min': pred_Y0.min(),
    'std_dev': pred_Y0.std()
}

print("\n\n")
print("Expected 1: ")
for key, value in summary_1.items():
    print(f"{key}: {value}")
    
print("\n\n")
print("Expected 0:")
for key, value in summary_0.items():
    print(f"{key}: {value}")



dataframes = []

input_folder = 'output'
for filename in files:
    if filename.endswith(".csv"):  
        #filepath = os.path.join(input_folder, filename)
        filepath = filename
        print(filepath)
        df = pd.read_csv(filepath, usecols=valuable_columns)
        dataframes.append(df)
        
df = pd.concat(dataframes, ignore_index=True)



failure_condition = df['failure'] == 0
duplicates = df.duplicated(subset="model", keep=False)

duplicates_with_failure_zero = duplicates & failure_condition

# Step 5: Randomly drop 50% of the filtered duplicates
def drop_half_duplicates(df, duplicate_mask, drop_fraction = 0.5):
    duplicates_df = df[duplicate_mask].copy()
    # Create a random mask to drop 50% of the duplicates
    random_mask = np.random.rand(len(duplicates_df)) < drop_fraction
    drop_indices = duplicates_df[random_mask].index
    return df.drop(index=drop_indices)

df = drop_half_duplicates(df, duplicates_with_failure_zero, 0.99992)

X = df.drop(columns=['failure', 'model'])
Y = df['failure']


#df.drop(columns=['failure', 'model'])

predictions = model.predict(X)
predictions = np.array([[1] if x[0] > global_mean else [0] for x in predictions])
predicted_labels = np.argmax(predictions, axis=1)



true_labels = np.argmax(Y, axis=0)

# Confusion matrix
cm = confusion_matrix(y_true=Y, y_pred=predictions,normalize='true')
plt.figure(figsize=(10, 7))
sns.heatmap(
    cm,
    annot=True,
    #mt="d",
    cmap="Blues",
    #xticklabels=label_names,
    #yticklabels=label_names,
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# Evaluate on test set


""" scaler = StandardScaler()
X = scaler.fit_transform(X)
X_0 = scaler.transform(X_0)

print("X_01:",X_0)
X_0_test = X_0.reshape((X_0.shape[0], 1, X_0.shape[1]))
print("X_02:",X_0)

X_test = X.reshape((X.shape[0], 1, X.shape[1]))
print("X: ",X)

loaded_model = load_model('model.keras')
print(loaded_model.summary())
print(loaded_model.predict(X_test)) 
print("With 0?")
print(loaded_model.predict(X_0_test)) """