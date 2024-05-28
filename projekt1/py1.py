import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from keras.layers import Flatten
from keras.layers import Dropout
from keras.optimizers import SGD
from sklearn.impute import SimpleImputer
from keras.optimizers import Adam

import tensorflow as tf

import numpy as np

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

datafiles = [
    '2023_Q4/2023-10-01.csv','2023_Q4/2023-10-02.csv','2023_Q4/2023-10-03.csv','2023_Q4/2023-10-04.csv',
    '2023_Q4/2023-10-05.csv','2023_Q4/2023-10-06.csv','2023_Q4/2023-10-07.csv','2023_Q4/2023-10-08.csv',
    '2023_Q4/2023-10-09.csv','2023_Q4/2023-10-10.csv','2023_Q4/2023-10-11.csv','2023_Q4/2023-10-12.csv',
    '2023_Q4/2023-10-13.csv','2023_Q4/2023-10-14.csv','2023_Q4/2023-10-15.csv','2023_Q4/2023-10-16.csv',
    '2023_Q4/2023-10-17.csv','2023_Q4/2023-10-18.csv','2023_Q4/2023-10-19.csv','2023_Q4/2023-10-20.csv',
    '2023_Q4/2023-10-21.csv','2023_Q4/2023-10-22.csv','2023_Q4/2023-10-23.csv','2023_Q4/2023-10-24.csv',
    '2023_Q4/2023-10-25.csv','2023_Q4/2023-10-26.csv','2023_Q4/2023-10-27.csv','2023_Q4/2023-10-28.csv',
    '2023_Q4/2023-10-29.csv','2023_Q4/2023-10-30.csv','2023_Q4/2023-10-31.csv',
    '2023_Q4/2023-11-01.csv','2023_Q4/2023-11-02.csv','2023_Q4/2023-11-03.csv','2023_Q4/2023-11-04.csv',
    '2023_Q4/2023-11-05.csv','2023_Q4/2023-11-06.csv','2023_Q4/2023-11-07.csv','2023_Q4/2023-11-08.csv',
    '2023_Q4/2023-11-09.csv','2023_Q4/2023-11-10.csv','2023_Q4/2023-11-11.csv','2023_Q4/2023-11-12.csv',
    '2023_Q4/2023-11-13.csv','2023_Q4/2023-11-14.csv','2023_Q4/2023-11-15.csv','2023_Q4/2023-11-16.csv',
    '2023_Q4/2023-11-17.csv','2023_Q4/2023-11-18.csv','2023_Q4/2023-11-19.csv','2023_Q4/2023-11-20.csv',
    '2023_Q4/2023-11-21.csv','2023_Q4/2023-11-22.csv','2023_Q4/2023-11-23.csv','2023_Q4/2023-11-24.csv',
    '2023_Q4/2023-11-25.csv','2023_Q4/2023-11-26.csv','2023_Q4/2023-11-27.csv','2023_Q4/2023-11-28.csv',
    '2023_Q4/2023-11-29.csv','2023_Q4/2023-11-30.csv',
    '2023_Q4/2023-12-01.csv','2023_Q4/2023-12-02.csv','2023_Q4/2023-12-03.csv','2023_Q4/2023-12-04.csv',
    '2023_Q4/2023-12-05.csv','2023_Q4/2023-12-06.csv','2023_Q4/2023-12-07.csv','2023_Q4/2023-12-08.csv',
    '2023_Q4/2023-12-09.csv','2023_Q4/2023-12-10.csv','2023_Q4/2023-12-11.csv','2023_Q4/2023-12-12.csv',
    '2023_Q4/2023-12-13.csv','2023_Q4/2023-12-14.csv','2023_Q4/2023-12-15.csv','2023_Q4/2023-12-16.csv',
    '2023_Q4/2023-12-17.csv','2023_Q4/2023-12-18.csv','2023_Q4/2023-12-19.csv','2023_Q4/2023-12-20.csv',
    '2023_Q4/2023-12-21.csv','2023_Q4/2023-12-22.csv','2023_Q4/2023-12-23.csv','2023_Q4/2023-12-24.csv',
    '2023_Q4/2023-12-25.csv','2023_Q4/2023-12-26.csv','2023_Q4/2023-12-27.csv','2023_Q4/2023-12-28.csv',
    '2023_Q4/2023-12-29.csv','2023_Q4/2023-12-30.csv','2023_Q4/2023-12-31.csv',
]

valuable_columns = [
    "model",
    "capacity_bytes",
    "failure",
    "smart_1_normalized",
    "smart_5_normalized",
    #"smart_8_normalized",
    #"smart_9_normalized",
    #"smart_12_normalized",
    #"smart_173_normalized",
    "smart_174_normalized",
    "smart_184_normalized",
    "smart_187_normalized",
    #"smart_194_normalized",
]

data_columns = [
    "smart_1_normalized",#"smart_1_raw",          # Read Error Rate
    "smart_5_normalized",#"smart_5_raw",          # Reallocated Sectors Count/Retired Block Count
    #"smart_8_normalized",#"smart_8_raw",          # Seek Time Performance
    #"smart_9_normalized",#"smart_9_raw",          # Power-On Hours
    #"smart_12_normalized",#"smart_12_raw",        # Power Cycle Count
    #"smart_173_normalized",#"smart_173_raw",      # Wear Leveling Count
    "smart_174_normalized",#"smart_174_raw",      # Unexpected power loss count
    "smart_184_normalized",#"smart_184_raw",      # End-to-End error / IOEDC
    "smart_187_normalized",#"smart_187_raw",      # Reported Uncorrectable Errors
    #"smart_194_normalized",#"smart_194_raw",      # Temperature
]

# Load the CSV file
df = pd.read_csv('2023_Q4/2023-10-01.csv',usecols=valuable_columns)

# Select valuable columns and drop rows with missing values

#df = df[valuable_columns].dropna()

# Convert the "failure" column to binary (0 or 1)
df['failure'] = df['failure'].astype(int)

# Split data into features and target
X = df.drop(columns=['failure', 'model'])
y = df['failure']

print("X",X)
print("y",y)

# Split the data into training and testing sets

(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.1, random_state=69420)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape input data for LSTM [samples, timesteps, features]
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Define the LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]),activation='relu' ))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=Adam(clipvalue=1.0), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model

for i in range (1, int(len(datafiles)/30)):
    print(datafiles[i])
    df = pd.read_csv(datafiles[i],usecols=valuable_columns)
    
    imputer = SimpleImputer(strategy='mean')
    imputer.fit(df[data_columns])

    df[data_columns] = imputer.transform(df[data_columns])
    
    X = df.drop(columns=['failure', 'model'])
    y = df['failure']
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.15, random_state=69420)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
    
    len_train = len(y_train)
    class_1_count = sum(y_train == 1)

    #weight_for_0 = 1.0 / (2 * (len_train - class_1_count))
    #weight_for_1 = 1.0 / (2 * class_1_count)
    #class_weights = {0: weight_for_0, 1: weight_for_1}
    
    total_count = len(y_train)
    class_0_count = sum(y_train == 0)
    class_1_count = sum(y_train == 1)
    class_weights = {0: total_count / (2 * class_0_count), 1: total_count / (2 * class_1_count)}
    
    #class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    
    model.fit(X_train_reshaped, y_train,shuffle=True, epochs=32, batch_size=256, validation_data=(X_test_reshaped, y_test),class_weight=class_weights)
    #model.fit(X_train_reshaped, y_train, epochs=16, batch_size=32, validation_data=(X_test_reshaped, y_test),class_weight=dict(enumerate(class_weights)))
    model.save('model.keras')

#model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, validation_data=(X_test_reshaped, y_test))
