import pandas as pd;
import numpy as np;
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

Sequential = tf.keras.models.Sequential
LSTM = tf.keras.layers.LSTM
Dense = tf.keras.layers.Dense

valuable_columns  = [
    #"date","serial_number","model",
    "capacity_bytes","failure",
    "smart_1_normalized",#"smart_1_raw",          # Read Error Rate
    "smart_5_normalized",#"smart_5_raw",          # Reallocated Sectors Count/Retired Block Count
    "smart_8_normalized",#"smart_8_raw",          # Seek Time Performance
    "smart_9_normalized",#"smart_9_raw",          # Power-On Hours
    "smart_12_normalized",#"smart_12_raw",        # Power Cycle Count
    "smart_173_normalized",#"smart_173_raw",      # Wear Leveling Count
    "smart_174_normalized",#"smart_174_raw",      # Unexpected power loss count
    "smart_184_normalized",#"smart_184_raw",      # End-to-End error / IOEDC
    "smart_187_normalized",#"smart_187_raw",      # Reported Uncorrectable Errors
    "smart_194_normalized",#"smart_194_raw",      # Temperature
]

data_columns = [
    "smart_1_normalized",#"smart_1_raw",          # Read Error Rate
    "smart_5_normalized",#"smart_5_raw",          # Reallocated Sectors Count/Retired Block Count
    "smart_8_normalized",#"smart_8_raw",          # Seek Time Performance
    "smart_9_normalized",#"smart_9_raw",          # Power-On Hours
    "smart_12_normalized",#"smart_12_raw",        # Power Cycle Count
    "smart_173_normalized",#"smart_173_raw",      # Wear Leveling Count
    "smart_174_normalized",#"smart_174_raw",      # Unexpected power loss count
    "smart_184_normalized",#"smart_184_raw",      # End-to-End error / IOEDC
    "smart_187_normalized",#"smart_187_raw",      # Reported Uncorrectable Errors
    "smart_194_normalized",#"smart_194_raw",      # Temperature
]


 
Q4_2023 = pd.read_csv('2023_Q4/2023-10-01.csv',usecols=valuable_columns)
                      
imputer = SimpleImputer(strategy='mean')
imputer.fit(Q4_2023[data_columns])

Q4_2023[data_columns] = imputer.transform(Q4_2023[data_columns])

#Q4_2023 = pd.get_dummies(Q4_2023, columns=["model"])

X = Q4_2023.drop(columns=["failure"])
Y = Q4_2023["failure"]

X = np.asarray(X).astype(np.longlong)
Y = np.array(Y).astype(np.longlong)

# Preprocess the data

print("X:",X,"Y:",Y)

(X_train, X_test, y_train, y_test) = train_test_split(X, Y, test_size=0.1, random_state=69420)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape input data for LSTM [samples, timesteps, features]
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(Y.reshape(-1, 1))



print("x_train",X_train)
print("y_train",y_train)

print("x_shape",X_train.shape)
print("y_shape",y_train.shape)


model = Sequential()

model.add(LSTM(64, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',loss="binary_crossentropy", metrics=['binary_accuracy'])
model.fit(X_train_reshaped, y_train, epochs=1, batch_size=8, validation_data=(X_test_reshaped, y_test))

loss = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
predictions = model.predict(X_test)

model.save("model.h5")