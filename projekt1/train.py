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
import seaborn as sns

import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

import os
import shutil

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

df_f = pd.read_csv('output/2023-10-01_preprocessed.csv',usecols=valuable_columns)

X = df_f.drop(columns=['failure', 'model'])
y = df_f['failure']

(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.15  , random_state=69420)

# Standardize the features
#scaler = StandardScaler()
#X_train_scaled = scaler.fit_transform(X_train)
#X_test_scaled = scaler.transform(X_test)

# Reshape input data for LSTM [samples, timesteps, features]
#X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
#X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

days = 180

dataframes = []

input_folder = 'output'
for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):  
        filepath = os.path.join(input_folder, filename)
        print(filepath)
        df = pd.read_csv(filepath, usecols=valuable_columns)
        dataframes.append(df)
        
    if days < 0:
        break
    days-=1
        
df = pd.concat(dataframes, ignore_index=True)



# TEEEEEEEEEEEEEEEEEEEEEEEEEEST
duplicates = df.duplicated(subset="model", keep=False)
        


failure_condition = df['failure'] == 0
duplicates_with_failure_zero = duplicates & failure_condition

# Step 5: Randomly drop 50% of the filtered duplicates
def drop_half_duplicates(df, duplicate_mask, drop_fraction = 0.5):
    duplicates_df = df[duplicate_mask].copy()
    # Create a random mask to drop 50% of the duplicates
    random_mask = np.random.rand(len(duplicates_df)) < drop_fraction
    drop_indices = duplicates_df[random_mask].index
    return df.drop(index=drop_indices)

df = drop_half_duplicates(df, duplicates_with_failure_zero, 0.99992)

# /TEEEEEEEEEEEEEEEEEEEEEEEEEEST
        
X = df.drop(columns=['failure', 'model'])
y = df['failure']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=69420)
#X_train_scaled = scaler.fit_transform(X_train)
#X_test_scaled = scaler.transform(X_test)
#X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
#X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

#len_train = len(y_train)
#class_1_count = sum(y_train == 1)
#weight_for_0 = 1.0 / (2 * (len_train - class_1_count))
#weight_for_1 = 1.0 / (2 * class_1_count)
#class_weights = {0: weight_for_0, 1: weight_for_1}
total_count = len(y_train)+len(y_test)
class_0_count = sum(y_train == 0)+sum(y_test==0)
class_1_count = total_count - class_0_count

    #class_weights = {0: total_count / (2 * (class_0_count)), 1: total_count / (2 * (class_1_count))}
class_weights = {
    0: 1.0/
        (1.0 + np.exp((total_count-(2*class_1_count))/(total_count/2))),
    1: 1.0/
        (1.0 + np.exp((total_count-(2*class_0_count))/(total_count/2)))
}
    
print(total_count)
print(class_0_count)
print(class_1_count)
print((total_count-(2*class_0_count))/(total_count/2))
print(class_weights)
print(class_weights[0]+class_weights[1])
#class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)

#model.fit(X_train, y_train,shuffle=True, epochs=32, batch_size=256, validation_data=(X_test, y_test),class_weight=class_weights)
history = model.fit(tf.convert_to_tensor(X_train, dtype=tf.float32),
          y_train,
          shuffle=True,
          epochs=64,
          batch_size=1,
          validation_data=(X_test, y_test),
          class_weight=class_weights)
#model.fit(X_train_reshaped, y_train, epochs=16, batch_size=32, validation_data=(X_test_reshaped, y_test),class_weight=dict(enumerate(class_weights)))
model.save('model.keras')
        
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# Plotting training and validation accuracy
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True, linestyle="--", color="grey")
plt.legend()

# Plotting training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True, linestyle="--", color="grey")
plt.legend()

plt.tight_layout()
plt.show()

# Label names in the order of the indices
#label_names = [
#    label for label, index in sorted(label_mapping.items(), key=lambda x: x[1])
#]
# Predict on test images
