import pandas as pd
import os
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler


Sequential = tf.keras.models.Sequential
LSTM = tf.keras.layers.LSTM
Dense = tf.keras.layers.Dense
Model = tf.keras.models.Model
Input = tf.keras.layers.Input
Adam = tf.keras.optimizers.Adam
Binary_crossentropy = tf.keras.losses.BinaryCrossentropy
Categorical_crossentropy = tf.keras.losses.CategoricalCrossentropy
Binary_accuracy = tf.keras.metrics.BinaryAccuracy

valuable_columns=[
    'failure',
    'smart_1_normalized',
    'smart_5_normalized',
    'smart_9_normalized',
    #'smart_12_normalized'
]

""" dataframes = []

input_folder = 'LSTM_ready'
for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):  
        filepath = os.path.join(input_folder, filename)
        print(filepath)
        df = pd.read_csv(filepath, usecols=valuable_columns)
        dataframes.append(df)
        
df = pd.concat(dataframes, ignore_index=True) """

#input_folder = 'LSTM_ready'
""" df = pd.DataFrame()

dataframes = []

for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):  
        filepath = os.path.join(input_folder, filename)
        print(filepath)
        df = pd.read_csv(filepath)
        dataframes.append(df)
        
df = pd.concat(dataframes, ignore_index=True)

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
 n_vars = 1 if type(data) is list else data.shape[1]
 df = pd.DataFrame(data)
 cols, names = list(), list()
 # input sequence (t-n, ... t-1)
 for i in range(n_in, 0, -1):
    cols.append(df.shift(i))
    names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
 for i in range(0, n_out):
    cols.append(df.shift(-i))
    if i == 0:
        names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
    else:
        names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
 # put it all together
 agg = pd.concat(cols, axis=1)
 agg.columns = names
 # drop rows with NaN values
 if dropnan:
    agg.dropna(inplace=True)
 return agg


prelstm_5 = series_to_supervised(df,1,5)

print(prelstm_5.head())

values = prelstm_5.values
 """
 
values = pd.read_csv('LSTM_ready/LSTM_ready.csv').values

print("\nvalues:",values[:5])

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(values)

print("\ndataset:",dataset[:5])
""" 
duplicates = values.duplicated(keep=False)
failure_condition = df['failure'] == 0
duplicates_with_failure_zero = duplicates & failure_condition

def drop_half_duplicates(df, duplicate_mask, drop_fraction = 0.5):
    duplicates_df = df[duplicate_mask].copy()
    # Create a random mask to drop 50% of the duplicates
    random_mask = np.random.rand(len(duplicates_df)) < drop_fraction
    drop_indices = duplicates_df[random_mask].index
    return df.drop(index=drop_indices)
values = drop_half_duplicates(values, duplicates_with_failure_zero, 1) 
"""

X, Y = dataset[:, :-5], dataset[:, -5:]



(train_X, test_X, train_Y, test_Y) = train_test_split(X, Y, test_size=0.2, random_state=69420)


#train = values[:n_train_hours, :]
#test = values[n_train_hours:, :]

# split into input and outputs
#train_X, train_y = train[:, :-5], train[:, -5:]
#test_X, test_y = test[:, :-5], test[:, -5:]


train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

train_Y = train_Y.T
test_Y  = test_Y.T



#print("Train: ",train_X[:3], train_Y[:3],"\nTest: ", test_X[:3], test_Y[:3])
#print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)


inputs = Input(shape=(train_X.shape[1], train_X.shape[2]))

lstm = LSTM(64, input_shape=(train_X.shape[1], train_X.shape[2]))
x=lstm(inputs)
x=Dense(32, activation='relu')(x)

failure_output = Dense(1, activation='sigmoid', name='failure_output')(x)
smart1_output = Dense(1, activation='linear', name='smart1_output')(x)
smart5_output = Dense(1, activation='linear', name='smart5_output')(x)
smart9_output = Dense(1, activation='linear', name='smart9_output')(x)
#smart12_output = Dense(1, activation='linear', name='smart12_output')(x)
outputs = [failure_output, smart1_output, smart5_output, smart9_output]#, smart12_output]

model = Model(inputs=inputs, outputs=outputs)


epochs = 64
batch_size = 32


model.compile(optimizer=Adam(learning_rate=0.01),
    loss={
        'failure_output': 'binary_crossentropy',
        'smart1_output': 'mse',
        'smart5_output': 'mse',
        'smart9_output': 'mse',
        #'smart12_output': 'mse'
        },
    metrics={
        'failure_output': Binary_accuracy(threshold=0.5),
        'smart1_output': 'msle',
        'smart5_output': 'msle',
        'smart9_output': 'msle',
        #'smart12_output': 'accuracy'
    }
)
              #loss={
              #      'dense_1': 'binary_crossentropy',
              #      'dense_2': 'categorical_crossentropy',
              #      'dense_3': 'categorical_crossentropy',
              #      'dense_4': 'categorical_crossentropy',
              #      'dense_5': 'categorical_crossentropy'
              #  },
              #metrics={
              #      'dense_1': 'accuracy',
              #      'dense_2': 'accuracy',
              #      'dense_3': 'accuracy',
              #      'dense_4': 'accuracy',
              #      'dense_5': 'accuracy'
              #  })

history = model.fit(train_X,
                    {'failure_output':train_Y[0],
                     'smart1_output': train_Y[1],
                     'smart5_output': train_Y[2],
                     'smart9_output': train_Y[3],
                     #'smart12_output': train_Y[4]
                     },
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(test_X,
                                    {
                                      'failure_output':test_Y[0],
                                      'smart1_output': test_Y[1],
                                      'smart5_output': test_Y[2],
                                      'smart9_output': test_Y[3],
                                      #'smart12_output': test_Y[4]
                                    }
                    ),
                    shuffle=True)

model.save('notes/modelLSTM_'+str(epochs)+'_'+str(batch_size)+'.keras')
          
          
          
          
print("\nPredykcja: ",model.predict(test_X[4100].reshape((test_X[4100].shape[0], 1, test_X[4100].shape[1]))))
          
          
notes_dir = 'notes/'
          
plt.figure(figsize=(14, 6))

print(history.history.keys())

plt.subplot(2, 2, 1)
plt.plot(history.history["loss"], label="Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True, linestyle="--", color="grey")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(history.history["failure_output_binary_accuracy"], label="Failure Accuracy")
plt.plot(history.history["failure_output_loss"], label="Failure Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True, linestyle="--", color="grey")
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(history.history["smart5_output_msle"], label="Smart5 MSLE")
plt.plot(history.history["smart9_output_msle"], label="Smart9 MSLE")
plt.xlabel("Epoch")
plt.ylabel("MSLE")
plt.grid(True, linestyle="--", color="grey")
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(history.history["smart5_output_loss"], label="Smart5 Loss")
plt.plot(history.history["smart9_output_loss"], label="Smart9 Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True, linestyle="--", color="grey")
plt.legend()

plt.tight_layout()

plt.savefig(notes_dir+"training_history_plots_"+str(epochs)+"_"+str(batch_size)+".png")

plt.show()