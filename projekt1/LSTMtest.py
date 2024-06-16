import os
import shutil
import pandas as pd

import numpy as np
import tensorflow as tf

from keras.saving import load_model
from sklearn.preprocessing import MinMaxScaler


model_name = 'modelLSTM_16_16.keras'

model = load_model('notes/'+model_name)

values = pd.read_csv('LSTM_ready/LSTM_ready.csv').values


# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(values)

X, Y = dataset[:, :-5], dataset[:, -5:]

X = X.reshape((X.shape[0], 1, X.shape[1]))


x = 24540
to_predict = X[x]

#print("X:",to_predict)

res = model.predict(to_predict.reshape(to_predict.shape[0], 1, to_predict.shape[1]))

print("Predicted Y: ", [res[0][0][0],res[1][0][0], res[2][0][0], res[3][0][0]])
print("Correct Y: ", Y[x])

x = 24988
to_predict = X[x]

#print("X:",to_predict)

res = model.predict(to_predict.reshape(to_predict.shape[0], 1, to_predict.shape[1]))

print("Predicted Y: ", [res[0][0][0],res[1][0][0], res[2][0][0], res[3][0][0]])
print("Correct Y: ", Y[x])


x = 2400
to_predict = X[x]

#print("X:",to_predict)

res = model.predict(to_predict.reshape(to_predict.shape[0], 1, to_predict.shape[1]))

print("Predicted Y: ", [res[0][0][0],res[1][0][0], res[2][0][0], res[3][0][0]])
print("Correct Y: ", Y[x])

x = 87028
to_predict = X[x]

#print("X:",to_predict)

res = model.predict(to_predict.reshape(to_predict.shape[0], 1, to_predict.shape[1]))

print("Predicted Y: ", [res[0][0][0],res[1][0][0], res[2][0][0], res[3][0][0]])
print("Correct Y: ", Y[x])