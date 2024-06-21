import pandas as pd
import os

input_folder = 'LSTM_temp'
df = pd.DataFrame()

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


prelstm_5 = series_to_supervised(df,5,1)

prelstm_5.to_csv('LSTM_ready/prelstm_5.csv', index=False)