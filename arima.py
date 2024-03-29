# -*- coding: utf-8 -*-
"""ARIMA-LSTM.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1IK5y7zp1D0DJMgzXfyp4sO2u1G2kuSLi
"""

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


df=pd.read_csv(r"C:\Users\user\systempr\bitcoin.csv", index_col='Date', parse_dates=True, usecols=['Date','Close'])
df=df.iloc[::-1]
split_size = int(0.8 * len(df))
testdf=df.iloc[split_size: , : ]
df=df.iloc[ :split_size, : ]
ax=df.plot()
testdf.plot(ax=ax)
plt.show()

adf = adfuller(df.Close)
print(adf[1])

d=0
while(adf[1]>0.05):
  dummy=df.diff().dropna()
  d+=1
  adf = adfuller(dummy.Close)
d

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8))
plot_acf(df, lags=15, zero=False, ax=ax1)
plot_pacf(df, lags=15, zero=False, ax=ax2)
plt.show()

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8))
plot_acf(df.diff().dropna(), lags=15, zero=False, ax=ax1)
plot_pacf(df.diff().dropna(), lags=15, zero=False, ax=ax2)
plt.show()

order_aic_bic =[]
for p in range(3):
  
  # Loop over q values from 0-3
    for q in range(3):
      try:
        # Create and fit ARMA(p,q) model
        model = ARIMA(df, order=(p,d,q))
        results = model.fit()
        
        # Print p, q, AIC, BIC
        order_aic_bic.append((p, q, results.aic))
        
      except:
        order_aic_bic.append((p, q, None))

order_df = pd.DataFrame(order_aic_bic, columns=['p','q','aic'])

print(order_df.sort_values('aic'))

p = order_df.sort_values('aic')[0:1].p.values[0]
q = order_df.sort_values('aic')[0:1].q.values[0]

if p==0 and q==0:
  p = order_df.sort_values('aic')[1:2].p.values[0]
  q = order_df.sort_values('aic')[1:2].q.values[0]

model = ARIMA(df,order=(p,d,q))
results = model.fit()

results.plot_diagnostics()
plt.show()

print(results.summary())

forecast = results.get_forecast(steps=293)
mean_forecast = forecast.predicted_mean
plt.plot(df.index, df.Close, label='observed')

plt.plot(mean_forecast.index, mean_forecast, color='r', label='forecast')
plt.xlabel('Date')
plt.ylabel('Bitcoin Price - USD')
plt.legend()
plt.show()

import numpy as np
mae = np.mean(np.abs(results.resid))
print(mae)

newdf = df.iloc[1:,:].Close + results.resid[1:].values
newdf.plot()
df.plot()
plt.show()

ax=newdf.plot()
df.plot(ax=ax)
plt.show()

adf = adfuller(testdf.Close)
print(adf[1])

d=0
while(adf[1]>0.05):
  dummy=testdf.diff().dropna()
  d+=1
  adf = adfuller(dummy.Close)
d

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8))
plot_acf(testdf, lags=15, zero=False, ax=ax1)
plot_pacf(testdf, lags=15, zero=False, ax=ax2)
plt.show()

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8))
plot_acf(testdf.diff().dropna(), lags=15, zero=False, ax=ax1)
plot_pacf(testdf.diff().dropna(), lags=15, zero=False, ax=ax2)
plt.show()

order_aic_bic =[]
for p in range(3):
  
  # Loop over q values from 0-3
    for q in range(3):
      try:
        # Create and fit ARMA(p,q) model
        model = ARIMA(testdf, order=(p,d,q))
        results = model.fit()
        
        # Print p, q, AIC, BIC
        order_aic_bic.append((p, q, results.aic))
        
      except:
        order_aic_bic.append((p, q, None))

order_df = pd.DataFrame(order_aic_bic, columns=['p','q','aic'])

print(order_df.sort_values('aic'))

p = order_df.sort_values('aic')[0:1].p.values[0]
q = order_df.sort_values('aic')[0:1].q.values[0]

if p==0 and q==0:
  p = order_df.sort_values('aic')[1:2].p.values[0]
  q = order_df.sort_values('aic')[1:2].q.values[0]

model_test = ARIMA(testdf,order=(p,d,q))
results_test = model_test.fit()

results_test.plot_diagnostics()
plt.show()

print(results_test.summary())

newtestdf = testdf.iloc[1:,:].Close + results_test.resid[1:].values
newtestdf.plot()
testdf.plot()
plt.show()

import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers

prices = newdf.values
timesteps = tf.convert_to_tensor(newdf.index.values.astype(np.int64))
timestepstest = tf.convert_to_tensor(testdf.index.values.astype(np.int64))
X_train, y_train = newdf.index, prices
X_test, y_test = newtestdf.index, newtestdf.values

HORIZON = 1 
WINDOW_SIZE = 7

def get_labelled_windows(x, horizon=1):
    return x[:, :-horizon], x[:, -horizon:]

def make_windows(x, window_size=7, horizon=1):
    window_step = np.expand_dims(np.arange(window_size + horizon), axis = 0)
    window_indexes = window_step + np.expand_dims(np.arange(len(x) - (window_size + horizon - 1)), axis = 0).T
    windowed_array = x[window_indexes]
    windows, labels = get_labelled_windows(windowed_array, horizon=horizon)
    return windows, labels

train_windows, train_labels = make_windows(prices, window_size=WINDOW_SIZE, horizon=HORIZON)
test_windows, test_labels = make_windows(y_test, window_size=WINDOW_SIZE, horizon=HORIZON)

len(test_labels)

x = tf.constant(train_windows[0])
expand_dims_layers = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1))
print(x.shape)
print(f'{expand_dims_layers(x).shape}')

tf.random.set_seed(42)

model_7 = tf.keras.Sequential([
    
    tf.keras.layers.Lambda(lambda x : tf.expand_dims(x, axis=1)),
    tf.keras.layers.Conv1D(filters=128, kernel_size=5, padding="causal", activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),

    tf. keras.layers.Dense(HORIZON, activation='linear'),
])
model_7.compile(loss='mae',
             optimizer='adam')
history_model_7 = model_7.fit(x=train_windows, y=train_labels, batch_size=50, epochs = 80, verbose=1)
model_7.summary()

def make_preds(model, input_data):
  forecast = model.predict(input_data)
  return tf.squeeze(forecast)

model_7_preds = make_preds(model_7, test_windows).numpy()
model_7_preds=np.append(model_7_preds,test_labels[-1])
test_labels = np.insert(test_labels,0,model_7_preds[0])

plt.figure(figsize=(15,7))
plt.plot(X_test[-len(test_labels):], test_labels,label='Test')
plt.xlabel("Time")
plt.ylabel("Bitcoin Price")
plt.legend(fontsize=10)
plt.grid(True)
plt.plot(X_test[-len(model_7_preds):], model_7_preds,label='Predictions')
plt.xlabel("Time")
plt.ylabel("Bitcoin Price")
plt.legend(fontsize=10)

plt.figure(figsize=(15,7))
plt.plot(X_test[-len(test_labels):-1], testdf[7:-1].Close.to_numpy(),label='Test')
plt.xlabel("Time")
plt.ylabel("Bitcoin Price")
plt.legend(fontsize=10)
plt.grid(True)
plt.plot(X_test[-len(model_7_preds):-1], model_7_preds[:-1],label='Predictions')
plt.xlabel("Time")
plt.ylabel("Bitcoin Price")
plt.legend(fontsize=10)

rmse = np.sqrt(np.mean(model_7_preds[:-1] - testdf[7:-1].Close.to_numpy())**2)
rmse

windows = train_labels[-7:]
model_30_preds = []
for i in range(30):
  windows = windows.reshape(1,7)
  single_preds = make_preds(model_7, windows).numpy()
  model_30_preds.append(single_preds)
  windows = np.delete(windows,0)
  windows = np.append(windows, single_preds)
  print(windows)
model_30_preds

model_7.save('keras_model.h5')