import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, SimpleRNN
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#data with noise
series = np.sin(0.1*np.arange(200)) + np.random.randn(200)*0.1
plt.plot(series)
plt.show()

#Lets make dataset
T = 10
D = 1
X = []
Y = []

for t in range(len(series)-T):
    x = series[t:t+T]
    X.append(x)
    y = series[t+T]
    Y.append(y)

X = np.array(X)
#X.shape

X = X.reshape(-1,T,1) #N x T x D
#X.shape

Y = np.array(Y)
N = len(X)

"""### Build model"""

i = Input(shape = (T,1))
x = SimpleRNN(5, activation='relu')(i) # , activation='relu'
x = Dense(1)(x)
model = Model(i,x)

model.compile(loss='mse',optimizer=Adam(learning_rate = 0.1))
r = model.fit(X[:-N//2], Y[:-N//2],
              epochs = 80,
              validation_data = (X[-N//2:], Y[-N//2:]))

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()

"""###One step Forecast"""

X[-N//2].shape

#Wrong way of forecasting

validation_target = Y[-N//2:]
validation_predictions = []

#index of first validation input
i = -N//2

while len(validation_predictions) < len(validation_target):
    p = model.predict(X[i].reshape(1,-1,1))[0,0]
    i += 1
    
    #update prediction list
    validation_predictions.append(p)

plt.plot(validation_target, label = 'Forecast target')
plt.plot(validation_predictions, label = 'Forecast prediction')
plt.legend()

"""###Corect way of forecast"""

validation_target = Y[-N//2:]
validation_predictions = []

#last train input
last_x = X[-N//2] #only last 10 training data is used

while len(validation_predictions) < len(validation_target):
     p = model.predict(last_x.reshape(1,-1,1))[0,0]

     validation_predictions.append(p)
     last_x = np.roll(last_x,-1)
     last_x[-1] = p

plt.plot(validation_target, label='validation_target')
plt.plot(validation_predictions, label='validation_predictions')
plt.legend()