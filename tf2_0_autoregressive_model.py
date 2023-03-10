import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Create your own data
series = np.sin(0.1 * np.arange(200)) #create sin series with 200 values
series

plt.plot(series)
plt.show()

### build the dataset
# lets see if we can use T past values to predict the next value
T = 10
X = []
Y = []
for t in range(len(series)-T):
    x = series[t:t+T] # eg.take 10 values from series
    X.append(x)
    y = series[t+T] # take 11th value from series
    Y.append(y)

X = np.array(X).reshape(-1,10) # make dimension of input N x T
Y = np.array(Y)
N = len(X)



### try autoregressive linear model
i = Input(shape = (T,))
x = Dense(1)(i)

model = Model(i,x)
model.compile(loss='mse', optimizer = Adam(lr=0.1),)

#train
#train on first half of data and validate on second half
r = model.fit(
    X[:-N//2], Y[:-N//2],
    epochs = 80,
    validation_data = (X[-N//2:], Y[-N//2:]),
)

#plot
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()

