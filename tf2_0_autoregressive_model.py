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

# Wrong forecast using true targets

validation_target = Y[-N//2:] #use second half of Y
validation_predictions = [] # empty list

#index of first validation input
i= -N//2

while len(validation_predictions) < len(validation_target):
     p = model.predict(X[i].reshape(1,-1))[0,0] 
     #X[i].reshape(1,-1) =====> 1xT (1 samples and T features)
     #again model.predict() returns N x K output (N samples and K output node)
     # 1x1 array ma index [0,0]------> scalar value 
     
     i += 1

     #update the predictions list
     validation_predictions.append(p)

plt.plot(validation_target, label='forecast Target')
plt.plot(validation_predictions, label = 'forecast Prediction')
plt.legend()

###Forecast future values (use only self-predictions for making future predictions)

validation_target = Y[-N//2:] #use second half of Y
validation_predictions = [] # empty list

#last train input
last_x = X[-N//2] #first input vector #1D array of length T #we wont use any new value from actual dataset

while len(validation_predictions) < len(validation_target):
     p = model.predict(last_x.reshape(1,-1))[0,0] 
     #again model.predict() returns N x K output (N samples and K output node)
     # 1x1 array ma index [0,0]------> scalar value 
     
     #update the predictions list
     validation_predictions.append(p)

     #make the new input
     last_x = np.roll(last_x, -1) # roll(last_x, -1) helps to shift everything to one step to left
     last_x[-1] = p

plt.plot(validation_target, label='forecast Target')
plt.plot(validation_predictions, label = 'forecast Prediction')
plt.legend()
