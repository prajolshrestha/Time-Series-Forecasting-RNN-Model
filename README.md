# DeepRecurentNeuralNetwork-RNN

This code is for time series forecasting using TensorFlow and Keras. 
The code contains implementation of four different models - Autoregressive Linear Model, SimpleRNN Model, GRU Model and LSTM Model.

Dataset
The code generates a synthetic dataset by taking the sine of every tenth value between 0 and 400. 
This dataset is then used for training and testing the models.

**Linear Model**
Autoregressive Linear Model
The autoregressive linear model is a simple linear regression model that uses the past sequence values to predict the next value. The model is trained using the first half of the data and tested on the second half. The model is a poor fit for the dataset, as can be seen from the predictions.

**NonLinear Model**
SimpleRNN Model
The SimpleRNN model is a recurrent neural network that uses the past sequence values to predict the next value. The model is trained using the first half of the data and tested on the second half. The model performs much better than the autoregressive linear model, as can be seen from the predictions.

LSTM and GRU networks are two popular types of RNNs that are designed to overcome the "vanishing gradient" problem that can occur with traditional RNNs.

LSTM Model
The LSTM model is a recurrent neural network that is capable of handling long-term dependencies. The model is trained using the first half of the data and tested on the second half. The model performs better than the SimpleRNN model and the autoregressive linear model, as can be seen from the predictions.

GRU Model
The GRU model is similar to LSTM but with few parameters.The model is trained on the first half of the data and validated on the second half. The GRU model does a similar job of forecasting to the LSTM model.

Conclusion
The LSTM model is the best fit for the dataset, as it is capable of handling long-term dependencies.
