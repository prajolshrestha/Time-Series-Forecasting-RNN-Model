# RecurentNeuralNetwork-RNN

Time Series Forecasting with Autoregressive Linear Model

This repository contains code for forecasting time series data using an autoregressive linear model in TensorFlow.

Overview
The code uses a simple autoregressive linear model to forecast future values of a time series based on its past values. The input data is a sin wave series with 200 values. The code creates a dataset of input/output pairs from the time series data and trains the autoregressive linear model to predict future values based on the past values. The model is then used to forecast future values of the time series based on its own predictions.

Dependencies
The code was written using Python 3. The following Python packages are required to run the code:

tensorflow
numpy
pandas
matplotlib

Code Structure
The code has the following structure:

Import the required libraries
Create the input time series data
Build the dataset of input/output pairs
Define the autoregressive linear model
Train the model
Plot the training and validation losses
Use the model to make predictions on the validation set
Plot the predicted values against the true values for the validation set
Use the model to make predictions on future values
Plot the predicted values against the true values for the future values

