import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.layers.recurrent import LSTM
import matplotlib.pyplot as plt
import pickle

# This is to fix any distributed dataset issues
from tensorflow.python.keras.engine import data_adapter

def _is_distributed_dataset(ds):
    return isinstance(ds, data_adapter.input_lib.DistributedDatasetSpec)

data_adapter._is_distributed_dataset = _is_distributed_dataset

# Load and preprocess the data
def load_and_preprocess_data(filename):
    df = pd.read_csv(filename)
    data = df['Close'].values
    data = data.reshape(-1, 1)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    prediction_days = 60  # You can adjust this window for more/fewer days
    X_train = []
    y_train = []
    
    # Prepare data for training (with a 60-day window)
    for i in range(prediction_days, len(scaled_data) - 7):
        X_train.append(scaled_data[i-prediction_days:i, 0])
        y_train.append(scaled_data[i:i+7, 0])
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    return X_train, y_train, scaler

# Build the LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=7))  # Predicting the next 7 days
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train the model
def train_model(model, X_train, y_train, epochs=50, batch_size=32):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

# Save the model and scaler
def save_model_and_scaler(model, scaler, model_filename='model.h5', scaler_filename='scaler.pkl'):
    # Save the model as an HDF5 file
    model.save(model_filename)  # Use model.h5 or SavedModel format
    
    # Save the scaler using pickle
    with open(scaler_filename, 'wb') as f:
        pickle.dump(scaler, f)

# Load the model and scaler
def load_model_and_scaler(model_filename='model.h5', scaler_filename='scaler.pkl'):
    # Load the model from the HDF5 file
    model = load_model(model_filename)
    
    # Load the scaler from pickle
    with open(scaler_filename, 'rb') as f:
        scaler = pickle.load(f)
    
    return model, scaler

# Predict future prices
def predict_future_prices(model, scaler, data, prediction_days=60):
    # Get the last 'prediction_days' days of data
    last_data = data[-prediction_days:].reshape(-1, 1)
    scaled_last_data = scaler.transform(last_data)

    # Reshape the data to match the model input
    X_test = np.reshape(scaled_last_data, (1, prediction_days, 1))

    # Predict the next 7 days
    predicted_scaled = model.predict(X_test)
    predicted = scaler.inverse_transform(predicted_scaled)
    
    return predicted

# Load and preprocess the data
X_train, y_train, scaler = load_and_preprocess_data("BTC-USD.csv")

# Build and train the model
model = build_lstm_model((X_train.shape[1], 1))
train_model(model, X_train, y_train, epochs=50)

# Save the model and scaler for later use
save_model_and_scaler(model, scaler)

# In another program, you can load the model and scaler and make predictions
model, scaler = load_model_and_scaler()

# Load the data again for prediction (or use live data)
df = pd.read_csv("BTC-USD.csv")
data = df['Close'].values

# Predict the next 7 days of prices
predicted_prices = predict_future_prices(model, scaler, data)

# Print the predicted prices
print("Predicted future prices for the next 7 days:")
print(predicted_prices)
