import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
import matplotlib.pyplot as plt
from tensorflow.python.keras.engine import data_adapter

# Fix distributed dataset error
from tensorflow.python.keras.engine import data_adapter

def _is_distributed_dataset(ds):
    return isinstance(ds, data_adapter.input_lib.DistributedDatasetSpec)

data_adapter._is_distributed_dataset = _is_distributed_dataset

# Load the trained LSTM model
model = load_model("model.h5")

# Function to load and preprocess the data
def load_and_preprocess_data(filename):
    df = pd.read_csv(filename)
    data = df['Close'].values
    data = data.reshape(-1, 1)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    prediction_days = 60
    X_train = []
    y_train = []
    
    for i in range(prediction_days, len(scaled_data) - 7):
        X_train.append(scaled_data[i-prediction_days:i, 0])
        y_train.append(scaled_data[i:i+7, 0])
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    return X_train, y_train, scaler, data  # Return the original data as well

# Load and preprocess the data
X_train, y_train, scaler, original_data = load_and_preprocess_data("BTC-USD.csv")

# Function to predict future prices for the next 30 days
def predict_future_prices(model, scaler, data, prediction_days=60, future_days=30):
    # Get the last 'prediction_days' days of data and reshape it to 2D
    last_60_days = data[-prediction_days:].reshape(-1, 1)

    # Scale the data (transform it into the range used during training)
    last_60_days_scaled = scaler.transform(last_60_days)

    # Reshape the data to fit the LSTM model input (3D: samples, timesteps, features)
    X_test = np.reshape(last_60_days_scaled, (1, prediction_days, 1))

    predicted_prices = []

    for _ in range(future_days // 7):
        # Predict the next 7 days
        predicted_scaled = model.predict(X_test)
        
        # Inverse transform the predicted values to get them back to the original scale
        predicted = scaler.inverse_transform(predicted_scaled.reshape(-1, 1))  # Reshape correctly for inverse scaling
        
        # Append the predicted prices to the list
        predicted_prices.extend(predicted.flatten())
        
        # Update the last 60 days with the predicted values for the next iteration
        # Use the last (60 - 7) actual values and append the newly predicted 7 values
        last_60_days_scaled = np.concatenate((last_60_days_scaled[7:], scaler.transform(predicted)), axis=0)
        
        # Reshape the data again to match LSTM input
        X_test = np.reshape(last_60_days_scaled, (1, prediction_days, 1))

    return np.array(predicted_prices)

# Make predictions for the next 30 days using the original data
predicted_prices = predict_future_prices(model, scaler, original_data, future_days=30)

# Print predicted prices
print("Predicted prices for the next 30 days:", predicted_prices)

# Plot the predicted prices
plt.plot(predicted_prices, label="Predicted Prices")
plt.title("Predicted Cryptocurrency Prices for the Next 30 Days")
plt.xlabel("Day")
plt.ylabel("Price")
plt.legend()
plt.show()
###

