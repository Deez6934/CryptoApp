import sys
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QComboBox, QPushButton, 
                             QVBoxLayout, QHBoxLayout, QFrame, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import LSTM, Dropout, Dense

# Fix distributed dataset error in TensorFlow
from tensorflow.python.keras.engine import data_adapter
def _is_distributed_dataset(ds):
    return isinstance(ds, data_adapter.input_lib.DistributedDatasetSpec)
data_adapter._is_distributed_dataset = _is_distributed_dataset

# Cryptocurrency data configuration
crypto_data = {
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "DOGE-USD": "Dogecoin",
    "ADA-USD": "Cardano",
    "BNB-USD": "Binance Coin",
    "SOL-USD": "Solana",
    "XRP-USD": "Ripple"
}

# DataFetchThread to fetch cryptocurrency data from yfinance
class DataFetchThread(QThread):
    data_fetched = pyqtSignal(pd.DataFrame)  # Signal to emit fetched data
    error_occurred = pyqtSignal(str)  # Signal to emit error messages

    def __init__(self, symbol):
        super().__init__()
        self.symbol = symbol

    def run(self):
        try:
            data = yf.download(self.symbol, period="1mo", interval="5m")
            if data.empty:
                self.error_occurred.emit(f"No data found for {self.symbol}.")
            else:
                data.to_csv(f"{self.symbol}.csv")
                self.data_fetched.emit(data)
        except Exception as e:
            self.error_occurred.emit(str(e))

# ModelTrainingThread to train an LSTM model
class ModelTrainingThread(QThread):
    training_completed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, symbol):
        super().__init__()
        self.symbol = symbol

    def run(self):
        try:
            # Load data
            df = pd.read_csv(f"{self.symbol}.csv")
            data = df['Close'].values.reshape(-1, 1)

            # Check if there is enough data
            if len(data) < 67:  # 60 days for input + 7 days for prediction
                self.error_occurred.emit("Not enough data to train the model. Please load more data.")
                return

            # Preprocess data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)
            prediction_days = 60
            X_train, y_train = [], []
            for i in range(prediction_days, len(scaled_data) - 7):
                X_train.append(scaled_data[i - prediction_days:i, 0])
                y_train.append(scaled_data[i:i + 7, 0])

            X_train = np.array(X_train).reshape(-1, prediction_days, 1)
            y_train = np.array(y_train)

            # Verify that the arrays are not empty
            if X_train.size == 0 or y_train.size == 0:
                self.error_occurred.emit("Training data is empty. Please ensure sufficient data is loaded.")
                return

            # Build the model
            model = Sequential([
                LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
                Dropout(0.2),
                LSTM(units=50, return_sequences=True),
                Dropout(0.2),
                LSTM(units=50, return_sequences=False),
                Dropout(0.2),
                Dense(units=7)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')

            # Train the model
            model.fit(X_train, y_train, epochs=50, batch_size=32)

            # Save the model and scaler
            model.save(f"{self.symbol}_model.h5")
            with open(f"{self.symbol}_scaler.pkl", 'wb') as f:
                pickle.dump(scaler, f)

            self.training_completed.emit(f"Model training for {self.symbol} completed.")
        except Exception as e:
            self.error_occurred.emit(str(e))

# PredictionThread to predict future prices
class PredictionThread(QThread):
    prediction_completed = pyqtSignal(np.ndarray)
    error_occurred = pyqtSignal(str)

    def __init__(self, symbol):
        super().__init__()
        self.symbol = symbol

    def run(self):
        try:
            # Load model and scaler
            model = load_model(f"{self.symbol}_model.h5")
            with open(f"{self.symbol}_scaler.pkl", 'rb') as f:
                scaler = pickle.load(f)

            # Load and preprocess data
            df = pd.read_csv(f"{self.symbol}.csv")
            data = df['Close'].values.reshape(-1, 1)
            last_60_days = data[-60:].reshape(-1, 1)
            last_60_days_scaled = scaler.transform(last_60_days)
            X_test = np.reshape(last_60_days_scaled, (1, 60, 1))

            # Predict future prices
            predicted_prices = model.predict(X_test)
            predicted_prices = scaler.inverse_transform(predicted_prices.reshape(-1, 1))

            self.prediction_completed.emit(predicted_prices.flatten())
        except Exception as e:
            self.error_occurred.emit(str(e))

# Main PyQt5 application
class CryptoPredictionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Crypto Prediction App")
        self.setGeometry(300, 300, 800, 600)
        self.setStyleSheet("background-color: #2c3e50; color: white;")
        self.init_ui()

    def init_ui(self):
        # Main layout
        self.layout = QVBoxLayout()

        # Title
        self.title = QLabel("Cryptocurrency Price Prediction", self)
        self.title.setStyleSheet("font-size: 24px; font-weight: bold; color: #ecf0f1;")
        self.title.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.title)

        # Dropdown for cryptocurrency selection
        self.input_frame = QFrame(self)
        self.input_layout = QHBoxLayout()
        self.symbol_label = QLabel("Select Symbol: ", self)
        self.symbol_dropdown = QComboBox(self)
        self.load_crypto_symbols()
        self.input_layout.addWidget(self.symbol_label)
        self.input_layout.addWidget(self.symbol_dropdown)
        self.input_frame.setLayout(self.input_layout)
        self.layout.addWidget(self.input_frame)

        # Load, train, and predict buttons
        self.load_button = QPushButton("Load Data", self)
        self.load_button.clicked.connect(self.fetch_data)
        self.layout.addWidget(self.load_button)

        self.train_button = QPushButton("Train Model", self)
        self.train_button.clicked.connect(self.train_model)
        self.layout.addWidget(self.train_button)

        self.predict_button = QPushButton("Predict", self)
        self.predict_button.clicked.connect(self.make_prediction)
        self.layout.addWidget(self.predict_button)

        # Chart frame
        self.chart_frame = QFrame(self)
        self.chart_layout = QVBoxLayout()
        self.chart_placeholder = QLabel("Prediction chart will appear here", self)
        self.chart_placeholder.setAlignment(Qt.AlignCenter)
        self.chart_layout.addWidget(self.chart_placeholder)
        self.chart_frame.setLayout(self.chart_layout)
        self.layout.addWidget(self.chart_frame)

        # Set the layout to the main window
        self.setLayout(self.layout)

    def load_crypto_symbols(self):
        for symbol, name in crypto_data.items():
            self.symbol_dropdown.addItem(f"{name} ({symbol})", symbol)

    def fetch_data(self):
        symbol = self.symbol_dropdown.currentData()
        self.data_thread = DataFetchThread(symbol)
        self.data_thread.data_fetched.connect(self.handle_data_fetched)
        self.data_thread.error_occurred.connect(self.show_error_message)
        self.data_thread.start()


    def train_model(self):
        symbol = self.symbol_dropdown.currentData()
        self.train_thread = ModelTrainingThread(symbol)
        self.train_thread.training_completed.connect(self.show_success_message)
        self.train_thread.error_occurred.connect(self.show_error_message)
        self.train_thread.start()

    def make_prediction(self):
        symbol = self.symbol_dropdown.currentData()
        self.predict_thread = PredictionThread(symbol)
        self.predict_thread.prediction_completed.connect(self.plot_chart)
        self.predict_thread.error_occurred.connect(self.show_error_message)
        self.predict_thread.start()

    def plot_chart(self, predicted_prices):
        self.chart_layout.removeWidget(self.chart_placeholder)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(predicted_prices, color="#1abc9c", label='Predicted Prices')
        ax.set_title('Predicted Prices for Next 7 Days', color='white')
        ax.set_facecolor('#34495e')
        fig.patch.set_facecolor('#2c3e50')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

        canvas = FigureCanvas(fig)
        self.chart_layout.addWidget(canvas)

    def handle_data_fetched(self, data):
        # Display a message indicating successful data fetching
        QMessageBox.information(self, "Info", f"Data fetched successfully for {self.symbol_dropdown.currentText()}.")
    
    def show_message(self, message):
        QMessageBox.information(self, "Info", message)

    def show_error_message(self, error_message):
        QMessageBox.critical(self, "Error", error_message)
    
    def show_success_message(self, message):
        # Display a success message in a message box
        QMessageBox.information(self, "Success", message)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    crypto_app = CryptoPredictionApp()
    crypto_app.show()
    sys.exit(app.exec_())
