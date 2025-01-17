import sys
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QComboBox, QPushButton, 
                             QVBoxLayout, QHBoxLayout, QFrame, QMessageBox, QProgressBar, QFileDialog)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import LSTM, Dropout, Dense
import mplcursors
import signal
import os

# Add at the beginning of the file after imports
def segfault_handler(signum, frame):
    print("Caught segmentation fault")
    # Restore default handler
    signal.signal(signal.SIGSEGV, signal.SIG_DFL)

# Register handler
signal.signal(signal.SIGSEGV, segfault_handler)

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
    # Add required signals
    data_fetched = pyqtSignal(pd.DataFrame)
    error_occurred = pyqtSignal(str)
    progress_updated = pyqtSignal(int)

    def __init__(self, symbol):
        super().__init__()
        self.symbol = symbol
        self.running = True
        
    def check_disk_space(self, path="."):
        import os
        st = os.statvfs(path)
        free_space = st.f_bavail * st.f_frsize
        return free_space > 1024 * 1024 * 100  # Require 100MB free

    def run(self):
        try:
            if not self.check_disk_space():
                self.error_occurred.emit("Insufficient disk space")
                return

            import resource
            memory_limit = 128 * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
            
            self.progress_updated.emit(10)
            
            try:
                chunk_sizes = ["5m"] * 6
                df_list = []
                
                for i, chunk_size in enumerate(chunk_sizes):
                    success = False
                    retries = 3
                    while not success and retries > 0:
                        import gc
                        gc.collect()
                        try:
                            chunk = yf.download(
                                self.symbol,
                                period=chunk_size,
                                interval="1m",
                                progress=False,
                                threads=False,
                                timeout=10
                            )
                            if not chunk.empty:
                                df_list.append(chunk)
                                success = True
                                progress = int((i + 1) / len(chunk_sizes) * 90)
                                self.progress_updated.emit(progress)
                        except OSError as e:
                            import errno
                            if e.errno == errno.EIO:
                                self.error_occurred.emit(f"Disk I/O error: {str(e)}")
                                return
                            retries -= 1
                            if retries == 0:
                                raise
                            import time
                            time.sleep(2)
                
                if df_list:
                    df = pd.concat(df_list)
                    try:
                        df.to_csv(f"{self.symbol}.csv", index=True)
                    except OSError as e:
                        import errno
                        if e.errno == errno.EIO:
                            self.error_occurred.emit(f"Failed saving file: {str(e)}")
                            return
                        raise
                    self.data_fetched.emit(df)
                else:
                    raise ValueError("No data received")
                    
            except Exception as e:
                self.error_occurred.emit(f"Data fetch error: {str(e)}")
            finally:
                del df_list
                if 'df' in locals():
                    del df
                gc.collect()
                
        except Exception as e:
            self.error_occurred.emit(f"Critical error: {str(e)}")
        finally:
            self.running = False

# ModelTrainingThread to train an LSTM model
class ModelTrainingThread(QThread):
    training_completed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    progress_updated = pyqtSignal(int)

    def __init__(self, symbol, csv_path=None, column_name=None):
        super().__init__()
        self.symbol = symbol
        self.csv_path = csv_path
        self.column_name = column_name

    def run(self):
        try:
            # Load data from the custom CSV if provided, otherwise from the default
            if self.csv_path:
                df = pd.read_csv(self.csv_path)
            else:
                df = pd.read_csv(f"{self.symbol}.csv")
            
            # Check if the specified column exists
            if self.column_name not in df.columns:
                self.error_occurred.emit(f"Selected column '{self.column_name}' not found in the CSV.")
                return

            # Use the specified column's data for training
            data = df[self.column_name].values.reshape(-1, 1)

            # Check if there is enough data
            if len(data) < 75:  # 60 days for input + 15 minutes for prediction
                self.error_occurred.emit("Not enough data to train the model. Please load more data.")
                return

            # Preprocess data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)
            prediction_days = 60
            X_train, y_train = [], []
            for i in range(prediction_days, len(scaled_data) - 15):
                X_train.append(scaled_data[i - prediction_days:i, 0])
                y_train.append(scaled_data[i:i + 15, 0])

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
                Dense(units=15)  # Changed from 7 to 15 units
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')

            # Train the model with progress updates
            for epoch in range(50):  # Assuming 50 epochs
                model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)
                progress = int((epoch + 1) / 50 * 100)
                self.progress_updated.emit(progress)

            # Save the model and scaler with unique names
            model_filename = f"{self.symbol}_model.h5"
            scaler_filename = f"{self.symbol}_scaler.pkl"
            model.save(model_filename)
            with open(scaler_filename, 'wb') as f:
                pickle.dump(scaler, f)

            self.training_completed.emit(f"Model training for {self.symbol} completed. Model saved as {model_filename}")
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

            # Predict future prices and verify shape
            predicted_prices = model.predict(X_test)
            print(f"Raw prediction shape: {predicted_prices.shape}")
            
            predicted_prices = predicted_prices.reshape(-1, 1)  # Reshape to (15, 1)
            predicted_prices = scaler.inverse_transform(predicted_prices)
            print(f"Transformed prediction shape: {predicted_prices.shape}")
            
            # Ensure we have exactly 15 predictions
            if len(predicted_prices) != 15:
                self.error_occurred.emit(f"Expected 15 predictions, got {len(predicted_prices)}")
                return
                
            self.prediction_completed.emit(predicted_prices.flatten())
        except Exception as e:
            self.error_occurred.emit(str(e))

# Main PyQt5 application
class CryptoPredictionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CryptoOracle")
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
        self.symbol_label = QLabel("Select Coin: ", self)
        self.symbol_dropdown = QComboBox(self)
        self.load_crypto_symbols()
        self.input_layout.addWidget(self.symbol_label)
        self.input_layout.addWidget(self.symbol_dropdown)
        self.input_frame.setLayout(self.input_layout)
        self.layout.addWidget(self.input_frame)

        # Progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.progress_bar)

        # Load, train, and predict buttons
        self.load_button = QPushButton("Load Data", self)
        self.load_button.clicked.connect(self.fetch_data)
        self.layout.addWidget(self.load_button)

        # Add button for loading a custom CSV file
        self.load_csv_button = QPushButton("Load Custom CSV", self)
        self.load_csv_button.clicked.connect(self.load_custom_csv)
        self.layout.addWidget(self.load_csv_button)

        # Add dropdown for selecting the column to use for training
        self.column_dropdown = QComboBox(self)
        self.column_dropdown.setStyleSheet("background-color: #ecf0f1; color: black; padding: 5px;")
        self.layout.addWidget(self.column_dropdown)
        self.column_dropdown.setEnabled(False)  # Initially disable the dropdown

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
            self.symbol_dropdown.addItem(name, symbol)

    # Update fetch_data method in CryptoPredictionApp
    def fetch_data(self):
        try:
            # Safely stop existing thread
            if hasattr(self, 'data_thread'):
                try:
                    self.data_thread.stop()
                    self.data_thread.wait()
                    self.data_thread.deleteLater()
                except:
                    pass
                
            # Create new thread with error handling
            try:
                symbol = self.symbol_dropdown.currentData()
                self.data_thread = DataFetchThread(symbol)
                
                # Connect signals with error handling
                self.data_thread.data_fetched.connect(self.handle_data_fetched, Qt.QueuedConnection)
                self.data_thread.error_occurred.connect(self.show_error_message, Qt.QueuedConnection)
                self.data_thread.progress_updated.connect(self.update_progress, Qt.QueuedConnection)
                
                # Reset and start
                self.progress_bar.setValue(0)
                self.data_thread.start()
                
            except Exception as e:
                self.show_error_message(f"Failed to start data fetch: {str(e)}")
                return
                
        except Exception as e:
            self.show_error_message(f"Critical error in fetch_data: {str(e)}")
            # Recover from error
            self.progress_bar.setValue(0)
            if hasattr(self, 'data_thread'):
                delattr(self, 'data_thread')

    def load_csv_columns(self, file_path):
        # Load the CSV to get column names
        try:
            df = pd.read_csv(file_path)
            self.column_dropdown.clear()
            self.column_dropdown.addItems(df.columns)
            self.column_dropdown.setEnabled(True)  # Enable the dropdown
        except Exception as e:
            self.show_error_message(str(e))

    def train_model(self):
        symbol = self.symbol_dropdown.currentData()
        csv_path = getattr(self, 'custom_csv_path', None)
        column_name = self.column_dropdown.currentText()  # Get the selected column name
        self.train_thread = ModelTrainingThread(symbol, csv_path, column_name)
        self.train_thread.training_completed.connect(self.show_success_message)
        self.train_thread.error_occurred.connect(self.show_error_message)
        self.train_thread.progress_updated.connect(self.update_progress)
        self.progress_bar.setValue(0)  # Reset progress bar
        self.train_thread.start()
    
    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def make_prediction(self):
        symbol = self.symbol_dropdown.currentData()
        self.predict_thread = PredictionThread(symbol)
        self.predict_thread.prediction_completed.connect(self.plot_chart)
        self.predict_thread.error_occurred.connect(self.show_error_message)
        self.progress_bar.setValue(0)  # Reset progress bar
        self.predict_thread.start()

    def plot_chart(self, predicted_prices):
        try:
            print(f"Number of predictions to plot: {len(predicted_prices)}")
            print(f"Prediction values: {predicted_prices}")
            
            # Clear existing widgets
            for i in reversed(range(self.chart_layout.count())):
                widget = self.chart_layout.itemAt(i).widget()
                if widget is not None:
                    widget.deleteLater()

            # Create figure and axis
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Plot with points and lines
            x_points = list(range(15))
            line = ax.plot(x_points, predicted_prices, color="#1abc9c", marker='o', 
                    linestyle='-', linewidth=2, markersize=8, label='Predicted Prices')[0]

            # Add hover annotations
            cursor = mplcursors.cursor(line, hover=True)
            cursor.connect(
                "add",
                lambda sel: sel.annotation.set_text(
                    f'Price: ${sel.target[1]:,.2f}'
                )
            )
            
            # Style annotations
            cursor.connect(
                "add",
                lambda sel: sel.annotation.get_bbox_patch().set(
                    fc="#34495e", 
                    alpha=0.8,
                    edgecolor="white"
                )
            )
            
            # Set annotation text color
            cursor.connect(
                "add",
                lambda sel: sel.annotation.set_color("white")
            )

            # Set x-axis ticks and labels
            ax.set_xticks(x_points)
            ax.set_xticklabels([f"{i+1} min" for i in range(15)])

            # Format y-axis to show full numbers
            formatter = plt.ScalarFormatter(useOffset=False)
            formatter.set_scientific(False)
            ax.yaxis.set_major_formatter(formatter)
            
            # Customize appearance
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_title(f'Predicted Prices for {self.symbol_dropdown.currentText()}', 
                        color='white', pad=20)
            ax.set_facecolor('#34495e')
            fig.patch.set_facecolor('#2c3e50')
            
            # Label axes
            ax.set_xlabel('Time (minutes)', color='white', labelpad=10)
            ax.set_ylabel('Price', color='white', labelpad=10)
            
            # Style ticks
            ax.tick_params(axis='x', colors='white', rotation=45)
            ax.tick_params(axis='y', colors='white')
            
            # Add legend
            ax.legend(loc='upper right')

            # Adjust layout to prevent label cutoff
            plt.tight_layout()

            # Create and add canvas
            canvas = FigureCanvas(fig)
            self.chart_layout.addWidget(canvas)

            # Add prediction label
            coin_label = QLabel(f"15-minute prediction for {self.symbol_dropdown.currentText()}")
            coin_label.setAlignment(Qt.AlignCenter)
            coin_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #ecf0f1;")
            self.chart_layout.addWidget(coin_label)

        except Exception as e:
            self.show_error_message(f"Error plotting chart: {str(e)}")

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

    def load_custom_csv(self):
        try:
            # Open file dialog
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select CSV File",
                "",
                "CSV Files (*.csv);;All Files (*)"
            )
            
            if file_path:
                # Save the path for later use
                self.custom_csv_path = file_path
                # Load columns into dropdown
                self.load_csv_columns(file_path)
                # Show success message
                self.show_message(f"Loaded CSV file: {file_path}")
                
        except Exception as e:
            self.show_error_message(f"Error loading CSV: {str(e)}")

if __name__ == '__main__':
    try:
        app = QApplication(sys.argv)
        window = CryptoPredictionApp()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Failed to start application: {str(e)}")
        sys.exit(1)