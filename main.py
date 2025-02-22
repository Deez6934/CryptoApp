import sys
import tensorflow as tf  # Update this import
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import gc  # Add this import
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QComboBox,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QMessageBox,
    QProgressBar,
    QFileDialog,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model  # type: ignore
from tensorflow.keras.layers import LSTM, Dropout, Dense  # type: ignore
import mplcursors
import signal
import os
from config import CONFIG, TECHNICAL_INDICATORS
from utils import (
    add_technical_indicators,
    validate_data,
    save_model_with_version,
    check_disk_space,
    clean_data,
)
import psutil  # Add this import
from tf_config import TensorFlowConfig

# Initialize TensorFlow configuration
TensorFlowConfig.configure()


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
    "XRP-USD": "Ripple",
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
        return check_disk_space(path, CONFIG["min_disk_space"])

    def check_memory(self):
        """Check if enough memory is available"""
        try:
            available_memory = psutil.virtual_memory().available
            return available_memory > CONFIG["memory_limit"]
        except Exception as e:
            print(f"Error checking memory: {e}")
            return True

    def run(self):
        try:
            if not self.check_disk_space():
                self.error_occurred.emit("Insufficient disk space")
                return

            if not self.check_memory():
                self.error_occurred.emit("Insufficient memory available")
                return

            self.progress_updated.emit(10)

            try:
                # Create Ticker object first
                ticker = yf.Ticker(self.symbol)

                # Then fetch historical data without the threading parameter
                df = ticker.history(
                    period="7d",
                    interval="1m",
                    actions=False,
                    timeout=30,
                )

                if df.empty:
                    self.error_occurred.emit("No data received from Yahoo Finance")
                    return

                # Clean data before processing
                df = clean_data(df)

                # Add technical indicators
                df = add_technical_indicators(df)

                # Validate data
                validation_results = validate_data(df)
                if not validation_results["valid"]:
                    self.error_occurred.emit("\n".join(validation_results["errors"]))
                    return

                for warning in validation_results["warnings"]:
                    print(f"Warning: {warning}")

                # Save to CSV
                df.to_csv(f"{self.symbol}.csv")

                self.progress_updated.emit(100)
                self.data_fetched.emit(df)

            except Exception as e:
                self.error_occurred.emit(f"Data fetch error: {str(e)}")
            finally:
                if "df" in locals():
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
            print("Starting model training...")

            # Load and validate data
            if self.csv_path:
                df = pd.read_csv(self.csv_path)
                print(f"Loaded custom CSV from: {self.csv_path}")
            else:
                df = pd.read_csv(f"{self.symbol}.csv")
                print(f"Loaded default CSV for symbol: {self.symbol}")

            if self.column_name not in df.columns:
                raise ValueError(f"Column '{self.column_name}' not found in CSV")

            # Convert data to numeric and handle NaN values
            df[self.column_name] = pd.to_numeric(df[self.column_name], errors="coerce")
            df = df.dropna(subset=[self.column_name])
            print(f"Data shape after cleaning: {df.shape}")

            # Prepare training data
            data = df[self.column_name].values.reshape(-1, 1)
            if len(data) < 75:
                raise ValueError(
                    "Insufficient data for training (minimum 75 rows required)"
                )

            # Scale the data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)

            # Prepare sequences
            prediction_days = 60
            X_train, y_train = [], []

            for i in range(prediction_days, len(scaled_data) - 15):
                X_train.append(scaled_data[i - prediction_days : i, 0])
                y_train.append(scaled_data[i : i + 15, 0])

            X_train = np.array(X_train)
            y_train = np.array(y_train)

            print(f"Training data shapes - X: {X_train.shape}, y: {y_train.shape}")

            # Build model
            model = Sequential(
                [
                    LSTM(
                        units=50,
                        return_sequences=True,
                        input_shape=(prediction_days, 1),
                    ),
                    Dropout(0.2),
                    LSTM(units=50, return_sequences=True),
                    Dropout(0.2),
                    LSTM(units=50),
                    Dropout(0.2),
                    Dense(units=15),
                ]
            )

            model.compile(optimizer="adam", loss="mean_squared_error")

            # Train the model with explicit progress updates
            epochs = 50
            batch_size = 32
            for epoch in range(epochs):
                print(f"Epoch {epoch+1}/{epochs}")
                history = model.fit(
                    X_train, y_train, epochs=1, batch_size=batch_size, verbose=0
                )
                progress = int((epoch + 1) / epochs * 100)
                self.progress_updated.emit(progress)
                print(f"Loss: {history.history['loss'][0]}")

            # Save model and scaler
            model_filename = f"{self.symbol}_model.h5"
            scaler_filename = f"{self.symbol}_scaler.pkl"

            model.save(model_filename)
            with open(scaler_filename, "wb") as f:
                pickle.dump(scaler, f)

            print("Training completed successfully")
            self.training_completed.emit(f"Model trained and saved as {model_filename}")

        except Exception as e:
            print(f"Training error: {str(e)}")
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
            model = load_model(
                f"{self.symbol}_model.h5", compile=False
            )  # Add compile=False
            model.compile(
                optimizer="adam",
                loss="mean_squared_error",
                run_eagerly=True,  # Add this parameter
            )
            with open(f"{self.symbol}_scaler.pkl", "rb") as f:
                scaler = pickle.load(f)

            # Load and preprocess data
            df = pd.read_csv(f"{self.symbol}.csv")
            data = df["Close"].values.reshape(-1, 1)
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
                self.error_occurred.emit(
                    f"Expected 15 predictions, got {len(predicted_prices)}"
                )
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
        self.column_dropdown.setStyleSheet(
            "background-color: #ecf0f1; color: black; padding: 5px;"
        )
        self.layout.addWidget(self.column_dropdown)
        self.column_dropdown.setEnabled(False)  # Initially disable the dropdown

        self.train_button = QPushButton("Train Model", self)
        self.train_button.clicked.connect(self.train_model)
        self.train_button.setStyleSheet(
            "QPushButton { background-color: #2ecc71; padding: 8px; border-radius: 4px; }"
            "QPushButton:disabled { background-color: #95a5a6; }"
        )
        self.layout.addWidget(self.train_button)

        self.predict_button = QPushButton("Predict", self)
        self.predict_button.clicked.connect(self.make_prediction)
        self.predict_button.setStyleSheet(
            "QPushButton { background-color: #3498db; padding: 8px; border-radius: 4px; }"
            "QPushButton:disabled { background-color: #95a5a6; }"
        )
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
            if hasattr(self, "data_thread"):
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
                self.data_thread.data_fetched.connect(
                    self.handle_data_fetched, Qt.QueuedConnection
                )
                self.data_thread.error_occurred.connect(
                    self.show_error_message, Qt.QueuedConnection
                )
                self.data_thread.progress_updated.connect(
                    self.update_progress, Qt.QueuedConnection
                )

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
            if hasattr(self, "data_thread"):
                delattr(self, "data_thread")

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
        try:
            # Disable the train button and update text
            self.train_button.setEnabled(False)
            self.train_button.setText("Training in Progress...")

            symbol = self.symbol_dropdown.currentData()
            csv_path = getattr(self, "custom_csv_path", None)
            column_name = self.column_dropdown.currentText()

            self.train_thread = ModelTrainingThread(symbol, csv_path, column_name)
            self.train_thread.training_completed.connect(self.handle_training_completed)
            self.train_thread.error_occurred.connect(self.handle_training_error)
            self.train_thread.progress_updated.connect(self.update_progress)
            self.progress_bar.setValue(0)
            self.train_thread.start()

        except Exception as e:
            self.handle_training_error(str(e))

    def handle_training_completed(self, message):
        """Handle successful training completion"""
        self.show_success_message(message)
        self.train_button.setEnabled(True)
        self.train_button.setText("Train Model")

    def handle_training_error(self, error_message):
        """Handle training errors"""
        self.show_error_message(error_message)
        self.train_button.setEnabled(True)
        self.train_button.setText("Train Model")

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
            line = ax.plot(
                x_points,
                predicted_prices,
                color="#1abc9c",
                marker="o",
                linestyle="-",
                linewidth=2,
                markersize=8,
                label="Predicted Prices",
            )[0]

            # Add hover annotations
            cursor = mplcursors.cursor(line, hover=True)
            cursor.connect(
                "add",
                lambda sel: sel.annotation.set_text(f"Price: ${sel.target[1]:,.2f}"),
            )

            # Style annotations
            cursor.connect(
                "add",
                lambda sel: sel.annotation.get_bbox_patch().set(
                    fc="#34495e", alpha=0.8, edgecolor="white"
                ),
            )

            # Set annotation text color
            cursor.connect("add", lambda sel: sel.annotation.set_color("white"))

            # Set x-axis ticks and labels
            ax.set_xticks(x_points)
            ax.set_xticklabels([f"{i+1} min" for i in range(15)])

            # Format y-axis to show full numbers
            formatter = plt.ScalarFormatter(useOffset=False)
            formatter.set_scientific(False)
            ax.yaxis.set_major_formatter(formatter)

            # Customize appearance
            ax.grid(True, linestyle="--", alpha=0.7)
            ax.set_title(
                f"Predicted Prices for {self.symbol_dropdown.currentText()}",
                color="white",
                pad=20,
            )
            ax.set_facecolor("#34495e")
            fig.patch.set_facecolor("#2c3e50")

            # Label axes
            ax.set_xlabel("Time (minutes)", color="white", labelpad=10)
            ax.set_ylabel("Price", color="white", labelpad=10)

            # Style ticks
            ax.tick_params(axis="x", colors="white", rotation=45)
            ax.tick_params(axis="y", colors="white")

            # Add legend
            ax.legend(loc="upper right")

            # Adjust layout to prevent label cutoff
            plt.tight_layout()

            # Create and add canvas
            canvas = FigureCanvas(fig)
            self.chart_layout.addWidget(canvas)

            # Add prediction label
            coin_label = QLabel(
                f"15-minute prediction for {self.symbol_dropdown.currentText()}"
            )
            coin_label.setAlignment(Qt.AlignCenter)
            coin_label.setStyleSheet(
                "font-size: 16px; font-weight: bold; color: #ecf0f1;"
            )
            self.chart_layout.addWidget(coin_label)

        except Exception as e:
            self.show_error_message(f"Error plotting chart: {str(e)}")

    def handle_data_fetched(self, data):
        # Display a message indicating successful data fetching
        QMessageBox.information(
            self,
            "Info",
            f"Data fetched successfully for {self.symbol_dropdown.currentText()}.",
        )

    def create_message_box(self, title, message, icon=QMessageBox.Information):
        """Create a message box with selectable text"""
        msg = QMessageBox(self)
        msg.setIcon(icon)
        msg.setWindowTitle(title)
        msg.setText(message)

        # Force the message box to be created so we can find the label
        msg.layout()

        # Find all labels in the message box and make their text selectable
        for child in msg.children():
            if isinstance(child, QLabel):
                child.setTextInteractionFlags(
                    Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard
                )
                # Ensure the cursor changes to indicate text is selectable
                child.setCursor(Qt.IBeamCursor)

        return msg

    def show_message(self, message):
        msg = self.create_message_box("Info", message)
        msg.exec_()

    def show_error_message(self, error_message):
        msg = self.create_message_box("Error", error_message, QMessageBox.Critical)
        msg.exec_()

    def show_success_message(self, message):
        msg = self.create_message_box("Success", message)
        msg.exec_()

    def load_custom_csv(self):
        try:
            # Open file dialog
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select CSV File", "", "CSV Files (*.csv);;All Files (*)"
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


if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        window = CryptoPredictionApp()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Failed to start application: {str(e)}")
        sys.exit(1)
