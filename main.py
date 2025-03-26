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
from metrics import evaluate_predictions  # Add this import

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


class AccuracyEvaluationThread(QThread):
    accuracy_calculated = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    progress_updated = pyqtSignal(int)

    def __init__(self, symbol, column_name="Close", test_days=3, custom_margin=None):
        super().__init__()
        self.symbol = symbol
        self.column_name = column_name
        self.test_days = test_days  # Number of days to use for accuracy testing
        self.custom_margin = custom_margin  # Custom margin of error

    def run(self):
        try:
            print(f"Starting accuracy evaluation for {self.symbol}...")
            self.progress_updated.emit(10)

            # Load model and scaler
            model_path = f"{self.symbol}_model.h5"
            scaler_path = f"{self.symbol}_scaler.pkl"

            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                raise FileNotFoundError(
                    f"Model or scaler file not found for {self.symbol}"
                )

            model = load_model(model_path, compile=False)
            model.compile(optimizer="adam", loss="mean_squared_error")

            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)

            self.progress_updated.emit(30)

            # Load historical data
            df = pd.read_csv(f"{self.symbol}.csv")
            if df.empty:
                raise ValueError(f"No data available for {self.symbol}")

            # Get the specified column data
            if self.column_name not in df.columns:
                raise ValueError(f"Column {self.column_name} not found in data")

            data = df[self.column_name].values.reshape(-1, 1)

            # We'll use only a portion of the data for testing
            # Use 1-minute data, so 1440 minutes per day
            test_size = min(self.test_days * 1440, len(data) // 2)
            train_data = data[:-test_size]
            test_data = data[-test_size:]

            self.progress_updated.emit(50)

            # Prepare the metrics
            mae_list = []
            mse_list = []
            mape_list = []

            # We'll make predictions at different points and compare
            prediction_days = 60  # Must match the model's input window
            prediction_window = 15  # 15-minute prediction window

            # Process in batches to avoid memory issues
            batch_size = 20
            total_batches = min(
                (test_size - prediction_days - prediction_window) // batch_size, 10
            )

            for batch in range(total_batches):
                batch_predictions = []
                batch_actuals = []

                for i in range(batch_size):
                    point = batch * batch_size + i
                    if point + prediction_days + prediction_window >= len(test_data):
                        break

                    # Get sequence and actual future values
                    sequence = test_data[point : point + prediction_days]
                    actuals = test_data[
                        point
                        + prediction_days : point
                        + prediction_days
                        + prediction_window,
                        0,
                    ]

                    # Scale the sequence
                    scaled_sequence = scaler.transform(sequence)
                    X_test = scaled_sequence.reshape(1, prediction_days, 1)

                    # Predict
                    scaled_predictions = model.predict(X_test, verbose=0)

                    # Reshape and inverse transform
                    scaled_predictions = scaled_predictions.reshape(-1, 1)
                    predictions = scaler.inverse_transform(scaled_predictions).flatten()

                    # Collect predictions and actuals
                    batch_predictions.extend(predictions)
                    batch_actuals.extend(actuals)

                progress = 50 + int(50 * (batch + 1) / total_batches)
                self.progress_updated.emit(progress)

            # Convert to numpy arrays
            predictions = np.array(batch_predictions)
            actuals = np.array(batch_actuals)

            # Use the enhanced evaluate_predictions function with custom margin
            from metrics import evaluate_predictions, calculate_accuracy_with_margin

            results = evaluate_predictions(actuals, predictions)

            # Always add custom margin accuracy if specified
            if self.custom_margin is not None:
                # Format the key correctly
                custom_margin_key = f"accuracy_{self.custom_margin}pct".replace(
                    ".", "_"
                )
                # Calculate and add to results
                results[custom_margin_key] = calculate_accuracy_with_margin(
                    actuals, predictions, self.custom_margin
                )
                # Add a flag to indicate this was user-selected
                results["user_selected_margin"] = self.custom_margin

                print(
                    f"Added custom margin accuracy for {self.custom_margin}%: {results[custom_margin_key]:.2f}%"
                )

            print("Accuracy evaluation completed")
            self.accuracy_calculated.emit(results)

        except Exception as e:
            print(f"Error during accuracy evaluation: {str(e)}")
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

        # Add margin of error settings
        margin_frame = QFrame(self)
        margin_layout = QHBoxLayout()
        margin_label = QLabel("Margin of Error (%): ", self)

        self.margin_dropdown = QComboBox(self)
        margin_options = ["0", "0.1", "0.5", "1", "2", "5", "10", "15", "20"]
        self.margin_dropdown.addItems(margin_options)
        self.margin_dropdown.setCurrentText("5")  # Default to 5%
        self.margin_dropdown.setStyleSheet(
            "background-color: #ecf0f1; color: black; padding: 5px;"
        )

        margin_layout.addWidget(margin_label)
        margin_layout.addWidget(self.margin_dropdown)
        margin_frame.setLayout(margin_layout)
        self.layout.addWidget(margin_frame)

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

        self.accuracy_button = QPushButton("Evaluate Accuracy", self)
        self.accuracy_button.clicked.connect(self.evaluate_accuracy)
        self.accuracy_button.setStyleSheet(
            "QPushButton { background-color: #9b59b6; padding: 8px; border-radius: 4px; }"
            "QPushButton:disabled { background-color: #95a5a6; }"
        )
        self.layout.addWidget(self.accuracy_button)

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
            return

    def load_csv_columns(self, file_path):
        try:
            # Load the CSV to get column names
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
            ax.set_facecolor("#34495e")
            fig.patch.set_facecolor("#2c3e50")
            ax.set_xlabel("Time (minutes)", color="white", labelpad=10)
            ax.set_ylabel("Price", color="white", labelpad=10)
            ax.set_title(
                f"Predicted Prices for {self.symbol_dropdown.currentText()}",
                color="white",
                pad=20,
            )
            ax.tick_params(axis="x", colors="white", rotation=45)
            ax.tick_params(axis="y", colors="white")
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

    def evaluate_accuracy(self):
        try:
            symbol = self.symbol_dropdown.currentData()

            # Check if model exists
            if not os.path.exists(f"{symbol}_model.h5"):
                self.show_error_message(
                    f"No trained model found for {symbol}. Please train a model first."
                )
                return

            # Get column name if available
            column_name = "Close"  # Default
            if self.column_dropdown.isEnabled() and self.column_dropdown.currentText():
                column_name = self.column_dropdown.currentText()

            # Get user-selected margin of error
            margin = float(self.margin_dropdown.currentText())

            # Create and start the accuracy evaluation thread
            self.accuracy_thread = AccuracyEvaluationThread(
                symbol, column_name, test_days=3, custom_margin=margin
            )
            self.accuracy_thread.accuracy_calculated.connect(
                self.display_accuracy_results
            )
            self.accuracy_thread.error_occurred.connect(self.show_error_message)
            self.accuracy_thread.progress_updated.connect(self.update_progress)

            # Disable button and update progress bar
            self.accuracy_button.setEnabled(False)
            self.progress_bar.setValue(0)

            # Start the thread
            self.accuracy_thread.start()

        except Exception as e:
            self.show_error_message(f"Error starting accuracy evaluation: {str(e)}")
            self.accuracy_button.setEnabled(True)

    def display_accuracy_results(self, results):
        try:
            # Format the results
            mae = results["mae"]
            mse = results["mse"]
            rmse = results["rmse"]
            mape = results["mape"]
            accuracy_0pct = results.get("accuracy_0pct", 0.0)
            accuracy_05pct = results.get("accuracy_0.5pct", 0.0)
            accuracy_1pct = results.get("accuracy_1pct", 0.0)
            accuracy_5pct = results.get("accuracy_5pct", 0.0)
            accuracy_10pct = results.get("accuracy_10pct", 0.0)
            directional_accuracy = results.get("directional_accuracy", 0.0)
            sample_size = results["sample_size"]

            # Get the user-selected margin of error
            user_selected_margin = results.get("user_selected_margin")

            # Build standard accuracies section
            accuracy_section = (
                f"Accuracy with margins of error:\n"
                f"• ±0% margin: {accuracy_0pct:.2f}%\n"
                f"• ±0.5% margin: {accuracy_05pct:.2f}%\n"
                f"• ±1% margin: {accuracy_1pct:.2f}%\n"
                f"• ±5% margin: {accuracy_5pct:.2f}%\n"
                f"• ±10% margin: {accuracy_10pct:.2f}%\n"
            )

            # Always add the user selected margin if it exists and is not already covered
            if user_selected_margin is not None and user_selected_margin not in [
                0,
                0.5,
                1,
                5,
                10,
            ]:
                custom_margin_key = f"accuracy_{user_selected_margin}pct".replace(
                    ".", "_"
                )
                custom_accuracy = results.get(custom_margin_key, 0.0)
                accuracy_section += f"• ±{user_selected_margin}% margin: {custom_accuracy:.2f}% (selected)\n"

            # Create a message with the results
            message = (
                f"Model Accuracy Metrics (sample size: {sample_size}):\n\n"
                f"{accuracy_section}\n"
                f"Price Direction Accuracy: {directional_accuracy:.2f}%\n\n"
                f"Error Metrics:\n"
                f"Mean Absolute Error (MAE): {mae:.6f}\n"
                f"Mean Squared Error (MSE): {mse:.6f}\n"
                f"Root Mean Squared Error (RMSE): {rmse:.6f}\n"
                f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%\n\n"
                f"Higher percentages indicate better performance for accuracy metrics.\n"
                f"Lower values indicate better performance for error metrics."
            )

            # Show the results
            self.show_accuracy_message("Accuracy Results", message)

        except Exception as e:
            self.show_error_message(f"Error displaying accuracy results: {str(e)}")
        finally:
            # Re-enable the accuracy button
            self.accuracy_button.setEnabled(True)

    def show_accuracy_message(self, title, message):
        msg = self.create_message_box(title, message)
        msg.setStyleSheet("QLabel { min-width: 400px; }")
        msg.exec_()


if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        window = CryptoPredictionApp()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Failed to start application: {str(e)}")
        sys.exit(1)
