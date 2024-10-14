import sys
import yfinance as yf
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QComboBox, QPushButton, 
                             QVBoxLayout, QHBoxLayout, QFrame, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import cryptodata  # Import the crypto data from cryptodata.py


# Create a thread to fetch data from yfinance
class DataFetchThread(QThread):
    data_fetched = pyqtSignal(object)  # Signal to emit data when fetched
    error_occurred = pyqtSignal(str)  # Signal to emit error messages

    def __init__(self, symbol):
        super().__init__()
        self.symbol = symbol

    def run(self):
        try:
            # Fetch cryptocurrency data from yfinance
            data = yf.download(self.symbol, period="1mo", interval="1d")
            if data.empty:
                self.error_occurred.emit(f"No data found for {self.symbol}.")
            else:
                self.data_fetched.emit(data)  # Emit data when fetched successfully
        except Exception as e:
            self.error_occurred.emit(str(e))


# Create a thread for saving the data fetched via cryptodata.py
class SaveDataThread(QThread):
    data_saved = pyqtSignal(str)  # Signal to emit when data is saved
    error_occurred = pyqtSignal(str)  # Signal to emit if an error occurs during saving

    def __init__(self, symbol):
        super().__init__()
        self.symbol = symbol

    def run(self):
        try:
            # Use the get_data function from cryptodata.py to save data
            flag = cryptodata.get_data(self.symbol)
            if flag == 0:
                self.data_saved.emit(f"Data fetched successfully and saved as {self.symbol}.csv")
            else:
                self.error_occurred.emit(f"Error fetching data for {self.symbol}.")
        except Exception as e:
            self.error_occurred.emit(str(e))


# Create the main PyQt5 window
class CryptoPredictionApp(QWidget):
    def __init__(self):
        super().__init__()

        # Window settings
        self.setWindowTitle("Crypto Prediction App")
        self.setGeometry(300, 300, 800, 600)
        self.setStyleSheet("background-color: #2c3e50; color: white;")

        # Layout for the main window
        self.layout = QVBoxLayout()

        # Create title and input box for selecting cryptocurrency symbol
        self.title = QLabel("Cryptocurrency Price Prediction", self)
        self.title.setStyleSheet("font-size: 24px; font-weight: bold; color: #ecf0f1;")
        self.title.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.title)

        # Create dropdown for selecting a cryptocurrency symbol
        self.input_frame = QFrame(self)
        self.input_layout = QHBoxLayout()
        self.symbol_label = QLabel("Select Symbol: ", self)
        self.symbol_label.setStyleSheet("font-size: 16px;")
        self.symbol_dropdown = QComboBox(self)
        self.symbol_dropdown.setStyleSheet("background-color: #ecf0f1; color: black; padding: 5px;")
        self.load_crypto_symbols()  # Load symbols from cryptodata.py

        self.input_layout.addWidget(self.symbol_label)
        self.input_layout.addWidget(self.symbol_dropdown)
        self.input_frame.setLayout(self.input_layout)
        self.layout.addWidget(self.input_frame)

        # Create a button for fetching and saving data
        self.load_button = QPushButton("Load Data", self)
        self.load_button.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                font-size: 16px;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #2ecc71;
            }
        """)
        self.load_button.clicked.connect(self.data_creation)
        self.layout.addWidget(self.load_button)

        # Create a button for predicting data
        self.predict_button = QPushButton("Predict", self)
        self.predict_button.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                font-size: 16px;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #2ecc71;
            }
        """)
        self.predict_button.clicked.connect(self.predict_price)
        self.layout.addWidget(self.predict_button)

        # Create a placeholder for displaying chart
        self.chart_frame = QFrame(self)
        self.chart_layout = QVBoxLayout()
        self.chart_placeholder = QLabel("Prediction chart will appear here", self)
        self.chart_placeholder.setAlignment(Qt.AlignCenter)
        self.chart_placeholder.setStyleSheet("font-size: 16px; border: 1px solid #7f8c8d; padding: 20px;")
        self.chart_layout.addWidget(self.chart_placeholder)
        self.chart_frame.setLayout(self.chart_layout)
        self.layout.addWidget(self.chart_frame)

        # Set the layout to the main window
        self.setLayout(self.layout)

    # Function to load cryptocurrency symbols from cryptodata.py
    def load_crypto_symbols(self):
        # Load data from cryptodata.py
        for symbol, name in cryptodata.crypto_data.items():
            self.symbol_dropdown.addItem(f"{name} ({symbol})", symbol)

    # Function to predict the price
    def predict_price(self):
        symbol = self.symbol_dropdown.currentData()  # Get the selected symbol
        if not symbol:
            self.chart_placeholder.setText("Please select a valid symbol.")
            return

        # Show loading message
        self.chart_placeholder.setText("Loading data, please wait...")

        # Start data fetching in a separate thread
        self.worker_thread = DataFetchThread(symbol)
        self.worker_thread.data_fetched.connect(self.plot_chart)  # Connect data to plot_chart function
        self.worker_thread.error_occurred.connect(self.show_error)  # Connect errors to error display function
        self.worker_thread.start()

    # Function to load and save cryptocurrency data
    def data_creation(self):
        symbol = self.symbol_dropdown.currentData()  # Get the selected symbol
        self.save_thread = SaveDataThread(symbol)
        self.save_thread.data_saved.connect(self.show_success_message)
        self.save_thread.error_occurred.connect(self.show_error_message)
        self.save_thread.start()

    # Function to plot chart after data is fetched
    def plot_chart(self, data):
        for i in reversed(range(self.chart_layout.count())):
            widget_to_remove = self.chart_layout.itemAt(i).widget()
            widget_to_remove.setParent(None)

        # Plot price data
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(data.index, data['Close'], color="#1abc9c", label='Closing Price')
        ax.set_title('Closing Price Trend', color='white')
        ax.set_facecolor('#34495e')
        fig.patch.set_facecolor('#2c3e50')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

        # Create a canvas to display the plot
        canvas = FigureCanvas(fig)
        self.chart_layout.addWidget(canvas)

        # Display the last price
        last_price = data['Close'].iloc[-1]
        self.chart_layout.addWidget(QLabel(f"Predicted Price: {last_price:.2f} USD", self))

    # Function to show error messages in a message box
    def show_error_message(self, error_message):
        error_box = QMessageBox()
        error_box.setIcon(QMessageBox.Critical)
        error_box.setWindowTitle("Error")
        error_box.setText(error_message)
        error_box.exec_()

    # Function to show success messages in a message box
    def show_success_message(self, success_message):
        success_box = QMessageBox()
        success_box.setIcon(QMessageBox.Information)
        success_box.setWindowTitle("Success")
        success_box.setText(success_message)
        success_box.exec_()

# Run the app
if __name__ == "__main__":
    app = QApplication(sys.argv)
    crypto_app = CryptoPredictionApp()
    crypto_app.show()
    sys.exit(app.exec_())
