import yfinance as yf
import datetime

crypto_data = {
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "DOGE-USD": "Dogecoin",
    "ADA-USD": "Cardano",
    "BNB-USD": "Binance Coin",
    "SOL-USD": "Solana",
    "XRP-USD": "Ripple"
}

def get_data(symbol):
    today = datetime.date.today()
    one_week_ago = today - datetime.timedelta(days=7)
    try:
        data = yf.download(symbol, start=one_week_ago.strftime("%Y-%m-%d"), end=datetime.datetime.today().strftime("%Y-%m-%d"), interval="1m")
        data.to_csv(f"{symbol}.csv")
        return 0
    except:
        return 1
    