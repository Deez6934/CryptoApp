CONFIG = {
    "prediction_window": 15,
    "training_window": 60,
    "epochs": 50,
    "batch_size": 32,
    "lstm_units": 50,
    "dropout_rate": 0.2,
    "memory_limit": 128 * 1024 * 1024,  # 128MB
    "min_disk_space": 100 * 1024 * 1024,  # 100MB
}

TECHNICAL_INDICATORS = {
    "SMA": 20,  # Simple Moving Average period
    "RSI": 14,  # Relative Strength Index period
    "MACD": {"fast": 12, "slow": 26, "signal": 9},
}
