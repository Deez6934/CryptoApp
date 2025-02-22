import pandas as pd
import numpy as np
from typing import Dict, Any
import os
import shutil


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataset by handling missing values and incorrect types"""
    df = df.copy()

    # Remove any non-numeric columns temporarily
    numeric_columns = df.select_dtypes(include=[np.number]).columns

    # Forward fill missing values first
    df[numeric_columns] = df[numeric_columns].ffill()

    # Then backward fill any remaining NaN at the beginning
    df[numeric_columns] = df[numeric_columns].bfill()

    # For any still remaining NaNs, replace with column mean
    for column in numeric_columns:
        df[column] = df[column].fillna(df[column].mean())

    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to the dataframe"""
    df = df.copy()

    # Ensure price columns are numeric
    price_columns = ["Open", "High", "Low", "Close", "Adj Close"]
    for col in price_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clean numeric data before calculating indicators
    df = clean_data(df)

    # SMA
    df["SMA20"] = df["Close"].rolling(window=20).mean()

    # RSI
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

    return df


def validate_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate input data and return validation results"""
    results = {"valid": True, "errors": [], "warnings": []}

    if df.empty:
        results["valid"] = False
        results["errors"].append("Dataset is empty")

    if df.isnull().values.any():
        results["warnings"].append("Dataset contains missing values")

    if len(df) < 75:
        results["valid"] = False
        results["errors"].append("Insufficient data points (minimum 75 required)")

    return results


def save_model_with_version(model, symbol: str, version: int) -> str:
    """Save model with versioning"""
    model_path = f"{symbol}_model_v{version}.h5"
    model.save(model_path)
    return model_path


def check_disk_space(path=".", required_space=100 * 1024 * 1024):  # 100MB default
    """Check if there's enough disk space on the drive containing path"""
    try:
        total, used, free = shutil.disk_usage(path)
        return free > required_space
    except Exception as e:
        print(f"Error checking disk space: {e}")
        return True  # Return True on error to allow operation to continue
