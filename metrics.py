import numpy as np


def calculate_mae(actual, predicted):
    """
    Calculate Mean Absolute Error

    Args:
        actual: numpy array of actual values
        predicted: numpy array of predicted values

    Returns:
        float: Mean Absolute Error
    """
    return np.mean(np.abs(actual - predicted))


def calculate_mse(actual, predicted):
    """
    Calculate Mean Squared Error

    Args:
        actual: numpy array of actual values
        predicted: numpy array of predicted values

    Returns:
        float: Mean Squared Error
    """
    return np.mean(np.square(actual - predicted))


def calculate_rmse(actual, predicted):
    """
    Calculate Root Mean Squared Error

    Args:
        actual: numpy array of actual values
        predicted: numpy array of predicted values

    Returns:
        float: Root Mean Squared Error
    """
    return np.sqrt(calculate_mse(actual, predicted))


def calculate_mape(actual, predicted, epsilon=1e-10):
    """
    Calculate Mean Absolute Percentage Error

    Args:
        actual: numpy array of actual values
        predicted: numpy array of predicted values
        epsilon: small value to avoid division by zero

    Returns:
        float: Mean Absolute Percentage Error (%)
    """
    return np.mean(np.abs((actual - predicted) / (actual + epsilon))) * 100


def calculate_accuracy_with_margin(actual, predicted, margin_percent=10):
    """
    Calculate percentage of predictions within a given margin of error

    Args:
        actual: numpy array of actual values
        predicted: numpy array of predicted values
        margin_percent: percentage margin for error tolerance (default 10%)

    Returns:
        float: Percentage of predictions within the margin
    """
    margin = margin_percent / 100.0
    differences = np.abs(actual - predicted)
    # Calculate the allowed error for each actual value
    allowed_errors = np.abs(actual) * margin
    # Count predictions within margin
    within_margin = np.sum(differences <= allowed_errors)
    return (within_margin / len(actual)) * 100


def calculate_directional_accuracy(actual, predicted):
    """
    Calculate accuracy of predicted direction (up/down) compared to actual direction

    Args:
        actual: numpy array of actual values
        predicted: numpy array of predicted values

    Returns:
        float: Percentage of correctly predicted directions
    """
    if len(actual) <= 1 or len(predicted) <= 1:
        return 0.0

    # Calculate actual and predicted directions
    actual_directions = np.diff(actual) >= 0  # True for up or flat, False for down
    predicted_directions = np.diff(predicted) >= 0

    # Compare directions and calculate accuracy
    correct_directions = np.sum(actual_directions == predicted_directions)
    return (correct_directions / len(actual_directions)) * 100


def evaluate_predictions(actual, predicted):
    """
    Evaluate predictions using multiple metrics with various margins of error

    Args:
        actual: numpy array of actual values
        predicted: numpy array of predicted values

    Returns:
        dict: Dictionary containing various metrics
    """
    return {
        "mae": calculate_mae(actual, predicted),
        "mse": calculate_mse(actual, predicted),
        "rmse": calculate_rmse(actual, predicted),
        "mape": calculate_mape(actual, predicted),
        "accuracy_0pct": calculate_accuracy_with_margin(actual, predicted, 0),
        "accuracy_0.5pct": calculate_accuracy_with_margin(actual, predicted, 0.5),
        "accuracy_1pct": calculate_accuracy_with_margin(actual, predicted, 1),
        "accuracy_5pct": calculate_accuracy_with_margin(actual, predicted, 5),
        "accuracy_10pct": calculate_accuracy_with_margin(actual, predicted, 10),
        "directional_accuracy": calculate_directional_accuracy(actual, predicted),
        "sample_size": len(actual),
    }
