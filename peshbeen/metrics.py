import numpy as np
import warnings
warnings.filterwarnings("ignore")

#------------------------------------------------------------------------------
# Evaluation Metrics
#------------------------------------------------------------------------------

def MAPE(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error (MAPE).

    Parameters:
    - y_true: Array of true values.
    - y_pred: Array of predicted values.

    Returns:
    - mape: Mean Absolute Percentage Error.
    """
    # Ensure both arrays have the same length
    if len(y_true) != len(y_pred):
        raise ValueError("Input arrays must have the same length.")
    # Convert to numpy arrays for element-wise operations
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate absolute percentage error
    return round(np.mean(np.abs((y_true - y_pred) / y_true)), 2)

def MAE(y_true, y_pred):
    """
    Calculate Mean Absolute Error (MAE).

    Parameters:
    - y_true: Array of true values.
    - y_pred: Array of predicted values.

    Returns:
    - mae: Mean Absolute Error.
    """
    # Ensure both arrays have the same length
    if len(y_true) != len(y_pred):
        raise ValueError("Input arrays must have the same length.")
    # Convert to numpy arrays for element-wise operations
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return np.mean(np.abs(y_true - y_pred))

def MSE(y_true, y_pred):
    """
    Calculate Mean Squared Error (MSE).

    Parameters:
    - y_true: Array of true values.
    - y_pred: Array of predicted values.

    Returns:
    - mse: Mean Squared Error.
    """
    # Ensure both arrays have the same length
    if len(y_true) != len(y_pred):
        raise ValueError("Input arrays must have the same length.")
    
    # Convert to numpy arrays for element-wise operations
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return np.mean((y_true - y_pred) ** 2)

def RMSE(y_true, y_pred):
    """
    Calculate Root Mean Square Error (RMSE).

    Parameters:
    - y_true: Array of true values.
    - y_pred: Array of predicted values.

    Returns:
    - rmse: Root Mean Square Error.
    """
    # Ensure both arrays have the same length
    if len(y_true) != len(y_pred):
        raise ValueError("Input arrays must have the same length.")
    # Convert to numpy arrays for element-wise operations
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def SMAPE(y_true, y_pred):
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE).
    
    Parameters:
    ------------
    y_true (array-like): Actual values.
    y_pred (array-like): Predicted values.
    
    Returns:
    ------------
    float: SMAPE value.
    """
    # Ensure both arrays have the same length
    if len(y_true) != len(y_pred):
        raise ValueError("Input arrays must have the same length.")
    # Convert to numpy arrays for element-wise operations
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return 1/len(y_true) * np.sum(2 * np.abs(y_pred-y_true) / (np.abs(y_true) + np.abs(y_pred))*100)

def MASE(y_true, y_pred, y_train):
    """
    Calculate Mean Absolute Scaled Error (MASE)
    
    Parameters:
    ------------
    y_true (array-like): Actual values
    y_pred (array-like): Predicted values
    y_train (array-like): Training data used to scale the error
    
    Returns:
    ------------
    float: MASE value
    """

    # Ensure both arrays have the same length
    if len(y_true) != len(y_pred):
        raise ValueError("Input arrays must have the same length.")
    # Convert to numpy arrays for element-wise operations
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_train = np.array(y_train)
    # Calculate the mean absolute error
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Calculate the scaled error
    scaled_error = np.mean(np.abs(np.diff(y_train)))
    
    # Calculate MASE

    return mae / scaled_error

# def MedianASE(y_true, y_pred, y_train):
#     """
#     Calculate Median Absolute Scaled Error (MASE)
    
#     Parameters:
#     y_true (array-like): Actual values
#     y_pred (array-like): Predicted values
#     y_train (array-like): Training data used to scale the error
    
#     Returns:
#     float: MASE value
#     """

#     # Ensure both arrays have the same length
#     if len(y_true) != len(y_pred):
#         raise ValueError("Input arrays must have the same length.")
#     # Convert to numpy arrays for element-wise operations
#     y_true = np.array(y_true)
#     y_pred = np.array(y_pred)
#     y_train = np.array(y_train)
#     # Calculate the mean absolute error
#     mae = np.median(np.abs(y_true - y_pred))
    
#     # Calculate the scaled error
#     scaled_error = np.median(np.abs(np.diff(y_train)))
    
#     # Calculate MASE
    
#     return mae / scaled_error


def CFE(y_true, y_pred):
    """
    Calculate Cumulative Forecast Error (CFE).
    Parameters:
    y_true (array-like): Actual values.
    y_pred (array-like): Predicted values.

    Returns:
    float: CFE value.
    """
    return np.cumsum([a - f for a, f in zip(y_true, y_pred)])[-1]

def CFE_ABS(y_true, y_pred):
    """
    Calculate Absolute Cumulative Forecast Error (CFE_ABS).
    Parameters:
    y_true (array-like): Actual values.
    y_pred (array-like): Predicted values.

    Returns:
    float: Absolute CFE value.
    """
    # Ensure both arrays have the same length
    if len(y_true) != len(y_pred):
        raise ValueError("Input arrays must have the same length.")
    # Convert to numpy arrays for element-wise operations
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # Calculate cumulative forecast error
    cfe_t = np.cumsum([a - f for a, f in zip(y_true, y_pred)])
    return np.abs(cfe_t[-1])

def WMAPE(y_true, y_pred):
    """
    Calculate Weighted Mean Absolute Percentage Error (WMAPE).
    
    Parameters:
    y_true (array-like): Actual values.
    y_pred (array-like): Forecasted values.
    
    Returns:
    float: WMAPE value.
    """
    # Ensure both arrays have the same length
    if len(y_true) != len(y_pred):
        raise ValueError("Input arrays must have the same length.")
    # Convert to numpy arrays for element-wise operations
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    return np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)