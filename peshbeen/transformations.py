from pyexpat import model
import pandas as pd
import numpy as np
from numba import jit
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from sklearn.linear_model import LinearRegression
from numba import jit
##Stationarity Check
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

#------------------------------------------------------------------------------
# Box-Cox Transformation Utility Functions
#------------------------------------------------------------------------------

def box_cox_transform(x, shift=False, box_cox_lmda = None):
    """
    Applies a Box-Cox transformation to a series x.

    Args:
        x (array-like): The input data.
        shift (bool): add 1 to each element of data if True, default is False.
        box_cox_lmda (float): The lambda parameter for the Box-Cox transformation.

    Returns:
        tuple: (transformed data, updated lambda)
    """
    if (box_cox_lmda == None):
        if shift ==True:
            transformed_data, lmbda = boxcox((np.array(x)+1))
        else:
            transformed_data, lmbda = boxcox(np.array(x))
            
    if (box_cox_lmda != None):
        if shift ==True:
            lmbda = box_cox_lmda
            transformed_data = boxcox((np.array(x)+1), lmbda)
        else:
            lmbda = box_cox_lmda
            transformed_data = boxcox(np.array(x), lmbda)
    return transformed_data, lmbda


def back_box_cox_transform(y_pred, lmda, shift=False, box_cox_biasadj=False):
    """
    Inverse Box-Cox transform.

    Args:
        y_pred (array-like): Transformed forecast.
        lmda (float): Box-Cox lambda.
        shift (bool): Whether the original data was shifted.
        box_cox_biasadj (bool): Whether bias adjustment is applied.
    
    Returns:
        array-like: Back-transformed forecast.
    """
    if (box_cox_biasadj==False):
        if shift == True:
            forecast = inv_boxcox(y_pred, lmda)-1
        else:
            forecast = inv_boxcox(y_pred, lmda)
            
    if (box_cox_biasadj== True):
        pred_var = np.var(y_pred)
        if shift == True:
            if lmda ==0:
                forecast = np.exp(y_pred)*(1+pred_var/2)-1
            else:
                forecast = ((lmda*y_pred+1)**(1/lmda))*(1+((1-lmda)*pred_var)/(2*((lmda*y_pred+1)**2)))-1
        else:
            if lmda ==0:
                forecast = np.exp(y_pred)*(1+pred_var/2)
            else:
                forecast = ((lmda*y_pred+1)**(1/lmda))*(1+((1-lmda)*pred_var)/(2*((lmda*y_pred+1)**2)))
    return forecast

#------------------------------------------------------------------------------
# Differencing Utility Functions
#------------------------------------------------------------------------------

def undiff_ts(original_data, differenced_data, n):
    """
    Inverts (ordinary) differencing.

    Args:
        original_data: The original time series.
        differenced_data: The n-differenced series.
        n (int): The degree of differencing.

    Returns:
        array-like: Undifferenced series.
    """
    undiff_data = np.array(differenced_data)
    if n > 1:
        for i in range(n-1, 0, -1):
            undiff_data = np.diff(original_data, i)[-1]+np.cumsum(undiff_data)
    
    return original_data[-1]+np.cumsum(undiff_data)

def seasonal_diff(data, seasonal_period):
    """
    Computes seasonal differences on array x.

    Args:
        x (array-like): The time series.
        seasonal_period (int): The seasonal period.
        
    Returns:
        array-like: Seasonally differenced time series.
    """
    orig_data = list(np.repeat(np.nan, seasonal_period))+[data[i] - data[i - seasonal_period] for i in range(seasonal_period, len(data))]
    return np.array(orig_data)

def invert_seasonal_diff(orig_data, diff_data, seasonal_period):

    """
    Inverts seasonal differencing.

    Args:
        orig (list): The original series values.
        diff (array-like): The differenced series.
        seasonal_period (int): The seasonal period.
        
    Returns:
        array-like: Reconstructed series.
    """
    # Start with the last seasonal_period original values
    result = list(orig_data[-seasonal_period:])
    for i in range(len(diff_data)):
        # Each new value is previous season value + diff
        val = diff_data[i] + result[i]
        result.append(val)
    # Only return the reconstructed values matching diff_data length
    return np.array(result[seasonal_period:])

#------------------------------------------------------------------------------
# Croston's Method metrics
#------------------------------------------------------------------------------

# @jit(nopython=True)
def nzInterval(data, lag=0):
    """
    Computes the intervals between non-zero values in a time series.

    Args:
        data (array-like): Input time series data.
        lag (int): Number of lags to consider. Default is 0.

    Returns:
        numpy.ndarray: An array of intervals between non-zero values, with NaNs for the first `lag` values.
    """
    intervals = []
    last_nonzero_times = []
    if lag !=0:
        arr = data[:-lag]
    else:
        arr = data

    for i, j in enumerate(arr):
        if j > 0.5:
            last_nonzero_times.append(i)
        
        if len(last_nonzero_times)==1:
            intervals.append(1)
        else:
            inter = last_nonzero_times[-1]-last_nonzero_times[-2]
            intervals.append(inter)
    
    if lag !=0:
        nas = list(np.repeat(np.nan, lag))
        intervals = nas+intervals
    return np.array(intervals)

# @jit(nopython=True)
def zeroCumulative(data, lag=0):
    """
    Computes the cumulative count of consecutive zeros in a time series.
    Args:
        data (array-like): Input time series data.
        lag (int): Number of lags to consider. Default is 0.
    Returns:
        numpy.ndarray: An array of cumulative counts of consecutive zeros, with NaNs for the first `lag` values.
    """
    if lag !=0:
        arr = data[:-lag]
    else:
        arr = data

    count = 0
    result = []
    
    for value in arr:
        if value < 0.5:
            count += 1
        else:
            count = 0
        result.append(count)

    if lag !=0:
        nas = list(np.repeat(np.nan, lag))
        result = nas+result
    return np.array(result)


#------------------------------------------------------------------------------
# Target Encoding Utility Functions
#------------------------------------------------------------------------------

def kfold_target_encoder(df, feature_col, target_col, n_splits=5):
    """
    Perform K-fold target encoding for a given feature.

    Args:
        df (pd.DataFrame): Input dataframe.
        feature_col (str): Column name of the categorical feature.
        target_col (str): Target column name.
        n_splits (int): Number of folds for target encoding.

    Returns:
        np.array: Encoded target values.
    """
    df_encoded = df.copy()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    encoded_col_name = f"{feature_col}_target_encoded"
    df_encoded[encoded_col_name] = np.nan
    feat_idx = df_encoded.columns.get_loc(feature_col)
    encode_idx = df_encoded.columns.get_loc(encoded_col_name)
    # Perform KFold encoding
    for train_idx, val_idx in kf.split(df):
        # Calculate target mean per category using only the training fold
        target_means = df.iloc[train_idx].groupby(feature_col)[target_col].mean()
        # Map means to the validation fold
        df_encoded.iloc[val_idx, encode_idx] = df_encoded.iloc[val_idx, feat_idx].map(target_means)
    # Fill missing values with overall target mean
    overall_mean = df[target_col].mean()
    df_encoded[encoded_col_name].fillna(overall_mean, inplace=True)
    return df_encoded[encoded_col_name].values

def target_encoder_for_test(train_df, test_df, feature_col):
    """
    Apply target encoding to the test set based on the training set.

    Args:
        train_df (pd.DataFrame): Training dataframe with encoded feature.
        test_df (pd.DataFrame): Test dataframe.
        feature_col (str): Column name to encode.

    Returns:
        np.array: Encoded values for the test set.
    """
    encoded_col_name = f"{feature_col}_target_encoded"
    target_means = train_df.groupby(feature_col)[encoded_col_name].mean()
    overall_mean = train_df[encoded_col_name].mean()
    test_encoded = test_df.copy()
    test_encoded[encoded_col_name] = test_df[feature_col].map(target_means)
    test_encoded[encoded_col_name].fillna(overall_mean, inplace=True)
    return test_encoded[encoded_col_name].values


#------------------------------------------------------------------------------
# Transformation Utility Functions
#------------------------------------------------------------------------------

def fourier_terms(
    start_end_index: tuple,
    period: int,
    num_terms: int
) -> pd.DataFrame:
    """
    Generate Fourier terms (sine and cosine) for a given period and number of terms.

    Args:
        start_end_index (tuple): (start, end) defining the index.
            - If integers: produces a RangeIndex from start to end (inclusive).
            - If datetimes (strings or pd.Timestamp): produces a DatetimeIndex from start to end (inclusive, daily freq).
        period (int): Seasonal period (e.g., 7 for weekly seasonality).
        num_terms (int): Number of Fourier pairs (sine/cosine).

    Returns:
        pd.DataFrame: DataFrame of Fourier terms indexed by generated index.
    """
    start, end = start_end_index

    # Decide index type
    if isinstance(start, (int, np.integer)) and isinstance(end, (int, np.integer)):
        index = pd.RangeIndex(start, end+1)  # exclusive of end
        t = np.arange(len(index))
    else:
        # Convert to datetime if string given
        start, end = pd.to_datetime(start), pd.to_datetime(end)
        index = pd.date_range(start, end, freq="D")
        t = np.arange(len(index))

    # Build Fourier terms
    terms = {
        f'sin_{k}_{period}': np.sin(2 * np.pi * k * t / period)
        for k in range(1, num_terms + 1)
    }
    terms.update({
        f'cos_{k}_{period}': np.cos(2 * np.pi * k * t / period)
        for k in range(1, num_terms + 1)
    })

    return pd.DataFrame(terms, index=index)

# @jit(nopython=True)
# Your transformation classes
class rolling_mean:
    """
    A class to compute rolling mean with a specified window size and minimum samples.
    Args:
        shift (int): The number of periods to shift time series data.
        window_size (int): The size of the rolling window.
        min_samples (int): Minimum number of samples required to compute the mean.
    """

    def __init__(self, window_size, shift=1, min_samples=1):
        self.shift = shift
        self.window_size = window_size
        self.min_samples = min_samples

    def __call__(self, data, is_forecast=False):
        """
        Compute the rolling mean of the data with the specified parameters.
        Args:
            data (pd.Series or np.ndarray): The input data to compute the rolling mean.
            is_forecast (bool): If True, adjust the shift for forecasting purposes.
        Returns:
            pd.Series: The rolling mean of the input data.
        """

        if is_forecast:
            # For example, if it's a forecast, for the forecasting next value, we might want to shift by one less than usual to align with the forecasted period
            return pd.Series(data).shift(self.shift-1).rolling(self.window_size, min_periods=self.min_samples).mean()
        # If not a forecast, apply the usual shift
        else:
            # If not a forecast, apply the usual shift
            return pd.Series(data).shift(self.shift).rolling(self.window_size, min_periods=self.min_samples).mean()
        
    def get_name(self):
        return f"rolling_mean_{self.window_size}_{self.shift}"


class rolling_quantile:
    """
    A class to compute rolling quantile with a specified window size and minimum samples.
    Args:
        shift (int): The number of periods to shift time series data.
        window_size (int): The size of the rolling window.
        min_samples (int): Minimum number of samples required to compute the quantile.
    """

    def __init__(self, window_size, quantile, shift=1, min_samples=1):
        self.shift = shift
        self.window_size = window_size
        self.quantile = quantile
        self.min_samples = min_samples

    def __call__(self, data, is_forecast=False):
        # Return the rolling quantile with the specified window size and minimum samples
        if is_forecast:
            return pd.Series(data).shift(self.shift-1).rolling(self.window_size, min_periods=self.min_samples).quantile(self.quantile)
        # If not a forecast, apply the usual shift
        else:
            # If not a forecast, apply the usual shift
            return pd.Series(data).shift(self.shift).rolling(self.window_size, min_periods=self.min_samples).quantile(self.quantile)
    def get_name(self):
        return f"rolling_quantile_{self.window_size}_{self.quantile}_{self.shift}"
    
class rolling_std:
    """
    A class to compute rolling std with a specified window size and minimum samples.
    Args:
        shift (int): The number of periods to shift time series data.
        window_size (int): The size of the rolling window.
        min_samples (int): Minimum number of samples required to compute the std.
    """

    def __init__(self, window_size, shift=1, min_samples=1):
        self.shift = shift
        self.window_size = window_size
        self.min_samples = min_samples

    def __call__(self, data, is_forecast=False):
        # Return the rolling std with the specified window size and minimum samples
        if is_forecast:
            return pd.Series(data).shift(self.shift-1).rolling(self.window_size, min_periods=self.min_samples).std()
        else:
            return pd.Series(data).shift(self.shift).rolling(self.window_size, min_periods=self.min_samples).std()
    def get_name(self):
        return f"rolling_std_{self.window_size}_{self.shift}"   

class expanding_std:
    """
    A class to compute expanding standard deviation.
    Args:
        shift (int): The number of periods to shift time series data.
    """
    def __init__(self, shift=1):
        self.shift = shift
    def __call__(self, data, is_forecast=False):
        if is_forecast:
            return pd.Series(data).shift(self.shift-1).expanding().std()
        else:
            return pd.Series(data).shift(self.shift).expanding().std()
    def get_name(self):
        return f"expanding_std_{self.shift}"

class expanding_mean:
    """
    A class to compute expanding mean.
    Args:
        shift (int): The number of periods to shift time series data.
    """
    def __init__(self, shift=1):
        self.shift = shift
    def __call__(self, data, is_forecast=False):
        if is_forecast:
            return pd.Series(data).shift(self.shift-1).expanding().mean()
        else:
            return pd.Series(data).shift(self.shift).expanding().mean()
    def get_name(self):
        return f"expanding_mean_{self.shift}"

class expanding_quantile:
    """
    A class to compute expanding quantile.
    Args:
        quantile (float): The quantile to compute.
    """
    def __init__(self, shift=1, quantile=0.5):
        self.shift = shift
        self.quantile = quantile

    def __call__(self, data, is_forecast=False):
        """
        Compute the expanding quantile.
        """
        if is_forecast:
            return pd.Series(data).shift(self.shift-1).expanding().quantile(self.quantile)
        else:
            return pd.Series(data).shift(self.shift).expanding().quantile(self.quantile)
    def get_name(self):
        return f"expanding_quantile_{self.quantile}_{self.shift}"   
        
class expanding_ets:
    """
    A class to compute expanding ETS.
    Args:
        shift (int): The number of periods to shift time series data.
        ets_params (dict): Parameters for the ExponentialSmoothing model.
        fit_params (dict): Parameters for the fit method of the ExponentialSmoothing model.
    """
    def __init__(self, ets_params, fit_params, shift=1):
        self.shift = shift
        self.ets_params = ets_params
        self.fit_params = fit_params
    def __call__(self, data, is_forecast=False):
        if is_forecast:
            self.shift -= 1
        model_ = ExponentialSmoothing(pd.Series(data).shift(self.shift).dropna(), **self.ets_params).fit(**self.fit_params)
        fitted = model_.fittedvalues
        fitted_aligned = pd.Series(np.nan, index=data.index) # Align the index with the original data to avoid misalignment
        fitted_aligned.iloc[self.shift:] = fitted.values
        return fitted_aligned
    
    def get_name(self):
        return f"expanding_ets_{self.shift}"
