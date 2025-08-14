from pyexpat import model
import pandas as pd
import numpy as np
from scipy import stats
from numba import jit
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from sklearn.linear_model import LinearRegression
from numba import jit
##Stationarity Check
from statsmodels.tsa.stattools import adfuller, kpss
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statistics import NormalDist
import warnings
warnings.filterwarnings("ignore")

#------------------------------------------------------------------------------
# Unit Root Test and Serial Correlation Check
#------------------------------------------------------------------------------

def unit_root_test(series, method = "ADF", n_lag = None):
    """
    Performs a unit root test on the given time series data to check for stationarity.
    Args:
        series (pd.Series): The time series data to be tested.
        method (str): The method for the unit root test, either "ADF" for Augmented Dickey-Fuller or "KPSS" for Kwiatkowski-Phillips-Schmidt-Shin.
        n_lag (int, optional): The number of lags to include in the test. If None, the default lag will be used.
    Returns:
        float: The p-value from the unit root test.
    """
    if method == "ADF":
        if n_lag ==None:
            adf = adfuller(series)[1]
        else:
            adf = adfuller(series, maxlag = n_lag)[1]        
        if adf < 0.05:
            return adf, print('ADF p-value: %f' % adf + " and data is stationary at 5% significance level")
        else:
            return adf, print('ADF p-value: %f' % adf + " and data is non-stationary at 5% significance level")
    elif method == "KPSS":
        if n_lag == None:
            kps = kpss(series)[1]
        else:
            kps = kpss(series, nlags = n_lag)[1]
        if kps < 0.05:
            return kps, print('KPSS p-value: %f' % kps + " and data is non-stationary at 5% significance level")
        else:
            return kps, print('KPSS p-value: %f' % kps + " and data is stationary at 5% significance level")
    else:
        return print('Enter a valid unit root test method')

## Serial Corelation Check

def plot_PACF_ACF(series, lag_num=40, figsize=(15, 8), pacf_method='ywm', alpha=0.05, **kwargs):
    """
    Plots the Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF) of a time series.

    Args:
        series (array-like): The time series data (pandas Series, numpy array, or list).
        lag_num (int): The number of lags to consider (default=40).
        figsize (tuple): Size of the figure for the plots.
        pacf_method (str): PACF method for statsmodels (default='ywm').
        alpha (float): Significance level for confidence intervals (default=0.05).
        show (bool): Whether to display the plot (default=True).
        **kwargs: Additional keyword arguments passed to plot_acf and plot_pacf.

    Returns:
        (fig, axes): Matplotlib Figure and axes array.
    """
    # Convert input to pandas Series if necessary
    if not isinstance(series, pd.Series):
        try:
            series = pd.Series(series)
        except Exception as e:
            raise ValueError("Input series must be convertible to a pandas Series or be a numpy array.") from e

    if not isinstance(lag_num, int) or lag_num < 1:
        raise ValueError("lag_num must be a positive integer.")

    fig, axes = plt.subplots(2, 1, figsize=figsize)
    plot_pacf(series, lags=lag_num, ax=axes[0], method=pacf_method, alpha=alpha, **kwargs)
    axes[0].set_title('Partial Autocorrelation Function (PACF)')
    axes[0].set_xlabel('Lag')
    axes[0].set_ylabel('Partial Autocorrelation')
    axes[0].grid(True)

    plot_acf(series, lags=lag_num, ax=axes[1], alpha=alpha, **kwargs)
    axes[1].set_title('Autocorrelation Function (ACF)')
    axes[1].set_xlabel('Lag')
    axes[1].set_ylabel('Autocorrelation')
    axes[1].grid(True)

    fig.tight_layout()
    plt.show()

def cross_autocorrelation(x, y, nlags, adjusted=True, alpha=None, bartlett_confint=False):
    """
    Compute the cross-autocorrelation between two time series.

    Parameters
    ----------
    x : array_like
        First time series.
    y : array_like
        Second time series.
    nlags : int
        Number of lags to compute.
    adjusted : bool, optional
        Whether to apply the adjustment factor (default is True).
    alpha : float, optional
        Significance level for confidence intervals (default is None).
    bartlett_confint : bool, optional
        Whether to use Bartlett's method for confidence intervals (default is False).

    Returns
    -------
    cc : ndarray
        Cross-autocorrelation values for each lag and confidence intervals if `alpha` is provided.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    n = len(x)
    if len(y) != n:
        raise ValueError("x and y must have the same length")
    
    # Variance (autocovariance at lag 0)
    var_x = np.sum((x - x_mean)**2)
    var_y = np.sum((y - y_mean)**2)

    # Autocovariance but make sure make adjusted, meaning applying 1/(n-k) or 1/n
    cc = np.empty(nlags + 1)
    for k in range(nlags + 1):
        num = np.sum((y[:n-k] - y_mean) * (x[k:] - x_mean))
        r = num / np.sqrt(var_x * var_y)
        if adjusted and k > 0:
            r *= n / (n - k)
        cc[k] = r

    # Confidence intervals (optional)
    if alpha is not None:
        z = NormalDist().inv_cdf(1 - alpha/2)
        se = np.zeros_like(cc)
        
        if bartlett_confint:
            # Bartlett approximation: SE_k = sqrt((1 + 2 * sum_{j=1}^{k-1} cc[j]^2) / n)
            if nlags >= 1:
                r_sq_cum = np.cumsum(cc[1:]**2)
                prev_sum = np.concatenate(([0.0], r_sq_cum[:-1]))
                se[1:] = np.sqrt((1.0 + 2.0 * prev_sum) / n)
            se[0] = 0.0
        else:
            se[1:] = 1.0 / np.sqrt(n)
            se[0] = 0.0

        confint = np.column_stack((cc - z*se, cc + z*se))
        return cc, confint
    else:
        return cc, None
    
    
def cross_autocorrelation_plot(x, y, nlags, adjusted=True, alpha=0.05, figsize=(8, 5), title="Cross-Autocorrelation"):
    """
    Plot the cross-autocorrelation between two time series.

    Parameters
    ----------
    x : array_like
        First time series.
    y : array_like
        Second time series.
    nlags : int
        Number of lags to compute.
    adjusted : bool, optional
        Whether to apply the adjustment factor (default is True).
    alpha : float, optional
        Significance level for confidence intervals (default is 0.05).
    figsize : tuple, optional
        Figure size for the plot (default is (8, 5)).
    title : str, optional
        Title for the plot (default is "Cross-Autocorrelation").

    Returns
    -------
    cc : ndarray
        Cross-autocorrelation plot.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    n = len(x)
    if len(y) != n:
        raise ValueError("x and y must have the same length")
    
    # Variance (autocovariance at lag 0)
    var_x = np.sum((x - x_mean)**2)
    var_y = np.sum((y - y_mean)**2)

    # Autocovariance but make sure make adjusted, meaning applying 1/(n-k) or 1/n
    cc = np.empty(nlags + 1)
    for k in range(nlags + 1):
        num = np.sum((y[:n-k] - y_mean) * (x[k:] - x_mean))
        r = num / np.sqrt(var_x * var_y)
        if adjusted and k > 0:
            r *= n / (n - k)
        cc[k] = r

    # Confidence intervals (optional)
    z = NormalDist().inv_cdf(1 - alpha/2)
    if adjusted:
        den = n-nlags
    else:
        den = n
    bound = z / np.sqrt(den)
    # Bar plot of cross-correlation
    plt.figure(figsize=figsize)
    plt.bar(np.arange(nlags + 1), cc, edgecolor='k', label='Cross-correlation')

    z = NormalDist().inv_cdf(1 - alpha/2)
    bound = z / np.sqrt(n - nlags) if adjusted else z / np.sqrt(n)
    lag_x = np.arange(nlags + 1)
    plt.fill_between(lag_x, -bound, bound, color='gray', alpha=0.15, label='Confidence Interval')
    plt.axhline(bound, color='red', linewidth=0.8, alpha=0.7, linestyle='--')
    plt.axhline(-bound, color='red', linewidth=0.8, alpha=0.7, linestyle='--')
    plt.xlabel("Lags of second time series")
    plt.grid(axis='y')
    plt.ylabel("Cross-correlation")
    plt.title(title)
    plt.legend()
    plt.show()

#------------------------------------------------------------------------------
# Transformation Utility Functions
#------------------------------------------------------------------------------

def fourier_terms(
    start: int,
    stop: int,
    period: int,
    num_terms: int,
    reference_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate Fourier terms (sine and cosine) for a given period and number of terms.

    Args:
        start (int): Start index (usually 0 for train, len(train) for test).
        stop (int): Stop index (len(train) for train, len(train)+len(test) for test).
        period (int): Seasonal period.
        num_terms (int): Number of Fourier pairs (sine/cosine) to generate.
        reference_df (pd.DataFrame): DataFrame whose index will be used for alignment. For example, this could be the training or test dataset.

    Returns:
        pd.DataFrame: DataFrame of Fourier terms, indexed as reference_df.
    """
    t = np.arange(start, stop)
    if len(t) != len(reference_df):
        raise ValueError("Length of generated time steps does not match reference DataFrame length.")
    data = {}
    for k in range(1, num_terms + 1):
        data[f'sin_{k}_{period}'] = np.sin(2 * np.pi * k * t / period)
        data[f'cos_{k}_{period}'] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(data, index=reference_df.index)

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

class rolling_quantile:
    """
    A class to compute rolling quantile with a specified window size and minimum samples.
    Args:
        shift (int): The number of periods to shift time series data.
        window_size (int): The size of the rolling window.
        min_samples (int): Minimum number of samples required to compute the quantile.
    """

    def __init__(self, window_size, quantile, shift=1, min_samples=3):
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
    
class rolling_std:
    """
    A class to compute rolling std with a specified window size and minimum samples.
    Args:
        shift (int): The number of periods to shift time series data.
        window_size (int): The size of the rolling window.
        min_samples (int): Minimum number of samples required to compute the std.
    """

    def __init__(self, window_size, shift=1, min_samples=3):
        self.shift = shift
        self.window_size = window_size
        self.min_samples = min_samples

    def __call__(self, data, is_forecast=False):
        # Return the rolling std with the specified window size and minimum samples
        if is_forecast:
            return pd.Series(data).shift(self.shift-1).rolling(self.window_size, min_periods=self.min_samples).std()
        else:
            return pd.Series(data).shift(self.shift).rolling(self.window_size, min_periods=self.min_samples).std()

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

#------------------------------------------------------------------------------
# Lag Selection Algorithms
#------------------------------------------------------------------------------

def forward_lag_selection(df, max_lags, n_folds, H, model, metrics,
                          step_size = None, verbose = False):
    """
    Performs forward lag selection for Regression models.
    Args:
        df (pd.DataFrame): DataFrame containing the time series data.
        max_lags (int): Maximum number of lags to consider.
        n_folds (int): Number of folds for cross-validation.
        H (int): Forecast horizon.
        model: The forecasting model to be used.
        model_params (dict): Parameters for the model.
        metrics (list): List of metrics to evaluate the model.
        step_size (int, optional): Step size for cross-validation. Defaults to None.
        verbose (bool, optional): Whether to print progress. Defaults to False.
    Returns:
        list: List of best lags selected.
    """
    max_lag = max_lags
    orj_lags = list(range(1, max_lags+1))
    lags = list(range(1, max_lags+1))
    best_lags = []
    
    best_score = list(np.repeat(float('inf'), len(metrics)))
    
    while max_lag>0:
        for i in range(max_lag):
            best_lag = None
            for lg in lags:
                current_lag = best_lags + [lg]
                current_lag.sort()
                model.n_lag = current_lag
                # lag_model = model(**model_params, n_lag = current_lag)
                my_cv = cross_validate(model=model, df=df, cv_split=n_folds,
                                       test_size=H, metrics=metrics, step_size=step_size)
                scores = my_cv["score"].tolist()
                if scores<best_score:
                    best_score = scores
                    best_lag = lg 
            if best_lag is not None:
                best_lags.append(best_lag)
                lags.remove(best_lag)
                best_lags.sort()
                if verbose == True:
            # print worst variable, lags and current score
                    print(f'best lag: {best_lag} with score: {best_score}')
            else:
                break
        if best_lag is None:
            break
        lags = [item for item in orj_lags if item not in best_lags]
        lags.sort()
        if lags == orj_lags:
            break
        else:
            orj_lags = [item for item in orj_lags if item not in best_lags]
            orj_lags.sort()
            max_lag = len(orj_lags)
    return best_lags

def backward_lag_selection(df, max_lags,min_lags, n_folds, H, model, metrics, step_size=None, forward_back=False, verbose = False):
    """
    Performs backward lag selection for Regression models.
    Args:
        df (pd.DataFrame): DataFrame containing the time series data.
        max_lags (int): Maximum number of lags to consider.
        min_lags (int): Minimum number of lags to retain.
        n_folds (int): Number of folds for cross-validation.
        H (int): Forecast horizon.
        model: The forecasting model to be used.
        metrics (list): List of metrics to evaluate the model.
        step_size (int, optional): Step size for cross-validation. Defaults to None.
        forward_back (bool, optional): Whether to perform forward-backward selection. Defaults to False.
        verbose (bool, optional): Whether to print progress. Defaults to False.
    Returns:
        dict: Dictionary of best lags for each variable.
    """
    max_lag = max_lags
    min_lag =min_lags
    lags = list(range(1, max_lag+1))
    
    best_score = list(np.repeat(float('inf'), len(metrics)))
    
    worst_lags = []
    best_lags = None
    while len(lags) >= min_lag:
        worst_lag=None
        for lg in lags:
            lags_to_test = [x for x in lags if x != lg]
            lags_to_test.sort()
            model.n_lag = lags_to_test
            my_cv = cross_validate(model=model, df=df, cv_split=n_folds,
                                   test_size=H, metrics=metrics, step_size=step_size)
            scores = my_cv["score"].tolist()
            if scores<best_score:
                best_lags = lags_to_test
                worst_lag = lg
        if worst_lag is not None:
            # lags.append(best_lag)
            lags.remove(worst_lag)
            worst_lags.append(worst_lag)
            # print(lags)
            lags.sort()
            best_lags.sort()
            if verbose == True:
        # print worst variable, lags and current score
                print(f'worst lag: {worst_lag} with score: {best_score}')
        else:
            break
    
    if forward_back ==True:
    
        orj_lags = worst_lags.copy()
        orj_lags.sort()
        len_worst = len(orj_lags)
        
        while len(worst_lags)>0:
            for i in range(len_worst):
                best_lag = None
                for lg in worst_lags:
                    current_lag = best_lags + [lg]
                    current_lag.sort()
                    # lag_model = model(**model_params, n_lag = current_lag)
                    model.n_lag = current_lag
                    my_cv = cross_validate(model=model, df=df, cv_split=n_folds,
                                           test_size=H, metrics=metrics, step_size=step_size)
                    scores = my_cv["score"].tolist()
                    if scores<best_score:
                        best_score = scores
                        best_lag = lg 
                        
                if best_lag is not None:
                    best_lags.append(best_lag)
                    worst_lags.remove(best_lag)
                    if verbose == True:
                # print worst variable, lags and current score
                        print(f'best lag after backward: {best_lag} with score: {best_score}')
                    best_lags.sort()
                    # if verbose = True:
                    #     print(best_lags)
                else:
                    break
            if best_lag is None:
                break
            worst_lags = [item for item in orj_lags if item not in best_lags]
            lags.sort()
            if worst_lags == orj_lags:
                break
            else:
                orj_lags = [item for item in orj_lags if item not in best_lags]
                orj_lags.sort()
    return best_lags


def var_forward_lag_selection(df, model, max_lags, target_col, n_folds, H, metrics, step_size=None, verbose = False):
    """
    Performs forward lag selection for Vektor Autoregressive models and bidirectional ml models
    Args:
        df (pd.DataFrame): DataFrame containing the time series data.
        model: The forecasting model to be used.
        max_lags (dict): Dictionary of maximum lags for each variable.
        n_folds (int): Number of folds for cross-validation.
        target_col (str): The target column for forecasting.
        H (int): Forecast horizon.
        model_params (dict): Parameters for the model.
        metrics (list): List of metrics to evaluate the model.
        step_size (int, optional): Step size for lag selection. Defaults to None
        verbose (bool, optional): Whether to print progress. Defaults to False.
    Returns:
        dict: Dictionary of best lags for each variable.
    """

    max_lag = sum(x for x in max_lags.values())
        
    orj_lags = {i:list(range(1, max_lags[i]+1)) for i in max_lags}
    # lags = list(range(1, max_lags+1))
    lags = {i:list(range(1, max_lags[i]+1)) for i in max_lags}
    best_lags = {i:[] for i in max_lags}
    
    best_score = list(np.repeat(float('inf'), len(metrics)))
    
    while max_lag>0:
        for i in range(max_lag):
            best_lag = None
            best_target = None
            for k, lg in lags.items():
                for x in lg:
                    current_lag = {a:b for a, b in best_lags.items()}
                    current_lag[k] = best_lags[k] + [x]
                    current_lag[k].sort()
                    model.n_lag = current_lag
                    my_cv = bidirectional_cross_validate(model=model, df=df, cv_split=n_folds,
                                                         test_size=H, metrics=metrics, step_size=step_size)

                    scores = my_cv[target_col].tolist()
                    if scores<best_score:
                        best_score = scores
                        best_lag = x
                        best_target = k
            if best_lag is not None:
                best_lags[best_target].append(best_lag)
                lags[best_target].remove(best_lag)
                best_lags[best_target].sort()
                max_lag = sum(len(x) for x in lags.values())
                if verbose == True:
            # print worst variable, lags and current score
                    print(f'Variable of best lag after backward: {best_target} with lag {best_lag}, current score: {best_score}')
            else:
                break
        lags = {i:[item for item in orj_lags[i] if item not in best_lags[i]] for i in max_lags.keys()}
        # lags.sort()
        if lags == orj_lags:
            break
        else:
            orj_lags = {i:[item for item in orj_lags[i] if item not in best_lags[i]] for i in max_lags.keys()}
            # orj_lags.sort()
            max_lag = sum(len(x) for x in orj_lags.values())

    return best_lags

def var_backward_lag_selection(df, model, max_lags, min_lags, n_folds,target_col, H, metrics, step_size=None, forward_back=False, verbose = False):
    """
    Performs backward lag selection for Vektor Autoregressive models.
    Args:
        df (pd.DataFrame): DataFrame containing the time series data.
        model: The forecasting model to be used.
        max_lags (dict): Dictionary of maximum lags for each variable.
        min_lags (int): Minimum number of lags to retain.
        n_folds (int): Number of folds for cross-validation.
        target_col (str): The target column for forecasting.
        H (int): Forecast horizon.
        model_params (dict): Parameters for the model.
        metrics (list): List of metrics to evaluate the model.
        forward_back (bool, optional): Whether to perform forward-backward selection. Defaults to False.
        verbose (bool, optional): Whether to print progress. Defaults to False.
    Returns:
        dict: Dictionary of best lags for each variable.
    """

    max_lag = sum(x for x in max_lags.values()) 
    
    orj_lags = {i:list(range(1, max_lags[i]+1)) for i in max_lags}
    # lags = list(range(1, max_lags+1))
    lags = {i:list(range(1, max_lags[i]+1)) for i in max_lags}
    best_lags = {i:[] for i in max_lags}
    
    best_score = list(np.repeat(float('inf'), len(metrics)))
    
    worst_lags = {i:[] for i in max_lags} # to store wost lags
    while max_lag >= min_lags: 
        worst_lag=None
        worst_k = None
        for k, lg in lags.items(): # Iterate over each variable's lags
            for r in lg: # try and test each lag
                lags_to_test = {a:b for a, b in lags.items()}
                lags_to_test[k] = [x for x in lg if x != r] # Remove the current lag r
                lags_to_test[k].sort()
                model.n_lag = lags_to_test
                my_cv = bidirectional_cross_validate(model=model, df=df, cv_split=n_folds,
                                                     test_size=H, metrics=metrics, step_size=step_size)
                scores = my_cv[target_col].tolist()
                if scores < best_score:
                    best_score = scores
                    worst_lag = r # update the worst lag
                    worst_k = k # update the worst variable whose lag was worst_lag
                    best_lags = lags_to_test # update the best lags
        if worst_lag is not None: # If a worst lag was found
            # lags.append(best_lag)
            lags[worst_k].remove(worst_lag) # Remove the worst lag from the variable worst_k
            lags[worst_k].sort()
            # best_lags[worst_k].remove(worst_lag)
            worst_lags[worst_k].append(worst_lag)
            worst_lags[worst_k].sort()
            if verbose == True:
                # print worst variable, lags and current score
                print(f'Worst lag\'s variable: {worst_k} with lag {worst_lag}, current score: {best_score}')
            max_lag = sum(len(x) for x in lags.values())
                
        else:
            break
    
    if forward_back ==True:
    
        orj_lags = worst_lags.copy()
        # orj_lags.sort()
        len_worst = sum(len(x) for x in worst_lags.values())
        
        while len_worst>0:
            for i in range(len_worst):
                best_lag = None
                best_k = None
                for k, lg in worst_lags.items():
                    for x in lg:
                        current_lag = {a:b for a, b in best_lags.items()}
                        current_lag[k] = best_lags[k] + [x]
                        current_lag[k].sort()
                        model.n_lag = current_lag
                        my_cv = bidirectional_cross_validate(model=model, df=df, cv_split=n_folds,
                                                             test_size=H, metrics=metrics, step_size=step_size)
                        scores = my_cv[target_col].tolist()
                        if scores < best_score:
                            best_score = scores
                            best_lag = x # update the best lag
                            best_k = k # update the best variable whose lag was best_lag

                if best_lag is not None:
                    best_lags[best_k].append(best_lag)
                    worst_lags[best_k].remove(best_lag)

                    best_lags[best_k].sort()
                    len_worst = sum(len(x) for x in worst_lags.values()) #  update len worst
                    if verbose == True:
                # print worst variable, lags and current score
                        print(f'Variable of best lag after backward: {best_k} with lag {best_lag}, current score: {best_score}')
                else:
                    break
                    
            worst_lags = {i:[item for item in orj_lags[i] if item not in best_lags[i]] for i in max_lags}
            len_worst = sum(len(x) for x in worst_lags.values())
            
            if worst_lags == orj_lags: #check if no lags is added
                break
            else:
                orj_lags = {i:[item for item in orj_lags[i] if item not in best_lags[i]] for i in max_lags}
    return best_lags


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

# invert difference
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
    conc_data = list(orig_data[-seasonal_period:]) + list(diff_data)
    for i in range(len(conc_data)-seasonal_period):
        conc_data[i+seasonal_period] = conc_data[i]+conc_data[i+seasonal_period]

    return np.array(conc_data[-len(diff_data):])

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
# Holt-Winters Exponential Smoothing Model Tuning
#------------------------------------------------------------------------------

def tune_ets(data, param_space, cv_splits, horizon, eval_metric, eval_num, step_size = None, verbose = False):
    """
    Tune ETS model hyperparameters using Hyperopt.

    Args:
        data (array-like): Time series data.
        param_space (dict): Hyperparameter search space.
        cv_splits (int): Number of cross-validation splits.
        horizon (int): Forecast horizon.
        step_size (int): Step size for time series cross-validation.
        eval_metric (function): Evaluation metric function.
        eval_num (int): Number of evaluations for hyperparameter tuning.
        verbose (bool): Whether to print progress.
    Returns:
        tuple: Best model parameters and fit parameters.
    """

    from sklearn.model_selection import TimeSeriesSplit
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval
    from hyperopt.pyll import scope
    tscv = ParametricTimeSeriesSplit(n_splits=cv_splits, test_size=horizon, step_size=step_size)
    # Define the objective function for hyperparameter tuning
    def objective(params):
        if (params.get("trend") != None) & (params.get("seasonal") != None): # Both trend and seasonal are specified
            alpha = params.get('smoothing_level') # Smoothing level for the level component
            beta = params.get('smoothing_trend') # Smoothing level for the trend component
            gamma = params.get('smoothing_seasonal') # Smoothing level for the seasonal component
            trend_type = params.get('trend') # Trend type
            season_type = params.get('seasonal') # Seasonal type
            S = params.get('seasonal_periods') # Seasonal periods
            if params.get("damped_trend"): # Damped trend
                damped_bool = params.get("damped_trend")
                damp_trend = params.get('damping_trend')
            else:
                damped_bool = params.get("damped_trend")
                damp_trend = None

        elif (params.get("trend") != None) & (params.get("seasonal") == None): # Trend is specified, seasonal is not
            alpha = params.get('smoothing_level')
            beta = params.get('smoothing_trend')
            gamma = None
            trend_type = params.get('trend')
            season_type = params.get('seasonal')
            S=None
            if params.get("damped_trend"):
                damped_bool = params.get("damped_trend")
                damp_trend = params.get('damping_trend')
            else:
                damped_bool = params.get("damped_trend")
                damp_trend = None

        elif (params.get("trend") == None) & (params.get("seasonal") != None): # Seasonal is specified, trend is not
            alpha = params.get('smoothing_level')
            beta = None
            gamma = params.get('smoothing_seasonal')
            trend_type = params.get('trend')
            season_type = params.get('seasonal')
            S = params.get('seasonal_periods')
            if params.get("damped_trend"):
                damped_bool = False
                damp_trend = None
            else:
                damped_bool = params.get("damped_trend")
                damp_trend = None
                
        else: # Neither trend nor seasonal is specified
            alpha = params.get('smoothing_level')
            beta = None
            gamma = None
            trend_type = params.get('trend')
            season_type = params.get('seasonal')
            S = None
            if params.get("damped_trend"):
                damped_bool = False
                damp_trend = None
            else:
                damped_bool = params.get("damped_trend")
                damp_trend = None
            

        metric = [] # List to store evaluation metrics
        # Perform time series cross-validation
        for train_index, test_index in tscv.split(data):
            train, test = data[train_index], data[test_index]
            # Fit the Holt-Winters model with the specified parameters
            hw_fit = ExponentialSmoothing(train ,seasonal_periods=S , seasonal=season_type, trend=trend_type, damped_trend = damped_bool).fit(smoothing_level = alpha, 
                                                                                                                      smoothing_trend = beta,
                                                                                                                      smoothing_seasonal = gamma,
                                                                                                damping_trend=damp_trend)
            
            hw_forecast = hw_fit.forecast(len(test))
            forecast_filled = np.nan_to_num(hw_forecast, nan=0)
            accuracy = eval_metric(test, forecast_filled)
            metric.append(accuracy)
            
        score = np.mean(metric)
        if verbose ==True:
            print ("SCORE:", score)
        return {'loss':score, 'status':STATUS_OK}
    
    # Perform hyperparameter optimization using Hyperopt
    trials = Trials()
    
    best_hyperparams = fmin(fn = objective,
                    space = param_space,
                    algo = tpe.suggest,
                    max_evals = eval_num,
                    trials = trials)
    best_params = space_eval(param_space, best_hyperparams)
    model_params = {
        "trend": best_params.get("trend"),
        "seasonal_periods": best_params.get("seasonal_periods"),
        "seasonal": best_params.get("seasonal"),
        "damped_trend": best_params.get("damped_trend")
    }
    fit_params = {
        "smoothing_level": best_params.get("smoothing_level"),
        "smoothing_trend": best_params.get("smoothing_trend"),
        "smoothing_seasonal": best_params.get("smoothing_seasonal"),
        "damping_trend": best_params.get("damping_trend")
    }

    # Remove all keys with value None in a single step
    model_params = {k: v for k, v in model_params.items() if v is not None}
    fit_params = {k: v for k, v in fit_params.items() if v is not None}
    fit_params = {k: v for k, v in fit_params.items()
                  if not (k == "damping_trend" and model_params.get("damped_trend") is False)}
    return model_params, fit_params

#------------------------------------------------------------------------------
# SARIMA Model Tuning
#------------------------------------------------------------------------------

def tune_sarima(y, d, D, season,p_range, q_range, P_range, Q_range, X=None):
    """
    Finds the best SARIMA parameters using AIC as the evaluation metric.
    Args:
        y (array-like): The time series data.
        d (int): The non-seasonal differencing order.
        D (int): The seasonal differencing order.
        season (int): The seasonal period.
        p_range (list): Range of non-seasonal AR orders to test.
        q_range (list): Range of non-seasonal MA orders to test.
        P_range (list): Range of seasonal AR orders to test.
        Q_range (list): Range of seasonal MA orders to test.
        X (array-like, optional): Exogenous variables. Defaults to None.
    Returns:
        pd.DataFrame: A DataFrame containing the combinations of parameters and their corresponding AIC values.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from tqdm import tqdm_notebook
    from itertools import product
    if X is not None:
        X = np.array(X, dtype = np.float64)
    p = p_range
    q = q_range # MA(q)
    P = P_range # Seasonal autoregressive order.
    Q = Q_range #Seasonal moving average order.
    parameters = product(p, q, P, Q) # combinations of parameters(cartesian product)
    parameters_list = list(parameters)
    result = []
    for param in tqdm_notebook(parameters_list):
        try:
            model = SARIMAX(endog=y, exog = X, order = (param[0], d, param[1]), seasonal_order= (param[2], D, param[3], season)).fit(disp = -1)
        except:
            continue
                            
        aic = model.aic
        result.append([param, aic])
    result_df = pd.DataFrame(result)
    result_df.columns = ["(p, q)x(P, Q)", "AIC"] 
    result_df = result_df.sort_values("AIC", ascending = True) #Sort in ascending order, lower AIC is better
    return result_df

#------------------------------------------------------------------------------
# ML tuning utility function
#------------------------------------------------------------------------------

def cross_validate(model, df, cv_split, test_size, metrics, step_size=None):
    """
    Run cross-validation using time series splits.

    Args:
        model (class): Machine learning model class (e.g., CatBoostRegressor, LGBMRegressor).
        df (pd.DataFrame): Input data.
        cv_split (int): Number of splits in TimeSeriesSplit.
        test_size (int): Size of test window.
        metrics (list): List of metric functions.
    
    Returns:
        pd.DataFrame: Performance metrics for CV.
    """
    tscv = ParametricTimeSeriesSplit(n_splits=cv_split, test_size=test_size, step_size=step_size)
    metrics_dict = {m.__name__: [] for m in metrics}
    for train_index, test_index in tscv.split(df):
        train, test = df.iloc[train_index], df.iloc[test_index]
        x_test = test.drop(columns=[model.target_col])
        y_test = np.array(test[model.target_col])
        model.fit(train)
        bb_forecast = model.forecast(test_size, x_test=x_test)
        # Evaluate each metric
        for m in metrics:
            if m.__name__ == "MASE":
                eval_val = m(y_test, bb_forecast, train[model.target_col])
            else:
                eval_val = m(y_test, bb_forecast)
            metrics_dict[m.__name__].append(eval_val)
    overall_performance = [[m.__name__, np.mean(metrics_dict[m.__name__])] for m in metrics]
    return pd.DataFrame(overall_performance).rename(columns={0: "eval_metric", 1: "score"})


def bidirectional_cross_validate(model, df, cv_split, test_size, metrics, step_size=None):
    """
    Cross-validate the bidirectional CatBoost model with time series split.
    Args:
        df (pd.DataFrame): Input dataframe.
        cv_split (int): Number of folds.
        test_size (int): Forecast window for each split.
        metrics (list): List of evaluation metric functions.
    Returns:
        pd.DataFrame: CV performance metrics for each target variable.
    """
    tscv = ParametricTimeSeriesSplit(n_splits=cv_split, test_size=test_size, step_size=step_size)
    metrics_dict = {m.__name__: [] for m in metrics}
    cv_fi = pd.DataFrame()
    cv_forecasts_df = pd.DataFrame()
    for i, (train_index, test_index) in enumerate(tscv.split(df)):
        train, test = df.iloc[train_index], df.iloc[test_index]
        x_test = test.drop(columns=model.target_cols)
        y_test1 = np.array(test[model.target_cols[0]])
        y_test2 = np.array(test[model.target_cols[1]])
        
        model.fit(train)

        forecast_vals1, forecast_vals2 = model.forecast(test_size, x_test=x_test)
        forecat_df = test[model.target_cols]
        forecat_df["forecasts1"] = forecast_vals1
        forecat_df["forecasts2"] = forecast_vals2
        cv_forecasts_df = pd.concat([cv_forecasts_df, forecat_df], axis=0)
        for m in metrics:
            if m.__name__ == "MASE":
                val1 = m(y_test1, forecast_vals1, train[model.target_cols[0]])
                val2 = m(y_test2, forecast_vals2, train[model.target_cols[1]])
            else:
                val1 = m(y_test1, forecast_vals1)
                val2 = m(y_test2, forecast_vals2)

            metrics_dict[m.__name__].append([val1, val2])

        cv_tr_df1 = pd.DataFrame({"feat_name": model.model1_fit.feature_names_in_,
                                "importance": model.model1_fit.feature_importances_}).sort_values(by="importance", ascending=False)
        cv_tr_df1["target"] = model.target_cols[0]
        cv_tr_df1["fold"] = i
        cv_tr_df2 = pd.DataFrame({"feat_name": model.model2_fit.feature_names_in_,
                                "importance": model.model2_fit.feature_importances_}).sort_values(by="importance", ascending=False)
        cv_tr_df2["target"] = model.target_cols[1]
        cv_tr_df2["fold"] = i
        cv_fi = pd.concat([cv_fi, cv_tr_df1, cv_tr_df2], axis=0)
    overall = [[m.__name__, np.mean([v[0] for v in metrics_dict[m.__name__]])] for m in metrics]
    # pd.DataFrame(overall).rename(columns={0: "eval_metric", 1: "score1", 2: "score2"})
    return pd.DataFrame(overall).rename(columns={0: "eval_metric", 1: model.target_cols[0], 2: model.target_cols[1]})

def cv_tune(
    model,
    df,
    cv_split,
    test_size,
    param_space,
    eval_metric,
    step_size=None,
    opt_horizon=None,
    eval_num=100,
    verbose=False,
):
    """
    Tune forecasting model hyperparameters using cross-validation and Bayesian optimization.

    Parameters
    ----------
    model : object
        Forecasting model object with .fit and .forecast methods and relevant attributes.
    df : pd.DataFrame
        Time series dataframe.
    cv_split : int
        Number of time series splits.
    test_size : int
        Size of test window for each split.
    param_space : dict
        Hyperopt parameter search space.
    eval_metric : callable
        Evaluation metric function.
    step_size : int, optional
        Step size for moving the window. Defaults to None (equal to test_size).
    opt_horizon : int, optional
        Evaluate only on last N points of each split. Defaults to None (all points).
    eval_num : int, optional
        Number of hyperopt evaluations. Defaults to 100.
    verbose : bool, optional
        Print progress. Defaults to False.

    Returns
    -------
    dict
        Best hyperparameter values found.
    """
    tscv = ParametricTimeSeriesSplit(n_splits=cv_split, test_size=test_size, step_size=step_size)

    def _set_model_params(params):
        # Handle special model parameters that are not passed to model constructor
        # and must be set directly on the forecasting model object
        if "n_lag" in params:
            if isinstance(params["n_lag"], int):
                model.n_lag = list(range(1, params["n_lag"] + 1))
            elif isinstance(params["n_lag"], list):
                model.n_lag = params["n_lag"]

        if "difference" in params:
            model.difference = params["difference"]
        if "box_cox" in params:
            model.box_cox = params["box_cox"]
        if "box_cox_lmda" in params:
            model.lamda = params["box_cox_lmda"]
        if "box_cox_biasadj" in params:
            model.biasadj = params["box_cox_biasadj"]

        # Handle ETS trend settings
        # if model.trend:
        #     if model.trend_type in ["ses", "feature_ses"]:
        #         model.ets_model = {
        #             k: params[k] for k in ["trend", "damped_trend", "seasonal", "seasonal_periods"] if k in params
        #         }
        #         model.ets_fit = {}
        #         for k in ["smoothing_level", "smoothing_trend", "smoothing_seasonal", "damping_trend"]:
        #             if k in params:
        #                 # Only set "damping_trend" if "damped_trend" is True
        #                 if (k == "damping_trend") and ("damped_trend" in params and not params["damped_trend"]):
        #                     continue
        #                 else:
        #                     model.ets_fit[k] = params[k]

    def _get_model_params_for_fit(params):
        # Exclude special parameters that should not be passed to the model constructor
        skip_keys = {
            "box_cox", "n_lag", "box_cox_lmda", "box_cox_biasadj",
            "trend", "damped_trend", "seasonal", "seasonal_periods",
            "smoothing_level", "smoothing_trend", "smoothing_seasonal", "damping_trend",
            "differencing_number"
        }
        return {k: v for k, v in params.items() if k not in skip_keys}

    def objective(params):
        _set_model_params(params)
        if isinstance(model.model, LinearRegression):
            # For LinearRegression, we don't need to set model_params
            model_params = None
        else:
            # For other models, get the parameters to set
            model_params = _get_model_params_for_fit(params)

        metrics = []
        for train_index, test_index in tscv.split(df):
            train, test = df.iloc[train_index], df.iloc[test_index]
            x_test = test.drop(columns=[model.target_col])
            y_test = np.array(test[model.target_col])

            if model_params is not None:
                model.model.set_params(**model_params)
            model.fit(train)
            
            y_pred = model.forecast(n_ahead=len(y_test), x_test=x_test)

            #Evaluate using the specified metric
            if eval_metric.__name__ == "MASE":
                score = eval_metric(y_test[-opt_horizon:] if opt_horizon else y_test,
                                    y_pred[-opt_horizon:] if opt_horizon else y_pred,
                                    train[model.target_col])
            else:
                score = eval_metric(
                        y_test[-opt_horizon:] if opt_horizon else y_test,
                        y_pred[-opt_horizon:] if opt_horizon else y_pred,
                    )
            metrics.append(score)

        mean_score = np.mean(metrics)
        if verbose:
            print("Score:", mean_score)
        return {"loss": mean_score, "status": STATUS_OK}

    trials = Trials()
    best_hyperparams = fmin(
        fn=objective,
        space=param_space,
        algo=tpe.suggest,
        max_evals=eval_num,
        trials=trials,
    )

    model.tuned_params = [
        space_eval(param_space, {k: v[0] for k, v in t["misc"]["vals"].items()})
        for t in trials.trials
    ]

    return space_eval(param_space, best_hyperparams)

#------------------------------------------------------------------------------
# Regression Detrending and Forecasting
#------------------------------------------------------------------------------

def regression_detrend(series):
    """
    Detrends a time series using linear regression.
    Args:
        series (array-like): The time series data to be detrended.
    Returns:
        np.ndarray: The detrended time series.
    """
    model = LinearRegression().fit(np.array(range(len(series))).reshape(-1, 1),series)
# Make predictions
    y_pred = model.predict(np.array(range(len(series))).reshape(-1, 1))
    return np.array(series)-y_pred

def forecast_trend(train_series, H):
    """ Forecasts the trend of a time series using linear regression.
    Args:
        train_series (array-like): The training time series data.
        H (int): The forecast horizon.
    Returns:
        np.ndarray: The forecasted trend values.
    """
    model = LinearRegression().fit(np.array(range(len(train_series))).reshape(-1, 1),train_series)
    return model.predict(np.array(range(len(train_series), len(train_series)+H)).reshape(-1, 1))


#------------------------------------------------------------------------------
# Parametric Time Series Split
#------------------------------------------------------------------------------
class ParametricTimeSeriesSplit:
    """
    Rolling window time series cross-validator with fixed step size

    Parameters:
        test_size (int): Number of test samples in each split.
        step_size (int): Number of steps to move backward for each split.
        n_splits (int, optional): Number of splits to generate.

    Yields:
        train_index, test_index: Indices for training and test sets.
    """
    def __init__(self, n_splits, test_size, step_size=None):
        self.test_size = test_size
        self.step_size = test_size if step_size is None else step_size
        self.n_splits = n_splits

    def split(self, X):
        n_samples = len(X)
        split_starts = []
        # Start the last test set at the last possible position
        last_test_start = n_samples - self.test_size
        # Build test starts, moving backward with fixed step_size
        current = last_test_start
        while current >= 0:
            split_starts.append(current)
            current -= self.step_size
        split_starts = split_starts[::-1]  # Reverse to start from earliest

        # Use only the last n_splits
        split_starts = split_starts[-self.n_splits:]

        for test_start in split_starts:
            test_end = test_start + self.test_size
            train_index = np.arange(0, test_start)
            test_index = np.arange(test_start, test_end)
            yield train_index, test_index


def cv_tune_bidirectional(
    model,
    df,
    forecast_col,
    cv_split,
    test_size,
    param_space,
    eval_metric,
    step_size=None,
    opt_horizon=None,
    eval_num=100,
    verbose=False,
):
    """
    Tune forecasting model hyperparameters using cross-validation and Bayesian optimization.

    Args:
        model: Forecasting model object with .fit and .forecast methods and relevant attributes.
        df (pd.DataFrame): Time series dataframe.
        forecast_col (str): Name of the target variable to test.
        cv_split (int): Number of time series splits.
        test_size (int): Size of test window for each split.
        step_size (int): Step size for moving the window. Defaults to None (equal to test_size).
        param_space (dict): Hyperopt parameter search space. params for lags, differencing, etc. can be {'n_lag': (hp.choice('lag_y1', [1,2,3]), hp.choice('lag_y2', [1,2]))}
        eval_metric (callable): Evaluation metric function.
        opt_horizon (int, optional): Evaluate only on last N points of each split. Defaults to None (all points).
        eval_num (int, optional): Number of hyperopt evaluations. Defaults to 100.
        verbose (bool, optional): Print progress. Defaults to False.

    Returns:
        dict: Best hyperparameter values found.
    """

    target_cols = model.target_cols
    tscv = ParametricTimeSeriesSplit(n_splits=cv_split, test_size=test_size, step_size=step_size)

# example: lgb_param_space_bi={'learning_rate': hp.quniform('learning_rate', 0.001, 0.8, 0.0001),
#             'num_leaves': scope.int(hp.quniform('num_leaves', 10, 200, 1)),
#            'max_depth':scope.int(hp.quniform('max_depth', 5, 100, 1)),

#     'n_lag': {
#         'attend': hp.choice('attend_lag', [2,3,[2,3]]),
#         'verified': hp.choice('verified_lag', [7,4,[3,4]])
#     },
#     'trend': {
#         'attend': hp.choice('attend', [None, "add", "mul"]),
#         'verified': hp.choice('verified', [None, "add", "mul"])
#     },
#     'smoothing_level': {
#         'attend': hp.uniform('attend', 0, 0.99),
#         'verified': hp.uniform('verified', 0, 0.99)
#     },
#     'smoothing_trend': {
#         'attend': hp.uniform('attend', 0, 0.99),
#         'verified': hp.uniform('verified', 0, 0.99)
#     }
        
#         }

    def _set_model_params(params):
        # Handle special model parameters that are not passed to model constructor
        # and must be set directly on the forecasting model object
        if "n_lag" in params:
            if isinstance(params["n_lag"], dict):
                if model.n_lag is None:
                    model.n_lag = {}
                # If n_lag is a dict, set it for each target column
                for target_col, lags in params["n_lag"].items():
                    if isinstance(lags, int):
                        model.n_lag[target_col] = list(range(1, lags + 1))
                    elif isinstance(lags, list):
                        model.n_lag[target_col] = lags

        if "lag_transform" in params:
            if isinstance(params["lag_transform"], dict):
                if model.lag_transform is None:
                    model.lag_transform = {}
                # If lag_transform is a dict, set it for each target column
                for target_col, lag_transform in params["lag_transform"].items():
                    model.lag_transform[target_col] = lag_transform

        if "difference" in params:
            if isinstance(params["difference"], dict):
                # If difference is a dict, set it for each target column
                for target_col, diff in params["difference"].items():
                    model.difference[target_col] = diff
        
        if "seasonal_length" in params:
            if isinstance(params["seasonal_length"], dict):
                # If seasonal_length is a dict, set it for each target column
                for target_col, seasonal_length in params["seasonal_length"].items():
                    model.season_diff[target_col] = seasonal_length

        if "box_cox" in params:
            if isinstance(params["box_cox"], dict):
                for target_col, box_cox in params["box_cox"].items():
                    model.box_cox[target_col] = box_cox
        if "box_cox_lmda" in params:
            if isinstance(params["box_cox_lmda"], dict):
                for target_col, lamda in params["box_cox_lmda"].items():
                    model.lamda[target_col] = lamda
        if "box_cox_biasadj" in params:
            if isinstance(params["box_cox_biasadj"], dict):
                for target_col, biasadj in params["box_cox_biasadj"].items():
                    model.biasadj[target_col] = biasadj
        # Handle ETS trend
        # in the model: ets_params (dict, optional): Dictionary of ETS model parameters (values are tuples of dictionaries of params) and fit settings for each target variable. Example: {'Target1': [{'trend': 'add', 'seasonal': 'add'}, {'damped_trend': True}], 'Target2': [{'trend': 'mul', 'seasonal': 'mul'}, {'damped_trend': False}]}.
        # if model.trend is not None:
        #     model.ets_params = {}
        #     for target_col in model.trend.keys(): # Iterate over each target column that has a trend (It can be 1 or 2 target columns)
        #         if (model.trend[target_col]) and (model.trend_type[target_col] in ["ses", "feature_ses"]):
        #             model.ets_params[target_col] = [{k: params[k][target_col] for k in ["trend", "damped_trend", "seasonal", "seasonal_periods"] if k in params}] # Set trend and seasonal parameters for each target column
        #             model.ets_fit = {}
        #             for k in ["smoothing_level", "smoothing_trend", "smoothing_seasonal", "damping_trend"]:
        #                 if k in params:
        #                     # Only set "damping_trend" if "damped_trend" is True
        #                     if (k == "damping_trend") and ("damped_trend" in params and not params["damped_trend"]):
        #                         continue
        #                     else:
        #                         model.ets_fit[k] = params[k][target_col]
        #             # append model.ets_fit to model.ets_params[target_col]
        #             model.ets_params[target_col].append(model.ets_fit)

    def _get_model_params_for_fit(params):
        # Exclude special parameters that should not be passed to the model constructor
        skip_keys = {
            "box_cox", "n_lag", "lag_transform", "box_cox_lmda", "box_cox_biasadj",
            "trend", "damped_trend", "seasonal", "seasonal_periods", "seasonal_length",
            "smoothing_level", "smoothing_trend", "smoothing_seasonal", "damping_trend",
            "difference"
        }
        return {k: v for k, v in params.items() if k not in skip_keys}

    def objective(params):
        _set_model_params(params)
        if isinstance(model.model, LinearRegression):
            # For LinearRegression, we don't need to set model_params
            model_params = None
        else:
            # For other models, get the parameters to set
            model_params = _get_model_params_for_fit(params)
        

        metrics = []
        for train_index, test_index in tscv.split(df):
            train, test = df.iloc[train_index], df.iloc[test_index]
            x_test = test.drop(columns=model.target_cols)
            y_test = np.array(test[forecast_col])

            if model_params is not None:
                model.model.set_params(**model_params)
            model.fit(train)

            y_pred = model.forecast(n_ahead=len(y_test), x_test=x_test)[forecast_col]

            #Evaluate using the specified metric
            if eval_metric.__name__ == "MASE":
                score = eval_metric(y_test[-opt_horizon:] if opt_horizon else y_test,
                                    y_pred[-opt_horizon:] if opt_horizon else y_pred,
                                    train[model.target_col])
            else:
                score = eval_metric(
                        y_test[-opt_horizon:] if opt_horizon else y_test,
                        y_pred[-opt_horizon:] if opt_horizon else y_pred,
                    )
            metrics.append(score)

        mean_score = np.mean(metrics)
        if verbose:
            print("Score:", mean_score)
        return {"loss": mean_score, "status": STATUS_OK}

    trials = Trials()
    best_hyperparams = fmin(
        fn=objective,
        space=param_space,
        algo=tpe.suggest,
        max_evals=eval_num,
        trials=trials,
    )

    model.tuned_params = [
        space_eval(param_space, {k: v[0] for k, v in t["misc"]["vals"].items()})
        for t in trials.trials
    ]

    return space_eval(param_space, best_hyperparams)



def prob_param_forecasts(model, H, train_df, test_df=None):
    prob_forecasts = []
    for params in model.tuned_params:
        if ('n_lag' in params) |('differencing_number' in params)|('box_cox' in params)|('box_cox_lmda' in params)|('box_cox_biasadj' in params):
            if ('n_lag' in params):
                if type(params["n_lag"]) is tuple:
                    model.n_lag = list(params["n_lag"])
                else:
                    model.n_lag = range(1, params["n_lag"]+1)

            if ('differencing_number' in params):
                model.difference = params["differencing_number"]
            if ('box_cox' in params):
                model.box_cox = params["box_cox"]
            if ('box_cox_lmda' in params):
                model.lmda = params["box_cox_lmda"]

            if ('box_cox_biasadj' in params):
                model.biasadj = params["box_cox_biasadj"]

        
        if model.model.__name__ != 'LinearRegression':
            model_params = {k: v for k, v in params.items() if (k not in ["box_cox", "n_lag", "box_cox_lmda", "box_cox_biasadj"])}
            model.fit(train_df, model_params)
        else:
            model.fit(train_df)
        if test_df is not None:
            forecasts = model.forecast(H, test_df)
        else:
            forecasts = model.forecast(H)

        prob_forecasts.append(forecasts)
    prob_forecasts = np.row_stack(prob_forecasts)
    prob_forecasts=pd.DataFrame(prob_forecasts)
    prob_forecasts.columns = ["horizon_"+str(i+1) for i in prob_forecasts.columns]
    return prob_forecasts