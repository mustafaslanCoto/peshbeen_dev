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

from matplotlib.ticker import MaxNLocator
from statsmodels.tsa.stattools import ccf


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


# def cross_autocorrelation_plot(x, y, nlags, adjusted=True, alpha=0.05, figsize=(8, 5), title="Cross-Autocorrelation"):
#     """
#     Plot the cross-autocorrelation between two time series.

#     Parameters
#     ----------
#     x : array_like
#         First time series.
#     y : array_like
#         Second time series.
#     nlags : int
#         Number of lags to compute.
#     adjusted : bool, optional
#         Whether to apply the adjustment factor (default is True).
#     alpha : float, optional
#         Significance level for confidence intervals (default is 0.05).
#     figsize : tuple, optional
#         Figure size for the plot (default is (8, 5)).
#     title : str, optional
#         Title for the plot (default is "Cross-Autocorrelation").

#     Returns
#     -------
#     cc : ndarray
#         Cross-autocorrelation plot.
#     """
#     x = np.asarray(x)
#     y = np.asarray(y)
#     x_mean = np.mean(x)
#     y_mean = np.mean(y)
#     n = len(x)
#     if len(y) != n:
#         raise ValueError("x and y must have the same length")
    
#     # Variance (autocovariance at lag 0)
#     var_x = np.sum((x - x_mean)**2)
#     var_y = np.sum((y - y_mean)**2)

#     # Autocovariance but make sure make adjusted, meaning applying 1/(n-k) or 1/n
#     cc = np.empty(nlags)
#     for k in range(nlags):
#         num = np.sum((y[:n-k] - y_mean) * (x[k:] - x_mean))
#         r = num / np.sqrt(var_x * var_y)
#         if adjusted and k > 0:
#             r *= n / (n - k)
#         cc[k] = r

#     # Confidence intervals (optional)
#     z = NormalDist().inv_cdf(1 - alpha/2)
#     if adjusted:
#         den = n-nlags
#     else:
#         den = n
#     bound = z / np.sqrt(den)
#     # Bar plot of cross-correlation
#     plt.figure(figsize=figsize)
#     plt.bar(np.arange(nlags), cc, edgecolor='k', label='Cross-correlation')

#     z = NormalDist().inv_cdf(1 - alpha/2)
#     bound = z / np.sqrt(n - nlags) if adjusted else z / np.sqrt(n)
#     lag_x = np.arange(nlags)
#     plt.fill_between(lag_x, -bound, bound, color='gray', alpha=0.15, label='Confidence Interval')
#     plt.axhline(bound, color='red', linewidth=0.8, alpha=0.7, linestyle='--')
#     plt.axhline(-bound, color='red', linewidth=0.8, alpha=0.7, linestyle='--')
#     plt.xlabel("Lags of second time series")
#     plt.grid(axis='y')
#     plt.ylabel("Cross-correlation")
#     plt.title(title)
#     plt.legend()
#     plt.show()



def plot_ccf(x, y, lags, alpha=0.05, figsize=[10, 5], adjusted=False):
    """
    Plot the cross-correlation function (CCF) between two time series.
    Args:
        x (array-like): First time series.
        y (array-like): Second time series.
        lags (int): Number of lags to include in the plot.
        alpha (float): Significance level for confidence intervals.
        figsize (tuple): Size of the figure.
        adjusted (bool): Whether to use adjusted confidence intervals (if true, divide z by sqrt(n - k) for k > 0)
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object with the plot.
    """
    # Compute CCF and confidence interval
    z = NormalDist().inv_cdf(1 - alpha/2)
    cross_corrs = ccf(x, y)
    # ci = 2 / np.sqrt(len(y))
    ci = z / np.sqrt(len(y) - lags) if adjusted else z / np.sqrt(len(y))
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.stem(range(0, lags + 1), cross_corrs[: lags + 1])
    ax.fill_between(range(0, lags + 1), ci, y2=-ci, alpha=0.2)
    ax.set_title("Cross-correlation")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    return ax