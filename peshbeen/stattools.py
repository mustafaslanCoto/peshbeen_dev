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
from statsmodels.tsa.stattools import pacf

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

## Cross Corelation Check

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
    cc = np.empty(nlags)
    for k in range(nlags):
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
            se = np.array([1.0 / np.sqrt(n) for _ in range(nlags)])

        confint = np.column_stack((cc - z*se, cc + z*se))
        return cc, confint
    else:
        return cc, None 
    


def pacf_exceedance(series, alpha=0.05, n_lags=5, adjusted=True):
    """
    Calculate the exceedance scores for the partial autocorrelation function (PACF) of a time series to identify powerful significant lags.
    It is calculated as (PACF value - bound) / bound for positive exceedances and (PACF value + bound) / bound for negative exceedances.

    Parameters:
    - series: The input time series data.
    - alpha: Significance level for the confidence intervals.
    - n_lags: Number of lags to consider for the PACF.
    - adjusted: Whether to use an adjusted bound for the PACF.

    Returns:
    - A DataFrame containing the exceedance scores for each lag. Also includes the absolute scores
    """
    pacf_vals = pacf(series, nlags=n_lags)[1:]
    n= len(series)   
    z = NormalDist().inv_cdf(1 - alpha/2)
    bound = z / np.sqrt(n - n_lags) if adjusted else z / np.sqrt(n)
    exceed_score = {}
    for i, j in enumerate(pacf_vals):
        if j > bound:
            exceed_score[i+1] = (j-bound)/bound
        elif j < -bound:
            exceed_score[i+1] = (j+bound)/bound
        else:
            exceed_score[i+1] = 0
    exceed_score = pd.DataFrame(exceed_score, index=[0]).T
    exceed_score.columns = ["exceedance_score"]
    exceed_score["abs_score"] = exceed_score["exceedance_score"].abs()
    exceed_score = exceed_score.sort_values(by="abs_score", ascending=False)
    exceed_score = exceed_score[exceed_score["abs_score"] > 0]
    exceed_score.index.name = "lag"
    return exceed_score
