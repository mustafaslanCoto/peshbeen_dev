import numpy as np
import pandas as pd
from statsforecast.models import ARIMA, AutoARIMA, TBATS, AutoTBATS
from peshbeen.model_selection import ParametricTimeSeriesSplit
from scipy.stats import gaussian_kde
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.api import VAR
import copy
rng_kde = np.random.default_rng(seed=42)

# --------------------------------------------------------------------- 
# Helper functions and classes for probabilistic forecasting with conformal prediction
# ---------------------------------------------------------------------

def get_conformal_quantiles(non_conform, n_calib, quantiles, y_forecast):
    """
    Generate conformal quantiles for future time steps.
    Args:
        non_conform: non-conformity scores from the conformal model
        n_calib: number of calibration samples
        quantiles (float or list): Quantiles to be calculated (e.g., 0.1, 0.5, 0.9).
        y_forecast: point forecasts
    Returns:
        pd.DataFrame: DataFrame containing point forecasts and conformal quantiles.
    """
    if isinstance(quantiles, float):
        if quantiles <0.5:
            q= (1-2*quantiles)
            q_which =np.ceil(q * (n_calib + 1)) / n_calib
            quantile_ = np.quantile(non_conform, q_which, method="higher", axis=1)
            y_quantile = y_forecast - quantile_
        elif quantiles == 0.5:
            y_quantile = y_forecast
        else:
            q= (2*quantiles-1)
            q_which = np.ceil(q * (n_calib + 1)) / n_calib
            quantile_ = np.quantile(non_conform, q_which, method="higher", axis=1)
            y_quantile = y_forecast + quantile_
        y_quantile = y_quantile[None, :]
    elif isinstance(quantiles, list):
        y_quantile = []
        for quantile in quantiles:
            if quantile < 0.5:
                q= (1-2*quantile)
                q_which = np.ceil(q * (n_calib + 1)) / n_calib # conformal quantile
                quantile_ = np.quantile(non_conform, q_which, method="lower", axis=1)
                y_q = y_forecast - quantile_
            elif quantile == 0.5:
                y_q = y_forecast
            else:
                q= (2*quantile-1)
                q_which = np.ceil(q * (n_calib + 1)) / n_calib # conformal quantile
                quantile_ = np.quantile(non_conform, q_which, method="lower", axis=1)
                y_q = y_forecast + quantile_
            y_quantile.append(y_q)
        y_quantile = np.array(y_quantile)
    
    if isinstance(quantiles, float):
        q_columns = ["point_forecast"]+[f'q_{int(quantiles*100)}']
    elif isinstance(quantiles, list):
        q_columns = ["point_forecast"]+[f'q_{int(quantile*100)}' for quantile in quantiles]
    else:
        raise ValueError("quantiles must be float or list of floats.")
    
    return pd.DataFrame(np.concatenate((y_forecast[None, :], y_quantile), axis=0).T, columns=q_columns)

def get_bootstrap_quantiles(samples, n_calib, quantiles, y_forecast):
    """
    Generate bootstrap quantiles for future time steps.
    Args:
        samples: bootstrap samples from the predictive distribution
        n_calib: number of calibration samples
        quantiles (float or list): Quantiles to be calculated (e.g., 0.1, 0.5, 0.9).
        y_forecast: point forecasts
    Returns:
        pd.DataFrame: DataFrame containing point forecasts and conformal quantiles.
    """
    if isinstance(quantiles, float):
        q_which =np.ceil(quantiles * (n_calib + 1)) / n_calib 
        quantile_ = np.quantile(samples, q_which, method="higher", axis=1)
        y_quantile = quantile_[None, :]
    elif isinstance(quantiles, list):
        y_quantile = []
        for quantile in quantiles:
            q_which = np.ceil(quantile * (n_calib + 1)) / n_calib # conformal quantile
            quantile_ = np.quantile(samples, q_which, method="lower", axis=1)
            y_quantile.append(quantile_)
        y_quantile = np.array(y_quantile)
    
    if isinstance(quantiles, float):
        q_columns = ["point_forecast"]+[f'q_{int(quantiles*100)}']
    elif isinstance(quantiles, list):
        q_columns = ["point_forecast"]+[f'q_{int(quantile*100)}' for quantile in quantiles]
    else:
        raise ValueError("quantiles must be float or list of floats.")
    
    return pd.DataFrame(np.concatenate((y_forecast[None, :], y_quantile), axis=0).T, columns=q_columns)

class ml_prob_forecasts():
    """
    Probabilistic forecasting for ML models. It generates prediction intervals for future time steps and approximates distribution of predictions using Kernel Density Estimation (KDE).
    Parameters:
    - model: forecasting model to be used
    - n_calibration: number of calibration windows
    - H: forecast horizon
    - sliding_window: size of the sliding window for cross-validation
    - verbose: whether to print progress messages
    """
    def __init__(self, model, n_calibration, H, sliding_window=1, verbose=False):
        self.model = model
        self.sliding_window = sliding_window
        self.n_calib = n_calibration
        self.verbose = verbose
        self.H = H

    def non_conformity_scores(self, df):
        c_actuals, c_forecasts = [], []
        # Create time series cross-validator that slides 1 time step for each training window
        tscv = ParametricTimeSeriesSplit(n_splits=self.n_calib, test_size=self.H, step_size=self.sliding_window)
        for train_index, test_index in tscv.split(df):
            train, test = df.iloc[train_index], df.iloc[test_index]
            x_test = test.drop(columns=[self.model.target_col])
            y_test = np.array(test[self.model.target_col])
            self.model.fit(train)
            H_forecasts = self.model.forecast(self.H, x_test)
            c_forecasts.append(H_forecasts)
            c_actuals.append(y_test)
            if self.verbose:
                print(f"Completed calibration window {len(c_forecasts)} out of {self.n_calib}")
        self.resid = np.column_stack(c_actuals) - np.column_stack(c_forecasts) # Residuals n_calib*H
        self.non_conform = np.abs(self.resid) # non-conformity scores
        self.c_actuals = np.column_stack(c_actuals)
        self.c_forecasts = np.column_stack(c_forecasts)

        
    def calculate_quantile(self, scores_calib):
        # Vectorized quantile calculation for list delta
        if isinstance(self.delta, float):
            which_quantile = np.ceil(self.delta * (self.n_calib + 1)) / self.n_calib
            return np.quantile(scores_calib, which_quantile, method="lower", axis=0)
        elif isinstance(self.delta, list):
            which_quantiles = np.ceil(np.array(self.delta) * (self.n_calib + 1)) / self.n_calib
            return np.array([np.quantile(scores_calib, q, method="lower", axis=0) for q in which_quantiles])
        else:
            raise ValueError("delta must be float or list of floats.")
    
    
    def calibrate(self, df, delta = 0.5):
        """
        Calibrate the conformal model using the calibration dataset.
        Args:
            df (pd.DataFrame): DataFrame containing the calibration data.
            delta (float or list): Significance level(s) for the prediction intervals.
        """
        self.delta = delta
        self.non_conformity_scores(df=df)
        h_quantiles = []
        for i in range(self.H):
            q_hat = self.calculate_quantile(self.non_conform[i])
            h_quantiles.append(q_hat)
        self.q_hat_D = np.array(h_quantiles)

    # Generate prediction intervals using the calibrated quantiles

    def generate_prediction_intervals(self, df, future_exog=None):
        '''
        Generate conformal prediction intervals for the forecasted values.
        Args:
            df (pd.DataFrame): DataFrame containing the training data.
            future_exog (pd.DataFrame, optional): Future exogenous variables for forecasting.
        '''
        # Only calibrate if not already done
        if not hasattr(self, 'q_hat_D'):
            raise RuntimeError("Conformalizer must be calibrated before generating prediction intervals. Run .calibrate(df_calibration) first.")
        self.model.fit(df)
        if future_exog is not None:
            y_forecast = np.array(self.model.forecast(self.H, future_exog))
        else:
            y_forecast = np.array(self.model.forecast(self.H))
        result = [y_forecast]
        col_names = ["point_forecast"]

        if isinstance(self.delta, float):
            y_lower, y_upper = y_forecast - self.q_hat_D, y_forecast + self.q_hat_D
            result.extend([y_lower, y_upper])
            col_names.extend([f'lower_{int(self.delta*100)}', f'upper_{int(self.delta*100)}'])
        elif isinstance(self.delta, list):
            for idx, d in enumerate(self.delta):
                y_lower = y_forecast - self.q_hat_D[:, idx]
                y_upper = y_forecast + self.q_hat_D[:, idx]
                result.extend([y_lower, y_upper])
                col_names.extend([f'lower_{int(d*100)}', f'upper_{int(d*100)}'])
        # distributions for each horizons. So add y_forecast array to each columns of self.resid and equal to self.dist
        dist = y_forecast[:, None] + self.resid
        self.dist = pd.DataFrame(dist.T, columns=[f'h_{i+1}' for i in range(self.H)])
        self.dist = self.dist.clip(lower=0.1)
        return pd.DataFrame(np.column_stack(result), columns=col_names)
        
    def conformal_quantiles(self, df, quantiles, future_exog=None):
        """
        Generate conformal quantiles for future time steps.
        Args:
            df (pd.DataFrame): DataFrame containing the training data.
            quantiles (float or list): Quantiles to be calculated (e.g., 0.1, 0.5, 0.9).
            future_exog (pd.DataFrame, optional): Future exogenous variables for forecasting.
        Returns:
            pd.DataFrame: DataFrame containing point forecasts and conformal quantiles.
        """
        # Only calibrate if not already done
        if not hasattr(self, 'q_hat_D'):
            raise RuntimeError("Conformalizer must be calibrated before generating prediction intervals. Run .calibrate(df_calibration) first.")
        self.model.fit(df)
        if future_exog is not None:
            y_forecast = np.array(self.model.forecast(self.H, future_exog))
        else:
            y_forecast = np.array(self.model.forecast(self.H))

        return get_conformal_quantiles(self.non_conform, self.n_calib, quantiles, y_forecast)

    def bootstrap(self, df, samples=1000, future_exog=None, approximate="kde"):
        """
        Generate samples from the predictive distribution generated by residuals from conformal prediction.
        The samples are drawn from a Gaussian kernel density estimate of the residuals.
        """
        # Return a random sample from Gaussian Kernel density estimation
        if not hasattr(self, 'resid'):
            raise RuntimeError("Residuals not available. Run .non_conformity_scores(df) .calibrate(df) or  first.")
        self.model.fit(df)
        if future_exog is not None:
            y_forecast = np.array(self.model.forecast(self.H, future_exog))
        else:
            y_forecast = np.array(self.model.forecast(self.H))

        # self.bootstrap_forecasts = np.column_stack([[gaussian_kde(self.resid[i]).resample(size=1)[0][0]+y_forecast[i]
        #                                    for i in range(self.H)] for _ in range(samples)]) # H x samples

        # ✅ Create a deep copy so that we don’t overwrite self
        new_instance = copy.deepcopy(self)
        if approximate == "kde":
            new_instance.bootstrap_forecasts = np.column_stack([[gaussian_kde(new_instance.resid[i]).resample(size=samples, seed=rng_kde)+y_forecast[i]]
                                        for i in range(new_instance.H)])[0] # H x samples
        elif approximate == "empirical":
            new_instance.bootstrap_forecasts = np.column_stack([np.random.choice(new_instance.resid[i], size=samples, replace=True)+y_forecast[i]
                for i in range(self.H)]).T
        else:
            raise ValueError("approximate must be 'kde' or 'empirical'.")

        new_instance.bootstrap_forecasts_df = pd.DataFrame(new_instance.bootstrap_forecasts.T, columns=[f'h_{i+1}' for i in range(self.H)])
        new_instance.y_forecast_b = y_forecast

        return new_instance

    def bootstrap_quantiles(self, quantiles, bootstrap_method='bootstrap'):
        """
        Generate bootstrap quantiles for future time steps.
        Args:
            quantiles (float or list): Quantiles to be calculated (e.g., 0.1, 0.5, 0.9).
            bootstrap_method (str): The method to use for bootstrapping or simulated correlated_forecasts ('bootstrap' or 'correlated').
        """
        ## Make sure bootstrap() method is called before calling this method
        if bootstrap_method == 'bootstrap':
            if (not hasattr(self, 'bootstrap')):
                raise RuntimeError("Bootstrap samples not available. Run .bootstrap(df) first.")
            return get_bootstrap_quantiles(self.bootstrap_forecasts, self.n_calib, quantiles, self.y_forecast_b)
        elif bootstrap_method == 'correlated':
            if (not hasattr(self, 'simulate_correlated_forecasts')):
                raise RuntimeError("Correlated forecast samples not available. Run .simulate_correlated_forecasts(df) first.")
            return get_bootstrap_quantiles(self.w_samples.T, self.n_calib, quantiles, self.y_forecast_c)

    def simulate_correlated_forecasts(self, df, samples=1000, future_exog=None):
        """
        Simulate correlated errors to generate forecasts
        Parameters:
        - df: DataFrame containing the training data
        - samples: number of samples to generate
        - future_exog: optional exogenous variables for forecasting
        Returns:
        - DataFrame containing the simulated forecasts
        """
        if not hasattr(self, 'resid'):
            raise RuntimeError("Residuals not available. Run .non_conformity_scores(df) .calibrate(df) or  first.")
        self.model.fit(df)
        if future_exog is not None:
            y_forecast = np.array(self.model.forecast(self.H, future_exog))
        else:
            y_forecast = np.array(self.model.forecast(self.H))

        mu = np.mean(self.resid.T, axis=0)
        sigma_ = np.cov(self.resid.T, rowvar=False, ddof=1)
        rng = np.random.default_rng(seed=42)
        samples = rng.multivariate_normal(mu, sigma_, size=samples)
        self.w_samples = samples + y_forecast
        self.correlated_forecasts = pd.DataFrame(self.w_samples, columns=[f'h_{i+1}' for i in range(self.H)])
        self.y_forecast_c = y_forecast
        return self

class var_prob_forecasts():
    """
    Probabilistic forecasting for vector autoregressive time series forecasting.
    It generates prediction intervals for future time steps and approximates distribution of predictions using Kernel Density Estimation (KDE).
    Parameters:
    - model: forecasting model to be used
    - target_col: one of the target columns in multivariate time series should be specified for conformalization
    - n_calibration: number of calibration windows
    - H: forecast horizon
    - sliding_window: size of the sliding window for cross-validation
    - verbose: whether to print progress messages
    """
    def __init__(self, model, target_col, n_calibration, H, sliding_window=1, verbose=False):
        self.model = model
        self.tar_col = target_col
        self.sliding_window = sliding_window
        self.n_calib = n_calibration
        self.verbose = verbose
        self.H = H

    def non_conformity_scores(self, df):
        c_actuals, c_forecasts = [], []
        # Create time series cross-validator that slides 1 time step for each training window
        tscv = ParametricTimeSeriesSplit(n_splits=self.n_calib, test_size=self.H, step_size=self.sliding_window)
        for train_index, test_index in tscv.split(df):
            train, test = df.iloc[train_index], df.iloc[test_index]
            x_test = test.drop(columns=self.model.target_cols)
            y_test = np.array(test[self.tar_col])
            self.model.fit(train)
            H_forecasts = self.model.forecast(self.H, x_test)[self.tar_col]
            c_forecasts.append(H_forecasts)
            c_actuals.append(y_test)
            if self.verbose:
                print(f"Completed calibration window {len(c_forecasts)} out of {self.n_calib}")
        self.resid = np.column_stack(c_actuals) - np.column_stack(c_forecasts) # Residuals n_calib*H
        self.non_conform = np.abs(self.resid) # non-conformity scores
        self.c_actuals = np.column_stack(c_actuals)
        self.c_forecasts = np.column_stack(c_forecasts)
        
    def calculate_quantile(self, scores_calib):
        # Vectorized quantile calculation for list delta
        if isinstance(self.delta, float):
            which_quantile = np.ceil(self.delta * (self.n_calib + 1)) / self.n_calib
            return np.quantile(scores_calib, which_quantile, method="lower", axis=0)
        elif isinstance(self.delta, list):
            which_quantiles = np.ceil(np.array(self.delta) * (self.n_calib + 1)) / self.n_calib
            return np.array([np.quantile(scores_calib, q, method="lower", axis=0) for q in which_quantiles])
        else:
            raise ValueError("delta must be float or list of floats.")
    
    
    def calibrate(self, df, delta = 0.5):
        """
        Calibrate the conformal model using the calibration dataset.
        Args:
            df (pd.DataFrame): DataFrame containing the calibration data.
            delta (float or list): Significance level(s) for the prediction intervals.
        """
        self.delta = delta
        self.non_conformity_scores(df=df)
        h_quantiles = []
        for i in range(self.H):
            q_hat = self.calculate_quantile(self.non_conform[i])
            h_quantiles.append(q_hat)
        self.q_hat_D = np.array(h_quantiles)

    # Generate prediction intervals using the calibrated quantiles

    def generate_prediction_intervals(self, df, future_exog=None):
        '''
        Generate conformal prediction intervals for the forecasted values.
        Args:
            df (pd.DataFrame): DataFrame containing the training data.
            future_exog (pd.DataFrame, optional): Future exogenous variables for forecasting.
        '''
        # Only calibrate if not already done
        if not hasattr(self, 'q_hat_D'):
            raise RuntimeError("Conformalizer must be calibrated before generating prediction intervals. Run .calibrate(df_calibration) first.")
        self.model.fit(df)
        if future_exog is not None:
            y_forecast = np.array(self.model.forecast(self.H, future_exog)[self.tar_col])
        else:
            y_forecast = np.array(self.model.forecast(self.H)[self.tar_col])
        result = [y_forecast]
        col_names = ["point_forecast"]

        if isinstance(self.delta, float):
            y_lower, y_upper = y_forecast - self.q_hat_D, y_forecast + self.q_hat_D
            result.extend([y_lower, y_upper])
            col_names.extend([f'lower_{int(self.delta*100)}', f'upper_{int(self.delta*100)}'])
        elif isinstance(self.delta, list):
            for idx, d in enumerate(self.delta):
                y_lower = y_forecast - self.q_hat_D[:, idx]
                y_upper = y_forecast + self.q_hat_D[:, idx]
                result.extend([y_lower, y_upper])
                col_names.extend([f'lower_{int(d*100)}', f'upper_{int(d*100)}'])
        # distributions for each horizons. So add y_forecast array to each columns of self.resid and equal to self.dist
        dist = y_forecast[:, None] + self.resid
        self.dist = pd.DataFrame(dist.T, columns=[f'h_{i+1}' for i in range(self.H)])
        self.dist = self.dist.clip(lower=0.1)
        return pd.DataFrame(np.column_stack(result), columns=col_names)


    def conformal_quantiles(self, df, quantiles, future_exog=None):
        """
        Generate conformal quantiles for future time steps.
        Args:
            df (pd.DataFrame): DataFrame containing the training data.
            quantiles (float or list): Quantiles to be calculated (e.g., 0.1, 0.5, 0.9).
            future_exog (pd.DataFrame, optional): Future exogenous variables for forecasting.
        Returns:
            pd.DataFrame: DataFrame containing point forecasts and conformal quantiles.
        """
        # Only calibrate if not already done
        if not hasattr(self, 'q_hat_D'):
            raise RuntimeError("Conformalizer must be calibrated before generating prediction intervals. Run .calibrate(df_calibration) first.")
        self.model.fit(df)
        if future_exog is not None:
            y_forecast = np.array(self.model.forecast(self.H, future_exog)[self.tar_col])
        else:
            y_forecast = np.array(self.model.forecast(self.H)[self.tar_col])

        return get_conformal_quantiles(self.non_conform, self.n_calib, quantiles, y_forecast)

    def bootstrap(self, df, samples=1000, future_exog=None , approximate="kde"):
        """
        Generate samples from the predictive distribution generated by residuals from conformal prediction.
        The samples are drawn from a Gaussian kernel density estimate of the residuals.
        """
        # Return a random sample from Gaussian Kernel density estimation
        if not hasattr(self, 'resid'):
            raise RuntimeError("Residuals not available. Run .non_conformity_scores(df) .calibrate(df) or  first.")
        self.model.fit(df)
        if future_exog is not None:
            y_forecast = np.array(self.model.forecast(self.H, future_exog)[self.tar_col])
        else:
            y_forecast = np.array(self.model.forecast(self.H)[self.tar_col])

        # ✅ Create a deep copy so that we don’t overwrite self
        new_instance = copy.deepcopy(self)
        if approximate == "kde":
            new_instance.bootstrap_forecasts = np.column_stack([[gaussian_kde(new_instance.resid[i]).resample(size=samples, seed=rng_kde)+y_forecast[i]]
                                        for i in range(new_instance.H)])[0] # H x samples
        elif approximate == "empirical":
            new_instance.bootstrap_forecasts = np.column_stack([np.random.choice(new_instance.resid[i], size=samples, replace=True)+y_forecast[i]
                for i in range(self.H)]).T
        else:
            raise ValueError("approximate must be 'kde' or 'empirical'.")

        new_instance.bootstrap_forecasts_df = pd.DataFrame(new_instance.bootstrap_forecasts.T, columns=[f'h_{i+1}' for i in range(self.H)])
        new_instance.y_forecast_b = y_forecast

        return new_instance

    def bootstrap_quantiles(self, quantiles, bootstrap_method='bootstrap'):
        """
        Generate bootstrap quantiles for future time steps.
        Args:
            quantiles (float or list): Quantiles to be calculated (e.g., 0.1, 0.5, 0.9).
            bootstrap_method (str): The method to use for bootstrapping or simulated correlated_forecasts ('bootstrap' or 'correlated').
        """
        ## Make sure bootstrap() method is called before calling this method
        if bootstrap_method == 'bootstrap':
            if (not hasattr(self, 'bootstrap')):
                raise RuntimeError("Bootstrap samples not available. Run .bootstrap(df) first.")
            return get_bootstrap_quantiles(self.bootstrap_forecasts, self.n_calib, quantiles, self.y_forecast_b)
        elif bootstrap_method == 'correlated':
            if (not hasattr(self, 'simulate_correlated_forecasts')):
                raise RuntimeError("Correlated forecast samples not available. Run .simulate_correlated_forecasts(df) first.")
            return get_bootstrap_quantiles(self.w_samples.T, self.n_calib, quantiles, self.y_forecast_c)
    
    def simulate_correlated_forecasts(self, df, samples=1000, future_exog=None):
        """
        Simulate correlated errors to generate forecasts
        Parameters:
        - df: DataFrame containing the training data
        - samples: number of samples to generate
        - future_exog: optional exogenous variables for forecasting
        Returns:
        - DataFrame containing the simulated forecasts
        """
        if not hasattr(self, 'resid'):
            raise RuntimeError("Residuals not available. Run .non_conformity_scores(df) .calibrate(df) or  first.")
        self.model.fit(df)
        if future_exog is not None:
            y_forecast = np.array(self.model.forecast(self.H, future_exog)[self.tar_col])
        else:
            y_forecast = np.array(self.model.forecast(self.H)[self.tar_col])

        mu = np.mean(self.resid.T, axis=0)
        sigma_ = np.cov(self.resid.T, rowvar=False, ddof=1)
        rng = np.random.default_rng(seed=42)
        samples = rng.multivariate_normal(mu, sigma_, size=samples)
        self.w_samples = samples + y_forecast
        self.correlated_forecasts = pd.DataFrame(self.w_samples, columns=[f'h_{i+1}' for i in range(self.H)])
        self.y_forecast_c = y_forecast
        return self
    
class hmm_prob_forecasts():
    """
    Probabilistic forecasting for Markov Switching Regression. It generates prediction intervals for future time steps and approximates distribution of predictions using Kernel Density Estimation (KDE).
    Parameters:
    - model: forecasting model to be used
    - n_calibration: number of calibration windows
    - H: forecast horizon
    - sliding_window: size of the sliding window for cross-validation
    - n_iter: number of iterations for HMM fitting
    - verbose: whether to print progress messages
    """
    def __init__(self, model, n_calibration, H, sliding_window=1, n_iter=1, verbose=False):
        self.model = model
        self.sliding_window = sliding_window
        self.n_calib = n_calibration
        self.n_iter = n_iter
        self.verbose = verbose
        self.H = H

    def non_conformity_scores(self, df):
        c_actuals, c_forecasts = [], []
        # Create time series cross-validator that slides 1 time step for each training window
        tscv = ParametricTimeSeriesSplit(n_splits=self.n_calib, test_size=self.H, step_size=self.sliding_window)
        for train_index, test_index in tscv.split(df):
            train, test = df.iloc[train_index], df.iloc[test_index]
            x_test = test.drop(columns=[self.model.target_col])
            y_test = np.array(test[self.model.target_col])
            self.model.fit(train, self.n_iter)
            H_forecasts = self.model.forecast(self.H, x_test)
            c_forecasts.append(H_forecasts)
            c_actuals.append(y_test)
            if self.verbose:
                print(f"Completed calibration window {len(c_forecasts)} out of {self.n_calib}")
        self.resid = np.column_stack(c_actuals) - np.column_stack(c_forecasts) # Residuals n_calib*H
        self.non_conform = np.abs(self.resid) # non-conformity scores
        self.c_actuals = np.column_stack(c_actuals)
        self.c_forecasts = np.column_stack(c_forecasts)

        
    def calculate_quantile(self, scores_calib):
        # Vectorized quantile calculation for list delta
        if isinstance(self.delta, float):
            which_quantile = np.ceil(self.delta * (self.n_calib + 1)) / self.n_calib
            return np.quantile(scores_calib, which_quantile, method="lower", axis=0)
        elif isinstance(self.delta, list):
            which_quantiles = np.ceil(np.array(self.delta) * (self.n_calib + 1)) / self.n_calib
            return np.array([np.quantile(scores_calib, q, method="lower", axis=0) for q in which_quantiles])
        else:
            raise ValueError("delta must be float or list of floats.")
    
    
    def calibrate(self, df, delta = 0.5):
        """
        Calibrate the conformal model using the calibration dataset.
        Args:
            df (pd.DataFrame): DataFrame containing the calibration data.
            delta (float or list): Significance level(s) for the prediction intervals.
        """
        self.delta = delta
        self.non_conformity_scores(df=df)
        h_quantiles = []
        for i in range(self.H):
            q_hat = self.calculate_quantile(self.non_conform[i])
            h_quantiles.append(q_hat)
        self.q_hat_D = np.array(h_quantiles)

    # Generate prediction intervals using the calibrated quantiles

    def generate_prediction_intervals(self, df, future_exog=None):
        '''
        Generate conformal prediction intervals for the forecasted values.
        Args:
            df (pd.DataFrame): DataFrame containing the training data.
            future_exog (pd.DataFrame, optional): Future exogenous variables for forecasting.
        '''
        # Only calibrate if not already done
        if not hasattr(self, 'q_hat_D'):
            raise RuntimeError("Conformalizer must be calibrated before generating prediction intervals. Run .calibrate(df_calibration) first.")
        self.model.fit(df, self.n_iter)
        if future_exog is not None:
            y_forecast = np.array(self.model.forecast(self.H, future_exog))
        else:
            y_forecast = np.array(self.model.forecast(self.H))
        result = [y_forecast]
        col_names = ["point_forecast"]

        if isinstance(self.delta, float):
            y_lower, y_upper = y_forecast - self.q_hat_D, y_forecast + self.q_hat_D
            result.extend([y_lower, y_upper])
            col_names.extend([f'lower_{int(self.delta*100)}', f'upper_{int(self.delta*100)}'])
        elif isinstance(self.delta, list):
            for idx, d in enumerate(self.delta):
                y_lower = y_forecast - self.q_hat_D[:, idx]
                y_upper = y_forecast + self.q_hat_D[:, idx]
                result.extend([y_lower, y_upper])
                col_names.extend([f'lower_{int(d*100)}', f'upper_{int(d*100)}'])
        # distributions for each horizons. So add y_forecast array to each columns of self.resid and equal to self.dist
        dist = y_forecast[:, None] + self.resid
        self.dist = pd.DataFrame(dist.T, columns=[f'h_{i+1}' for i in range(self.H)])
        self.dist = self.dist.clip(lower=0.1)
        return pd.DataFrame(np.column_stack(result), columns=col_names)

    def conformal_quantiles(self, df, quantiles, future_exog=None):
        """
        Generate conformal quantiles for future time steps.
        Args:
            df (pd.DataFrame): DataFrame containing the training data.
            quantiles (float or list): Quantiles to be calculated (e.g., 0.1, 0.5, 0.9).
            future_exog (pd.DataFrame, optional): Future exogenous variables for forecasting.
        Returns:
            pd.DataFrame: DataFrame containing point forecasts and conformal quantiles.
        """
        # Only calibrate if not already done
        if not hasattr(self, 'q_hat_D'):
            raise RuntimeError("Conformalizer must be calibrated before generating prediction intervals. Run .calibrate(df_calibration) first.")
        self.model.fit(df, self.n_iter)
        if future_exog is not None:
            y_forecast = np.array(self.model.forecast(self.H, future_exog))
        else:
            y_forecast = np.array(self.model.forecast(self.H))

        return get_conformal_quantiles(self.non_conform, self.n_calib, quantiles, y_forecast)

    def bootstrap(self, df, samples=1000, future_exog=None , approximate="kde"):
        """
        Generate samples from the predictive distribution generated by residuals from conformal prediction.
        The samples are drawn from a Gaussian kernel density estimate of the residuals.
        """
        # Return a random sample from Gaussian Kernel density estimation
        if not hasattr(self, 'resid'):
            raise RuntimeError("Residuals not available. Run .non_conformity_scores(df) .calibrate(df) or  first.")
        self.model.fit(df, self.n_iter)
        if future_exog is not None:
            y_forecast = np.array(self.model.forecast(self.H, future_exog))
        else:
            y_forecast = np.array(self.model.forecast(self.H))

        # ✅ Create a deep copy so that we don’t overwrite self
        new_instance = copy.deepcopy(self)
        if approximate == "kde":
            new_instance.bootstrap_forecasts = np.column_stack([[gaussian_kde(new_instance.resid[i]).resample(size=samples, seed=rng_kde)+y_forecast[i]]
                                        for i in range(new_instance.H)])[0] # H x samples
        elif approximate == "empirical":
            new_instance.bootstrap_forecasts = np.column_stack([np.random.choice(new_instance.resid[i], size=samples, replace=True)+y_forecast[i]
                for i in range(self.H)]).T
        else:
            raise ValueError("approximate must be 'kde' or 'empirical'.")

        new_instance.bootstrap_forecasts_df = pd.DataFrame(new_instance.bootstrap_forecasts.T, columns=[f'h_{i+1}' for i in range(self.H)])
        new_instance.y_forecast_b = y_forecast

        return new_instance

    def bootstrap_quantiles(self, quantiles, bootstrap_method='bootstrap'):
        """
        Generate bootstrap quantiles for future time steps.
        Args:
            quantiles (float or list): Quantiles to be calculated (e.g., 0.1, 0.5, 0.9).
            bootstrap_method (str): The method to use for bootstrapping or simulated correlated_forecasts ('bootstrap' or 'correlated').
        """
        ## Make sure bootstrap() method is called before calling this method
        if bootstrap_method == 'bootstrap':
            if (not hasattr(self, 'bootstrap')):
                raise RuntimeError("Bootstrap samples not available. Run .bootstrap(df) first.")
            return get_bootstrap_quantiles(self.bootstrap_forecasts, self.n_calib, quantiles, self.y_forecast_b)
        elif bootstrap_method == 'correlated':
            if (not hasattr(self, 'simulate_correlated_forecasts')):
                raise RuntimeError("Correlated forecast samples not available. Run .simulate_correlated_forecasts(df) first.")
            return get_bootstrap_quantiles(self.w_samples.T, self.n_calib, quantiles, self.y_forecast_c)
    
    def simulate_correlated_forecasts(self, df, samples=1000, future_exog=None):
        """
        Simulate correlated errors to generate forecasts
        Parameters:
        - df: DataFrame containing the training data
        - samples: number of samples to generate
        - future_exog: optional exogenous variables for forecasting
        Returns:
        - DataFrame containing the simulated forecasts
        """
        if not hasattr(self, 'resid'):
            raise RuntimeError("Residuals not available. Run .non_conformity_scores(df) .calibrate(df) or  first.")
        self.model.fit(df, self.n_iter)
        if future_exog is not None:
            y_forecast = np.array(self.model.forecast(self.H, future_exog))
        else:
            y_forecast = np.array(self.model.forecast(self.H))

        mu = np.mean(self.resid.T, axis=0)
        sigma_ = np.cov(self.resid.T, rowvar=False, ddof=1)

        rng = np.random.default_rng(seed=42)
        samples = rng.multivariate_normal(mu, sigma_, size=samples)
        self.w_samples = samples + y_forecast
        self.correlated_forecasts = pd.DataFrame(self.w_samples, columns=[f'h_{i+1}' for i in range(self.H)])
        self.y_forecast_c = y_forecast
        return self

class hmm_var_prob_forecasts():
    """
    Probabilistic forecasting for Markov Switching VAR models. It generates prediction intervals for future time steps and approximates distribution of predictions using Kernel Density Estimation (KDE).
    Parameters:
    - model: forecasting model to be used
    - n_calibration: number of calibration windows
    - H: forecast horizon
    - sliding_window: size of the sliding window for cross-validation
    - n_iter: number of iterations for HMM fitting
    - verbose: whether to print progress messages
    """
    def __init__(self, model, target_col, n_calibration, H, sliding_window=1, n_iter=1, verbose=False):
        self.model = model
        self.tar_col = target_col
        self.sliding_window = sliding_window
        self.n_calib = n_calibration
        self.n_iter = n_iter
        self.verbose = verbose
        self.H = H

    def non_conformity_scores(self, df):
        c_actuals, c_forecasts = [], []
        # Create time series cross-validator that slides 1 time step for each training window
        tscv = ParametricTimeSeriesSplit(n_splits=self.n_calib, test_size=self.H, step_size=self.sliding_window)
        for train_index, test_index in tscv.split(df):
            train, test = df.iloc[train_index], df.iloc[test_index]
            x_test = test.drop(columns=self.model.target_col)
            y_test = np.array(test[self.tar_col])
            self.model.fit(train, self.n_iter)
            H_forecasts = self.model.forecast(self.H, x_test)[self.tar_col]
            c_forecasts.append(H_forecasts)
            c_actuals.append(y_test)
            if self.verbose:
                print(f"Completed calibration window {len(c_forecasts)} out of {self.n_calib}")
        self.resid = np.column_stack(c_actuals) - np.column_stack(c_forecasts) # Residuals n_calib*H
        self.non_conform = np.abs(self.resid) # non-conformity scores
        self.c_actuals = np.column_stack(c_actuals)
        self.c_forecasts = np.column_stack(c_forecasts)

        
    def calculate_quantile(self, scores_calib):
        # Vectorized quantile calculation for list delta
        if isinstance(self.delta, float):
            which_quantile = np.ceil(self.delta * (self.n_calib + 1)) / self.n_calib
            return np.quantile(scores_calib, which_quantile, method="lower", axis=0)
        elif isinstance(self.delta, list):
            which_quantiles = np.ceil(np.array(self.delta) * (self.n_calib + 1)) / self.n_calib
            return np.array([np.quantile(scores_calib, q, method="lower", axis=0) for q in which_quantiles])
        else:
            raise ValueError("delta must be float or list of floats.")
    
    
    def calibrate(self, df, delta = 0.5):
        """
        Calibrate the conformal model using the calibration dataset.
        Args:
            df (pd.DataFrame): DataFrame containing the calibration data.
            delta (float or list): Significance level(s) for the prediction intervals.
        """
        self.delta = delta
        self.non_conformity_scores(df=df)
        h_quantiles = []
        for i in range(self.H):
            q_hat = self.calculate_quantile(self.non_conform[i])
            h_quantiles.append(q_hat)
        self.q_hat_D = np.array(h_quantiles)

    # Generate prediction intervals using the calibrated quantiles

    def generate_prediction_intervals(self, df, future_exog=None):
        '''
        Generate conformal prediction intervals for the forecasted values.
        Args:
            df (pd.DataFrame): DataFrame containing the training data.
            future_exog (pd.DataFrame, optional): Future exogenous variables for forecasting.
        '''
        
        if not hasattr(self, 'q_hat_D'):
            raise RuntimeError("Conformalizer must be calibrated before generating prediction intervals. Run .calibrate(df_calibration) first.")
        self.model.fit(df, self.n_iter)
        if future_exog is not None:
            y_forecast = np.array(self.model.forecast(self.H, future_exog)[self.tar_col])
        else:
            y_forecast = np.array(self.model.forecast(self.H)[self.tar_col])
        result = [y_forecast]
        col_names = ["point_forecast"]

        if isinstance(self.delta, float):
            y_lower, y_upper = y_forecast - self.q_hat_D, y_forecast + self.q_hat_D
            result.extend([y_lower, y_upper])
            col_names.extend([f'lower_{int(self.delta*100)}', f'upper_{int(self.delta*100)}'])
        elif isinstance(self.delta, list):
            for idx, d in enumerate(self.delta):
                y_lower = y_forecast - self.q_hat_D[:, idx]
                y_upper = y_forecast + self.q_hat_D[:, idx]
                result.extend([y_lower, y_upper])
                col_names.extend([f'lower_{int(d*100)}', f'upper_{int(d*100)}'])
        # distributions for each horizons. So add y_forecast array to each columns of self.resid and equal to self.dist
        dist = y_forecast[:, None] + self.resid
        self.dist = pd.DataFrame(dist.T, columns=[f'h_{i+1}' for i in range(self.H)])
        self.dist = self.dist.clip(lower=0.1)
        return pd.DataFrame(np.column_stack(result), columns=col_names)

    def conformal_quantiles(self, df, quantiles, future_exog=None):
        """
        Generate conformal quantiles for future time steps.
        Args:
            df (pd.DataFrame): DataFrame containing the training data.
            quantiles (float or list): Quantiles to be calculated (e.g., 0.1, 0.5, 0.9).
            future_exog (pd.DataFrame, optional): Future exogenous variables for forecasting.
        Returns:
            pd.DataFrame: DataFrame containing point forecasts and conformal quantiles.
        """
        # Only calibrate if not already done
        if not hasattr(self, 'q_hat_D'):
            raise RuntimeError("Conformalizer must be calibrated before generating prediction intervals. Run .calibrate(df_calibration) first.")
        self.model.fit(df, self.n_iter)
        if future_exog is not None:
            y_forecast = np.array(self.model.forecast(self.H, future_exog)[self.tar_col])
        else:
            y_forecast = np.array(self.model.forecast(self.H)[self.tar_col])

        return get_conformal_quantiles(self.non_conform, self.n_calib, quantiles, y_forecast)

    def bootstrap(self, df, samples=1000, future_exog=None, approximate="kde"):
        """
        Generate samples from the predictive distribution generated by residuals from conformal prediction.
        The samples are drawn from a Gaussian kernel density estimate of the residuals.
        """
        # Return a random sample from Gaussian Kernel density estimation
        if not hasattr(self, 'resid'):
            raise RuntimeError("Residuals not available. Run .non_conformity_scores(df) .calibrate(df) or  first.")
        self.model.fit(df, self.n_iter)
        if future_exog is not None:
            y_forecast = np.array(self.model.forecast(self.H, future_exog)[self.tar_col])
        else:
            y_forecast = np.array(self.model.forecast(self.H)[self.tar_col])

        # ✅ Create a deep copy so that we don’t overwrite self
        new_instance = copy.deepcopy(self)
        if approximate == "kde":
            new_instance.bootstrap_forecasts = np.column_stack([[gaussian_kde(new_instance.resid[i]).resample(size=samples, seed=rng_kde)+y_forecast[i]]
                                        for i in range(new_instance.H)])[0] # H x samples
        elif approximate == "empirical":
            new_instance.bootstrap_forecasts = np.column_stack([np.random.choice(new_instance.resid[i], size=samples, replace=True)+y_forecast[i]
                for i in range(self.H)]).T
        else:
            raise ValueError("approximate must be 'kde' or 'empirical'.")

        new_instance.bootstrap_forecasts_df = pd.DataFrame(new_instance.bootstrap_forecasts.T, columns=[f'h_{i+1}' for i in range(self.H)])
        new_instance.y_forecast_b = y_forecast

        return new_instance

    def bootstrap_quantiles(self, quantiles, bootstrap_method='bootstrap'):
        """
        Generate bootstrap quantiles for future time steps.
        Args:
            quantiles (float or list): Quantiles to be calculated (e.g., 0.1, 0.5, 0.9).
            bootstrap_method (str): The method to use for bootstrapping or simulated correlated_forecasts ('bootstrap' or 'correlated').
        """
        ## Make sure bootstrap() method is called before calling this method
        if bootstrap_method == 'bootstrap':
            if (not hasattr(self, 'bootstrap')):
                raise RuntimeError("Bootstrap samples not available. Run .bootstrap(df) first.")
            return get_bootstrap_quantiles(self.bootstrap_forecasts, self.n_calib, quantiles, self.y_forecast_b)
        elif bootstrap_method == 'correlated':
            if (not hasattr(self, 'simulate_correlated_forecasts')):
                raise RuntimeError("Correlated forecast samples not available. Run .simulate_correlated_forecasts(df) first.")
            return get_bootstrap_quantiles(self.w_samples.T, self.n_calib, quantiles, self.y_forecast_c)
    
    def simulate_correlated_forecasts(self, df, samples=1000, future_exog=None):
        """
        Simulate correlated errors to generate forecasts
        Parameters:
        - df: DataFrame containing the training data
        - samples: number of samples to generate
        - future_exog: optional exogenous variables for forecasting
        Returns:
        - DataFrame containing the simulated forecasts
        """
        if not hasattr(self, 'resid'):
            raise RuntimeError("Residuals not available. Run .non_conformity_scores(df) .calibrate(df) or  first.")
        self.model.fit(df, self.n_iter)
        if future_exog is not None:
            y_forecast = np.array(self.model.forecast(self.H, future_exog)[self.tar_col])
        else:
            y_forecast = np.array(self.model.forecast(self.H)[self.tar_col])

        mu = np.mean(self.resid.T, axis=0)
        sigma_ = np.cov(self.resid.T, rowvar=False, ddof=1)
        rng = np.random.default_rng(seed=42)
        samples = rng.multivariate_normal(mu, sigma_, size=samples)
        self.w_samples = samples + y_forecast
        self.correlated_forecasts = pd.DataFrame(self.w_samples, columns=[f'h_{i+1}' for i in range(self.H)])
        self.y_forecast_c = y_forecast
        return self

class ets_prob_forecasts():
    """
    Probabilistic forecasting for Exponential Smoothing State Space Model (ETS). It generates prediction intervals for future time steps and approximates distribution of predictions using Kernel Density Estimation (KDE).
    Parameters:
    - ets_param: parameters for ETS model. A tuble of (dict, dict) where first dict is for ExponentialSmoothing() and second dict is for .fit()
    - n_calibration: number of calibration windows
    - H: forecast horizon
    - sliding_window: size of the sliding window for cross-validation
    - verbose: whether to print progress messages
    """
    def __init__(self, ets_param, n_calibration,
                 H, sliding_window=1, verbose=False):
        self.ets_param = ets_param
        self.sliding_window = sliding_window
        self.n_calib = n_calibration
        self.verbose = verbose
        self.H = H

    def non_conformity_scores(self, series):
        c_actuals, c_forecasts = [], []
        # Create time series cross-validator that slides 1 time step for each training window
        tscv = ParametricTimeSeriesSplit(n_splits=self.n_calib, test_size=self.H, step_size=self.sliding_window)

        for train_index, test_index in tscv.split(series):
            train, test = series[train_index], series[test_index]
            y_test = np.array(test)
            self.model = ExponentialSmoothing(train,
                            **self.ets_param[0]).fit(**self.ets_param[1])
            H_forecasts = np.array(self.model.forecast(self.H))
            c_forecasts.append(H_forecasts)
            c_actuals.append(y_test)
            if self.verbose:
                print(f"Completed calibration window {len(c_forecasts)} out of {self.n_calib}")
        self.resid = np.column_stack(c_actuals) - np.column_stack(c_forecasts) # Residuals n_calib*H
        self.non_conform = np.abs(self.resid) # non-conformity scores
        self.c_actuals = np.column_stack(c_actuals)
        self.c_forecasts = np.column_stack(c_forecasts)

    def calculate_quantile(self, scores_calib):
        # Vectorized quantile calculation for list delta
        if isinstance(self.delta, float):
            which_quantile = np.ceil(self.delta * (self.n_calib + 1)) / self.n_calib
            return np.quantile(scores_calib, which_quantile, method="lower", axis=0)
        elif isinstance(self.delta, list):
            which_quantiles = np.ceil(np.array(self.delta) * (self.n_calib + 1)) / self.n_calib
            return np.array([np.quantile(scores_calib, q, method="lower", axis=0) for q in which_quantiles])
        else:
            raise ValueError("delta must be float or list of floats.")
    
    def calibrate(self, series, delta = 0.5):
        """
        Calibrate the conformal model using the calibration dataset.
        Args:
            series: pd.Series or np.array containing the calibration data.
            delta (float or list): Significance level(s) for the prediction intervals.
        """
        self.delta = delta
        self.non_conformity_scores(series)
        h_quantiles = []
        for i in range(self.H):
            q_hat = self.calculate_quantile(self.non_conform[i])
            h_quantiles.append(q_hat)
        self.q_hat_D = np.array(h_quantiles)

    # Generate prediction intervals using the calibrated quantiles

    def generate_prediction_intervals(self, series):
        '''
        Generate conformal prediction intervals for the forecasted values.
        Args:
            series: pd.Series or np.array containing the training data.
        '''
        if not hasattr(self, 'q_hat_D'):
            raise RuntimeError("Conformalizer must be calibrated before generating prediction intervals. Run .calibrate(df_calibration) first.")
        self.model = ExponentialSmoothing(series,
                            **self.ets_param[0]).fit(**self.ets_param[1])
        y_forecast = np.array(self.model.forecast(self.H))
        result = [y_forecast]
        col_names = ["point_forecast"]

        if isinstance(self.delta, float):
            y_lower, y_upper = y_forecast - self.q_hat_D, y_forecast + self.q_hat_D
            result.extend([y_lower, y_upper])
            col_names.extend([f'lower_{int(self.delta*100)}', f'upper_{int(self.delta*100)}'])
        elif isinstance(self.delta, list):
            for idx, d in enumerate(self.delta):
                y_lower = y_forecast - self.q_hat_D[:, idx]
                y_upper = y_forecast + self.q_hat_D[:, idx]
                result.extend([y_lower, y_upper])
                col_names.extend([f'lower_{int(d*100)}', f'upper_{int(d*100)}'])
        # distributions for each horizons. So add y_forecast array to each columns of self.resid and equal to self.dist
        dist = y_forecast[:, None] + self.resid
        self.dist = pd.DataFrame(dist.T, columns=[f'h_{i+1}' for i in range(self.H)])
        self.dist = self.dist.clip(lower=0.1)
        return pd.DataFrame(np.column_stack(result), columns=col_names)

    def conformal_quantiles(self, series, quantiles):
        """
        Generate conformal quantiles for future time steps.
        Args:
            series: pd.Series or np.array containing the training data.
            quantiles (float or list): Quantiles to be calculated (e.g., 0.1, 0.5, 0.9).
            future_exog (pd.DataFrame, optional): Future exogenous variables for forecasting.
        Returns:
            pd.DataFrame: DataFrame containing point forecasts and conformal quantiles.
        """
        if not hasattr(self, 'q_hat_D'):
            raise RuntimeError("Conformalizer must be calibrated before generating prediction intervals. Run .calibrate(df_calibration) first.")
        self.model = ExponentialSmoothing(series,
                            **self.ets_param[0]).fit(**self.ets_param[1])
        y_forecast = np.array(self.model.forecast(self.H))

        return get_conformal_quantiles(self.non_conform, self.n_calib, quantiles, y_forecast)

    def bootstrap(self, series, samples=1000, approximate="kde"):
        """
        Generate samples from the predictive distribution generated by residuals from conformal prediction.
        The samples are drawn from a Gaussian kernel density estimate of the residuals.
        """
        # Return a random sample from Gaussian Kernel density estimation
        if not hasattr(self, 'resid'):
            raise RuntimeError("Residuals not available. Run .non_conformity_scores(df) .calibrate(df) or  first.")

        self.model = ExponentialSmoothing(series,
                            **self.ets_param[0]).fit(**self.ets_param[1])
        y_forecast = np.array(self.model.forecast(self.H))

        # ✅ Create a deep copy so that we don’t overwrite self
        new_instance = copy.deepcopy(self)
        if approximate == "kde":
            new_instance.bootstrap_forecasts = np.column_stack([[gaussian_kde(new_instance.resid[i]).resample(size=samples, seed=rng_kde)+y_forecast[i]]
                                        for i in range(new_instance.H)])[0] # H x samples
        elif approximate == "empirical":
            new_instance.bootstrap_forecasts = np.column_stack([np.random.choice(new_instance.resid[i], size=samples, replace=True)+y_forecast[i]
                for i in range(self.H)]).T
        else:
            raise ValueError("approximate must be 'kde' or 'empirical'.")

        new_instance.bootstrap_forecasts_df = pd.DataFrame(new_instance.bootstrap_forecasts.T, columns=[f'h_{i+1}' for i in range(self.H)])
        new_instance.y_forecast_b = y_forecast

        return new_instance

    def bootstrap_quantiles(self, quantiles, bootstrap_method='bootstrap'):
        """
        Generate bootstrap quantiles for future time steps.
        Args:
            quantiles (float or list): Quantiles to be calculated (e.g., 0.1, 0.5, 0.9).
            bootstrap_method (str): The method to use for bootstrapping or simulated correlated_forecasts ('bootstrap' or 'correlated').
        """
        ## Make sure bootstrap() method is called before calling this method
        if bootstrap_method == 'bootstrap':
            if (not hasattr(self, 'bootstrap')):
                raise RuntimeError("Bootstrap samples not available. Run .bootstrap(df) first.")
            return get_bootstrap_quantiles(self.bootstrap_forecasts, self.n_calib, quantiles, self.y_forecast_b)
        elif bootstrap_method == 'correlated':
            if (not hasattr(self, 'simulate_correlated_forecasts')):
                raise RuntimeError("Correlated forecast samples not available. Run .simulate_correlated_forecasts(df) first.")
            return get_bootstrap_quantiles(self.w_samples.T, self.n_calib, quantiles, self.y_forecast_c)

    def simulate_correlated_forecasts(self, series, samples=1000):
        """
        Simulate correlated errors to generate forecasts
        Parameters:
        - df: DataFrame containing the training data
        - samples: number of samples to generate
        - future_exog: optional exogenous variables for forecasting
        Returns:
        - DataFrame containing the simulated forecasts
        """
        if not hasattr(self, 'resid'):
            raise RuntimeError("Residuals not available. Run .non_conformity_scores(df) .calibrate(df) or  first.")

        self.model = ExponentialSmoothing(series,
                            **self.ets_param[0]).fit(**self.ets_param[1])
        y_forecast = np.array(self.model.forecast(self.H))

        mu = np.mean(self.resid.T, axis=0)
        sigma_ = np.cov(self.resid.T, rowvar=False, ddof=1)

        rng = np.random.default_rng(seed=42)
        samples = rng.multivariate_normal(mu, sigma_, size=samples)
        self.w_samples = samples + y_forecast
        self.correlated_forecasts = pd.DataFrame(self.w_samples, columns=[f'h_{i+1}' for i in range(self.H)])
        self.y_forecast_c = y_forecast
        return self
    
class naive_prob_forecasts():
    """
    Probabilistic forecasting for Naive forecasting model. It generates prediction intervals for future time steps and approximates distribution of predictions using Kernel Density Estimation (KDE).
    - n_calibration: number of calibration windows
    - H: forecast horizon
    - sliding_window: size of the sliding window for cross-validation
    - season_period: seasonal period for naive forecasting
    - verbose: whether to print progress messages
    """
    def __init__(self, n_calibration,
                 H, sliding_window=1, season_period=None, verbose=False):
        self.sliding_window = sliding_window
        self.n_calib = n_calibration
        self.verbose = verbose
        self.H = H
        self.season_period = season_period

    def naive(self, series, H, season_period=None):
        """Generate naive forecasts.

        Parameters:
        series (pd.Series): Time series data.
        H (int): Forecast horizon.
        season_period (int, optional): Seasonal period. If None, non-seasonal naive is used.
        Returns:
        np.ndarray: Forecasted values for the next H periods.
        """
        s = series.dropna()
        if season_period is None:
            # Non-seasonal naive: repeat last observed value H times
            if len(s) == 0:
                y_forecast = np.full(H, np.nan)
            else:
                last_val = s.iloc[-1]
                y_forecast = np.full(H, last_val)
        else:
            # Seasonal naive: use values from one season back, repeated/cycled
            if len(s) < season_period:
                y_forecast = np.full(H, np.nan)
            else:
                last_season = s.iloc[-season_period:]
                repeats = H // season_period
                remainder = H % season_period
                y_forecast = np.tile(last_season.values, repeats)
                if remainder > 0:
                    y_forecast = np.concatenate([y_forecast, last_season.values[:remainder]])
        return y_forecast


    def non_conformity_scores(self, series):
        c_actuals, c_forecasts = [], []
        # Create time series cross-validator that slides 1 time step for each training window
        tscv = ParametricTimeSeriesSplit(n_splits=self.n_calib, test_size=self.H, step_size=self.sliding_window)

        for train_index, test_index in tscv.split(series):
            train, test = series[train_index], series[test_index]
            y_test = np.array(test)
            H_forecasts = np.array(self.naive(train, self.H, season_period=self.season_period))
            c_forecasts.append(H_forecasts)
            c_actuals.append(y_test)
            if self.verbose:
                print(f"Completed calibration window {len(c_forecasts)} out of {self.n_calib}")
        self.resid = np.column_stack(c_actuals) - np.column_stack(c_forecasts) # Residuals n_calib*H
        self.non_conform = np.abs(self.resid) # non-conformity scores
        self.c_actuals = np.column_stack(c_actuals)
        self.c_forecasts = np.column_stack(c_forecasts)

    def calculate_quantile(self, scores_calib):
        # Vectorized quantile calculation for list delta
        if isinstance(self.delta, float):
            which_quantile = np.ceil(self.delta * (self.n_calib + 1)) / self.n_calib
            return np.quantile(scores_calib, which_quantile, method="lower", axis=0)
        elif isinstance(self.delta, list):
            which_quantiles = np.ceil(np.array(self.delta) * (self.n_calib + 1)) / self.n_calib
            return np.array([np.quantile(scores_calib, q, method="lower", axis=0) for q in which_quantiles])
        else:
            raise ValueError("delta must be float or list of floats.")
    
    def calibrate(self, series, delta = 0.5):
        """
        Calibrate the conformal model using the calibration dataset.
        Args:
            series: pd.Series or np.array containing the calibration data.
            delta (float or list): Significance level(s) for the prediction intervals.
        """
        self.delta = delta
        self.non_conformity_scores(series)
        h_quantiles = []
        for i in range(self.H):
            q_hat = self.calculate_quantile(self.non_conform[i])
            h_quantiles.append(q_hat)
        self.q_hat_D = np.array(h_quantiles)

    # Generate prediction intervals using the calibrated quantiles

    def generate_prediction_intervals(self, series):
        '''
        Generate conformal prediction intervals for the forecasted values.
        Args:
            series: pd.Series or np.array containing the training data.
        '''
        if not hasattr(self, 'q_hat_D'):
            raise RuntimeError("Conformalizer must be calibrated before generating prediction intervals. Run .calibrate(df_calibration) first.")
    
        y_forecast = np.array(self.naive(series, self.H, season_period=self.season_period))
        result = [y_forecast]
        col_names = ["point_forecast"]

        if isinstance(self.delta, float):
            y_lower, y_upper = y_forecast - self.q_hat_D, y_forecast + self.q_hat_D
            result.extend([y_lower, y_upper])
            col_names.extend([f'lower_{int(self.delta*100)}', f'upper_{int(self.delta*100)}'])
        elif isinstance(self.delta, list):
            for idx, d in enumerate(self.delta):
                y_lower = y_forecast - self.q_hat_D[:, idx]
                y_upper = y_forecast + self.q_hat_D[:, idx]
                result.extend([y_lower, y_upper])
                col_names.extend([f'lower_{int(d*100)}', f'upper_{int(d*100)}'])
        # distributions for each horizons. So add y_forecast array to each columns of self.resid and equal to self.dist
        dist = y_forecast[:, None] + self.resid
        self.dist = pd.DataFrame(dist.T, columns=[f'h_{i+1}' for i in range(self.H)])
        self.dist = self.dist.clip(lower=0.1)
        return pd.DataFrame(np.column_stack(result), columns=col_names)

    def conformal_quantiles(self, series, quantiles):
        """
        Generate conformal quantiles for future time steps.
        Args:
            series: pd.Series or np.array containing the training data.
            quantiles (float or list): Quantiles to be calculated (e.g., 0.1, 0.5, 0.9).
            future_exog (pd.DataFrame, optional): Future exogenous variables for forecasting.
        Returns:
            pd.DataFrame: DataFrame containing point forecasts and conformal quantiles.
        """
        if not hasattr(self, 'q_hat_D'):
            raise RuntimeError("Conformalizer must be calibrated before generating prediction intervals. Run .calibrate(df_calibration) first.")

        y_forecast = np.array(self.naive(series, self.H, season_period=self.season_period))

        return get_conformal_quantiles(self.non_conform, self.n_calib, quantiles, y_forecast)

    def bootstrap(self, series, samples=1000, approximate="kde"):
        """
        Generate samples from the predictive distribution generated by residuals from conformal prediction.
        The samples are drawn from a Gaussian kernel density estimate of the residuals.
        """
        # Return a random sample from Gaussian Kernel density estimation
        if not hasattr(self, 'resid'):
            raise RuntimeError("Residuals not available. Run .non_conformity_scores(df) .calibrate(df) or  first.")

        y_forecast = np.array(self.naive(series, self.H, season_period=self.season_period))

        # ✅ Create a deep copy so that we don’t overwrite self
        new_instance = copy.deepcopy(self)
        if approximate == "kde":
            new_instance.bootstrap_forecasts = np.column_stack([[gaussian_kde(new_instance.resid[i]).resample(size=samples, seed=rng_kde)+y_forecast[i]]
                                        for i in range(new_instance.H)])[0] # H x samples
        elif approximate == "empirical":
            new_instance.bootstrap_forecasts = np.column_stack([np.random.choice(new_instance.resid[i], size=samples, replace=True)+y_forecast[i]
                for i in range(self.H)]).T
        else:
            raise ValueError("approximate must be 'kde' or 'empirical'.")

        new_instance.bootstrap_forecasts_df = pd.DataFrame(new_instance.bootstrap_forecasts.T, columns=[f'h_{i+1}' for i in range(self.H)])
        new_instance.y_forecast_b = y_forecast

        return new_instance

    def bootstrap_quantiles(self, quantiles, bootstrap_method='bootstrap'):
        """
        Generate bootstrap quantiles for future time steps.
        Args:
            quantiles (float or list): Quantiles to be calculated (e.g., 0.1, 0.5, 0.9).
            bootstrap_method (str): The method to use for bootstrapping or simulated correlated_forecasts ('bootstrap' or 'correlated').
        """
        ## Make sure bootstrap() method is called before calling this method
        if bootstrap_method == 'bootstrap':
            if (not hasattr(self, 'bootstrap')):
                raise RuntimeError("Bootstrap samples not available. Run .bootstrap(df) first.")
            return get_bootstrap_quantiles(self.bootstrap_forecasts, self.n_calib, quantiles, self.y_forecast_b)
        elif bootstrap_method == 'correlated':
            if (not hasattr(self, 'simulate_correlated_forecasts')):
                raise RuntimeError("Correlated forecast samples not available. Run .simulate_correlated_forecasts(df) first.")
            return get_bootstrap_quantiles(self.w_samples.T, self.n_calib, quantiles, self.y_forecast_c)

    def simulate_correlated_forecasts(self, series, samples=1000):
        """
        Simulate correlated errors to generate forecasts
        Parameters:
        - df: DataFrame containing the training data
        - samples: number of samples to generate
        - future_exog: optional exogenous variables for forecasting
        Returns:
        - DataFrame containing the simulated forecasts
        """
        if not hasattr(self, 'resid'):
            raise RuntimeError("Residuals not available. Run .non_conformity_scores(df) .calibrate(df) or  first.")

        y_forecast = np.array(self.naive(series, self.H, season_period=self.season_period))

        mu = np.mean(self.resid.T, axis=0)
        sigma_ = np.cov(self.resid.T, rowvar=False, ddof=1)

        rng = np.random.default_rng(seed=42)
        samples = rng.multivariate_normal(mu, sigma_, size=samples)
        self.w_samples = samples + y_forecast
        self.correlated_forecasts = pd.DataFrame(self.w_samples, columns=[f'h_{i+1}' for i in range(self.H)])
        self.y_forecast_c = y_forecast
        return self
    
class arima_prob_forecasts():
    """
    Probabilistic forecasting for ARIMA model. It generates prediction intervals for future time steps and approximates distribution of predictions using Kernel Density Estimation (KDE).
    Parameters:
    - model: ARIMA model instance supported by statsforecast (Nixtla)
    - n_calibration: number of calibration windows
    - H: forecast horizon
    - sliding_window: size of the sliding window for cross-validation
    - verbose: whether to print progress messages
    """
    def __init__(self, model, n_calibration, H, sliding_window=1, verbose=False):
        self.model = model
        self.sliding_window = sliding_window
        self.n_calib = n_calibration
        self.verbose = verbose
        self.H = H

    def non_conformity_scores(self, df):
        c_actuals, c_forecasts = [], []
        # Create time series cross-validator that slides 1 time step for each training window
        tscv = ParametricTimeSeriesSplit(n_splits=self.n_calib, test_size=self.H, step_size=self.sliding_window)
        for train_index, test_index in tscv.split(df):
            train, test = df.iloc[train_index], df.iloc[test_index]
            y_test = np.array(test[self.target_col])
            H_forecasts = self.model.forecast(y=np.array(train[self.target_col]), h=self.H, X=np.array(train.drop(columns=[self.target_col])),
                                X_future=np.array(test.drop(columns=[self.target_col])))["mean"]
            c_forecasts.append(H_forecasts)
            c_actuals.append(y_test)
            if self.verbose:
                print(f"Completed calibration window {len(c_forecasts)} out of {self.n_calib}")
        self.resid = np.column_stack(c_actuals) - np.column_stack(c_forecasts) # Residuals n_calib*H
        self.non_conform = np.abs(self.resid) # non-conformity scores
        self.c_actuals = np.column_stack(c_actuals)
        self.c_forecasts = np.column_stack(c_forecasts)

        
    def calculate_quantile(self, scores_calib):
        # Vectorized quantile calculation for list delta
        if isinstance(self.delta, float):
            which_quantile = np.ceil(self.delta * (self.n_calib + 1)) / self.n_calib
            return np.quantile(scores_calib, which_quantile, method="lower", axis=0)
        elif isinstance(self.delta, list):
            which_quantiles = np.ceil(np.array(self.delta) * (self.n_calib + 1)) / self.n_calib
            return np.array([np.quantile(scores_calib, q, method="lower", axis=0) for q in which_quantiles])
        else:
            raise ValueError("delta must be float or list of floats.")
    
    def calibrate(self, df, target_col, delta = 0.5):
        """
        Calibrate the conformal model using the calibration dataset.
        Args:
            df (pd.DataFrame): DataFrame containing the calibration data.
            target_col (str): Name of the target column in the DataFrame.
            delta (float or list): Significance level(s) for the prediction intervals.
        """
        self.delta = delta
        self.target_col = target_col
        self.non_conformity_scores(df=df)
        h_quantiles = []
        for i in range(self.H):
            q_hat = self.calculate_quantile(self.non_conform[i])
            h_quantiles.append(q_hat)
        self.q_hat_D = np.array(h_quantiles)

    # Generate prediction intervals using the calibrated quantiles

    def generate_prediction_intervals(self, df, future_exog=None):
        '''
        Generate conformal prediction intervals for the forecasted values.
        Args:
            df (pd.DataFrame): DataFrame containing the training data.
            future_exog (pd.DataFrame, optional): Future exogenous variables for forecasting.
        '''
        if not hasattr(self, 'q_hat_D'):
            raise RuntimeError("Conformalizer must be calibrated before generating prediction intervals. Run .calibrate(df_calibration) first.")
        y_train = df[self.target_col]
        x_train = df.drop(columns=[self.target_col])
        if future_exog is not None:
            y_forecast = self.model.forecast(y=np.array(y_train), h=self.H, X=np.array(x_train),
                    X_future=np.array(future_exog))["mean"]
        else:
            y_forecast = self.model.forecast(y=np.array(y_train), h=self.H, X=np.array(x_train))["mean"]
        result = [y_forecast]
        col_names = ["point_forecast"]

        if isinstance(self.delta, float):
            y_lower, y_upper = y_forecast - self.q_hat_D, y_forecast + self.q_hat_D
            result.extend([y_lower, y_upper])
            col_names.extend([f'lower_{int(self.delta*100)}', f'upper_{int(self.delta*100)}'])
        elif isinstance(self.delta, list):
            for idx, d in enumerate(self.delta):
                y_lower = y_forecast - self.q_hat_D[:, idx]
                y_upper = y_forecast + self.q_hat_D[:, idx]
                result.extend([y_lower, y_upper])
                col_names.extend([f'lower_{int(d*100)}', f'upper_{int(d*100)}'])
        # distributions for each horizons. So add y_forecast array to each columns of self.resid and equal to self.dist
        dist = y_forecast[:, None] + self.resid
        self.dist = pd.DataFrame(dist.T, columns=[f'h_{i+1}' for i in range(self.H)])
        self.dist = self.dist.clip(lower=0.1)
        return pd.DataFrame(np.column_stack(result), columns=col_names)

    def conformal_quantiles(self, df, quantiles, future_exog=None):
        """
        Generate conformal quantiles for future time steps.
        Args:
            df (pd.DataFrame): DataFrame containing the training data.
            quantiles (float or list): Quantiles to be calculated (e.g., 0.1, 0.5, 0.9).
            future_exog (pd.DataFrame, optional): Future exogenous variables for forecasting.
        Returns:
            pd.DataFrame: DataFrame containing point forecasts and conformal quantiles.
        """
        # Only calibrate if not already done
        if not hasattr(self, 'q_hat_D'):
            raise RuntimeError("Conformalizer must be calibrated before generating prediction intervals. Run .calibrate(df_calibration) first.")
        y_train = df[self.target_col]
        x_train = df.drop(columns=[self.target_col])
        if future_exog is not None:
            y_forecast = self.model.forecast(y=np.array(y_train), h=self.H, X=np.array(x_train),
                    X_future=np.array(future_exog))["mean"]
        else:
            y_forecast = self.model.forecast(y=np.array(y_train), h=self.H, X=np.array(x_train))["mean"]

        return get_conformal_quantiles(self.non_conform, self.n_calib, quantiles, y_forecast)

    def bootstrap(self, df, samples=1000, future_exog=None, approximate="kde"):
        """
        Generate samples from the predictive distribution generated by residuals from conformal prediction.
        The samples are drawn from a Gaussian kernel density estimate of the residuals.
        """
        # Return a random sample from Gaussian Kernel density estimation
        if not hasattr(self, 'resid'):
            raise RuntimeError("Residuals not available. Run .non_conformity_scores(df) .calibrate(df) or  first.")
        y_train = df[self.target_col]
        x_train = df.drop(columns=[self.target_col])
        if future_exog is not None:
            y_forecast = self.model.forecast(y=np.array(y_train), h=self.H, X=np.array(x_train),
                    X_future=np.array(future_exog))["mean"]
        else:
            y_forecast = self.model.forecast(y=np.array(y_train), h=self.H, X=np.array(x_train))["mean"]

        # ✅ Create a deep copy so that we don’t overwrite self
        new_instance = copy.deepcopy(self)
        if approximate == "kde":
            new_instance.bootstrap_forecasts = np.column_stack([[gaussian_kde(new_instance.resid[i]).resample(size=samples, seed=rng_kde)+y_forecast[i]]
                                        for i in range(new_instance.H)])[0] # H x samples
        elif approximate == "empirical":
            new_instance.bootstrap_forecasts = np.column_stack([np.random.choice(new_instance.resid[i], size=samples, replace=True)+y_forecast[i]
                for i in range(self.H)]).T
        else:
            raise ValueError("approximate must be 'kde' or 'empirical'.")

        new_instance.bootstrap_forecasts_df = pd.DataFrame(new_instance.bootstrap_forecasts.T, columns=[f'h_{i+1}' for i in range(self.H)])
        new_instance.y_forecast_b = y_forecast

        return new_instance

    def bootstrap_quantiles(self, quantiles, bootstrap_method='bootstrap'):
        """
        Generate bootstrap quantiles for future time steps.
        Args:
            quantiles (float or list): Quantiles to be calculated (e.g., 0.1, 0.5, 0.9).
            bootstrap_method (str): The method to use for bootstrapping or simulated correlated_forecasts ('bootstrap' or 'correlated').
        """
        ## Make sure bootstrap() method is called before calling this method
        if bootstrap_method == 'bootstrap':
            if (not hasattr(self, 'bootstrap')):
                raise RuntimeError("Bootstrap samples not available. Run .bootstrap(df) first.")
            return get_bootstrap_quantiles(self.bootstrap_forecasts, self.n_calib, quantiles, self.y_forecast_b)
        elif bootstrap_method == 'correlated':
            if (not hasattr(self, 'simulate_correlated_forecasts')):
                raise RuntimeError("Correlated forecast samples not available. Run .simulate_correlated_forecasts(df) first.")
            return get_bootstrap_quantiles(self.w_samples.T, self.n_calib, quantiles, self.y_forecast_c)

    def simulate_correlated_forecasts(self, df, samples=1000, future_exog=None):
        """
        Simulate correlated errors to generate forecasts
        Parameters:
        - df: DataFrame containing the training data
        - samples: number of samples to generate
        - future_exog: optional exogenous variables for forecasting
        Returns:
        - DataFrame containing the simulated forecasts
        """
        if not hasattr(self, 'resid'):
            raise RuntimeError("Residuals not available. Run .non_conformity_scores(df) .calibrate(df) or  first.")
        y_train = df[self.target_col]
        x_train = df.drop(columns=[self.target_col])
        if future_exog is not None:
            y_forecast = self.model.forecast(y=np.array(y_train), h=self.H, X=np.array(x_train),
                    X_future=np.array(future_exog))["mean"]
        else:
            y_forecast = self.model.forecast(y=np.array(y_train), h=self.H, X=np.array(x_train))["mean"]

        mu = np.mean(self.resid.T, axis=0)
        sigma_ = np.cov(self.resid.T, rowvar=False, ddof=1)

        rng = np.random.default_rng(seed=42)
        samples = rng.multivariate_normal(mu, sigma_, size=samples)
        self.w_samples = samples + y_forecast
        self.correlated_forecasts = pd.DataFrame(self.w_samples, columns=[f'h_{i+1}' for i in range(self.H)])
        self.y_forecast_c = y_forecast
        return self
    
class bidirect_ts_conformalizer():
    def __init__(self, delta, train_df, col_index, n_windows, model, H, calib_metric = "mae", model_param=None):
        self.delta = delta
        self.model = model
        self.train_df = train_df
        self.n_windows = n_windows
        self.n_calib = n_windows
        self.H = H
        self.calib_metric = calib_metric
        self.param = model_param
        self.col=col_index
        self.calibrate()
    def backtest(self):
        actuals = []
        predictions = []
        for i in range(self.n_windows):
            x_back = self.train_df[:-self.H-i]
            if i !=0:
                test_y = self.train_df[-self.H-i:-i].iloc[:, self.col]
                if len(self.train_df.columns)>1:
                    test_x = self.train_df[-self.H-i:-i].iloc[:, 2:]
                else:
                    test_x = None
            else:
                test_y = self.train_df[-self.H:].iloc[:, self.col]
                if len(self.train_df.columns)>1:
                    test_x = self.train_df[-self.H:].iloc[:, 2:]
                else:
                    test_x = None
                
#             mod_arima = ARIMA(y_back, exog=x_back, order = (0,1,2), seasonal_order=(0,1,1, 7)).fit()
#             y_pred = mod_arima.forecast(self.H, exog = test_x)
            
            if self.param is not None:
                self.model.fit(x_back, param=self.param)
            else:
                self.model.fit(x_back)
            if test_x is not None:
                forecast = self.model.forecast(self.H, test_x)[self.col]
            else:
                forecast = self.model.forecast(self.H)[self.col]
            
            test_y = np.array(test_y)
            predictions.append(forecast)
            actuals.append(test_y)
            print("model "+str(i+1)+" is completed")
        return np.row_stack(actuals), np.row_stack(predictions)
    
    def calculate_qunatile(self, scores_calib):
        # Calculate the quantile values for each delta and non-conformity scores
        delta_q = []
        for i in self.delta:
            which_quantile = np.ceil((i)*(self.n_calib+1))/self.n_calib
            q_data = np.quantile(scores_calib, which_quantile, method = "lower")
            delta_q.append(q_data)
        self.delta_q = delta_q
        return delta_q
    
    def non_conformity_func(self):
        acts, preds = self.backtest()
        horizon_scores = []
        dists = []
        for i in range(self.H):
            # calculating metrics horizon i
            mae =np.abs(acts[:,i] - preds[:,i]) 
            smape = 2*mae/(np.abs(acts[:,i])+np.abs(preds[:,i]))
            mape = mae/acts[:,i]
            metrics = np.stack((smape,  mape, mae), axis=1)
            horizon_scores.append(metrics)
            dist = 2*acts[:,i] - preds[:,i]
            dists.append(dist)
        self.cp_dist = np.stack(dists).T
        return horizon_scores
    
    
    def calibrate(self):
         # Calibrate the conformalizer to calculate q_hat
        scores_calib = self.non_conformity_func()
        self.q_hat_D = []
        for d in range(len(self.delta)):
            q_hat_H = []
            for i in range(self.H):
                scores_i = scores_calib[i]
                if self.calib_metric == "smape":
                    q_hat = self.calculate_qunatile(scores_i[:, 0])[d]
                elif self.calib_metric == "mape":
                    q_hat = self.calculate_qunatile(scores_i[:, 1])[d]
                elif self.calib_metric == "mae":
                    q_hat = self.calculate_qunatile(scores_i[:, 2])[d]
                else:
                    raise ValueError("not a valid metric")
                q_hat_H.append(q_hat)
            self.q_hat_D.append(q_hat_H)
            
    def forecast(self, X=None):
        if self.param is not None:
            self.model.fit(self.train_df, param=self.param)
        else:
            self.model.fit(self.train_df)
            
        if X is not None:
            y_pred = self.model.forecast(self.H, X)[self.col]
        else:
            y_pred = self.model.forecast(self.H)[self.col]
            
        result = []
        result.append(y_pred)
        for i in range(len(self.delta)):
            if self.calib_metric == "mae":
                y_lower, y_upper = y_pred - np.array(self.q_hat_D[i]).flatten(), y_pred + np.array(self.q_hat_D[i]).flatten()
            elif self.calib_metric == "mape":
                y_lower, y_upper = y_pred/(1+np.array(self.q_hat_D[i]).flatten()), y_pred/(1-np.array(self.q_hat_D[i]).flatten())
            elif self.calib_metric == "smape":
                y_lower = y_pred*(2-np.array(self.q_hat_D[i]).flatten())/(2+np.array(self.q_hat_D[i]).flatten())
                y_upper = y_pred*(2+np.array(self.q_hat_D[i]).flatten())/(2-np.array(self.q_hat_D[i]).flatten())
            else:
                raise ValueError("not a valid metric")
            result.append(y_lower)
            result.append(y_upper)
        CPs = pd.DataFrame(result).T
        CPs.rename(columns = {0:"point_forecast"}, inplace = True)
        for i in range(0, 2*len(self.delta), 2):
            d_index = round(i/2)
            CPs.rename(columns = {i+1:"lower_"+str(round(self.delta[d_index]*100)), i+2:"upper_"+str(round(self.delta[d_index]*100))}, inplace = True)
        return CPs
    
class bag_boost_aggr_conformalizer():
    def __init__(self, delta, train_df, n_windows, models, cat_cols, H, calib_metric = "mae", model_param=None):
        self.delta = delta
        self.models = models
        self.train_df = train_df
        self.cats = cat_cols
        self.n_windows = n_windows
        self.n_calib = n_windows
        self.H = H
        self.calib_metric = calib_metric
        self.param = model_param
        self.cols = [m.target_col for m in self.models]
        
        self.models_f = self.models.copy()
        for i, j in zip(self.cols, range(len(self.cols))):
            train = pd.concat([self.train_df[i], self.train_df[self.cats]], axis=1)
            # self.models[i] = ExponentialSmoothing(self.train[i], **self.params[i][0]).fit(**self.params[i][1])
            self.models_f[j].fit(train, self.param[i])

        
        self.calibrate()
    def backtest(self):
        actuals = []
        predictions = []
        for i in range(self.n_windows):
            sum_forecasts = np.zeros((self.H,))
            sum_actuals = np.zeros((self.H,))
            for m in self.models:
                bactest_df = pd.concat([self.train_df[m.target_col], self.train_df[self.cats]], axis=1)
                x_back = bactest_df[:-self.H-i]
                if i !=0:
                    test_y = bactest_df[-self.H-i:-i][m.target_col]
                    if len(bactest_df.columns)>1:
                        test_x = bactest_df[-self.H-i:-i].iloc[:, 1:]
                    else:
                        test_x = None
                else:
                    test_y = bactest_df[-self.H:][m.target_col]
                    if len(bactest_df.columns)>1:
                        test_x = bactest_df[-self.H:].iloc[:, 1:]
                    else:
                        test_x = None
                    

                
                if self.param is not None:
                    m.fit(x_back, param=self.param[m.target_col])
                else:
                    m.fit(x_back)
                if test_x is not None:
                    forecast = m.forecast(self.H, test_x)
                else:
                    forecast = m.forecast(self.H)
                sum_forecasts +=forecast
                sum_actuals +=np.array(test_y)
            
            predictions.append(sum_forecasts)
            actuals.append(sum_actuals)
            print("model "+str(i+1)+" is completed")
        self.predictions = np.row_stack(predictions)
        self.actuals = np.row_stack(actuals)
        return np.row_stack(actuals), np.row_stack(predictions)
    
    def calculate_qunatile(self, scores_calib):
        # Calculate the quantile values for each delta and non-conformity scores
        delta_q = []
        for i in self.delta:
            which_quantile = np.ceil((i)*(self.n_calib+1))/self.n_calib
            q_data = np.quantile(scores_calib, which_quantile, method = "lower")
            delta_q.append(q_data)
        self.delta_q = delta_q
        return delta_q
    
    def non_conformity_func(self):
        acts, preds = self.backtest()
        horizon_scores = []
        dists = []
        for i in range(self.H):
            # calculating metrics horizon i
            mae =np.abs(acts[:,i] - preds[:,i]) 
            smape = 2*mae/(np.abs(acts[:,i])+np.abs(preds[:,i]))
            mape = mae/acts[:,i]
            metrics = np.stack((smape,  mape, mae), axis=1)
            horizon_scores.append(metrics)
            dist = 2*acts[:,i] - preds[:,i]
            dists.append(dist)
        self.cp_dist = np.stack(dists).T
        return horizon_scores
    
    
    def calibrate(self):
         # Calibrate the conformalizer to calculate q_hat
        scores_calib = self.non_conformity_func()
        self.q_hat_D = []
        for d in range(len(self.delta)):
            q_hat_H = []
            for i in range(self.H):
                scores_i = scores_calib[i]
                if self.calib_metric == "smape":
                    q_hat = self.calculate_qunatile(scores_i[:, 0])[d]
                elif self.calib_metric == "mape":
                    q_hat = self.calculate_qunatile(scores_i[:, 1])[d]
                elif self.calib_metric == "mae":
                    q_hat = self.calculate_qunatile(scores_i[:, 2])[d]
                else:
                    raise ValueError("not a valid metric")
                q_hat_H.append(q_hat)
            self.q_hat_D.append(q_hat_H)
            
    def forecast(self, X=None):

        y_pred = np.zeros((self.H,))
        for f in self.models_f:
            # y_pred += self.models_f[i].forecast(self.H,  x_test= X)
            if X is not None:
                y_pred += f.forecast(self.H,  X)
            else:
                y_pred += f.forecast(self.H)
            
        result = []
        result.append(y_pred)
        for i in range(len(self.delta)):
            if self.calib_metric == "mae":
                y_lower, y_upper = y_pred - np.array(self.q_hat_D[i]).flatten(), y_pred + np.array(self.q_hat_D[i]).flatten()
            elif self.calib_metric == "mape":
                y_lower, y_upper = y_pred/(1+np.array(self.q_hat_D[i]).flatten()), y_pred/(1-np.array(self.q_hat_D[i]).flatten())
            elif self.calib_metric == "smape":
                y_lower = y_pred*(2-np.array(self.q_hat_D[i]).flatten())/(2+np.array(self.q_hat_D[i]).flatten())
                y_upper = y_pred*(2+np.array(self.q_hat_D[i]).flatten())/(2-np.array(self.q_hat_D[i]).flatten())
            else:
                raise ValueError("not a valid metric")
            result.append(y_lower)
            result.append(y_upper)
        CPs = pd.DataFrame(result).T
        CPs.rename(columns = {0:"point_forecast"}, inplace = True)
        for i in range(0, 2*len(self.delta), 2):
            d_index = round(i/2)
            CPs.rename(columns = {i+1:"lower_"+str(round(self.delta[d_index]*100)), i+2:"upper_"+str(round(self.delta[d_index]*100))}, inplace = True)
        return CPs
    
class bidirect_aggr_conformalizer():
    def __init__(self, delta, train_df, col_index, n_windows, models, cat_cols, H, calib_metric = "mae", model_param=None):
        self.delta = delta
        self.models = models
        self.cats = cat_cols
        self.idx = col_index
        self.train_df = train_df
        self.n_windows = n_windows
        self.n_calib = n_windows
        self.H = H
        self.calib_metric = calib_metric
        self.param = model_param
        # self.col=col_index
        
        self.models_f = self.models.copy()
        for f, j in zip(self.models_f, range(len(self.models_f))):
            train = pd.concat([self.train_df[f.target_col], self.train_df[self.cats]], axis=1)
            # self.models[i] = ExponentialSmoothing(self.train[i], **self.params[i][0]).fit(**self.params[i][1])
            self.models_f[j].fit(train, self.param[f.target_col[self.idx]])
        self.calibrate()
    def backtest(self):
        actuals = []
        predictions = []
        for i in range(self.n_windows):
            sum_forecasts = np.zeros((self.H,))
            sum_actuals = np.zeros((self.H,))
            for m in self.models:
                bactest_df = pd.concat([self.train_df[m.target_col], self.train_df[self.cats]], axis=1)
                x_back = bactest_df[:-self.H-i]
                if i !=0:
                    test_y = bactest_df[-self.H-i:-i][m.target_col[self.idx]]
                    if len(bactest_df.columns)>1:
                        test_x = bactest_df[-self.H-i:-i].iloc[:, 2:]
                    else:
                        test_x = None
                else:
                    test_y = bactest_df[-self.H:][m.target_col[self.idx]]
                    if len(bactest_df.columns)>1:
                        test_x = bactest_df[-self.H:].iloc[:, 2:]
                    else:
                        test_x = None
                    
    #             mod_arima = ARIMA(y_back, exog=x_back, order = (0,1,2), seasonal_order=(0,1,1, 7)).fit()
    #             y_pred = mod_arima.forecast(self.H, exog = test_x)
                
                if self.param is not None:
                    m.fit(x_back, param=self.param[m.target_col[self.idx]])
                else:
                    m.fit(x_back)
                if test_x is not None:
                    forecast = m.forecast(self.H, test_x)[self.idx]
                else:
                    forecast = m.forecast(self.H)[self.idx]
                sum_forecasts +=forecast
                sum_actuals +=np.array(test_y)
            
            predictions.append(sum_forecasts)
            actuals.append(sum_actuals)
            print("model "+str(i+1)+" is completed")
        self.predictions = np.row_stack(predictions)
        self.actuals = np.row_stack(actuals)
        return np.row_stack(actuals), np.row_stack(predictions)
    
    def calculate_qunatile(self, scores_calib):
        # Calculate the quantile values for each delta and non-conformity scores
        delta_q = []
        for i in self.delta:
            which_quantile = np.ceil((i)*(self.n_calib+1))/self.n_calib
            q_data = np.quantile(scores_calib, which_quantile, method = "lower")
            delta_q.append(q_data)
        self.delta_q = delta_q
        return delta_q
    
    def non_conformity_func(self):
        acts, preds = self.backtest()
        horizon_scores = []
        dists = []
        for i in range(self.H):
            # calculating metrics horizon i
            mae =np.abs(acts[:,i] - preds[:,i]) 
            smape = 2*mae/(np.abs(acts[:,i])+np.abs(preds[:,i]))
            mape = mae/acts[:,i]
            metrics = np.stack((smape,  mape, mae), axis=1)
            horizon_scores.append(metrics)
            dist = 2*acts[:,i] - preds[:,i]
            dists.append(dist)
        self.cp_dist = np.stack(dists).T
        return horizon_scores
    
    
    def calibrate(self):
         # Calibrate the conformalizer to calculate q_hat
        scores_calib = self.non_conformity_func()
        self.q_hat_D = []
        for d in range(len(self.delta)):
            q_hat_H = []
            for i in range(self.H):
                scores_i = scores_calib[i]
                if self.calib_metric == "smape":
                    q_hat = self.calculate_qunatile(scores_i[:, 0])[d]
                elif self.calib_metric == "mape":
                    q_hat = self.calculate_qunatile(scores_i[:, 1])[d]
                elif self.calib_metric == "mae":
                    q_hat = self.calculate_qunatile(scores_i[:, 2])[d]
                else:
                    raise ValueError("not a valid metric")
                q_hat_H.append(q_hat)
            self.q_hat_D.append(q_hat_H)
            
    def forecast(self, X=None):

        y_pred = np.zeros((self.H,))
        for f in self.models_f:
            # y_pred += self.models_f[i].forecast(self.H,  x_test= X)
            if X is not None:
                y_pred += f.forecast(self.H,  x_test= X)[self.idx]
            else:
                y_pred += f.forecast(self.H)[self.idx]
            
        result = []
        result.append(y_pred)
        for i in range(len(self.delta)):
            if self.calib_metric == "mae":
                y_lower, y_upper = y_pred - np.array(self.q_hat_D[i]).flatten(), y_pred + np.array(self.q_hat_D[i]).flatten()
            elif self.calib_metric == "mape":
                y_lower, y_upper = y_pred/(1+np.array(self.q_hat_D[i]).flatten()), y_pred/(1-np.array(self.q_hat_D[i]).flatten())
            elif self.calib_metric == "smape":
                y_lower = y_pred*(2-np.array(self.q_hat_D[i]).flatten())/(2+np.array(self.q_hat_D[i]).flatten())
                y_upper = y_pred*(2+np.array(self.q_hat_D[i]).flatten())/(2-np.array(self.q_hat_D[i]).flatten())
            else:
                raise ValueError("not a valid metric")
            result.append(y_lower)
            result.append(y_upper)
        CPs = pd.DataFrame(result).T
        CPs.rename(columns = {0:"point_forecast"}, inplace = True)
        for i in range(0, 2*len(self.delta), 2):
            d_index = round(i/2)
            CPs.rename(columns = {i+1:"lower_"+str(round(self.delta[d_index]*100)), i+2:"upper_"+str(round(self.delta[d_index]*100))}, inplace = True)
        return CPs

class ets_aggr_conformalizer():
    def __init__(self, train_data, tar_cols, delta, params, n_windows, H, calib_metric = "mae"):
        self.delta = delta
        self.train = train_data
        self.n_windows = n_windows
        self.n_calib = n_windows
        self.H = H
        self.cols = tar_cols
        self.params = params
        self.calib_metric = calib_metric
        self.models = {}
        for i in self.cols:
            self.models[i] = ExponentialSmoothing(self.train[i], **self.params[i][0]).fit(**self.params[i][1])
        self.calibrate()
    def backtest(self):
        #making H-step-ahead forecast n_windows times for each 1-step backward sliding window.
        # We can the think of n_windows as the size of calibration set for each H horizon 
        actuals = []
        predictions = []
        for i in range(self.n_windows):
            sum_forecasts = np.zeros((self.H,))
            sum_actuals = np.zeros((self.H,))
            for j in self.cols:
                y_train = self.train[:-self.H-i][j]
                if i !=0:
                    y_test = self.train[-self.H-i:-i][j]
                else:
                    y_test = self.train[-self.H:][j]
    
                model_ets = ExponentialSmoothing(y_train, **self.params[j][0])
                fit_ets = model_ets.fit(**self.params[j][1])
        
                y_pred = np.array(fit_ets.forecast(self.H))
                sum_forecasts +=y_pred
                sum_actuals +=np.array(y_test)
            
            predictions.append(sum_forecasts)
            actuals.append(sum_actuals)
            print("model "+str(i+1)+" is completed")
        self.predictions = np.row_stack(predictions)
        self.actuals = np.row_stack(actuals)
        return np.row_stack(actuals), np.row_stack(predictions)

    
    def calculate_qunatile(self, scores_calib):
        # Calculate the quantile values for each delta value
        delta_q = []
        for i in self.delta:
            which_quantile = np.ceil((i)*(self.n_calib+1))/self.n_calib
            q_data = np.quantile(scores_calib, which_quantile, method = "lower")
            delta_q.append(q_data)
        self.delta_q = delta_q
        return delta_q
    
    def non_conformity_func(self):
        #Calculate non-conformity scores (mae, smape and mape for now) for each forecasted horizon
        acts, preds = self.backtest()
        horizon_scores = []
        dists = []
        for i in range(self.H):
            mae =np.abs(acts[:,i] - preds[:,i])
            smape = 2*mae/(np.abs(acts[:,i])+np.abs(preds[:,i]))
            mape = mae/acts[:,i]
            metrics = np.stack((smape,  mape, mae), axis=1)
            horizon_scores.append(metrics)
            dist = 2*acts[:,i] - preds[:,i]
            dists.append(dist)
        self.cp_dist = np.stack(dists).T
        return horizon_scores
    
    
    def calibrate(self):
         # Calibrate the conformalizer to calculate q_hat for all given delta values
        scores_calib = self.non_conformity_func()
        self.q_hat_D = []
        for d in range(len(self.delta)):
            q_hat_H = []
            for i in range(self.H):
                scores_i = scores_calib[i]
                if self.calib_metric == "smape":
                    qhat = self.calculate_qunatile(scores_i[:, 0])[d]
                elif self.calib_metric == "mape":
                    qhat = self.calculate_qunatile(scores_i[:, 1])[d]
                elif self.calib_metric == "mae":
                    q_hat = self.calculate_qunatile(scores_i[:, 2])[d]
                else:
                    raise ValueError("not a valid metric")
                q_hat_H.append(q_hat)
            self.q_hat_D.append(q_hat_H)
            
    def forecast(self):
        y_pred = np.zeros((self.H,))
        for i in self.cols:
            y_pred += np.array(self.models[i].forecast(self.H))

        result = []
        result.append(y_pred)
        #Calculate the prediction intervals given the calibration metric used for non-conformity score
        for i in range(len(self.delta)):
            if self.calib_metric == "mae":
                y_lower, y_upper = y_pred - np.array(self.q_hat_D[i]).flatten(), y_pred + np.array(self.q_hat_D[i]).flatten()
            elif self.calib_metric == "mape":
                y_lower, y_upper = y_pred/(1+np.array(self.q_hat_D[i]).flatten()), y_pred/(1-np.array(self.q_hat_D[i]).flatten())
            elif self.calib_metric == "smape":
                y_lower = y_pred*(2-np.array(self.q_hat_D[i]).flatten())/(2+np.array(self.q_hat_D[i]).flatten())
                y_upper = y_pred*(2+np.array(self.q_hat_D[i]).flatten())/(2-np.array(self.q_hat_D[i]).flatten())
            else:
                raise ValueError("not a valid metric")
            result.append(y_lower)
            result.append(y_upper)
        CPs = pd.DataFrame(result).T
        CPs.rename(columns = {0:"point_forecast"}, inplace = True)
        for i in range(0, 2*len(self.delta), 2):
            d_index = round(i/2)
            CPs.rename(columns = {i+1:"lower_"+str(round(self.delta[d_index]*100)), i+2:"upper_"+str(round(self.delta[d_index]*100))}, inplace = True)
        return CPs
    
class s_arima_aggr_conformalizer():
    def __init__(self, train_data, tar_cols, orders, delta, n_windows, H, exog = None, calib_metric = "mae"):
        self.delta = delta
        self.orders = orders
        self.cols = tar_cols
        self.train = train_data
        self.exog = exog
        self.n_windows = n_windows
        self.n_calib = n_windows
        self.H = H
        self.calib_metric = calib_metric

        self.models = {}
        for i in self.cols:
            # self.models[i] = ExponentialSmoothing(self.train[i], **self.params[i][0]).fit(**self.params[i][1])
            self.models[i] = ARIMA(order=(self.orders[i][0],self.orders[i][5],self.orders[i][1]),
                                   seasonal_order=(self.orders[i][2],self.orders[i][6],self.orders[i][3]),
                                   season_length=self.orders[i][4]).fit(y=np.array(self.train[i]), X=np.array(self.exog, dtype = np.float64))

            # y_pred = arima_m.forecast(y=y_train, h=test_size, X=x_train, X_future=x_test)["mean"]
            # self.models[i] = self.models[i].fit(y=y_train, X=x_train)
            
        self.calibrate()
    def backtest(self):
        #making H-step-ahead forecast n_windows times for each 1-step backward sliding window.
        # We can the think of n_windows as the size of calibration set for each H horizon 
        actuals = []
        predictions = []
        for i in range(self.n_windows):
            sum_forecasts = np.zeros((self.H,))
            sum_actuals = np.zeros((self.H,))
            for j in self.cols:
                y_back = np.array(self.train[:-self.H-i][j])
                if self.exog is not None:
                    x_back = np.array(self.exog[:-self.H-i], dtype = np.float64)
                else:
                    x_back = None
                if i !=0:
                    test_y = np.array(self.train[-self.H-i:-i][j])
                    if self.exog is not None:
                        test_x = np.array(self.exog[-self.H-i:-i], np.float64)
                    else:
                        test_x = None
                else:
                    test_y = np.array(self.train[-self.H:][j])
                    if self.exog is not None:
                        test_x = np.array(self.exog[-self.H:], np.float64)
                    else:
                        test_x = None
                    
                # mod_arima = self.model(y_back, exog=x_back, order = self.order, seasonal_order=self.S_order).fit()
                mod_arima = ARIMA(order=(self.orders[j][0],self.orders[j][5],self.orders[j][1]),
                                   seasonal_order=(self.orders[j][2],self.orders[j][6],self.orders[j][3]),
                                   season_length=self.orders[j][4]).fit(y=y_back, X=x_back)
                # y_pred = mod_arima.forecast(self.H, exog = test_x)
                y_pred = mod_arima.predict(self.H,  X=test_x)["mean"]
                sum_forecasts +=y_pred
                sum_actuals +=test_y
            
            predictions.append(sum_forecasts)
            actuals.append(sum_actuals)
            print("model "+str(i+1)+" is completed")
        self.predictions = np.row_stack(predictions)
        self.actuals = np.row_stack(actuals)
        return np.row_stack(actuals), np.row_stack(predictions)
    
    def calculate_qunatile(self, scores_calib):
        # Calculate the quantile values for each delta value
        delta_q = []
        for i in self.delta:
            which_quantile = np.ceil((i)*(self.n_calib+1))/self.n_calib
            q_data = np.quantile(scores_calib, which_quantile, method = "lower")
            delta_q.append(q_data)
        self.delta_q = delta_q
        return delta_q
    
    def non_conformity_func(self):
        #Calculate non-conformity scores (mae, smape and mape for now) for each forecasted horizon
        acts, preds = self.backtest()
        horizon_scores = []
        dists = []
        for i in range(self.H):
            mae =np.abs(acts[:,i] - preds[:,i])
            smape = 2*mae/(np.abs(acts[:,i])+np.abs(preds[:,i]))
            mape = mae/acts[:,i]
            metrics = np.stack((smape,  mape, mae), axis=1)
            horizon_scores.append(metrics)
            dist = 2*acts[:,i] - preds[:,i]
            dists.append(dist)
        self.cp_dist = np.stack(dists).T
        return horizon_scores
    
    
    def calibrate(self):
         # Calibrate the conformalizer to calculate q_hat for all given delta values
        scores_calib = self.non_conformity_func()
        self.q_hat_D = []
        for d in range(len(self.delta)):
            q_hat_H = []
            for i in range(self.H):
                scores_i = scores_calib[i]
                if self.calib_metric == "smape":
                    qhat = self.calculate_qunatile(scores_i[:, 0])[d]
                elif self.calib_metric == "mape":
                    qhat = self.calculate_qunatile(scores_i[:, 1])[d]
                elif self.calib_metric == "mae":
                    q_hat = self.calculate_qunatile(scores_i[:, 2])[d]
                else:
                    raise ValueError("not a valid metric")
                q_hat_H.append(q_hat)
            self.q_hat_D.append(q_hat_H)
            
    def forecast(self, exog = None):
        y_pred = np.zeros((self.H,))
        for i in self.cols:
            y_pred += self.models[i].predict(self.H,  X=np.array(exog, np.float64))["mean"]
            # y_pred += np.array(self.models[i].forecast(self.H))

        result = []
        result.append(y_pred)
        #Calculate the prediction intervals given the calibration metric used for non-conformity score
        for i in range(len(self.delta)):
            if self.calib_metric == "mae":
                y_lower, y_upper = y_pred - np.array(self.q_hat_D[i]).flatten(), y_pred + np.array(self.q_hat_D[i]).flatten()
            elif self.calib_metric == "mape":
                y_lower, y_upper = y_pred/(1+np.array(self.q_hat_D[i]).flatten()), y_pred/(1-np.array(self.q_hat_D[i]).flatten())
            elif self.calib_metric == "smape":
                y_lower = y_pred*(2-np.array(self.q_hat_D[i]).flatten())/(2+np.array(self.q_hat_D[i]).flatten())
                y_upper = y_pred*(2+np.array(self.q_hat_D[i]).flatten())/(2-np.array(self.q_hat_D[i]).flatten())
            else:
                raise ValueError("not a valid metric")
            result.append(y_lower)
            result.append(y_upper)
        CPs = pd.DataFrame(result).T
        CPs.rename(columns = {0:"point_forecast"}, inplace = True)
        for i in range(0, 2*len(self.delta), 2):
            d_index = round(i/2)
            CPs.rename(columns = {i+1:"lower_"+str(round(self.delta[d_index]*100)), i+2:"upper_"+str(round(self.delta[d_index]*100))}, inplace = True)
        return CPs
    
class var_aggr_conformalizer():
    def __init__(self, train_data, exogs, tar_columns, delta, n_windows, H, col_index, lags, diffs, calib_metric = "mae"):
        self.delta = delta
        self.lag_order = lags
        self.train = train_data
        self.exogs = exogs
        self.cols = tar_columns
        self.n_windows = n_windows
        self.n_calib = n_windows
        self.H = H
        self.idx = col_index
        self.diffs = diffs
        self.calib_metric = calib_metric
        self.models = {}
        self.last_vals = {i: [] for i in tar_columns}
        for i in self.cols:
            # self.models[i] = ExponentialSmoothing(self.train[i], **self.params[i][0]).fit(**self.params[i][1])
            tar_col = self.train.columns[self.train.columns.str.contains(i)].tolist()
            dfi = self.train[tar_col]
            if self.diffs[i][0] ==1:
                dfi[tar_col[0]] = dfi[tar_col[0]].diff()
                self.last_vals[i].append(self.train[tar_col[0]][-1])
            else:
                self.last_vals[i].append(None)
            if self.diffs[i][1] ==1:
                dfi[tar_col[1]] = dfi[tar_col[1]].diff()
                self.last_vals[i].append(self.train[tar_col[1]][-1])
            else:
                self.last_vals[i].append(None)
                
            if (self.diffs[i][0] ==1) | (self.diffs[i][1] ==1):
                exog_f = exogs[1:]
            else:
                exog_f = exogs
            dfi = dfi.dropna()

            self.models[i] = VAR(dfi, exog=exog_f).fit(self.lag_order)
        self.calibrate()
    def backtest(self):
        #making H-step-ahead forecast n_windows times for each 1-step backward sliding window.
        # We can the think of n_windows as the size of calibration set for each H horizon 
        actuals = []
        predictions = []
        for i in range(self.n_windows):
            sum_forecasts = np.zeros((self.H,))
            sum_actuals = np.zeros((self.H,))
            for j in self.cols:
                tar_col = self.train.columns[self.train.columns.str.contains(j)].tolist()
                dfj = self.train[tar_col]
                if self.diffs[j][0] ==1:
                    dfj[tar_col[0]] = dfj[tar_col[0]].diff()
                if self.diffs[j][1] ==1:
                    dfj[tar_col[1]] = dfj[tar_col[1]].diff()
                    
                y_back = dfj[:-self.H-i]
                x_back = self.exogs[:-self.H-i]
                
                if i !=0:
                    if self.diffs[j][self.idx] ==1:
                        test_y = np.array(self.train[tar_col[self.idx]])[-self.H-i:-i]
                        last_train = np.array(self.train[tar_col[self.idx]])[:-self.H-i][-1]
                    else:
                        test_y = np.array(self.train[tar_col[self.idx]])[-self.H-i:-i]
                   
                    test_x = self.exogs[-self.H-i:-i]

                else:
                    if self.diffs[j][self.idx] ==1:
                        test_y = np.array(self.train[tar_col[self.idx]])[-self.H:]
                        last_train = np.array(self.train[tar_col[self.idx]])[:-self.H-i][-1]
                    else:
                        test_y = np.array(self.train[tar_col[self.idx]])[-self.H:]
                    test_x = self.exogs[-self.H:]
                
                y_back = y_back.dropna()
                # y_back[self.idx].fillna(y_back[self.idx][17:33].mean())
                # y_back = y_back.fillna(y_back.rolling(window=16, min_periods=1).mean().shift(-16 + 1)

                if (self.diffs[j][0] ==1) | (self.diffs[j][1] ==1):
                    x_back = x_back[1:]
                    
                var_result = VAR(y_back, exog=x_back).fit(self.lag_order)
             
                y_pred = var_result.forecast(y = y_back.values[-self.lag_order:], steps = self.H, exog_future = np.array(test_x))[:, self.idx]
                
                if self.diffs[j][self.idx] ==1:
                    pred_dif = np.insert(y_pred, 0, last_train)
                    sum_forecasts += np.cumsum(pred_dif)[-self.H:]
                else:
                    sum_forecasts += y_pred
                    
                sum_forecasts +=y_pred
                sum_actuals +=test_y
            
            predictions.append(sum_forecasts)
            actuals.append(sum_actuals)
            print("model "+str(i+1)+" is completed")
        self.predictions = np.row_stack(predictions)
        self.actuals = np.row_stack(actuals)
        return np.row_stack(actuals), np.row_stack(predictions)
    
    def calculate_qunatile(self, scores_calib):
        # Calculate the quantile values for each delta value
        delta_q = []
        for i in self.delta:
            which_quantile = np.ceil((i)*(self.n_calib+1))/self.n_calib
            q_data = np.quantile(scores_calib, which_quantile, method = "lower")
            delta_q.append(q_data)
        self.delta_q = delta_q
        return delta_q
    
    def non_conformity_func(self):
        #Calculate non-conformity scores (mae, smape and mape for now) for each forecasted horizon
        acts, preds = self.backtest()
        horizon_scores = []
        dists = []
        for i in range(self.H):
            mae =np.abs(acts[:,i] - preds[:,i])
            smape = 2*mae/(np.abs(acts[:,i])+np.abs(preds[:,i]))
            mape = mae/acts[:,i]
            metrics = np.stack((smape,  mape, mae), axis=1)
            horizon_scores.append(metrics)
            dist = 2*acts[:,i] - preds[:,i]
            dists.append(dist)
        self.cp_dist = np.stack(dists).T
        return horizon_scores
    
    
    def calibrate(self):
         # Calibrate the conformalizer to calculate q_hat for all given delta values
        scores_calib = self.non_conformity_func()
        self.q_hat_D = []
        for d in range(len(self.delta)):
            q_hat_H = []
            for i in range(self.H):
                scores_i = scores_calib[i]
                if self.calib_metric == "smape":
                    qhat = self.calculate_qunatile(scores_i[:, 0])[d]
                elif self.calib_metric == "mape":
                    qhat = self.calculate_qunatile(scores_i[:, 1])[d]
                elif self.calib_metric == "mae":
                    q_hat = self.calculate_qunatile(scores_i[:, 2])[d]
                else:
                    raise ValueError("not a valid metric")
                q_hat_H.append(q_hat)
            self.q_hat_D.append(q_hat_H)
            
    def forecast(self, X = None):
        y_pred = np.zeros((self.H,))
        for i in self.cols:
            y_train = self.models[i].endog
            # x_train = self.models[i].exog
            fore_var = self.models[i].forecast(y = y_train[-self.lag_order:], steps = self.H, exog_future = np.array(X))[:, self.idx]
            if self.diffs[i][self.idx] ==1:
                last_origin = self.last_vals[i][self.idx]
                add_orig = np.insert(fore_var, 0, last_origin)
                y_pred += np.cumsum(add_orig)[-self.H:]
            else:
                y_pred += fore_var

        result = []
        result.append(y_pred)
        #Calculate the prediction intervals given the calibration metric used for non-conformity score
        for i in range(len(self.delta)):
            if self.calib_metric == "mae":
                y_lower, y_upper = y_pred - np.array(self.q_hat_D[i]).flatten(), y_pred + np.array(self.q_hat_D[i]).flatten()
            elif self.calib_metric == "mape":
                y_lower, y_upper = y_pred/(1+np.array(self.q_hat_D[i]).flatten()), y_pred/(1-np.array(self.q_hat_D[i]).flatten())
            elif self.calib_metric == "smape":
                y_lower = y_pred*(2-np.array(self.q_hat_D[i]).flatten())/(2+np.array(self.q_hat_D[i]).flatten())
                y_upper = y_pred*(2+np.array(self.q_hat_D[i]).flatten())/(2-np.array(self.q_hat_D[i]).flatten())
            else:
                raise ValueError("not a valid metric")
            result.append(y_lower)
            result.append(y_upper)
        CPs = pd.DataFrame(result).T
        CPs.rename(columns = {0:"point_forecast"}, inplace = True)
        for i in range(0, 2*len(self.delta), 2):
            d_index = round(i/2)
            CPs.rename(columns = {i+1:"lower_"+str(round(self.delta[d_index]*100)), i+2:"upper_"+str(round(self.delta[d_index]*100))}, inplace = True)
        return CPs