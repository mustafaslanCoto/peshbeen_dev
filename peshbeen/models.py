#!/usr/bin/env python3
"""
ML Forecasting Package
=======================

This module contains various forecasting classes (using CatBoost, LightGBM, XGBoost,
RandomForest, etc.) and utility functions for cross-validation and hyperparameter
tuning for time-series forecasting.
"""
from typing import List, Dict, Optional, Callable, Tuple, Any, Union
from sklearn.base import clone
from tabnanny import verbose
import numpy as np
import pandas as pd
import copy
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.model_selection import TimeSeriesSplit, KFold
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, HistGradientBoostingRegressor
from peshbeen.transformations import (box_cox_transform, back_box_cox_transform, undiff_ts, seasonal_diff,
                        invert_seasonal_diff, kfold_target_encoder, target_encoder_for_test,
                        rolling_quantile, rolling_mean, rolling_std,
                        expanding_mean, expanding_std, expanding_quantile)
from peshbeen.model_selection import ParametricTimeSeriesSplit
from peshbeen.stattools import lr_trend_model, forecast_trend
from catboost import CatBoostRegressor
from cubist import Cubist
# dot not show warnings
import warnings
warnings.filterwarnings("ignore")
import copy
import statsmodels.api as sm
from scipy.stats import norm, multivariate_normal
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy.special import logsumexp


class ml_forecaster:
    """
    ml Forecaster for time series forecasting.

    Args:
        model (class): Machine learning model class (e.g., CatBoostRegressor, LGBMRegressor).
        target_col (str): Name of the target variable.
        cat_variables (list, optional): List of categorical features.
        target_encode (bool, optional): Whether to use target encoding for categorical features. Default is False.
        lags (list or int, optional): Lag(s) to include as features.
        difference (int, optional): Order of difference (e.g. 1 for first difference).
        seasonal_length (int, optional): Seasonal period for seasonal differencing.
        trend (bool, optional): Whether to remove trend.
        ets_params (tuple, optional): A tuple (model_params, fit_params) for exponential smoothing. Ex.g. ({'trend': 'add', 'seasonal': 'add'}, {'damped_trend': True}).
        change_points (list, optional): List of change points for piecewise regression if trend is passed as : "linear".
        box_cox (bool, optional): Whether to perform a Box–Cox transformation.
        box_cox_lmda (float, optional): The lambda value for Box–Cox.
        box_cox_biasadj (bool, optional): If True, adjust bias after Box–Cox inversion. Default is False.
        lag_transform (list, optional): List specifying additional lag transformations.
    """
    def __init__(self, model, target_col, cat_variables=None, target_encode=False, lags=None, difference=None, seasonal_diff=None,
                 trend=None, ets_params=None, change_points=None, box_cox=False, box_cox_lmda=None,
                 box_cox_biasadj=False, lag_transform=None):

        self.target_col = target_col
        self.cat_variables = cat_variables
        self.target_encode = target_encode  # whether to use target encoding for categorical variables
        self.n_lag = lags
        if self.n_lag is not None:
            if isinstance(self.n_lag, int):
                self.n_lag = list(range(1, self.n_lag + 1))  # convert to list if single integer
            elif isinstance(self.n_lag, list):
                if not all(isinstance(l, int) for l in self.n_lag):
                    raise TypeError("n_lag list must contain only integers")
                # No assignment needed here, already a list
            else:
                raise TypeError("n_lag must be an integer or a list of integers")
        self.difference = difference
        self.season_diff = seasonal_diff
        self.trend = trend
        if ets_params is not None:
            self.ets_model = ets_params[0]
            self.ets_fit = ets_params[1]
        else:
            self.ets_model = None
            self.ets_fit = None
        self.cps = change_points
        self.box_cox = box_cox
        self.lamda = box_cox_lmda
        self.biasadj = box_cox_biasadj
        self.lag_transform = lag_transform
        
        # Set default tuned parameters and placeholders for fitted attributes
        self.tuned_params = None
        self.actuals = None
        self.prob_forecasts = None
        self.model = model  # the chosen ML model
    
    def data_prep(self, df):
        """
        Prepare the data with lag features, differencing, trend-removal, and Box–Cox transformation.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Dataframe ready for model fitting.
        """
        dfc = df.copy()

        # Process categorical variables if provided

        if self.cat_variables is not None:
            if self.target_encode ==True:
                for col in self.cat_variables:
                    encode_col = col+"_target_encoded"
                    dfc[encode_col] = kfold_target_encoder(dfc, col, self.target_col, 36)
                self.df_encode = dfc.copy()
                dfc = dfc.drop(columns = self.cat_variables)
                # If target encoding is not used, convert categories to dummies    
            else:
                if isinstance(self.model, (CatBoostRegressor, LGBMRegressor)):
                    for col in self.cat_variables:
                        dfc[col] = dfc[col].astype('category')

                else:
                    for col, cat in self.cat_var.items():
                        dfc[col] = dfc[col].astype('category')
                        # Set categories for categorical columns
                        dfc[col] = dfc[col].cat.set_categories(cat)
                    dfc = pd.get_dummies(dfc)
                    if isinstance(self.model, (LinearRegression, Ridge, Lasso, ElasticNet)):
                        for i in self.drop_categ:
                            dfc.drop(list(dfc.filter(regex=i)), axis=1, inplace=True)

        
        if self.target_col in dfc.columns:
            # Apply Box–Cox transformation if specified
            if self.box_cox:
                self.is_zero = np.any(np.array(dfc[self.target_col]) < 1) # check for zero or negative values
                trans_data, self.lamda = box_cox_transform(x=dfc[self.target_col],
                                                        shift=self.is_zero,
                                                        box_cox_lmda=self.lamda)
                dfc[self.target_col] = trans_data
            # Detrend the series if specified
            if self.trend is not None:
                self.len = len(df)
                self.target_orig = dfc[self.target_col] # Store original values for later use during forecasting
                if self.trend in ["linear", "feature_lr"]:
                    # self.lr_model = LinearRegression().fit(np.arange(self.len).reshape(-1, 1), dfc[self.target_col])
                    if self.cps is not None:
                        trend, self.lr_model = lr_trend_model(dfc[self.target_col], breakpoints=self.cps, type='piecewise')
                    else:
                        trend, self.lr_model = lr_trend_model(dfc[self.target_col])
                    if self.trend == "linear":
                        dfc[self.target_col] = dfc[self.target_col] - trend
                if self.trend in ["ets", "feature_ets"]:
                    self.ets_model_fit = ExponentialSmoothing(dfc[self.target_col], **self.ets_model).fit(**self.ets_fit)
                    if self.trend == "ets":
                        dfc[self.target_col] = dfc[self.target_col] - self.ets_model_fit.fittedvalues.values

            # Apply differencing if specified
            if self.difference is not None or self.season_diff is not None:
                self.orig = dfc[self.target_col].tolist()
                if self.difference is not None:
                    dfc[self.target_col] = np.diff(dfc[self.target_col], n=self.difference,
                                                prepend=np.repeat(np.nan, self.difference))
                if self.season_diff is not None:
                    self.orig_d = dfc[self.target_col].tolist()
                    dfc[self.target_col] = seasonal_diff(dfc[self.target_col], self.season_diff)

            # Create lag features based on n_lag parameter
            if self.n_lag is not None:
                for lag in self.n_lag:
                    dfc[f"{self.target_col}_lag_{lag}"] = dfc[self.target_col].shift(lag)
            # Create additional lag transformations if specified
            if self.lag_transform is not None:
                for func in self.lag_transform:
                    if isinstance(func, (expanding_std, expanding_mean)):
                        dfc[f"{func.__class__.__name__}_shift_{func.shift}"] = func(dfc[self.target_col])
                    elif isinstance(func, expanding_quantile):
                        dfc[f"{func.__class__.__name__}_shift_{func.shift}_q{func.quantile}"] = func(dfc[self.target_col])
                    elif isinstance(func, rolling_quantile):
                        dfc[f"{func.__class__.__name__}_{func.window_size}_shift_{func.shift}_q{func.quantile}"] = func(dfc[self.target_col])
                    else:
                        dfc[f"{func.__class__.__name__}_{func.window_size}_shift_{func.shift}"] = func(dfc[self.target_col])
            if self.trend is not None:
                if self.trend == "feature_lr":
                    dfc["trend"] = trend
                if self.trend == "feature_ets":
                    dfc["trend"] = self.ets_model.fittedvalues.values
        return dfc.dropna()
        

    def fit(self, df):
        """
        Fit the ml model.

        Args:
            df (pd.DataFrame): Input dataframe.
        """
        model_ = self.model

        if isinstance(self.model, (XGBRegressor, RandomForestRegressor, Cubist, HistGradientBoostingRegressor, AdaBoostRegressor, LinearRegression, Ridge, Lasso, ElasticNet)):
            if (self.cat_variables is not None) and (self.target_encode == False):
                # If categorical variables are provided, create a dictionary of categories
                self.cat_var = {c: sorted(df[c].drop_duplicates().tolist()) for c in self.cat_variables}
                # Create a list of the first category for each categorical variable
                if isinstance(self.model, (LinearRegression, Ridge, Lasso, ElasticNet)):
                    self.drop_categ= [sorted(df[i].drop_duplicates().tolist(), key=lambda x: x[0])[0] for i in self.cat_variables]

        model_df = self.data_prep(df)
        self.X = model_df.drop(columns=[self.target_col])
        self.y = model_df[self.target_col]
        # Fit the model (passing the categorical features if provided)
        if isinstance(self.model, LGBMRegressor):
            self.model_fit = model_.fit(self.X, self.y, categorical_feature=self.cat_variables)
        elif isinstance(self.model, CatBoostRegressor):
            self.model_fit = model_.fit(self.X, self.y, cat_features=self.cat_variables, verbose=True)
        else:
            self.model_fit = model_.fit(self.X, self.y)

    def copy(self):
        return copy.deepcopy(self)

    def forecast(self, H, exog=None):
        """
        Forecast H time steps.

        Args:
            H (int): Number of forecast steps.
            exog (pd.DataFrame, optional): Exogenous variables for forecasting.

        Returns:
            np.array: Forecasted values.
        """
        if exog is not None:  # if external regressors are provided
            if self.cat_variables is not None:
                if self.target_encode:
                    for col in self.cat_variables:
                        encode_col = col + "_target_encoded"
                        exog[encode_col] = target_encoder_for_test(self.df_encode, exog, col)
                    exog = exog.drop(columns=self.cat_variables)
                else:
                    if isinstance(self.model, (XGBRegressor, RandomForestRegressor, Cubist, HistGradientBoostingRegressor, AdaBoostRegressor, LinearRegression, Ridge, Lasso, ElasticNet)):
                        exog = self.data_prep(exog)

        lags = self.y.tolist() # to keep the latest values for lag features
        predictions = []
        
        # Compute trend forecasts if needed
        if self.trend:
            # orig_lags = self.target_orig.tolist()
            if self.trend in ["feature_lr", "linear"]:
                # future_time = np.arange(self.len, self.len + H).reshape(-1, 1)
                # trend_forecast = np.array(self.lr_model.predict(future_time)) # Predicting trend
                trend_forecast= forecast_trend(model = self.lr_model, H=H, start=self.len, breakpoints=self.cps)
            else:  # ets or feature_ets
                trend_forecast = np.array(self.ets_model_fit.forecast(H))

        for i in range(H):
            # If external regressors are provided, extract the i-th row
            if exog is not None:
                x_var = exog.iloc[i, :].tolist()
            else:
                x_var = []

            # Build lag-based features from the latest forecast–history
            inp_lag = []
            if self.n_lag is not None:
                inp_lag.extend([lags[-lag] for lag in self.n_lag])

            # Similarly compute additional lag transforms if available
            transform_lag = []
            if self.lag_transform is not None:
                transform_lag = []
                series_array = np.array(lags)
                for func in self.lag_transform:
                    transform_lag.append(func(series_array, is_forecast=True).to_numpy()[-1])
                
                
            # If using trend as a feature, add the forecasted trend component
            trend_var = []
            if self.trend is not None:
                if self.trend in ["feature_ets", "feature_lr"]:
                    trend_var.append(trend_forecast[i]) # Add the trend forecast as a featured

            # Concatenate all features for the forecast step
            inp = x_var + inp_lag + transform_lag + trend_var
            # Ensure that the input is a DataFrame with the same columns as the training data
            df_inp = pd.DataFrame(np.array(inp).reshape(1, -1), columns=self.X.columns)
            if isinstance(self.model, (LGBMRegressor, CatBoostRegressor)):
                df_inp = df_inp.astype({col: 'category' if col in self.cat_variables else 'float64' for col in df_inp.columns})
            # Get the forecast via the model
            pred = self.model_fit.predict(df_inp)[0]
            lags.append(pred)  # update lag history
            predictions.append(pred)

        forecasts = np.array(predictions)
        # If trend as ets is applied, add the trend component (ets_forecast) to the prediction
        if self.trend is not None:
            if self.trend in ["ets", "linear"]:
                forecasts += trend_forecast
        # Revert seasonal differencing if applied
        if self.season_diff is not None:
            forecasts = invert_seasonal_diff(self.orig_d, forecasts, self.season_diff)
        # Revert ordinary differencing if applied
        if self.difference is not None:
            forecasts = undiff_ts(self.orig, forecasts, self.difference)
        # Ensure forecasts are nonnegative
        forecasts = np.array([max(0, x) for x in forecasts])
        # Finally, invert Box-Cox transform if it was applied
        if self.box_cox:
            forecasts = back_box_cox_transform(y_pred=forecasts, lmda=self.lamda,
                                                shift=self.is_zero,
                                                box_cox_biasadj=self.biasadj)
        return forecasts

class VARModel:
    """
    Vector Autoregressive Model class supporting data preprocessing, fitting, forecasting, and cross-validation.

    Parameters
    ----------
    target_cols : List[str]
        Target columns for the VAR model (dependent variables).
    lags : Dict[str, List[int]]
        Dictionary specifying lags for each target variable. For example, {'target1': [1, 2], 'target2': [1, 2, 3]} or {'target1': 3, 'target2': 5}.
    lag_transform : Optional[Dict[str, List[tuple]]] (default=None)
        Dictionary specifying lag transformations per target (e.g. rolling, quantile), each as a list of tuples:
        (lag, func, window, [quantile]). For example, { 'target1': [(1, rolling_mean, 30), (2, rolling_quantile, 30, 0.5)] }.
    difference : Optional[Dict[str, int]] (default=None)
        Dictionary specifying order of differencing for each variable. For example, {'target1': 1, 'target2': 2}.
    seasonal_diff : Optional[Dict[str, int]] (default=None)
        Dictionary specifying seasonal differencing for each variable. For example, {'target1': 12, 'target2': 7}.
    trend : Optional[Dict[str, bool]] (default=None)
        Dictionary specifying trend type for each variable: "linear", "ses", "feature_lr", or "feature_ses".
    ets_params : Optional[Dict[str, tuple]] (default=None)
        Dictionary specifying params for ExponentialSmoothing per variable.
        For example, {'target1': ({'trend': 'add', 'seasonal': 'add'}, {'damped_trend': True}), 'target2': ({'trend': 'add', 'seasonal': 'add'}, {'damped_trend': True})}.
    cps: Optional[Dict[str, List[int]]] (default=None)
        Dictionary specifying change points for each variable.
    box_cox : Optional[Dict[str, bool]] (default=None)
        Dictionary specifying which variables require Box-Cox transform.
    box_cox_lmda : Optional[Dict[str, float]] (default=None)
        Dictionary of Box-Cox lambdas for each variable.
    box_cox_biasadj : bool or Dict[str, bool] (default=False)
        Whether to use bias adjustment for Box-Cox for each variable.
    add_constant : bool (default=True)
        If True, add a constant column to exogenous variables.
    cat_variables : Optional[List[str]] (default=None)
        List of categorical columns to one-hot encode.
    verbose : bool (default=False)
        If True, print verbose messages.

    Methods
    -------
    data_prep(df)
        Prepare the data for VAR model.
    fit(df_train)
        Fit the VAR model to training data.
    forecast(H, exog=None)
        Forecast H steps ahead.
    predict(X)
        Predict with model coefficients.
    cv_var(df, target_col, cv_split, test_size, metrics)
        Cross-validate VAR model.

    Notes
    -----
    - Assumes external utility functions exist for Box-Cox, seasonal differencing, etc.
    """

    def __init__(
        self,
        target_cols: List[str],
        lags: Dict[str, List[int]],
        lag_transform: Optional[Dict[str, List[tuple]]] = None,
        difference: Optional[Dict[str, int]] = None,
        seasonal_diff: Optional[Dict[str, int]] = None,
        trend: Optional[Dict[str, bool]] = None,
        ets_params: Optional[Dict[str, tuple]] = None,
        change_points: Optional[Dict[str, List[int]]] = None,
        box_cox: Optional[Dict[str, bool]] = None,
        box_cox_lmda: Optional[Dict[str, float]] = None,
        box_cox_biasadj: Any = False,
        add_constant: bool = True,
        cat_variables: Optional[List[str]] = None,
        verbose: bool = False
    ):
        self.target_cols = target_cols
        self.n_lag = lags
        self.lag_transform = lag_transform
        self.diffs = difference
        self.season_diffs = seasonal_diff
        self.ets_params = ets_params
        self.cps = change_points
        self.box_cox = box_cox
        self.lamdas = box_cox_lmda
        self.biasadj = box_cox_biasadj
        self.cons = add_constant
        self.cat_variables = cat_variables
        self.verbose = verbose

        # Handle box_cox bias adjustment dict default
        if self.box_cox is not None and not isinstance(self.box_cox, dict):
            raise TypeError("box_cox must be a dictionary of target values")
        if isinstance(self.box_cox, dict) and not isinstance(self.biasadj, dict):
            self.biasadj = {k: False for k in self.box_cox}
        
        # Handle trend default types
        self.trend = trend
        if self.trend is not None:
            if not isinstance(self.trend, dict):
                raise TypeError("trend must be a dictionary of target values")

    def data_prep(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare the data according to the model configuration.
        Applies categorical encoding, Box-Cox, detrending, differencing, seasonal differencing, lags, and lag transforms.
        Drops rows with any NaN after transformation.
        """
        dfc = df.copy()
        # Handle categorical variables
        if self.cat_variables is not None:
            for col, cats in self.cat_var.items():
                dfc[col] = pd.Categorical(dfc[col], categories=cats)
            dfc = pd.get_dummies(dfc, dtype=float)
            for i in self.drop_categ:
                dfc.drop(list(dfc.filter(regex=i)), axis=1, inplace=True)
        
        # Check all target columns exist
        if all(col in dfc.columns for col in self.target_cols):

            # Box-Cox transformation
            if self.box_cox is not None:
                if self.lamdas is None:
                    self.lamdas = {i: None for i in self.box_cox}
                self.is_zeros = {i: None for i in self.lamdas}
                for k, lm in self.lamdas.items():
                    self.is_zeros[k] = (dfc[k] < 1).any()
                    trans_data, self.lamdas[k] = box_cox_transform(
                        x=dfc[k], shift=self.is_zeros[k], box_cox_lmda=lm
                    )
                    if self.box_cox.get(k, False):
                        dfc[k] = trans_data

            # Detrending
            if self.trend is not None:
                self.len = df.shape[0]
                self.orig_targets = {i: dfc[i] for i in self.trend.keys()}  # Store original values for later use during forecasting
                self.trend_models = {}
                for k, v in self.trend.items():
                    if v == "linear": # If trend removal is required for this target
                        # trend, model_fit = LinearRegression().fit(np.arange(self.len).reshape(-1, 1), self.orig_targets[k])
                        # dfc[k] = dfc[k] - model_fit.predict(np.arange(self.len).reshape(-1, 1))
                        # self.trend_models[k] = model_fit

                            # self.lr_model = LinearRegression().fit(np.arange(self.len).reshape(-1, 1), dfc[self.target_col])
                        if self.cps is not None:
                            if k in self.cps and self.cps[k]:
                                trend, model_fit = lr_trend_model(self.orig_targets[k], breakpoints=self.cps[k], type='piecewise')
                        else:
                            trend, model_fit = lr_trend_model(self.orig_targets[k])

                        dfc[k] = dfc[k] - trend
                        self.trend_models[k] = model_fit

                    elif v == "ets": # ets
                        model_fit = ExponentialSmoothing(self.orig_targets[k], **self.ets_params[k][0]).fit(**self.ets_params[k][1])
                        dfc[k] = dfc[k] - model_fit.fittedvalues.values
                        self.trend_models[k] = model_fit
                    else:
                        raise ValueError(f"Unknown trend type: {v} for target {k}. Use 'linear' or 'ets'.")

            # Differencing
            if self.diffs is not None:
                self.origs = {i: dfc[i].tolist() for i in self.diffs}
                for x, d in self.diffs.items():
                    dfc[x] = np.diff(dfc[x], n=d, prepend=np.repeat(np.nan, d))

            # Seasonal differencing
            if self.season_diffs is not None:
                self.orig_ds = {i: dfc[i].tolist() for i in self.season_diffs}
                for w, s in self.season_diffs.items():
                    dfc[w] = seasonal_diff(dfc[w], s)

            # Lag features
            if self.n_lag is not None:
                for a, lags in self.n_lag.items():
                    lag_used = lags if isinstance(lags, list) else range(1, lags + 1) # Ensure lags is a list, even if a single int
                    for lg in lag_used:
                        dfc[f"{a}_lag_{lg}"] = dfc[a].shift(lg)

            # Lag transforms

            if self.lag_transform is not None:
                for idx, (target, funcs) in enumerate(self.lag_transform.items()):
                    for func in funcs:
                        if isinstance(func, (expanding_std, expanding_mean)):
                            dfc[f"trg{idx}_{func.__class__.__name__}_shift_{func.shift}"] = func(dfc[target])
                        elif isinstance(func, expanding_quantile):
                            dfc[f"trg{idx}_{func.__class__.__name__}_shift_{func.shift}_q{func.quantile}"] = func(dfc[target])
                        elif isinstance(func, rolling_quantile):
                            dfc[f"trg{idx}_{func.__class__.__name__}_{func.window_size}_shift_{func.shift}_q{func.quantile}"] = func(dfc[target])
                        else:
                            dfc[f"trg{idx}_{func.__class__.__name__}_{func.window_size}_shift_{func.shift}"] = func(dfc[target])
                            
        dfc = dfc.dropna()
        return dfc

    def fit(self, df_train: pd.DataFrame) -> None:
        """
        Fit the VAR model to the training data.

        Parameters
        ----------
        df_train : pd.DataFrame
            Training data.
        """
        if self.cat_variables is not None:
            self.cat_var = {c: sorted(df_train[c].drop_duplicates().tolist()) for c in self.cat_variables}
            self.drop_categ = [self.cat_var[c][0] for c in self.cat_variables]

        df = self.data_prep(df_train)
        X = df.drop(columns=self.target_cols)
        if self.cons:
            X = sm.add_constant(X)
        X = X.apply(pd.to_numeric, errors='raise')
        self.X = np.array(X)
        self.y = np.array(df[self.target_cols])
        self.coeffs = np.linalg.lstsq(self.X, self.y, rcond=None)[0]

    def predict(self, X: List[float]) -> np.ndarray:
        """
        Predict the model output for input X.

        Parameters
        ----------
        X : pd.DataFrame or List[float] or np.ndarray
            Feature DataFrame for prediction.

        Returns
        -------
        np.ndarray
            Model predictions for each target.
        """
        arr = np.array(X)
        return np.dot(self.coeffs.T, arr.T)

    def copy(self):
        return copy.deepcopy(self)

    def forecast(self, H: int, exog: Optional[pd.DataFrame] = None) -> Dict[str, np.ndarray]:
        """
        Forecast H steps ahead.

        Parameters
        ----------
        H : int
            Number of steps to forecast.
        exog : Optional[pd.DataFrame]
            Exogenous variables for forecasting.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary of forecasts for each target variable.
        """
        y_lists = {j: self.y[:, i].tolist() for i, j in enumerate(self.target_cols)} # Initialize lists for each target variable
        if exog is not None:
            if self.cons:
                if exog.shape[0] == 1:
                    exog.insert(0, 'const', 1)
                else:
                    exog = sm.add_constant(exog)
            exog = np.array(self.data_prep(exog))

        forecasts = {i: [] for i in self.target_cols}

        # Store original values for later estimate trend components
        if self.trend is not None:
            # orig_targets = {k: self.orig_targets[k].tolist() for k in self.trend.keys()}
            trend_forecasts = {}
            for ff in self.trend:
                if self.trend.get(ff) == "linear":
                    if ff in self.cps and self.cps[ff]:
                        trend_forecast= forecast_trend(model = self.trend_models[ff], H=H, start=self.len, breakpoints=self.cps[ff])
                    else:
                        trend_forecast= forecast_trend(model = self.trend_models[ff], H=H, start=self.len)  
                elif self.trend.get(ff) == "ets":
                    trend_forecast = np.array(self.trend_models[ff].forecast(H))

                trend_forecasts[ff] = trend_forecast

        for t in range(H):
            # Exogenous input for step t
            if exog is not None:
                exo_inp = exog[t].tolist()
            else:
                exo_inp = [1] if self.cons else []

            # Lagged features
            lags = []
            if self.n_lag is not None:
                for tr, vals in y_lists.items():
                    if tr in self.lags:
                        lag_used = self.n_lag[tr] if isinstance(self.n_lag[tr], list) else range(1, self.n_lag[tr] + 1)
                        ys = [vals[-x] for x in lag_used]
                        lags += ys
            # Lag transforms
            transform_lag = []
            if self.lag_transform is not None:
                for target, funcs in self.lag_transform.items():
                    series_array = np.array(y_lists[target])
                    for func in funcs:
                        transform_lag.append(func(series_array, is_forecast=True).to_numpy()[-1])


            inp = exo_inp + lags + transform_lag
            pred = self.predict(inp)
            # Add back trend
            
            for id_, ff in enumerate(self.forecasts.keys()):
                forecasts[ff].append(pred[id_])
                y_lists[ff].append(pred[id_])
        
        # add trend if not none
        if self.trend is not None:
            for ff in self.trend.keys():
                forecasts[ff] += trend_forecasts[ff]

        # Invert seasonal difference
        if self.season_diffs is not None:
            for s in self.orig_ds:
                forecasts[s] = invert_seasonal_diff(self.orig_ds[s], np.array(forecasts[s]), self.season_diffs[s])

        # Invert difference
        if self.diffs is not None:
            for d in self.diffs:
                forecasts[d] = undiff_ts(self.origs[d], np.array(forecasts[d]), self.diffs[d])


        # Non-negativity
        for f in forecasts:
            forecasts[f] = np.array([max(0, x) for x in forecasts[f]])

        # Invert Box-Cox
        if self.box_cox is not None:
            for k, lmd in self.lamdas.items():
                if self.box_cox.get(k, False):
                    forecasts[k] = back_box_cox_transform(
                        y_pred=forecasts[k], lmda=lmd, shift=self.is_zeros[k], box_cox_biasadj=self.biasadj[k]
                    )

        return forecasts

    def cv_var(
        self,
        df: pd.DataFrame,
        target_col: str,
        cv_split: int,
        test_size: int,
        step_size: None,
        metrics: List[Callable]
    ) -> pd.DataFrame:
        """
        Perform cross-validation for VAR model.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe.
        target_col : str
            Target variable for evaluation.
        cv_split : int
            Number of cross-validation folds.
        test_size : int
            Test size per fold.
        step_size : int
            Step size for rolling window. Default is None. Test size is applied
        metrics : List[Callable]
            List of metric functions.

        Returns
        -------
        pd.DataFrame
            DataFrame with averaged cross-validation metric scores.
        """
        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
        tscv = ParametricTimeSeriesSplit(n_splits=cv_split, test_size=test_size, step_size=step_size)
        self.metrics_dict = {m.__name__: [] for m in metrics}
        self.cv_forecasts_df = pd.DataFrame()

        for train_index, test_index in tscv.split(df):
            train, test = df.iloc[train_index], df.iloc[test_index]
            x_test, y_test = test.drop(columns=self.target_cols), np.array(test[target_col])
            self.fit(train)
            bb_forecast = self.forecast(H=test_size, exog=x_test)[target_col]

            forecast_df = test[target_col].to_frame()
            forecast_df["forecasts"] = bb_forecast
            self.cv_forecasts_df = pd.concat([self.cv_forecasts_df, forecast_df], axis=0)

            for m in metrics:
                if m.__name__ == 'MASE':
                    eval_score = m(y_test, bb_forecast, train[target_col])
                else:
                    eval_score = m(y_test, bb_forecast)
                self.metrics_dict[m.__name__].append(eval_score)

        overall_perform = [[m.__name__, np.mean(self.metrics_dict[m.__name__])] for m in metrics]
        return pd.DataFrame(overall_perform, columns=["eval_metric", "score"])
    

class ml_bidirect_forecaster:
    """
    Bidirectional ml Forecaster for time-series forecasting.

    Args:
         target_cols (list): Names of the target variables.
         cat_variables (list, optional): List of categorical variable names.
         n_lag (dict, optional): Dictionary specifying the number of lags or list of lags for each target variable. Default is None. Example: {'target1': 3, 'target2': [1, 2, 3]}.
         difference (dict, optional): Dictionary specifying the order of ordinary differencing for each target variable. Default is None. Example: {'target1': 1, 'target2': 2}.
         seasonal_length (dict, optional): Seasonal differencing period. Example: {'target1': 7, 'target2': 7}.
         trend (dict, optional): Trend handling strategy; one of 'linear' or 'ets'. Default is None. Example: {'Target1': 'linear', 'Target2': 'ets'}.
         ets_params (dict, optional): Dictionary of ETS model parameters (values are lists of dictionaries of params) and fit settings for each target variable. Example: {'Target1': [{'trend': 'add', 'seasonal': 'add'}, {'damped_trend': True}], 'Target2': [{'trend': 'mul', 'seasonal': 'mul'}, {'damped_trend': False}]}.
         target_encode (dict, optional): Flag determining if target encoding is used for categorical features for each target variable. Default is False. Example: {'Target1': True, 'Target2': False}.
         box_cox (dict, optional): Whether to apply a Box–Cox transformation for each target variable. Default is False. Example: {'Target1': True, 'Target2': False}.
         box_cox_lmda (dict, optional): Lambda parameter for the Box–Cox transformation for each target variable. Example: {'Target1': 0.5, 'Target2': 0.5}.
         box_cox_biasadj (dict, optional): Whether to adjust bias when inverting the Box–Cox transform for each target variable. Default is False. Example: {'Target1': True, 'Target2': False}.
         lag_transform (dict, optional): Dictionary specifying additional lag transformation functions for each target variable. List of functions to apply for each target variable. Example: {'Target1': [func1, func2], 'Target2': [func3]}.
    """
    def __init__(self, model, target_cols, cat_variables=None, lags=None, difference=None, seasonal_length=None,
                 trend=None,ets_params=None, target_encode=False,
                 box_cox=None, box_cox_lmda=None, box_cox_biasadj=None, lag_transform=None):
        self.model = model
        self.target_cols = target_cols
        self.cat_variables = cat_variables
        self.n_lag = lags
        if self.n_lag is not None:
            if not isinstance(self.n_lag, dict):
                raise TypeError("n_lag must be a dictionary of target values")
            for col, lags in self.n_lag.items():
                if isinstance(lags, int):
                    self.n_lag[col] = list(range(1, lags + 1))
                elif isinstance(lags, list):
                    self.n_lag[col] = lags
                else:
                    raise TypeError("n_lag values must be int or list of ints")

        if lag_transform is not None:
            if not isinstance(lag_transform, dict):
                raise TypeError("lag_transform must be a dictionary of target values")
            for col in lag_transform.keys():
                if not isinstance(lag_transform[col], list):
                    raise TypeError("lag_transform values must be a list of functions")
            self.lag_transform = lag_transform
        else:
            self.lag_transform = lag_transform
        
        # if difference is None, set it to None for all target columns
        if difference is None:
            self.difference = {col: None for col in target_cols}
        else:
            if not isinstance(difference, dict):
                raise TypeError("difference must be a dictionary of target values")
            self.difference = difference
            for col in target_cols:
                if col not in self.difference:
                    self.difference[col] = None
        # if seasonal_length is None, set it to None for all target columns
        # if seasonal_length is a dictionary, it must contain all target columns
        if seasonal_length is None:
            self.season_diff = {col: None for col in target_cols}
        else:
            if not isinstance(seasonal_length, dict):
                raise TypeError("seasonal_length must be a dictionary of target values")
            self.season_diff = seasonal_length
            # if any target column is not in the seasonal_length, set it to None
            for col in target_cols:
                if col not in self.season_diff:
                    self.season_diff[col] = None
        # if trend is None, set it to False for all target columns
        # if trend is a dictionary, it must contain all target columns
        self.trend = trend
        if trend is not None:
            if not isinstance(trend, dict):
                raise TypeError("trend must be a dictionary of target values")

        # if ets_params is not None:
        #     self.ets_model1 = ets_params[target_cols[0]][0]
        #     self.ets_fit1 = ets_params[target_cols[0]][1]
        #     self.ets_model2 = ets_params[target_cols[1]][0]
        #     self.ets_fit2 = ets_params[target_cols[1]][1]
        self.ets_params = ets_params
        self.target_encode = target_encode
        if box_cox is None:
            self.box_cox = {col: False for col in target_cols}
        else:
            if not isinstance(box_cox, dict):
                raise TypeError("box_cox must be a dictionary of target values")
            self.box_cox = box_cox

        if box_cox_lmda is None:
            self.lamda = {col: None for col in target_cols}
        else:
            if not isinstance(box_cox_lmda, dict):
                raise TypeError("box_cox_lmda must be a dictionary of target values")
            self.lamda = box_cox_lmda
            # if any target column is not in the box_cox_lmda, set it to None
            for col in target_cols:
                if col not in self.lamda:
                    self.lamda[col] = None
        
        if box_cox_biasadj is None:
            self.biasadj = {col: False for col in target_cols}
        else:
            if not isinstance(box_cox_biasadj, dict):
                raise TypeError("box_cox_biasadj must be a dictionary of target values")
            self.biasadj = box_cox_biasadj
            # if any target column is not in the box_cox_biasadj, set it to False
            for col in target_cols:
                if col not in self.biasadj:
                    self.biasadj[col] = False
        self.tuned_params = None
        self.actuals = None
        self.prob_forecasts = None


    def data_prep(self, df):
        """
        Prepare the data and handle categorical encoding, lag generation, trend removal, and differencing.
        """
        dfc = df.copy()
        if isinstance(self.model, (LGBMRegressor, CatBoostRegressor)):
                        # Process categorical variables if provided
            if self.cat_variables is not None:
                for col in self.cat_variables:
                    dfc[col] = dfc[col].astype('category')
        else:
            # Handle categorical variables
            if self.cat_variables is not None:
                for col, cats in self.cat_var.items():
                    dfc[col] = pd.Categorical(dfc[col], categories=cats)
                dfc = pd.get_dummies(dfc, dtype=float)
                if isinstance(self.model, (LinearRegression, Ridge, Lasso, ElasticNet)):
                    for i in self.drop_categ:
                        dfc.drop(list(dfc.filter(regex=i)), axis=1, inplace=True)

        if all(col in dfc.columns for col in self.target_cols):
        # Box-Cox transformation if flag is set
            if self.box_cox[self.target_cols[0]]:
                self.is_zero1 = np.any(np.array(dfc[self.target_cols[0]]) < 1) # Check if any values are less than 1 for Box-Cox
                trans_data1, self.lamda1 = box_cox_transform(x=dfc[self.target_cols[0]],
                                                            shift=self.is_zero1,
                                                            box_cox_lmda=self.lamda[self.target_cols[0]])
                dfc[self.target_cols[0]] = trans_data1
            if self.box_cox[self.target_cols[1]]:
                self.is_zero2 = np.any(np.array(dfc[self.target_cols[1]]) < 1)
                trans_data2, self.lamda2 = box_cox_transform(x=dfc[self.target_cols[1]],
                                                            shift=self.is_zero2,
                                                            box_cox_lmda=self.lamda[self.target_cols[1]])
                dfc[self.target_cols[1]] = trans_data2

            # Handle trend removal if specified
            # if atleast one target column has a trend, we need to apply the trend removal
            if self.trend is not None:
                self.len = len(df) # Store the length of the dataframe for later use
                if self.trend.get(self.target_cols[0]) in ["linear", "feature_lr"]:
                    self.orig_target1 = dfc[self.target_cols[0]] # Store original values for later use during forecasting
                    self.lr_model1 = LinearRegression().fit(np.arange(self.len).reshape(-1, 1), self.orig_target1)
                    if self.trend.get(self.target_cols[0]) == "linear":
                        dfc[self.target_cols[0]] = dfc[self.target_cols[0]] - self.lr_model1.predict(np.arange(self.len).reshape(-1, 1))

                if self.trend.get(self.target_cols[0]) in ["ets", "feature_ets"]:
                    self.orig_target1 = dfc[self.target_cols[0]] # Store original values for later use during forecasting
                    self.ses_model1 = ExponentialSmoothing(self.orig_target1, **self.ets_params[self.target_cols[0]][0]).fit(**self.ets_params[self.target_cols[0]][1])
                    if self.trend.get(self.target_cols[0]) == "ets":
                        dfc[self.target_cols[0]] = dfc[self.target_cols[0]] - self.ses_model1.fittedvalues.values

                # If the second target column has a trend, apply the same logic
                if self.trend.get(self.target_cols[1]) in ["linear", "feature_lr"]:
                    self.orig_target2 = df[self.target_cols[1]] # Store original values for later use during forecasting
                    self.lr_model2 = LinearRegression().fit(np.arange(self.len).reshape(-1, 1), self.orig_target2)
                    if self.trend.get(self.target_cols[1]) == "linear":
                        dfc[self.target_cols[1]] = dfc[self.target_cols[1]] - self.lr_model2.predict(np.arange(self.len).reshape(-1, 1))

                if self.trend.get(self.target_cols[1]) in ["ets", "feature_ets"]:
                    self.orig_target2 = df[self.target_cols[1]] # Store original values for later use during forecasting
                    self.ses_model2 = ExponentialSmoothing(self.orig_target2, **self.ets_params[self.target_cols[1]][0]).fit(**self.ets_params[self.target_cols[1]][1])
                    if self.trend.get(self.target_cols[1]) == "ets":
                        dfc[self.target_cols[1]] = dfc[self.target_cols[1]] - self.ses_model2.fittedvalues.values

            # Handle differencing if specified
            if self.difference[self.target_cols[0]] is not None:
                    self.orig1 = df[self.target_cols[0]].tolist()
                    dfc[self.target_cols[0]] = np.diff(dfc[self.target_cols[0]], n=self.difference[self.target_cols[0]],
                                                    prepend=np.repeat(np.nan, self.difference[self.target_cols[0]]))
            if self.difference[self.target_cols[1]] is not None:
                self.orig2 = df[self.target_cols[1]].tolist()
                dfc[self.target_cols[1]] = np.diff(dfc[self.target_cols[1]], n=self.difference[self.target_cols[1]],
                                                prepend=np.repeat(np.nan, self.difference[self.target_cols[1]]))
            if self.season_diff[self.target_cols[0]] is not None:
                self.orig_d1 = dfc[self.target_cols[0]].tolist()
                dfc[self.target_cols[0]] = seasonal_diff(dfc[self.target_cols[0]], self.season_diff[self.target_cols[0]])
            if self.season_diff[self.target_cols[1]] is not None:
                self.orig_d2 = dfc[self.target_cols[1]].tolist()
                dfc[self.target_cols[1]] = seasonal_diff(dfc[self.target_cols[1]], self.season_diff[self.target_cols[1]])

            # Create lag features based on n_lag parameter
            if self.n_lag is not None:
                for target, lags in self.n_lag.items():
                    for lag in lags:
                        dfc[f"{target}_lag_{lag}"] = dfc[target].shift(lag)

            # Create additional lag transformations if specified (check this later)
            if self.lag_transform is not None:
                for idx, (target, funcs) in enumerate(self.lag_transform.items()):
                    for func in funcs:
                        if isinstance(func, (expanding_std, expanding_mean)):
                            dfc[f"trg{idx}_{func.__class__.__name__}_shift_{func.shift}"] = func(dfc[target])
                        elif isinstance(func, expanding_quantile):
                            dfc[f"trg{idx}_{func.__class__.__name__}_shift_{func.shift}_q{func.quantile}"] = func(dfc[target])
                        elif isinstance(func, rolling_quantile):
                            dfc[f"trg{idx}_{func.__class__.__name__}_{func.window_size}_shift_{func.shift}_q{func.quantile}"] = func(dfc[target])
                        else:
                            dfc[f"trg{idx}_{func.__class__.__name__}_{func.window_size}_shift_{func.shift}"] = func(dfc[target])
            # Add trend features if specified
            if self.trend is not None:
                # if self.target_cols[0] in dfc.columns:
                if self.trend.get(self.target_cols[0]) == "feature_lr":
                    dfc["trend1"] = self.lr_model1.predict(np.arange(self.len).reshape(-1, 1))
                if self.trend.get(self.target_cols[0]) == "feature_ses":
                    dfc["trend1"] = self.ses_model1.fittedvalues.values
                # if self.target_cols[1] in dfc.columns:
                if self.trend.get(self.target_cols[1]) == "feature_lr":
                    dfc["trend2"] = self.lr_model2.predict(np.arange(self.len).reshape(-1, 1))
                if self.trend.get(self.target_cols[1]) == "feature_ses":
                    dfc["trend2"] = self.ses_model2.fittedvalues.values

        return dfc.dropna()
            
    def fit(self, df):
        # Fit the model to the dataframe
        model1_ = clone(self.model)
        model2_ = clone(self.model)
        if isinstance(self.model, (XGBRegressor, RandomForestRegressor, Cubist, HistGradientBoostingRegressor, AdaBoostRegressor, LinearRegression, Ridge, Lasso, ElasticNet)):
            if self.cat_variables is not None:
                self.cat_var = {c: sorted(df[c].drop_duplicates().tolist(), key=lambda x: x[0]) for c in self.cat_variables}
                if isinstance(self.model, (LinearRegression, Ridge, Lasso, ElasticNet)):
                    self.drop_categ= [sorted(df[i].drop_duplicates().tolist(), key=lambda x: x[0])[0] for i in self.cat_variables]
        model_train = self.data_prep(df)
        self.X = model_train.drop(columns=self.target_cols)
        self.y1 = model_train[self.target_cols[0]]
        self.y2 = model_train[self.target_cols[1]]
        if isinstance(self.model, CatBoostRegressor):
            self.model1_fit = model1_.fit(self.X, self.y1, cat_features=self.cat_variables, verbose = True)
            self.model2_fit = model2_.fit(self.X, self.y2, cat_features=self.cat_variables, verbose = True)
        elif isinstance(self.model, LGBMRegressor):
            self.model1_fit = model1_.fit(self.X, self.y1, categorical_feature=self.cat_variables)
            self.model2_fit = model2_.fit(self.X, self.y2, categorical_feature=self.cat_variables)
        else:
            self.model1_fit = model1_.fit(self.X, self.y1)
            self.model2_fit = model2_.fit(self.X, self.y2)

    def copy(self):
        return copy.deepcopy(self)

    def forecast(self, H, exog=None):
        """
        Forecast future values for H periods.

        Args:
            H (int): Number of periods to forecast.
            exog (pd.DataFrame, optional): Exogenous variables.

        Returns:
            np.array: Forecasted values.
        """
        if isinstance(self.model, (XGBRegressor, RandomForestRegressor, Cubist, HistGradientBoostingRegressor, AdaBoostRegressor, LinearRegression, Ridge, Lasso, ElasticNet)):
            if exog is not None:
                exog = self.data_prep(exog)

        target1_lags = self.y1.tolist()
        target2_lags = self.y2.tolist()
        tar1_forecasts = []
        tar2_forecasts = []


        if self.trend is not None:

            if self.trend.get(self.target_cols[0]) in ["ets", "feature_ets"]:
                trend_forecast1 = np.array(self.ses_model1.forecast(H)) # Forecasting H step
            elif self.trend.get(self.target_cols[0]) in ["linear", "feature_lr"]:
                future_time = np.arange(self.len, self.len + H).reshape(-1, 1)
                trend_forecast1 = self.lr_model1.predict(future_time) # Predicting the next value trend

            ## Second target variable

            if self.trend.get(self.target_cols[1]) in ["ets", "feature_ets"]:
                trend_forecast2 = np.array(self.ses_model2.forecast(H)) # Forecasting H step
            elif self.trend.get(self.target_cols[1]) in ["linear", "feature_lr"]:
                future_time = np.arange(self.len, self.len + H).reshape(-1, 1)
                trend_forecast2 = self.lr_model2.predict(future_time) # Predicting the next value trend

        # Forecast recursively one step at a time
        for i in range(H):
            if exog is not None:
                x_var = exog.iloc[i, :].tolist()
            else:
                x_var = []
                
            inp_lag = []
            if self.n_lag is not None:
                # For the first target variable
                for col, lags in self.n_lag.items():
                    if col == self.target_cols[0]:
                        inp_lag.extend([target1_lags[-lag] for lag in lags])
                    else:
                        inp_lag.extend([target2_lags[-lag] for lag in lags])

            transform_lag = []
            if self.lag_transform is not None:
                for target, funcs in self.lag_transform.items():
                    series_array = np.array(target1_lags if target == self.target_cols[0] else target2_lags)
                    for func in funcs:
                        transform_lag.append(func(series_array, is_forecast=True).to_numpy()[-1])

            # Trend feature
            trend_var = []

            if self.trend is not None:
                # First target variable
                if self.trend.get(self.target_cols[0]) in ["feature_lr", "feature_ets"]:
                    trend_var.append(trend_forecast1[i])

                ## Second target variable
                
                if self.trend.get(self.target_cols[1]) in ["feature_lr", "feature_ets"]:
                    trend_var.append(trend_forecast2[i])

            inp = x_var + inp_lag + transform_lag + trend_var
            # print(f"len of x_var: {len(x_var)}, len of inp_lag1: {len(inp_lag1)}, len of inp_lag2: {len(inp_lag2)}, len of transform_lag: {len(transform_lag)}, len of inp: {len(inp)}")

            df_inp = pd.DataFrame(inp).T
            df_inp.columns = self.X.columns
            if isinstance(self.model, (LGBMRegressor, CatBoostRegressor)):
                df_inp = df_inp.astype({col: 'category' if col in self.cat_variables else 'float64' for col in df_inp.columns})
            pred1 = self.model1_fit.predict(df_inp)[0]
            target1_lags.append(pred1)
            pred2 = self.model2_fit.predict(df_inp)[0]
            target2_lags.append(pred2)

            tar1_forecasts.append(pred1)
            tar2_forecasts.append(pred2)

        forecasts1 = np.array(tar1_forecasts)
        forecasts2 = np.array(tar2_forecasts)
        # If trend is applied, add the trend forecast to the prediction
        if self.trend is not None:
            # Revert trend if applied
            if self.trend.get(self.target_cols[0]) in ["ets", "linear"]:
                forecasts1 += trend_forecast1
            if self.trend.get(self.target_cols[1]) in ["ets", "linear"]:
                forecasts2 += trend_forecast2
        # Revert seasonal differencing if applied
        if self.season_diff[self.target_cols[0]] is not None:
            forecasts1 = invert_seasonal_diff(self.orig_d1, forecasts1, self.season_diff[self.target_cols[0]])
        if self.season_diff[self.target_cols[1]] is not None:
            forecasts2 = invert_seasonal_diff(self.orig_d2, forecasts2, self.season_diff[self.target_cols[1]])
            
        if self.difference[self.target_cols[0]] is not None:
            forecasts1 = undiff_ts(self.orig1, forecasts1, self.difference[self.target_cols[0]])
        if self.difference[self.target_cols[1]] is not None:
            forecasts2 = undiff_ts(self.orig2, forecasts2, self.difference[self.target_cols[1]])

        forecasts1 = np.array([max(0, x) for x in forecasts1])
        forecasts2 = np.array([max(0, x) for x in forecasts2])
        if self.box_cox[self.target_cols[0]]:
            forecasts1 = back_box_cox_transform(y_pred=forecasts1,
                                                lmda=self.lamda1,
                                                shift=self.is_zero1,
                                                box_cox_biasadj=self.biasadj[self.target_cols[0]])
        if self.box_cox[self.target_cols[1]]:
            forecasts2 = back_box_cox_transform(y_pred=forecasts2,
                                                lmda=self.lamda2,
                                                shift=self.is_zero2,
                                                box_cox_biasadj=self.biasadj[self.target_cols[1]])
        forecasts = {
            self.target_cols[0]: forecasts1,
            self.target_cols[1]: forecasts2
        }
        return forecasts

# Hidden Markov Model with Regression
class MsHmmRegression:
    """
    Hidden Markov Model Regression for time series with EM parameter estimation.

    Args:
        n_components (int): Number of hidden states.
        target_col (str): Name of the target variable.
        lag_list (list): List of integer lags to include as features.
        method (str): 'posterior' for soft state assignment, 'viterbi' for hard paths.
        startprob_prior (float): Prior for initial state probabilities.
        transmat_prior (float): Prior for transition matrix.
        ets_params (tuple, optional): A tuple (model_params, fit_params) for exponential smoothing. Ex.g. ({'trend': 'add', 'seasonal': 'add'}, {'damped_trend': True}). If trend is "ets", this will be used.
        change_points (list or None): List of change points for piecewise linear regression to handle trend
        add_constant (bool): Whether to add constant to regressors.
        difference (int or None): Order of differencing to apply to target.
        trend (str or None): Type of trend to remove ('linear', 'ets', etc.). Default is None.
        cat_variables (list or None): List of categorical columns.
        n_iter (int): Maximum number of EM iterations.
        tol (float): Convergence tolerance for EM.
        coefficients (np.ndarray or None): Initial regression coefficients.
        stds (np.ndarray or None): Initial state std deviations.
        init_state (np.ndarray or None): Initial state distribution.
        trans_matrix (np.ndarray or None): Initial transition matrix.
        random_state (int or None): Random seed.
        verbose (bool): Print progress if True.
    """

    def __init__(self, n_components, target_col, lags, method="posterior",
                 startprob_prior=1e3, transmat_prior=1e5, add_constant=True,
                 difference=None, trend=None, ets_params = None, change_points=None,
                 cat_variables=None, lag_transform=None, n_iter=100, tol=1e-6,
                 coefficients=None, stds=None, init_state=None, trans_matrix=None,
                 box_cox=False, lamda=None, box_cox_biasadj=False, season_diff=None,
                 random_state=None, verbose=False):
        self.N = n_components
        self.target_col = target_col
        self.diff = difference
        self.cons = add_constant
        self.cat_variables = cat_variables
        # lags must be a list of integers or an integer
        if not isinstance(lags, (int, list)):
            raise ValueError("Lags must be an integer or a list of integers.")

        self.lags = [i for i in range(1, lags + 1)] if isinstance(lags, int) else lags
        self.method = method
        self.box_cox = box_cox
        self.lamda = lamda
        self.biasadj = box_cox_biasadj
        self.trend = trend
        if ets_params is not None:
            self.ets_model = ets_params[0]
            self.ets_fit = ets_params[1]
        else:
            self.ets_model = None
            self.ets_fit = None
        
        self.cps = change_points
        self.season_diff = season_diff
        self.lag_transform = lag_transform
        self.iter = n_iter
        self.tol = tol
        self.verb = verbose


        # RNG for reproducibility
        self.rng = np.random.default_rng(random_state)
        if init_state is None:
            self.sp = startprob_prior
            self.alpha_p = np.repeat(self.sp, self.N)
            self.pi = self.rng.dirichlet(self.alpha_p) # Initial state probabilities using Dirichlet distribution
        else:
            self.pi = np.array(init_state)
        if trans_matrix is None:
            self.tm = transmat_prior
            self.alpha_t = np.repeat(self.tm, self.N) # 
            self.A = self.rng.dirichlet(self.alpha_t, size=self.N)
        else:
            self.A = np.array(trans_matrix)

        self.coeffs = coefficients
        self.stds = stds


    def data_prep(self, df):
        """
        Prepare the data: encode categoricals, add lags, trend, differencing.
        """
        dfc = df.copy()
        # Categorical variable encoding
        if self.cat_variables is not None:
            # if self.target_encode ==True:
            #     for col in self.cat_variables:
            #         encode_col = col+"_target_encoded"
            #         dfc[encode_col] = kfold_target_encoder(dfc, col, self.target_col, 36)
            #     self.df_encode = dfc.copy()
            #     dfc = dfc.drop(columns = self.cat_variables)
            #     # If target encoding is not used, convert categories to dummies    

            # else:
            for col, cat in self.cat_var.items():
                dfc[col] = dfc[col].astype('category')
                # Set categories for categorical columns
                dfc[col] = dfc[col].cat.set_categories(cat)
            dfc = pd.get_dummies(dfc, dtype=np.float64)

            for i in self.drop_categ:
                dfc.drop(list(dfc.filter(regex=i)), axis=1, inplace=True)
        
        if self.target_col in dfc.columns:
            # Apply Box–Cox transformation if specified
            if self.box_cox:
                self.is_zero = np.any(np.array(dfc[self.target_col]) < 1) # check for zero or negative values
                trans_data, self.lamda = box_cox_transform(x=dfc[self.target_col],
                                                        shift=self.is_zero,
                                                        box_cox_lmda=self.lamda)
                dfc[self.target_col] = trans_data

            if self.trend is not None:
                self.len = len(dfc)
                self.target_orig = dfc[self.target_col] # Store original values for later use during forecasting
                if self.trend == "linear":
                    # self.lr_model = LinearRegression().fit(np.arange(self.len).reshape(-1, 1), self.target_orig)
                    # dfc[self.target_col] = dfc[self.target_col] - self.lr_model.predict(np.arange(self.len).reshape(-1, 1))
                    if self.cps is not None:
                        trend, self.lr_model = lr_trend_model(self.target_orig, breakpoints=self.cps, type='piecewise')
                    else:
                        trend, self.lr_model = lr_trend_model(self.target_orig)
                    dfc[self.target_col] = dfc[self.target_col] - trend
                if self.trend == "ets":
                    self.ets_model_fit = ExponentialSmoothing(self.target_orig, **self.ets_model).fit(**self.ets_fit)
                    dfc[self.target_col] = dfc[self.target_col] - self.ets_model_fit.fittedvalues.values


            # Apply differencing if specified
            if self.diff is not None or self.season_diff is not None:
                self.orig = dfc[self.target_col].tolist()
                if self.diff is not None:
                    dfc[self.target_col] = np.diff(dfc[self.target_col], n=self.diff,
                                                prepend=np.repeat(np.nan, self.diff))
                if self.season_diff is not None:
                    self.orig_d = dfc[self.target_col].tolist()
                    dfc[self.target_col] = seasonal_diff(dfc[self.target_col], self.season_diff)

            # Create lag features based on lags parameter
            if self.lags is not None:
                for lag in self.lags:
                    dfc[f"{self.target_col}_lag_{lag}"] = dfc[self.target_col].shift(lag)
            # Create additional lag transformations if specified
            if self.lag_transform is not None:
                for func in self.lag_transform:
                    if isinstance(func, (expanding_std, expanding_mean)):
                        dfc[f"{func.__class__.__name__}_shift_{func.shift}"] = func(dfc[self.target_col])
                    elif isinstance(func, expanding_quantile):
                        dfc[f"{func.__class__.__name__}_shift_{func.shift}_q{func.quantile}"] = func(dfc[self.target_col])
                    elif isinstance(func, rolling_quantile):
                        dfc[f"{func.__class__.__name__}_{func.window_size}_shift_{func.shift}_q{func.quantile}"] = func(dfc[self.target_col])
                    else:
                        dfc[f"{func.__class__.__name__}_{func.window_size}_shift_{func.shift}"] = func(dfc[self.target_col])
                        
            self.df = dfc.dropna()
            self.X = self.df.drop(columns=self.target_col)
            self.y = self.df[self.target_col]
            if self.cons:
                self.X = sm.add_constant(self.X)
            self.col_names = self.X.columns.tolist() if hasattr(self.X, 'columns') else [f"x{i}" for i in range(self.X.shape[1])]
            self.X = np.array(self.X)
            self.y = np.array(self.y)
            self.T = len(self.y)
            if self.coeffs is None or self.stds is None:
                # Initial fit: use unweighted least squares for all states
                coeffs = []
                stds = []
                for i in range(self.N):
                    # Least squares fit
                    coeff_i = np.linalg.lstsq(self.X, self.y, rcond=None)[0]
                    coeffs.append(coeff_i)
                    y_pred = self.X @ coeff_i
                    resid = self.y - y_pred
                    var_i = np.mean(resid ** 2)
                    stds.append(np.sqrt(var_i))
                self.coeffs = np.row_stack(coeffs)
                self.stds = np.array(stds)

        else:
            return dfc.dropna()


    def compute_coeffs(self, ridge=1e-5, var_floor=1e-5, w_floor=1e-5):

        # Update regression coefficients and stds for each state

        # If posterior probabilities are shorter than the number of observations, make self.X and self.y same length visa vis make posterier same as self.X and self.y length
        if self.posterior.shape[1] < self.X.shape[0]:
            # Truncate self.X and self.y to match the length of self.posterior[i]
            self.X = self.X[:self.posterior.shape[1]]
            self.y = self.y[:self.posterior.shape[1]]
        if self.posterior.shape[1] > self.X.shape[0]:
            # Truncate self.posterior to match the length of self.X and self.y
            self.posterior = self.posterior[:, -self.X.shape[0]:]

        coeffs = []
        stds = []
        X = self.X
        for s in range(self.N):
            # Add floor so state isn’t “killed”
            w = self.posterior[s] + w_floor
            w /= w.sum()
            sw = np.sqrt(w)
            Xw = X * sw[:, None]
            yw = self.y * sw
            XtX = Xw.T @ Xw + ridge*np.eye(X.shape[1])
            Xty = Xw.T @ yw
            beta_s = np.linalg.lstsq(XtX, Xty, rcond=None)[0]
            coeffs.append(beta_s)
            resid = self.y - X @ beta_s
            var_s = (w * resid**2).sum() / max((w.sum()-beta_s.shape[0]), 1.0)
            stds.append(np.sqrt(max(var_s, var_floor)))
        self.coeffs = np.row_stack(coeffs)
        self.stds = np.array(stds)


# Hidden Markov Model with Vector Autoregressive (VAR)

    def _log_emissions(self):
        # logB[s,t] = log p(y_t | state s)
        N, T = self.N, self.T
        logB = np.empty((N, T))
        self.fitted = np.empty((N, T))
        # self.compute_coeffs()
        for s in range(N):
            mu = self.X @ self.coeffs[s]           # (T,)
            self.fitted[s, :] = mu
            logB[s, :] = norm.logpdf(self.y, loc=mu, scale=self.stds[s])
        return logB

    def _e_step_log(self):
        N, T = self.N, self.T
        logA  = np.log(self.A + 1e-300)
        logpi = np.log(self.pi + 1e-300)
        logB  = self._log_emissions()

        # Forward
        log_alpha = np.empty((N, T))
        log_alpha[:, 0] = logpi + logB[:, 0]
        for t in range(1, T):
            # log_alpha[:,t] = logB[:,t] + logsumexp_i( log_alpha[i,t-1] + logA[i,:] )
            log_alpha[:, t] = logB[:, t] + logsumexp(log_alpha[:, t-1][:, None] + logA, axis=0)

        # Log-likelihood
        loglik = logsumexp(log_alpha[:, -1])

        # Backward
        log_beta = np.full((N, T), 0.0)
        for t in range(T-2, -1, -1):
            # log_beta[:,t] = logsumexp_j( logA + logB[:,t+1] + log_beta[:,t+1] , axis=1 )
            log_beta[:, t] = logsumexp(logA + (logB[:, t+1] + log_beta[:, t+1])[None, :], axis=1)

        # Gamma
        log_gamma = log_alpha + log_beta - loglik
        # normalize per time to kill rounding; columns sum to 1 after exp
        log_gamma -= logsumexp(log_gamma, axis=0)
        gamma = np.exp(log_gamma)

        # Xi
        log_xi = np.empty((N, N, T-1))
        for t in range(T-1):
            tmp = log_alpha[:, t][:, None] + logA + (logB[:, t+1] + log_beta[:, t+1])[None, :]
            tmp -= logsumexp(tmp)      # normalize this slice
            log_xi[:, :, t] = tmp
        xi = np.exp(log_xi)

        # print("logB min/max:", np.min(logB), np.max(logB))
        # print("Any NaN in logB?", np.any(np.isnan(logB)))
        # print("Any Inf in logB?", np.any(np.isinf(logB)))
        # sanity checks (soft)
        assert np.allclose(gamma.sum(axis=0), 1.0, atol=1e-8)
        assert np.allclose(xi.sum(axis=(0,1)), 1.0, atol=1e-8)

        self.log_forward  = log_alpha
        self.log_backward = log_beta
        self.posterior    = gamma
        self.loglik       = loglik
        return loglik, gamma, xi


    def _m_step(self, gamma, xi):
        numer = xi.sum(axis=2)                         # (N,N)
        denom = gamma[:, :-1].sum(axis=1, keepdims=True)  # (N,1)
        A = numer / (denom + 1e-12)
        A = np.maximum(A, 1e-12)
        A /= A.sum(axis=1, keepdims=True)
        self.A = A
        self.compute_coeffs() 


    def EM(self):
        loglik, gamma, xi = self._e_step_log()
        self._m_step(gamma, xi)
        self.LL = loglik
        # return loglik
        
    def fit_em(self, df_train):
        """
        Run EM iterations until convergence (log-domain version).
        """

        # Handle categorical variable encoding
        if self.cat_variables is not None:
            self.cat_var = {c: sorted(df_train[c].drop_duplicates().tolist(), key=lambda x: str(x))
                            for c in self.cat_variables}
            self.drop_categ = [sorted(df_train[col].drop_duplicates().tolist(), key=lambda x: str(x))[0]
                               for col in self.cat_variables]
        self.data_prep(df_train)

        prev_ll = -np.inf
        # store intermediate log-likelihoods
        self.log_likelihoods = []
        for it in range(self.iter):
            # loglik, gamma, xi = self._e_step_log()
            # self._m_step(gamma, xi)
            self.EM()
            if self.verb:
                print(f"Iter {it}: loglik={self.LL:.4f}")
            if it > 10:
                if abs(self.LL - prev_ll) < self.tol:
                    if self.verb:
                        print("Converged.")
                    break
            self.log_likelihoods.append(self.LL)
            prev_ll = self.LL
        return self.LL

    def fit(self, df):
        """
        Refit the HMM regression model on new training data (log-domain version).
        """
        self.data_prep(df)
        self.EM()

        # N, T = self.N, self.T
        # logpi = np.log(self.pi + 1e-300)
        # logA = np.log(self.A + 1e-300)
        # logB = self._log_emissions()
        # log_alpha = np.zeros((N, T))

        # # Initialization
        # for i in range(N):
        #     log_alpha[i, 0] = logpi[i] + logB[i, 0]

        # # Recursion
        # for t in range(1, T):
        #     for j in range(N):
        #         log_alpha[j, t] = logB[j, t] + logsumexp(log_alpha[:, t-1] + logA[:, j])

        # # Sequence log-likelihood
        # self.LL = logsumexp(log_alpha[:, -1])
        # self.log_alpha = log_alpha
        return self.LL
    
    def predict_states(self):
        return np.argmax(self.posterior, axis=0)
    def predict_proba(self):
        return self.posterior
    
    # AIC computation    
    @property
    def AIC(self):
        k = self.N * self.X.shape[1] + self.N ** 2 + self.N - 1
        return 2 * k - 2 * self.LL

    @property
    def BIC(self):
        
        k = self.N * self.X.shape[1] + self.N ** 2 + self.N - 1
        
        n = self.T  # effective number of observations
        return -2 * self.LL + k * np.log(n)

    def copy(self):
        return copy.deepcopy(self)

    def forecast(self, H, exog=None):
        """
        Forecast H periods ahead using fitted HMM regression (log-domain version), with advanced post-processing.

        Handles:
        - Trend re-adjustment (linear/ETS)
        - Seasonal differencing reversal
        - Regular differencing reversal
        - Box-Cox back-transform
        - Exogenous variables
        - Lag transformations
        """
        y_list = self.y.tolist()
        forecasts_ = []
        N = self.N

        # Prepare exogenous future regressors if provided
        if exog is not None:
            if self.cons:
                if exog.shape[0] == 1:
                    exog.insert(0, 'const', 1)
                else:
                    exog = sm.add_constant(exog)
            exog = np.array(self.data_prep(exog))


        # Init with last forward distribution (in log)
        log_forward_last = self.log_forward[:, -1]
        logA = np.log(self.A + 1e-300)

        # Forward
        log_alpha = np.empty((N, H))
        log_alpha[:, 0] = logsumexp(log_forward_last[:, None] + logA, axis=0)
        for t in range(1, H):
            # log_alpha[:,t] = logB[:,t] + logsumexp_i( log_alpha[i,t-1] + logA[i,:] )
            log_alpha[:, t] = logsumexp(log_alpha[:, t-1][:, None] + logA, axis=0)

        log_alpha -= logsumexp(log_alpha, axis=0)
        self.forecast_forward = np.exp(log_alpha)
        self.state_forecasts = np.argmax(self.forecast_forward, axis=0)

        # Prepare for trend adjustment
        # This assumes you stored original target (pre-trend removal) in self.target_orig
        if hasattr(self, 'target_orig') and self.trend is not None:
            if self.trend == "linear":
                # future_time = np.arange(len(self.target_orig), len(self.target_orig) + H).reshape(-1, 1)
                # trend_forecast = np.array(self.lr_model.predict(future_time))
                trend_forecast= forecast_trend(model = self.lr_model, H=H, start=self.len, breakpoints=self.cps)
            elif self.trend == "ets":
                trend_forecast = np.array(self.ets_model_fit.forecast(H))

        for t in range(H):
            if exog is not None:
                exo_inp = exog[t].tolist()
            else:
                exo_inp = [1] if self.cons else []
            lags = [y_list[-l] for l in self.lags]
            transform_lag = []
            if self.lag_transform is not None:
                series_array = np.array(y_list)
                for func in self.lag_transform:
                    transform_lag.append(func(series_array, is_forecast=True).to_numpy()[-1])
            inp = np.array(exo_inp + lags + transform_lag)

            state_preds = np.zeros(N)
            for j in range(N):
                mu = np.dot(self.coeffs[j], inp)
                state_preds[j] = mu

            # normalize to probabilities

            # normalize to probabilities
            pred_w = np.sum(self.forecast_forward[:, t] * state_preds)
            forecasts_.append(pred_w)
            y_list.append(pred_w)

            # log_forward_last = log_f_t.copy()

        forecasts = np.array(forecasts_)

        if self.trend is not None:
            forecasts += trend_forecast

        # --- Revert seasonal differencing if applied ---
        if self.season_diff is not None:
            forecasts = invert_seasonal_diff(self.orig_d, forecasts, self.season_diff)

        # --- Revert regular differencing if applied ---
        if self.diff is not None:
            forecasts = undiff_ts(self.orig, forecasts, self.diff)

        # --- Box-Cox back-transform if applied ---
        if self.box_cox:
            forecasts = back_box_cox_transform(
                y_pred=forecasts, lmda=self.lamda,
                shift=self.is_zero, box_cox_biasadj=self.biasadj
            )

        return forecasts
# Hidden Markov Model with Vector Autoregressive (VAR)

class MsHmmVar:
    """
    Hidden Markov Model with Vector Autoregressive (VAR) emission distributions.

    This model can be used for time series with multiple targets, capturing regime-switching dynamics
    using multivariate Gaussian emissions whose means depend on lagged values (VAR).

    Args:
        n_components (int): Number of hidden states.
        target_col (list of str): List of target variable names.
        lags (dict): Dict mapping target column to list of lag values.
        difference (dict): Dict mapping target column to integer difference order.
        seasonal_diff (dict): Dict mapping target column to seasonal difference order.
        lag_transform (dict): Dict mapping target column to lag transformation order.
        trend (dict): Dict mapping target column to trend removal method.
        ets_params : Optional[Dict[str, tuple]] (default=None)
        Dictionary specifying params for ExponentialSmoothing per variable.
        For example, {'target1': ({'trend': 'add', 'seasonal': 'add'}, {'damped_trend': True}), 'target2': ({'trend': 'add', 'seasonal': 'add'}, {'damped_trend': True})}.
        change_points (list or None): List of change points for piecewise linear regression to handle trend
        method (str): 'posterior' for soft state assignment, 'viterbi' for hard paths.
        covariance_type (str): 'full' (default) or 'diag' for emission covariances. default is "diag".
        startprob_prior, transmat_prior: Dirichlet prior values.
        add_constant (bool): Whether to add constant/intercept to regressors.
        cat_variables (list): List of categorical columns to encode.
        n_iter (int): Maximum EM iterations.
        tol (float): Convergence tolerance.
        coefficients (list of np.ndarray): Initial state-wise VAR coefficient matrices.
        init_state (np.ndarray): Initial state probabilities.
        trans_matrix (np.ndarray): Initial transition matrix.
        random_state (int): Seed.
        verbose (bool): Print progress if True.
    """
    def __init__(self, n_components, target_col, lags, difference=None, method="posterior", covariance_type="full",
                 startprob_prior=1e3, transmat_prior=1e5, add_constant=True, cat_variables=None, lag_transform=None, seasonal_diff=None, trend=None,
                 ets_params = None, change_points=None,
                 n_iter=100, tol=1e-6, coefficients=None, init_state=None, trans_matrix=None, box_cox=None, lamda=None, box_cox_biasadj=False,
                 random_state=None, verbose=False):

        self.N = n_components
        self.target_col = target_col
        self.diffs = difference
        self.season_diff = seasonal_diff
        self.cons = add_constant
        # make sure lags is a dict mapping target columns to lists of lags even the lag is integer
        if not isinstance(lags, dict):
            raise ValueError("Lags must be a dictionary mapping target columns to lists of lags or a single integer.")
        self.lags = {col: lags[col] if isinstance(lags[col], list) else list(range(1, lags[col] + 1)) for col in lags}
        self.cat_variables = cat_variables
        self.lag_transform = lag_transform
        self.method = method
        self.iter = n_iter
        self.tol = tol
        self.verb = verbose
        self.cvr = covariance_type
        self.coeffs = coefficients
        self.box_cox = box_cox
        if lamda is None:
            self.lamda = {col: None for col in self.target_col}
        else:
            # must be dictionary mapping target columns to Box-Cox lambda values
            if not isinstance(lamda, dict):
                raise ValueError("Lambda must be a dictionary mapping target columns to Box-Cox lambda values.")
            self.lamda = lamda
        if box_cox_biasadj == False:
            self.biasadj = {col: False for col in self.target_col}
        else:
            # must be dictionary mapping target columns to boolean values
            if not isinstance(box_cox_biasadj, dict):
                raise ValueError("Box-Cox bias adjustment must be a dictionary mapping target columns to boolean values.")
            self.biasadj = box_cox_biasadj

        # Handle trend default types
        self.trend = trend
        if self.trend is not None:
            if not isinstance(self.trend, dict):
                raise TypeError("trend must be a dictionary of target values")
        self.ets_params = ets_params
        # Initialization of state, transition, coefficients and covariances
        self.cps = change_points
        self.rng = np.random.default_rng(random_state)
        if init_state is None:
            self.alpha_p = np.repeat(startprob_prior, self.N)
            self.pi = self.rng.dirichlet(self.alpha_p)
        else:
            self.pi = np.array(init_state)

        if trans_matrix is None:
            self.alpha_t = np.repeat(transmat_prior, self.N)
            self.A = self.rng.dirichlet(self.alpha_t, size=self.N)
        else:
            self.A = np.array(trans_matrix)

    # -----------------------------
    # Data preparation
    # -----------------------------
    def data_prep(self, df):
        """
        Prepare data: encode categoricals, handle differencing, add lags.
        Returns DataFrame with targets and features.
        """
        dfc = df.copy()
        if self.cat_variables is not None:
            for col, cat in self.cat_var.items():
                dfc[col] = pd.Categorical(dfc[col], categories=cat)
            dfc = pd.get_dummies(dfc, dtype=np.float64)
            for i in self.drop_categ:
                dfc.drop(list(dfc.filter(regex=i)), axis=1, inplace=True)
        if all(elem in dfc.columns for elem in self.target_col):

            # Apply Box–Cox transformation if specified
            if self.box_cox is not None:
                self.is_zero = {col: np.any(np.array(dfc[col]) < 1) for col in self.box_cox}  # check for zero or negative values
                for col in self.box_cox:
                    if self.box_cox[col]:
                        self.is_zero[col] = np.any(np.array(dfc[col]) < 1)  # check for zero or negative values
                        trans_data, self.lamda[col] = box_cox_transform(x=dfc[col],
                                                                shift=self.is_zero[col],
                                                                box_cox_lmda=self.lamda[col])
                        dfc[col] = trans_data


            if self.trend is not None:
                self.len = df.shape[0]
                self.orig_targets = {i: dfc[i] for i in self.trend.keys()}  # Store original values for later use during forecasting
                self.trend_models = {}
                for k, v in self.trend.items():
                    if v == "linear":
                        # model_fit = LinearRegression().fit(np.arange(self.len).reshape(-1, 1), self.orig_targets[k])
                        # dfc[k] = dfc[k] - model_fit.predict(np.arange(self.len).reshape(-1, 1))
                        # self.trend_models[k] = model_fit
                            # self.lr_model = LinearRegression().fit(np.arange(self.len).reshape(-1, 1), dfc[self.target_col])
                        if self.cps is not None:
                            if k in self.cps and self.cps[k]:
                                trend, model_fit = lr_trend_model(self.orig_targets[k], breakpoints=self.cps[k], type='piecewise')
                        else:
                            trend, model_fit = lr_trend_model(self.orig_targets[k])

                        dfc[k] = dfc[k] - trend
                        self.trend_models[k] = model_fit

                    elif v == "ets": # ets
                        model_fit = ExponentialSmoothing(self.orig_targets[k], **self.ets_params[k][0]).fit(**self.ets_params[k][1])
                        dfc[k] = dfc[k] - model_fit.fittedvalues.values
                        self.trend_models[k] = model_fit
                    else:
                        raise ValueError(f"Unknown trend type: {v}")

            # Apply differencing if specified
            if self.diffs is not None:
                # Save original values for inverse-differencing later for forecasting
                self.origs = {}
                for col in self.diffs.keys():
                    if self.diffs[col] is not None:
                        self.origs[col] = dfc[col].tolist()
                        dfc[col] = np.diff(dfc[col], n=self.diffs[col],
                                                    prepend=np.repeat(np.nan, self.diffs[col]))
            # Apply seasonal differencing if specified
            if self.season_diff is not None:
                # Save original values for inverse-differencing later for forecasting
                self.orig_d = {}
                for col in self.season_diff.keys():
                    if self.season_diff[col] is not None:
                        self.orig_d[col] = dfc[col].tolist()
                        dfc[col] = seasonal_diff(dfc[col], self.season_diff[col])

            # Add lags for each target
            if self.lags is not None:
                for col, lags in self.lags.items():
                    for lag in lags:
                        dfc[f"{col}_lag_{lag}"] = dfc[col].shift(lag)
            # Add lag transformations if specified
            if self.lag_transform is not None:
                for idx, (target, funcs) in enumerate(self.lag_transform.items()):
                    for func in funcs:
                        if isinstance(func, (expanding_std, expanding_mean)):
                            dfc[f"trg{idx}_{func.__class__.__name__}_shift_{func.shift}"] = func(dfc[target])
                        elif isinstance(func, expanding_quantile):
                            dfc[f"trg{idx}_{func.__class__.__name__}_shift_{func.shift}_q{func.quantile}"] = func(dfc[target])
                        elif isinstance(func, rolling_quantile):
                            dfc[f"trg{idx}_{func.__class__.__name__}_{func.window_size}_shift_{func.shift}_q{func.quantile}"] = func(dfc[target])
                        else:
                            dfc[f"trg{idx}_{func.__class__.__name__}_{func.window_size}_shift_{func.shift}"] = func(dfc[target])

            self.df = dfc.dropna()
            self.X = self.df.drop(columns=self.target_col)
            self.y = self.df[self.target_col]
            if self.cons:
                self.X = sm.add_constant(self.X)
            self.col_names = self.X.columns.tolist() if hasattr(self.X, 'columns') else [f"x{i}" for i in range(self.X.shape[1])]
            self.X = np.array(self.X)
            self.y = np.array(self.y)
            self.T = len(self.y)

            # Init coeffs/stds if None
            if self.coeffs is None:
                coeffs, covs = [], []
                for _ in range(self.N):
                    beta = np.linalg.lstsq(self.X, self.y, rcond=None)[0]
                    coeffs.append(beta)
                    resid = self.y - self.X @ beta
                    eff_df = len(resid) - self.X.shape[1]
                    eff_df = max(eff_df, 1)
                    cov_i = (resid.T @ resid) / eff_df
                    covs.append(cov_i)
                self.coeffs = np.stack(coeffs, axis=0)
                self.covs = np.array(covs)

        else:
            return dfc.dropna()

    # -----------------------------
    # Emissions
    # -----------------------------
    def _log_emissions(self):
        N, T = self.N, self.T
        logB = np.empty((N, T))
        self.fitted = np.empty((N, T, self.y.shape[1]))   # (N, T, m)
        for s in range(N):
            mus = self.X @ self.coeffs[s]
            self.fitted[s, :, :] = mus
            for t in range(T):
                logB[s, t] = multivariate_normal(mean=mus[t], cov=self.covs[s]).logpdf(self.y[t])
        return logB
    
    # -----------------------------
    # E-step: perform expectation step: forward and backward and posterior update
    # -----------------------------

    def _e_step_log(self):
        N, T = self.N, self.T
        logA  = np.log(self.A + 1e-300)
        logpi = np.log(self.pi + 1e-300)
        logB  = self._log_emissions()

        # Forward
        log_alpha = np.empty((N, T))
        log_alpha[:, 0] = logpi + logB[:, 0]
        for t in range(1, T):
            # log_alpha[:,t] = logB[:,t] + logsumexp_i( log_alpha[i,t-1] + logA[i,:] )
            log_alpha[:, t] = logB[:, t] + logsumexp(log_alpha[:, t-1][:, None] + logA, axis=0)

        # Log-likelihood
        loglik = logsumexp(log_alpha[:, -1])

        # Backward
        log_beta = np.full((N, T), 0.0)
        for t in range(T-2, -1, -1):
            # log_beta[:,t] = logsumexp_j( logA + logB[:,t+1] + log_beta[:,t+1] , axis=1 )
            log_beta[:, t] = logsumexp(logA + (logB[:, t+1] + log_beta[:, t+1])[None, :], axis=1)

        # Gamma
        log_gamma = log_alpha + log_beta - loglik
        # normalize per time to kill rounding; columns sum to 1 after exp
        log_gamma -= logsumexp(log_gamma, axis=0)
        gamma = np.exp(log_gamma)

        # Xi
        log_xi = np.empty((N, N, T-1))
        for t in range(T-1):
            tmp = log_alpha[:, t][:, None] + logA + (logB[:, t+1] + log_beta[:, t+1])[None, :]
            tmp -= logsumexp(tmp)      # normalize this slice
            log_xi[:, :, t] = tmp
        xi = np.exp(log_xi)

        # print("logB min/max:", np.min(logB), np.max(logB))
        # print("Any NaN in logB?", np.any(np.isnan(logB)))
        # print("Any Inf in logB?", np.any(np.isinf(logB)))
        # sanity checks (soft)
        assert np.allclose(gamma.sum(axis=0), 1.0, atol=1e-8)
        assert np.allclose(xi.sum(axis=(0,1)), 1.0, atol=1e-8)

        self.log_forward  = log_alpha
        self.log_backward = log_beta
        self.posterior    = gamma
        self.loglik       = loglik
        return loglik, gamma, xi
    
    # -----------------------------
    # M-step: perform expectation step: forward and backward and posterior update
    # -----------------------------

    def _m_step(self, gamma, xi):
        numer = xi.sum(axis=2)                         # (N,N)
        denom = gamma[:, :-1].sum(axis=1, keepdims=True)  # (N,1)
        A = numer / (denom + 1e-12)
        A = np.maximum(A, 1e-12)
        A /= A.sum(axis=1, keepdims=True)
        self.A = A
        self.compute_coeffs() 


    def EM(self):
        loglik, gamma, xi = self._e_step_log()
        self._m_step(gamma, xi)
        self.LL = loglik
        # return loglik
    
    # -----------------------------
    # perform iterations using EM
    # -----------------------------
        
    def fit_em(self, df_train):
        """
        Run EM iterations until convergence (log-domain version).
        """

        # Handle categorical variable encoding
        if self.cat_variables is not None:
            self.cat_var = {c: sorted(df_train[c].drop_duplicates().tolist(), key=lambda x: str(x))
                            for c in self.cat_variables}
            self.drop_categ = [sorted(df_train[col].drop_duplicates().tolist(), key=lambda x: str(x))[0]
                               for col in self.cat_variables]
        self.data_prep(df_train)

        prev_ll = -np.inf
        # store intermediate log-likelihoods
        self.log_likelihoods = []
        for it in range(self.iter):
            # loglik, gamma, xi = self._e_step_log()
            # self._m_step(gamma, xi)
            self.EM()
            if self.verb:
                print(f"Iter {it}: loglik={self.LL:.4f}")
            if it > 10:
                if abs(self.LL - prev_ll) < self.tol:
                    if self.verb:
                        print("Converged.")
                    break
            self.log_likelihoods.append(self.LL)
            prev_ll = self.LL
        # self.LL = self.LL
        return self.LL

    # -----------------------------
    # Compute coefficients and covariances
    # -----------------------------

    def compute_coeffs(self, ridge=1e-5, var_floor=1e-5, w_floor=1e-6):

        # Update regression coefficients and stds for each state

        # If posterior probabilities are shorter than the number of observations, make self.X and self.y same length visa vis make posterier same as self.X and self.y length
        if self.posterior.shape[1] < self.X.shape[0]:
            # Truncate self.X and self.y to match the length of self.posterior[i]
            self.X = self.X[:self.posterior.shape[1]]
            self.y = self.y[:self.posterior.shape[1]]
        if self.posterior.shape[1] > self.X.shape[0]:
            # Truncate self.posterior to match the length of self.X and self.y
            self.posterior = self.posterior[:, -self.X.shape[0]:]

        covs, coeffs = [], []
        X, y = self.X, self.y
        for s in range(self.N):
            w = self.posterior[s] + w_floor
            w /= w.sum()
            sw = np.sqrt(w)
            self.sw = sw
            Xw = X * sw[:, None]
            yw = y * sw[:, None]  # <-- THIS IS THE FIX
            XtX = Xw.T @ Xw + ridge * np.eye(X.shape[1])
            Xty = Xw.T @ yw
            beta_s = np.linalg.lstsq(XtX, Xty, rcond=None)[0]
            coeffs.append(beta_s)
            resid = y - X @ beta_s
            eff_df = np.sum(w) - X.shape[1]
            eff_df = max(eff_df, 1)
            cov_i = (w[:, None] * resid).T @ resid / eff_df
            # Regularize covariance matrix
            # cov_i += var_floor * np.eye(cov_i.shape[0])
            eigvals, eigvecs = np.linalg.eigh(cov_i)
            eigvals = np.clip(eigvals, var_floor, None)
            cov_i = eigvecs @ np.diag(eigvals) @ eigvecs.T

            covs.append(cov_i)
            # var_s = (w * resid**2).sum() / max(w.sum(), 1.0)
        self.coeffs = np.stack(coeffs)
        if self.cvr == "full":
            self.covs = covs
        elif self.cvr == "diag":
            self.covs = [np.diag(np.diag(cov_i)) for cov_i in covs]

    # -----------------------------
    # Fit model with learned parameters
    # -----------------------------

    def fit(self, df_train):
        """
        Refit the HMM-VAR model on new training data (log-domain version).
        """
        self.data_prep(df_train)
        # N, T = self.N, self.T
        # logpi = np.log(self.pi + 1e-300)
        # logA = np.log(self.A + 1e-300)
        # log_alpha = np.zeros((N, T))
        # logB  = self._log_emissions()

        # # Initialization
        # for i in range(N):
        #     log_alpha[i, 0] = logpi[i] + logB[i, 0]

        # # Recursion

        # for t in range(1, T):
        #     for j in range(N):
        #         log_alpha[j, t] = logB[j, t] + logsumexp(log_alpha[:, t-1] + logA[:, j])

        # # Sequence log-likelihood
        # self.LL = logsumexp(log_alpha[:, -1])
        # self.log_forward = log_alpha
        self.EM()
        return self.LL

    def predict_states(self):
        return np.argmax(self.posterior, axis=0)
    def predict_proba(self):
        return self.posterior

    # AIC computation    
    @property
    def AIC(self):
        d = self.y.shape[1]     # number of dependent variables
        r = self.X.shape[1]     # number of predictors (lags + exogenous, intercept included)
        
        # Parameters per regime (state):
        # regression coefficients + covariance parameters
        per_regime = r * d + d * (d + 1) / 2  
        
        # Total parameters:
        # - regime-specific: N * per_regime
        # - transition matrix: N*(N-1)
        # - initial distribution: (N-1)
        n_params = self.N * per_regime + self.N * (self.N - 1) + (self.N - 1)
        
        return 2 * n_params - 2 * self.LL
    
    @property
    def BIC(self):
        d = self.y.shape[1]  # number of dependent variables
        r = self.X.shape[1]  # number of predictors (lags + exogenous, intercept included)
        
        per_regime = r * d + d * (d + 1) / 2  # regression + covariance per state
        k = self.N * per_regime + self.N*(self.N-1) + (self.N-1)  # total params
        
        n = self.T  # effective number of observations
        return -2 * self.LL + k * np.log(n)
    
    def copy(self):
        return copy.deepcopy(self)
    
    # -----------------------------
    # Forecasting
    # -----------------------------
    def forecast(self, H, exog=None):
        """
        Forecast H periods ahead using fitted HMM-VAR model (log-domain version).
        Includes trend adjustment, differencing reversal, seasonal differencing,
        and Box-Cox back-transform.
        """
        if exog is not None:
            if self.cons:
                if exog.shape[0] == 1:
                    exog.insert(0, 'const', 1)
                else:
                    exog = sm.add_constant(exog)
            exog = np.array(self.data_prep(exog))
        y_dict = {col: self.y[:, i].tolist() for i, col in enumerate(self.target_col)}
        forecasts = {col: [] for col in self.target_col}

        # Init with last forward distribution (in log)
        log_forward_last = self.log_forward[:, -1]
        logA = np.log(self.A + 1e-300)
        N = self.N
        # Forward
        log_alpha = np.empty((N, H))
        log_alpha[:, 0] = logsumexp(log_forward_last[:, None] + logA, axis=0)
        for t in range(1, H):
            # log_alpha[:,t] = logB[:,t] + logsumexp_i( log_alpha[i,t-1] + logA[i,:] )
            log_alpha[:, t] = logsumexp(log_alpha[:, t-1][:, None] + logA, axis=0)

        log_alpha -= logsumexp(log_alpha, axis=0)
        self.forecast_forward = np.exp(log_alpha)
        self.state_forecasts = np.argmax(self.forecast_forward, axis=0)

        # Keep original targets if trend adjustments are needed
        if self.trend is not None:
            # orig_targets = {col: self.orig_targets[col].tolist() for col in self.trend.keys()}

            # --- Trend re-addition ---
            trend_forecasts = {}
            for col in self.trend:
                if self.trend[col] == "linear":
                    if col in self.cps and self.cps[col]:
                        trend_forecasts[col]= forecast_trend(model = self.trend_models[col], H=H, start=self.len, breakpoints=self.cps[col])
                    else:
                        trend_forecasts[col]= forecast_trend(model = self.trend_models[col], H=H, start=self.len)
                elif self.trend[col] == "ets":
                    trend_forecasts[col] = np.array(self.trend_models[col].forecast(H))

        for t in range(H):
            if exog is not None:
                exo_inp = exog[t].tolist()
            else:
                exo_inp = [1] if self.cons else []

            # Construct lag inputs
            lags = []
            for col in y_dict.keys():
                ys = [y_dict[col][-x] for x in self.lags[col]]
                lags.extend(ys)

            # Lag transforms
            transform_lag = []
            if self.lag_transform is not None:
                for target, funcs in self.lag_transform.items():
                    series_array = np.array(y_dict[target])
                    for func in funcs:
                        transform_lag.append(func(series_array, is_forecast=True).to_numpy()[-1])

            inp = np.array(exo_inp + lags + transform_lag)
            state_preds = {col: np.zeros(self.N) for col in self.target_col}

            for j in range(self.N):
                mus = inp @ self.coeffs[j]  # shape: (n_targets,)
                for i, col in enumerate(self.target_col):
                    state_preds[col][j] = mus[i]


            # Weighted prediction per target
            for col in self.target_col:
                pred = np.sum(self.forecast_forward[:, t] * state_preds[col])
                forecasts[col].append(pred)
                y_dict[col].append(pred)

        # trend adjustments
        if self.trend is not None:
            for col in self.trend.keys():
                forecasts[col] += trend_forecasts[col]

        # --- Post-processing transforms ---
        if self.season_diff is not None:
            for col in self.season_diff.keys():
                forecasts[col] = invert_seasonal_diff(self.orig_d[col], forecasts[col], self.season_diff[col])

        if self.diffs is not None:
            for col in self.diffs.keys():
                forecasts[col] = undiff_ts(self.origs[col], forecasts[col], self.diffs[col])

        if self.box_cox is not None:
            for col in self.box_cox:
                if self.box_cox[col]:
                    forecasts[col] = back_box_cox_transform(
                        y_pred=forecasts[col],
                        lmda=self.lamda[col],
                        shift=self.is_zero[col],
                        box_cox_biasadj=self.biasadj[col]
                    )

        return forecasts

### Slow version of forecast with ets and LR

    # def forecast(self, H, exog=None):
    #     """
    #     Forecast H periods ahead using fitted HMM regression (log-domain version), with advanced post-processing.

    #     Handles:
    #     - Trend re-adjustment (linear/ETS)
    #     - Seasonal differencing reversal
    #     - Regular differencing reversal
    #     - Box-Cox back-transform
    #     - Exogenous variables
    #     - Lag transformations
    #     """
    #     y_list = self.y.tolist()
    #     forecasts_ = []
    #     N = self.N

    #     # Prepare exogenous future regressors if provided
    #     if exog is not None:
    #         if self.cons:
    #             if exog.shape[0] == 1:
    #                 exog.insert(0, 'const', 1)
    #             else:
    #                 exog = sm.add_constant(exog)
    #         exog = np.array(self.data_prep(exog))

    #     # Prepare for trend adjustment
    #     # This assumes you stored original target (pre-trend removal) in self.target_orig
    #     if hasattr(self, 'target_orig') and self.trend is not None:
    #         orig_targets = self.target_orig.tolist()  # Used to re-add trend during forecasting

    #     # Init with last forward distribution (in log)
    #     log_forward_last = self.log_forward[:, -1]
    #     logA = np.log(self.A + 1e-300)

    #     # Forward
    #     log_alpha = np.empty((N, H))
    #     log_alpha[:, 0] = logsumexp(log_forward_last[:, None] + logA, axis=0)
    #     for t in range(1, H):
    #         # log_alpha[:,t] = logB[:,t] + logsumexp_i( log_alpha[i,t-1] + logA[i,:] )
    #         log_alpha[:, t] = logsumexp(log_alpha[:, t-1][:, None] + logA, axis=0)

    #     log_alpha -= logsumexp(log_alpha, axis=0)
    #     self.forecast_forward = np.exp(log_alpha)
    #     self.state_forecasts = np.argmax(self.forecast_forward, axis=0)


    #     for t in range(H):
    #         if exog is not None:
    #             exo_inp = exog[t].tolist()
    #         else:
    #             exo_inp = [1] if self.cons else []
    #         lags = [y_list[-l] for l in self.lags]
    #         transform_lag = []
    #         if self.lag_transform is not None:
    #             series_array = np.array(y_list)
    #             for func in self.lag_transform:
    #                 transform_lag.append(func(series_array, is_forecast=True).to_numpy()[-1])
    #         inp = np.array(exo_inp + lags + transform_lag)

    #         state_preds = np.zeros(N)
    #         for j in range(N):
    #             mu = np.dot(self.coeffs[j], inp)
    #             state_preds[j] = mu

    #         # normalize to probabilities

    #         # normalize to probabilities
    #         pred_w = np.sum(self.forecast_forward[:, t] * state_preds)
    #         forecasts_.append(pred_w)
    #         y_list.append(pred_w)

    #         # --- Trend re-adjustment ---
    #         if self.trend is not None:
    #             if self.trend == "linear":
    #                 # Fit a linear trend on original targets
    #                 trend_fit = LinearRegression().fit(
    #                     np.arange(len(orig_targets)).reshape(-1, 1),
    #                     np.array(orig_targets)
    #                 )
    #                 trend_forecast = trend_fit.predict(np.array([[len(orig_targets)]]))[0]
    #             elif self.trend == "ets":
    #                 # Fit an ETS model and forecast next point
    #                 trend_fit = ExponentialSmoothing(
    #                     np.array(orig_targets),
    #                     **self.ets_model
    #                 ).fit(**self.ets_fit)
    #                 trend_forecast = trend_fit.forecast(1)[0]
    #             else:
    #                 trend_forecast = 0.0  # fallback

    #             orig_forecast = pred_w + trend_forecast
    #             forecasts_[-1] = orig_forecast    # overwrite with trend-adjusted value
    #             orig_targets.append(orig_forecast)  # update for next lag/trend step

    #         # log_forward_last = log_f_t.copy()

    #     forecasts = np.array(forecasts_)

    #     # --- Revert seasonal differencing if applied ---
    #     if self.season_diff is not None:
    #         forecasts = invert_seasonal_diff(self.orig_d, forecasts, self.season_diff)

    #     # --- Revert regular differencing if applied ---
    #     if self.diff is not None:
    #         forecasts = undiff_ts(self.orig, forecasts, self.diff)

    #     # --- Box-Cox back-transform if applied ---
    #     if self.box_cox:
    #         forecasts = back_box_cox_transform(
    #             y_pred=forecasts, lmda=self.lamda,
    #             shift=self.is_zero, box_cox_biasadj=self.biasadj
    #         )

    #     return forecasts
