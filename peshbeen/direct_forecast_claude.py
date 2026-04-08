class ml_direct_forecaster:

    def __init__(
        self,
        model: Any,
        target_col: str,
        H: int,
        lags: Optional[Union[int, List[int]]] = None,
        lag_transform: Optional[list] = None,
        difference: Optional[int] = None,
        seasonal_diff: Optional[int] = None,
        trend: Optional[str] = None,
        pol_degree: int = 1,
        ets_params: Optional[Dict[str, Any]] = None,
        change_points: Optional[List[int]] = None,
        box_cox: Union[bool, float, int] = False,
        box_cox_biasadj: bool = False,
        cat_variables: Optional[List[str]] = None,
        target_encode: bool = False,
    ) -> None:

        """
        Initialize the ml_direct_forecaster with the specified model and preprocessing options.
        Unlike ml_forecaster, this class uses a direct forecasting strategy: a separate model
        is trained for each horizon h=1, ..., H, eliminating recursive error accumulation.

        Parameters
        ----------
        model : Any
            A regression model object (e.g. LGBMRegressor(), XGBRegressor(), LinearRegression(), etc.)
        target_col : str
            Name of the target variable column in the input DataFrame.
        H : int
            Forecast horizon. A separate model is trained for each step 1 through H.
        lags : int or list of int, optional
            Lags to include as features. Default is None.
        lag_transform : list of function objects, optional
            Lag-transform functions to apply to the target variable. Default is None.
        difference : int, optional
            Order of ordinary differencing. Default is None.
        seasonal_diff : int, optional
            Seasonal period for seasonal differencing. Default is None.
        trend : str, optional
            Trend strategy: 'linear' or 'ets'. Default is None.
        pol_degree : int, optional
            Polynomial degree for linear trend. Default is 1.
        ets_params : dict, optional
            Parameters for ExponentialSmoothing when trend='ets'. Default is None.
        change_points : list of int, optional
            Breakpoint indices for piecewise linear trend. Default is None.
        box_cox : bool or float or int, optional
            Box-Cox transformation. If float/int, used as lambda. If True, lambda is estimated. Default is False.
        box_cox_biasadj : bool, optional
            Bias adjustment when inverting Box-Cox. Default is False.
        cat_variables : list of str, optional
            Categorical feature column names. Default is None.
        target_encode : bool, optional
            Use target encoding instead of one-hot encoding. Default is False.
        """

        self.model = model
        self.target_col = target_col
        self.H = H
        self.cat_variables = cat_variables
        self.target_encode = target_encode
        self.cps = change_points
        self.pol = pol_degree

        if isinstance(box_cox, (float, int)) and box_cox is not True and box_cox is not False:
            self.box_cox = True
            self.lamda = box_cox
        else:
            self.box_cox = box_cox
            self.lamda = None

        self.biasadj = box_cox_biasadj
        self.difference = difference
        self.season_diff = seasonal_diff
        self.lag_transform = lag_transform

        # ── trend ─────────────────────────────────────────────────────────────
        self.trend = trend
        if self.trend == "ets":
            self.ets_model = {}
            self.ets_fit = {}
            if ets_params is not None:
                if not isinstance(ets_params, dict):
                    raise TypeError("ets_params must be a dictionary.")
                constructor_params = ["trend", "damped_trend", "seasonal", "seasonal_periods",
                                      "initialization_method", "initial_level", "initial_trend",
                                      "initial_seasonal", "bounds", "dates", "freq", "missing"]
                fit_params = ["optimized", "smoothing_level", "smoothing_trend", "smoothing_seasonal",
                              "damping_trend", "remove_bias", "start_params", "method",
                              "minimize_kwargs", "use_brute"]
                for param in constructor_params:
                    if param in ets_params:
                        self.ets_model[param] = ets_params[param]
                for param in fit_params:
                    if param in ets_params:
                        self.ets_fit[param] = ets_params[param]

        # ── lags ──────────────────────────────────────────────────────────────
        if lags is None:
            self.n_lag = None
        elif isinstance(lags, int):
            self.n_lag = list(range(1, lags + 1))
        elif isinstance(lags, list):
            if not all(isinstance(l, int) for l in lags):
                raise TypeError("lags list must contain only integers.")
            self.n_lag = lags
        else:
            raise TypeError("lags must be an int or a list of ints.")

        # ── placeholders ──────────────────────────────────────────────────────
        self.tuned_params = None
        self.actuals = None
        self.direct_models = {}   # stores one fitted model per horizon h

    # ─────────────────────────────────────────────────────────────────────────
    # DATA PREPARATION
    # identical to ml_forecaster.data_prep — same transformations, same order
    # ─────────────────────────────────────────────────────────────────────────

    def data_prep(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare the data by applying transformations and lag feature engineering.
        Identical pipeline to ml_forecaster — Box-Cox, trend removal, differencing,
        lag features, lag transforms — in the same order.
        """
        dfc = df.copy()

        # ── categorical encoding ──────────────────────────────────────────────
        if self.cat_variables is not None:
            if self.target_encode:
                for col in self.cat_variables:
                    encode_col = col + "_target_encoded"
                    dfc[encode_col] = kfold_target_encoder(dfc, col, self.target_col, 36)
                self.df_encode = dfc.copy()
                dfc = dfc.drop(columns=self.cat_variables)
            else:
                if isinstance(self.model, (CatBoostRegressor, LGBMRegressor)):
                    for col in self.cat_variables:
                        dfc[col] = dfc[col].astype('category')
                else:
                    for col, cats in self.cat_var.items():
                        dfc[col] = pd.Categorical(dfc[col], categories=cats)
                    dfc = pd.get_dummies(dfc, dtype=float)
                    if isinstance(self.model, (LinearRegression, Ridge, Lasso, ElasticNet)):
                        for pat in self.drop_categ_patterns:
                            cols = list(dfc.filter(regex=pat).columns)
                            if cols:
                                dfc.drop(cols, axis=1, inplace=True)

        if self.target_col not in dfc.columns:
            return dfc.dropna()

        self.orig_target = dfc[self.target_col].values

        # ── Box-Cox ───────────────────────────────────────────────────────────
        if self.box_cox:
            self.is_zero = np.any(np.array(dfc[self.target_col]) < 1)
            self.trans_data, self.lamda = box_cox_transform(
                x=dfc[self.target_col], shift=self.is_zero, box_cox_lmda=self.lamda
            )
            dfc[self.target_col] = self.trans_data

        # ── Trend removal ─────────────────────────────────────────────────────
        if self.trend is not None:
            self.len = len(df)
            self.target_orig = dfc[self.target_col].copy()
            if self.trend == "linear":
                if self.cps is not None:
                    self.trend_vals, self.lr_model, self.X_trend = lr_trend_model(
                        dfc[self.target_col], degree=self.pol,
                        breakpoints=self.cps, type='piecewise'
                    )
                else:
                    self.trend_vals, self.lr_model, self.X_trend = lr_trend_model(
                        dfc[self.target_col], degree=self.pol
                    )
            elif self.trend == "ets":
                self.ets_model_fit = ExponentialSmoothing(
                    dfc[self.target_col], **self.ets_model
                ).fit(**self.ets_fit)
                self.trend_vals = self.ets_model_fit.fittedvalues.values
            else:
                raise ValueError(f"Unknown trend type '{self.trend}'. Use 'linear' or 'ets'.")
            dfc[self.target_col] = dfc[self.target_col] - self.trend_vals

        # ── Ordinary differencing ─────────────────────────────────────────────
        if self.difference is not None or self.season_diff is not None:
            self.orig = dfc[self.target_col].tolist()
            if self.difference is not None:
                dfc[self.target_col] = np.diff(
                    dfc[self.target_col], n=self.difference,
                    prepend=np.repeat(np.nan, self.difference)
                )
            if self.season_diff is not None:
                self.orig_d = dfc[self.target_col].tolist()
                dfc[self.target_col] = seasonal_diff(dfc[self.target_col], self.season_diff)

        # ── Lag features ──────────────────────────────────────────────────────
        if self.n_lag is not None:
            for lag in self.n_lag:
                dfc[f"{self.target_col}_lag_{lag}"] = dfc[self.target_col].shift(lag)

        # ── Lag transforms ────────────────────────────────────────────────────
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

        return dfc.dropna()

    # ─────────────────────────────────────────────────────────────────────────
    # FIT
    # Train one model per horizon h by shifting the target h steps forward
    # ─────────────────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit a separate model for each horizon h in 1..H.
        For each h, the target is shifted h steps forward so the model learns
        to predict the value h steps ahead directly from the current lag features,
        bypassing recursive error accumulation.

        Parameters
        ----------
        df : pd.DataFrame
            Training DataFrame containing the target and any feature columns.
        """
        # Build categorical lookup for non-native-cat models
        if isinstance(self.model, (
            XGBRegressor, RandomForestRegressor, Cubist,
            HistGradientBoostingRegressor, AdaBoostRegressor,
            LinearRegression, Ridge, Lasso, ElasticNet
        )):
            if self.cat_variables is not None and not self.target_encode:
                self.cat_var = {
                    c: sorted(df[c].drop_duplicates().tolist())
                    for c in self.cat_variables
                }
                if isinstance(self.model, (LinearRegression, Ridge, Lasso, ElasticNet)):
                    self.drop_categ_patterns = []
                    for c in self.cat_variables:
                        base = sorted(df[c].drop_duplicates().tolist())[0]
                        self.drop_categ_patterns.append(rf"^{re.escape(c)}_{re.escape(str(base))}$")

        # Run data_prep once to get the transformed feature matrix (lags, transforms, etc.)
        # We reuse X across all horizons — only y changes (shifted per horizon h)
        base_df = self.data_prep(df)
        self.X = base_df.drop(columns=[self.target_col])  # features are horizon-agnostic
        self.feature_cols = self.X.columns.tolist()

        # Store state needed for forecast post-processing
        self.y_base = base_df[self.target_col]

        self.direct_models = {}

        for h in range(1, self.H + 1):
            # Shift target h steps forward: row i now contains the value h steps ahead
            df_h = base_df.copy()
            df_h[self.target_col] = base_df[self.target_col].shift(-h)
            df_h = df_h.dropna()

            X_h = df_h.drop(columns=[self.target_col])
            y_h = df_h[self.target_col]

            model_h = copy.deepcopy(self.model)

            if isinstance(model_h, LGBMRegressor):
                fitted_h = model_h.fit(X_h, y_h, categorical_feature=self.cat_variables)
            elif isinstance(model_h, CatBoostRegressor):
                fitted_h = model_h.fit(X_h, y_h, cat_features=self.cat_variables, verbose=False)
            else:
                fitted_h = model_h.fit(X_h, y_h)

            self.direct_models[h] = fitted_h

    # ─────────────────────────────────────────────────────────────────────────
    # FORECAST
    # Each horizon h is predicted by its own model using today's lag features
    # ─────────────────────────────────────────────────────────────────────────

    def forecast(
        self,
        H: int,
        exog: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """
        Generate direct multi-step forecasts. Each horizon h is predicted
        independently by its own model using the most recent lag features —
        no predictions are fed back as inputs.

        Parameters
        ----------
        H : int
            Forecast horizon. Must be <= self.H (models are only trained up to self.H).
        exog : pd.DataFrame or None
            Optional future exogenous variables (H rows).

        Returns
        -------
        np.ndarray
            Forecast values of length H.
        """
        if H > self.H:
            raise ValueError(
                f"H={H} exceeds the training horizon self.H={self.H}. "
                f"Re-fit with a larger H to forecast further ahead."
            )

        if not self.direct_models:
            raise ValueError("Model has not been fitted yet. Call .fit() before .forecast().")

        # ── Prepare exog ──────────────────────────────────────────────────────
        if exog is not None:
            if self.cat_variables is not None:
                if self.target_encode:
                    for col in self.cat_variables:
                        encode_col = col + "_target_encoded"
                        exog[encode_col] = target_encoder_for_test(self.df_encode, exog, col)
                    exog = exog.drop(columns=self.cat_variables)
                else:
                    if isinstance(self.model, (
                        XGBRegressor, RandomForestRegressor, Cubist,
                        HistGradientBoostingRegressor, AdaBoostRegressor,
                        LinearRegression, Ridge, Lasso, ElasticNet
                    )):
                        exog = self.data_prep(exog)

        # ── Build input row from most recent lags ─────────────────────────────
        # Use the last row of the training feature matrix as the forecast input.
        # This contains the most recent lag values computed during fit.
        last_features = self.X.iloc[[-1]]  # shape (1, n_features)

        # ── Pre-compute trend forecasts ───────────────────────────────────────
        if self.trend is not None:
            if self.trend == "linear":
                trend_forecast, _ = forecast_trend(
                    model=self.lr_model, H=H, start=self.len,
                    degree=self.pol, breakpoints=self.cps
                )
            else:
                trend_forecast = np.array(self.ets_model_fit.forecast(H))

        # ── Direct forecast: one model call per horizon ───────────────────────
        predictions = []

        for h in range(1, H + 1):
            # Merge exog features for this step if provided
            if exog is not None:
                exog_row = exog.iloc[[h - 1]].reset_index(drop=True)
                inp = pd.concat(
                    [exog_row, last_features.reset_index(drop=True)], axis=1
                )[self.feature_cols]
            else:
                inp = last_features.copy()

            if isinstance(self.model, (LGBMRegressor, CatBoostRegressor)):
                for c in inp.columns:
                    if c in (self.cat_variables or []):
                        inp[c] = inp[c].astype(int).astype('category')
                    else:
                        inp[c] = inp[c].astype('float64')

            pred = self.direct_models[h].predict(inp)[0]
            predictions.append(pred)

        # ── Post-processing (same order as ml_forecaster) ─────────────────────
        forecasts = np.array(predictions)

        if self.trend is not None:
            forecasts = forecasts + trend_forecast

        if self.season_diff is not None:
            forecasts = invert_seasonal_diff(self.orig_d, forecasts, self.season_diff)

        if self.difference is not None:
            forecasts = undiff_ts(self.orig, forecasts, self.difference)

        forecasts = np.array([max(0, x) for x in forecasts])

        if self.box_cox:
            forecasts = back_box_cox_transform(
                y_pred=forecasts, lmda=self.lamda,
                shift=self.is_zero, box_cox_biasadj=self.biasadj
            )

        return forecasts

    def copy(self):
        return copy.deepcopy(self)

    def get_name(self):
        return "ml_direct_forecaster"