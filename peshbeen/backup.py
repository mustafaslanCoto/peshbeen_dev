import copy
import statsmodels.api as sm
from scipy.stats import norm, multivariate_normal
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy.special import logsumexp

class MsHmmRegression_tst:
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
        switching_var (bool): If True, allows if variance to switch between states.
        verbose (bool): Print progress if True.
    """

    def __init__(self, n_components, target_col, lags, method="posterior",
                 startprob_prior=1e3, transmat_prior=1e5, add_constant=True,
                 difference=None, trend=None, ets_params = None, change_points=None,
                 cat_variables=None, lag_transform=None, n_iter=100, tol=1e-6,
                 coefficients=None, stds=None, init_state=None, trans_matrix=None,
                 box_cox=False, lamda=None, box_cox_biasadj=False, season_diff=None, 
                 random_state=None, switching_var=True, verbose=False):
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
        self.switching_var = switching_var
        self.verb = verbose


        # RNG for reproducibility
        self.rng = np.random.default_rng(random_state)
        if init_state is None:
            self.sp = startprob_prior
            self.alpha_p = np.repeat(self.sp, self.N)
            self.pi = self.rng.dirichlet(self.alpha_p) # Initial state probabilities using Dirichlet distribution
        else:
            self.pi = np.array(init_state)
        # if trans_matrix is None:
        #     self.tm = transmat_prior
        #     self.alpha_t = np.repeat(self.tm, self.N) # 
        #     self.A = self.rng.dirichlet(self.alpha_t, size=self.N)
        # else:
        #     self.A = np.array(trans_matrix)

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
        weighted_resid_all = []
        weights_all = []
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
            # var_s = (w * resid**2).sum() / max((w.sum()-beta_s.shape[0]), 1.0)
            # stds.append(np.sqrt(max(var_s, var_floor)))
            weighted_resid_all.append(w * resid**2)
            weights_all.append(w)

            if self.switching_var:
                var_s = (w * resid**2).sum() / max((w.sum()-beta_s.shape[0]), 1.0)
                stds.append(np.sqrt(max(var_s, var_floor)))

        if not self.switching_var:
            # pooled variance across all states
            pooled_var = (np.sum(weighted_resid_all)) / (np.sum(weights_all) - X.shape[1])
            stds = [np.sqrt(max(pooled_var, var_floor))] * self.N

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

    def _e_step_log(self, tm):
        N, T = self.N, self.T
        logA  = np.log(tm + 1e-300)
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

        # print("logB min/max:", np.min(logB), np.max(logB))
        # print("Any NaN in logB?", np.any(np.isnan(logB)))
        # print("Any Inf in logB?", np.any(np.isinf(logB)))
        # sanity checks (soft)
        assert np.allclose(gamma.sum(axis=0), 1.0, atol=1e-8)

        self.log_forward  = log_alpha
        self.log_backward = log_beta
        self.posterior    = gamma
        self.loglik       = loglik
        return loglik, gamma

    def tm(self):
        alpha = np.ones(self.N)
        matrix = np.zeros((self.N, self.N))
        for i in range(self.N):
            matrix[i, :] = np.random.dirichlet(alpha)
        return matrix

    def _m_step(self, gamma):
        self.pi = gamma[:, 0] / gamma[:, 0].sum()
        self.compute_coeffs() 


    def EM(self, tm):
        loglik, gamma = self._e_step_log(tm)
        self._m_step(gamma)
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
        self.best_LL = -np.inf
        # store intermediate log-likelihoods
        self.log_likelihoods = []
        for it in range(self.iter):
            # loglik, gamma, xi = self._e_step_log()
            # self._m_step(gamma, xi)
            tm = self.tm()
            self.EM(tm)
            if self.LL > prev_ll:
                self.A = tm
                if it > 10 and abs(self.LL - prev_ll) < self.tol:
                    break
                prev_ll = self.LL
                self.best_LL = self.LL
                if self.verb:
                    print(f"Iter {it}: loglik={prev_ll:.4f}")
            self.log_likelihoods.append(self.LL)
        return self.best_LL

    def fit(self, df, n_iter=1):
        """
        Refit the HMM regression model on new training data (log-domain version).
        """
        if n_iter < 1:
            raise ValueError("n_iter must be at least 1.")
        
        self.data_prep(df)
        if n_iter > 1:
            prev_ll = self.best_LL
            for _ in range(n_iter):
                self.EM(self.A)
                if abs(self.LL - prev_ll) < self.tol:
                    break
                else:
                    prev_ll = self.LL
        else:
            self.EM(self.A)

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
        #per state forecasts
        self.forecast_ps = np.zeros((N, H))

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
            self.forecast_ps[:, t] = state_preds

            # normalize to probabilities

            # normalize to probabilities
            pred_w = np.sum(self.forecast_forward[:, t] * state_preds)
            forecasts_.append(pred_w)
            y_list.append(pred_w)

            # log_forward_last = log_f_t.copy()

        forecasts = np.array(forecasts_)

        if self.trend is not None:
            forecasts += trend_forecast
            self.forecast_ps += trend_forecast

        # --- Revert seasonal differencing if applied ---
        if self.season_diff is not None:
            forecasts = invert_seasonal_diff(self.orig_d, forecasts, self.season_diff)
            # Also revert seasonal differencing for per state forecasts
            for s in range(self.N):
                self.forecast_ps[s] = invert_seasonal_diff(self.orig_d, self.forecast_ps[s], self.season_diff)

        # --- Revert regular differencing if applied ---
        if self.diff is not None:
            forecasts = undiff_ts(self.orig, forecasts, self.diff)
            # Also revert differencing for per state forecasts
            for s in range(self.N):
                self.forecast_ps[s] = undiff_ts(self.orig, self.forecast_ps[s], self.diff)

        # --- Box-Cox back-transform if applied ---
        if self.box_cox:
            forecasts = back_box_cox_transform(
                y_pred=forecasts, lmda=self.lamda,
                shift=self.is_zero, box_cox_biasadj=self.biasadj
            )
            for s in range(self.N):
                self.forecast_ps[s] = back_box_cox_transform(
                    y_pred=self.forecast_ps[s], lmda=self.lamda,
                    shift=self.is_zero, box_cox_biasadj=self.biasadj
                )

        return forecasts