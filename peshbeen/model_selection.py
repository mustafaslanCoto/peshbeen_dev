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
# Feature Selection Algorithms
# ------------------------------------------------------------------------------

def forward_feature_selection(df, n_folds = None, H = None, model = None, metrics = None,
                                  lags_to_consider = None, candidate_features = None, transformations = None, 
                                    step_size = None, verbose = False):
    """
    Performs forward lag/feature/transform selection for Regression models.
    Parameters:
        df (pd.DataFrame): DataFrame containing the time series data.
        n_folds (int, optional): Number of cross-validation folds.
        H (int, optional): Forecast horizon.
        model: Model to be used for training and evaluation.
        metrics (list, optional): List of metrics to evaluate the model.
        lags_to_consider (list, optional): List of lags to consider for feature selection.
        candidate_features (list, optional): List of candidate exogenous features.
        transformations (list, optional): List of transformations to apply.
        step_size (int, optional): Step size for rolling window.
        verbose (bool, optional): Whether to print progress messages.
    Returns:
        dict: Dictionary of best features
    """


    if lags_to_consider is not None:
        remaining_lags = list(range(1, lags_to_consider + 1))
        model.n_lag = None # Start with no lags
    if candidate_features is not None:
        candidate_features = candidate_features.copy()
        df = df.drop(columns=candidate_features)
        df_orig = df.copy() # Keep original for feature add-back
    if transformations is not None:
        transformations = transformations.copy()
        model.lag_transform = None # Start with no transformations
    best_features = {"best_lags": [], "best_exogs": [], "best_transforms": []}
    best_score = [float('inf')] * len(metrics)

    while True:
        improvement = False
        candidate = {'type': None, 'name': None}
        scores = best_score

        # Test Lags
        if lags_to_consider is not None:
            for lag in remaining_lags:
                current_lags = sorted(best_features["best_lags"] + [lag])
                model_test = model.copy()
                model_test.n_lag = current_lags
                my_cv = cross_validate(model=model_test, df=df, cv_split=n_folds,
                                       test_size=H, metrics=metrics, step_size=step_size)
                score = my_cv["score"].tolist()
                # print(f'testing lag: {lag} with score: {score}')
                if score < scores:
                    scores = score
                    candidate = {'type': 'lag', 'name': lag}
                    # print(candidate["type"], candidate["name"], score)
                    improvement = True

        # Test Exogenous Features
        if candidate_features is not None:
            for feat in candidate_features:
                df_test = df.copy()
                df_test[feat] = df_orig[feat]
                model_test = model.copy()
                my_cv = cross_validate(model=model_test, df=df_test, cv_split=n_folds,
                                       test_size=H, metrics=metrics, step_size=step_size)
                score = my_cv["score"].tolist()
                if score < scores:
                    scores = score
                    candidate = {'type': 'exog', 'name': feat}
                    improvement = True

        # Test Transformations
        if transformations is not None:
            for trans in transformations:
                model_test = model.copy()
                lag_transform = (model_test.lag_transform or []) + [trans]
                model_test.lag_transform = lag_transform
                my_cv = cross_validate(model=model_test, df=df, cv_split=n_folds,
                                       test_size=H, metrics=metrics, step_size=step_size)
                score = my_cv["score"].tolist()
                # print(f'testing transformation: {trans.get_name()} with score: {score}')
                if score < scores:
                    scores = score
                    candidate = {'type': 'transform', 'name': trans}
                    # print(candidate["type"], candidate["name"].get_name(), score)
                    improvement = True

        # Update best features
        if improvement:
            best_score = scores
            if candidate['type'] == 'lag':
                best_features["best_lags"].append(candidate['name'])
                remaining_lags.remove(candidate['name'])
            elif candidate['type'] == 'exog':
                best_features["best_exogs"].append(candidate['name'])
                candidate_features.remove(candidate['name'])
                df[candidate['name']] = df_orig[candidate['name']]
            elif candidate['type'] == 'transform':
                best_features["best_transforms"].append(candidate['name'])
                transformations.remove(candidate['name'])
                if model.lag_transform is None:
                    model.lag_transform = [candidate['name']]
                else:
                    model.lag_transform.append(candidate['name'])

            if verbose:
                if candidate['type'] == 'transform':
                    print(f"Added {candidate['type']}: {candidate['name'].get_name()} with score: {best_score}")
                else:
                    print(f"Added {candidate['type']}: {candidate['name']} with score: {best_score}")
        else:
            break  # No improvement

    if transformations is not None and best_features["best_transforms"]:
        best_features["best_transforms"] = [trans.get_name() for trans in best_features["best_transforms"]]
        
    if lags_to_consider is not None and best_features["best_lags"]:
        best_features["best_lags"].sort()

    return best_features



def backward_feature_selection(df, n_folds = None, H = None, model = None, metrics = None,
                                  lags_to_consider = None, candidate_features = None, transformations = None, 
                                    step_size = None, verbose = False):
    """
    Performs backward lag selection for Regression models.
    Parameters:
        df (pd.DataFrame): DataFrame containing the time series data.
        n_folds (int, optional): Number of cross-validation folds.
        H (int, optional): Forecast horizon.
        model: Model to be used for training and evaluation.
        metrics (list, optional): List of metrics to evaluate the model.
        lags_to_consider (list, optional): List of lags to consider for feature selection.
        candidate_features (list, optional): List of candidate exogenous features.
        transformations (list, optional): List of transformations to apply.
        step_size (int, optional): Step size for rolling window.
        verbose (bool, optional): Whether to print progress messages.
    Returns:
        dict: Dictionary of best features
    """
    remaining_lags = list(range(1, lags_to_consider + 1)) if lags_to_consider is not None else []
    candidate_features = candidate_features.copy() if candidate_features is not None else []
    transformations = transformations.copy() if transformations is not None else None
    best_features = {"best_lags": remaining_lags, "best_exogs": candidate_features, "best_transforms": transformations}

    ## setting the full model
    # model_full = model.copy()
    if lags_to_consider is not None:
        model.n_lag = remaining_lags # Start with all lags to consider
    if transformations is not None:
        model.lag_transform = transformations # Start with all transformations

    best_score = list(np.repeat(float('inf'), len(metrics)))

    # best_lags = None
    while True:
        improvement = False
        candidate = {'type': None, 'name': None}
        scores = best_score
        if best_features["best_lags"]:
            for lg in best_features["best_lags"]:
                lags_to_test = [x for x in best_features["best_lags"] if x != lg]
                lags_to_test.sort()
                model_test = model.copy()
                model_test.n_lag = lags_to_test
                my_cv = cross_validate(model=model_test, df=df, cv_split=n_folds,
                                    test_size=H, metrics=metrics, step_size=step_size)
                score = my_cv["score"].tolist()
                # print(f"len of lags_to_test: {len(lags_to_test)} and score: {score}")
                if score < scores:
                    scores = score
                    candidate = {'type': 'lag', 'name': lg}
                    improvement = True
        if best_features["best_transforms"]:
            for trans in best_features["best_transforms"]:
                trans_to_test = [x for x in best_features["best_transforms"] if x != trans]
                model_test = model.copy()
                model_test.lag_transform = trans_to_test
                my_cv = cross_validate(model=model_test, df=df, cv_split=n_folds,
                                    test_size=H, metrics=metrics, step_size=step_size)
                score = my_cv["score"].tolist()
                if score < scores:
                    scores = score
                    candidate = {'type': 'transform', 'name': trans}
                    improvement = True
        if best_features["best_exogs"]:
            for feat in best_features["best_exogs"]:
                # feat_to_test = [x for x in candidate_features if x != feat]
                df_test = df.drop(columns=feat)
                model_test = model.copy()
                my_cv = cross_validate(model=model_test, df=df_test, cv_split=n_folds,
                                    test_size=H, metrics=metrics, step_size=step_size)
                score = my_cv["score"].tolist()
                if score < scores:
                    scores = score
                    candidate = {'type': 'exog', 'name': feat}
                    improvement = True

        # Update best features
        if improvement and candidate['type']:
            best_score = scores
            if candidate['type'] == 'lag':
                best_features["best_lags"].remove(candidate['name'])
            elif candidate['type'] == 'exog':
                best_features["best_exogs"].remove(candidate['name'])
                df = df.drop(columns=candidate['name'])
            elif candidate['type'] == 'transform':
                best_features["best_transforms"].remove(candidate['name'])
                if not best_features["best_transforms"]:
                    model.lag_transform = best_features["best_transforms"]
                else:
                    model.lag_transform = None

            if verbose:
                if candidate['type'] == 'transform':
                    print(f"Removed {candidate['type']}: {candidate['name'].get_name()} with score: {best_score}")
                else:
                    print(f"Removed {candidate['type']}: {candidate['name']} with score: {best_score}")
        else:
            break  # No improvement

    if transformations is not None and best_features["best_transforms"]:
        best_features["best_transforms"] = [trans.get_name() for trans in best_features["best_transforms"]]
    if lags_to_consider is not None and best_features["best_lags"]:
        best_features["best_lags"].sort()
    return best_features


def mv_forward_feature_selection(df, target_col, n_folds = None, H = None, model = None, metrics = None,
                                  lags_to_consider = None, candidate_features = None, transformations = None, 
                                    step_size = None, verbose = False):
    """
    Performs forward lag selection for multivariate models 
    Parameters:
        df (pd.DataFrame): DataFrame containing the time series data.
        target_col (str): The target column for accuracy evaluation.
        n_folds (int): Number of folds for cross-validation.
        H (int): Forecast horizon.
        model: The forecasting model to be used.
        metrics (list): List of metrics to evaluate the model.
        lags_to_consider (dict): Dictionary of maximum lags for each variable.
        candidate_features (list): List of candidate exogenous features.
        transformations (list): List of transformations to consider.
        step_size (int, optional): Step size for lag selection. Defaults to None.
        verbose (bool, optional): Whether to print progress. Defaults to False.
    Returns:
        dict: Dictionary of best features for each variable.
    """

    # max_lag = sum(x for x in max_lags.values())
    
    # lags = list(range(1, max_lags+1))

    best_features = {"best_lags": {i: [] for i in lags_to_consider if lags_to_consider is not None}, "best_transforms": {i: [] for i in transformations if transformations is not None}, "best_exogs": []}
    remaining_lags = {i:list(range(1, j+1)) for i, j in lags_to_consider.items()}
    best_score = list(np.repeat(float('inf'), len(metrics)))

    # Keep original for feature add-back
    df_orig = df.copy()

    # Drop candidate features initially
    if candidate_features:
        df = df.drop(columns=candidate_features) # Drop candidate features to start with feature selection
    if transformations is not None:
        model.lag_transform = None # Start with no transformations
    if lags_to_consider is not None:
        model.n_lag = None # Start with no lags

    # while max_lag>0:
    while True:
        improvement = False
        candidate = {'target': None, 'type': None, 'name': None}
        scores = best_score
        if lags_to_consider is not None:
            for k, lg in remaining_lags.items():
                for x in lg:
                    model_test = model.copy()
                    current_lag = {a:b for a, b in best_features['best_lags'].items()}
                    current_lag[k] = best_features['best_lags'][k] + [x]
                    current_lag[k].sort()
                    model_test.n_lag = current_lag
                    my_cv = mv_cross_validate(model=model_test, df=df, cv_split=n_folds,
                                                         test_size=H, metrics=metrics, step_size=step_size)

                    score = my_cv[target_col].tolist()
                    if score < scores:
                        scores = score
                        candidate = {'target': k, 'type': 'lag', 'name': x}
                        improvement = True

        # Test Exogenous Features
        if candidate_features is not None:
            for feat in candidate_features:
                df_test = df.copy()
                df_test[feat] = df_orig[feat]
                model_test = model.copy()
                my_cv = mv_cross_validate(model=model_test, df=df_test, cv_split=n_folds, test_size=H,
                                                     metrics=metrics, step_size=step_size)
                score = my_cv[target_col].tolist()
                if score < scores:
                    scores = score
                    candidate = {'target': None, 'type': 'exog', 'name': feat}
                    improvement = True

            # Test Transformations
        if transformations is not None:
            for k, trans in transformations.items():
                for t in trans:
                    model_test = model.copy()
                    lag_transform = (model_test.lag_transform[k] or []) + [t]
                    model_test.lag_transform[k] = lag_transform
                    my_cv = mv_cross_validate(model=model_test, df=df, cv_split=n_folds,
                                                         test_size=H, metrics=metrics, step_size=step_size)
                    score = my_cv[target_col].tolist()
                    if score < scores:
                        scores = score
                        candidate = {'target': k, 'type': 'transform', 'name': t}
                        improvement = True

        # Update best features
        if improvement:
            best_score = scores
            if candidate['type'] == 'lag':
                best_features["best_lags"][candidate['target']].append(candidate['name']) # store lags by target
                remaining_lags[candidate['target']].remove(candidate['name'])
            elif candidate['type'] == 'exog':
                best_features["best_exogs"].append(candidate['name'])
                candidate_features.remove(candidate['name'])
                df[candidate['name']] = df_orig[candidate['name']]
            elif candidate['type'] == 'transform':
                best_features["best_transforms"][candidate['target']].append(candidate['name'])
                transformations[candidate['target']].remove(candidate['name'])
                if model.lag_transform is None:
                    transform_dict = {candidate['target']: [candidate['name']]}
                    model.lag_transform = transform_dict
                else:
                    if candidate['target'] not in model.lag_transform:
                        model.lag_transform[candidate['target']] = [candidate['name']]
                    else:
                        model.lag_transform[candidate['target']].append(candidate['name'])

            if verbose:
                if candidate['type'] == 'transform':
                    print(f"Added {candidate['type']} for target {candidate['target']}: {candidate['name'].get_name()} with score: {best_score}")
                else:
                    print(f"Added {candidate['type']} for target {candidate['target']}: {candidate['name']} with score: {best_score}")
        else:
            break  # No improvement

    if transformations is not None:
        for key, trans in best_features["best_transforms"].items():
            if trans:  # only process non-empty lists
                best_features["best_transforms"][key] = [t.get_name() for t in trans]

    if lags_to_consider is not None:
        # sort the lags for each variable
        for key in best_features["best_lags"]:
            best_features["best_lags"][key].sort()

    return best_features


def mv_backward_feature_selection(df, target_col, n_folds = None, H = None, model = None, metrics = None,
                                  lags_to_consider = None, candidate_features = None, transformations = None, 
                                    step_size = None, verbose = False):
    """
    Performs backward lag selection for multivariate models.
    Parameters:
        df (pd.DataFrame): DataFrame containing the time series data.
        target_col (str): The target column for accuracy evaluation.
        n_folds (int, optional): Number of cross-validation folds.
        H (int, optional): Forecast horizon.
        model: The forecasting model to be used.
        metrics (list): List of metrics to evaluate the model.
        step_size (int, optional): Step size for cross-validation. Defaults to None.
        verbose (bool, optional): Whether to print progress. Defaults to False.
    Returns:
        dict: Dictionary of best features for each variable.

    """

    # remaining_lags = {i:list(range(1, j+1)) for i, j in lags_to_consider.items()}
    # best_lags = {i:[] for i in max_lags}
    best_features = {
        "best_lags": {i: list(range(1, j+1)) for i, j in (lags_to_consider or {}).items()},
        "best_exogs": candidate_features.copy() if candidate_features is not None else [],
        "best_transforms": {i: j for i, j in (transformations or {}).items()}
}
    
    ## setting the full model
    if lags_to_consider is not None:
        model.n_lag = best_features["best_lags"]  # Start with all lags to consider
    if transformations is not None:
        model.lag_transform = best_features["best_transforms"]  # Start with all transformations to consider
    # exogenous variables should be in df before passing df
    best_score = list(np.repeat(float('inf'), len(metrics)))
    
    while True:
        improvement = False
        candidate = {'target': None, 'type': None, 'name': None}
        scores = best_score
        if lags_to_consider is not None:
            for targ_l, lags in best_features["best_lags"].items():
                for lg in lags:
                    lags_to_test = {a:b for a, b in lags.items()}
                    # Remove the current lag lg from current target
                    lags_to_test[targ_l] = [x for x in lags if x != lg]
                    lags_to_test[targ_l].sort()
                    model_test = model.copy()
                    model_test.n_lag = lags_to_test
                    my_cv = mv_cross_validate(model=model_test, df=df, cv_split=n_folds,
                                                         test_size=H, metrics=metrics, step_size=step_size)
                    score = my_cv[target_col].tolist()
                    if score < scores:
                        scores = score
                        candidate = {'target': targ_l, 'type': 'lag', 'name': lg}
                        improvement = True
        if transformations is not None:
            for targ_t, trans in best_features["best_transforms"].items():
                for tr in trans:
                    trans_to_test = {a:b for a, b in best_features["best_transforms"].items()}
                    trans_to_test[targ_t] = [x for x in trans if x != tr]
                    model_test = model.copy()
                    # model_test.lags = remaining_lags
                    model_test.lag_transform = trans_to_test
                    my_cv = mv_cross_validate(model=model_test, df=df, cv_split=n_folds,
                                                         test_size=H, metrics=metrics, step_size=step_size)
                    scores = my_cv[target_col].tolist()
                    if score < scores:
                        scores = score
                        candidate = {'target': targ_t, 'type': 'transform', 'name': trans}
                        improvement = True
        if candidate_features is not None:
            for feat in best_features["best_exogs"]:
                # feat_to_test = [x for x in candidate_features if x != feat]
                df_test = df.drop(columns=feat)
                model_test = model.copy()
                model_test.data_prep(df_test) # update data preparation because if new lags to be consistent with coefficients
                model_test.compute_coeffs() # update model coefficients because of new lags
                my_cv = mv_cross_validate(model=model_test, df=df_test, cv_split=n_folds,
                                                         test_size=H, metrics=metrics, step_size=step_size)
                score = my_cv[target_col].tolist()
                if score < scores:
                    scores = score
                    candidate = {'target': None, 'type': 'exog', 'name': feat}
                    improvement = True

        # Update best features
        if improvement and candidate['type']:
            best_score = scores
            if candidate['type'] == 'lag':
                best_features["best_lags"][candidate['target']].remove(candidate['name'])
            elif candidate['type'] == 'exog':
                best_features["best_exogs"].remove(candidate['name'])
                df = df.drop(columns=candidate['name'])
            elif candidate['type'] == 'transform':
                best_features["best_transforms"][candidate['target']].remove(candidate['name'])
                if any(best_features["best_transforms"][key] for key in best_features["best_transforms"]):
                    best_features["best_transforms"] = {k: v for k, v in best_features["best_transforms"].items() if not len(v) == 0}
                    model.lag_transform = best_features["best_transforms"]
                else:
                    model.lag_transform = None

            if verbose:
                if candidate['type'] == 'transform':
                    print(f"Removed {candidate['type']} for target {candidate['target']}: {candidate['name'].get_name()} with score: {best_score}")
                else:
                    print(f"Removed {candidate['type']} for target {candidate['target']}: {candidate['name']} with score: {best_score}")
        else:
            break  # No improvement

    # if transformations is not None and at least one key is not empty get their names
    if transformations is not None:
        for key, trans in best_features["best_transforms"].items():
            if trans:  # only process non-empty lists
                best_features["best_transforms"][key] = [t.get_name() for t in trans]
    if lags_to_consider is not None:
        # sort the lags for each variable
        for key in best_features["best_lags"]:
            best_features["best_lags"][key].sort()


    return best_features

# ------------------------------------------------------------------------------
# Forward Feature Selection for HMM
# ------------------------------------------------------------------------------

def hmm_forward_feature_selection(df, n_folds = None, H = None, model = None, metrics = None,
                                  lags_to_consider = None, candidate_features = None, transformations = None, 
                                    step_size = None, start_lag = None, start_transform = None,
                                    validation_type = "cv", iterations = 10, tol = 1e-4, verbose = False):
    """
    Performs forward lag/feature/transform selection for Regression models.
    Parameters:
        df (pd.DataFrame): DataFrame containing the time series data.
        n_folds (int, optional): Number of cross-validation folds.
        H (int, optional): Forecast horizon.
        model: Model to be used for training and evaluation.
        metrics (list, optional): List of metrics to evaluate the model. Even one metric, should be provided in a list.
        lags_to_consider (list, optional): List of lags to consider for feature selection.
        candidate_features (list, optional): List of candidate exogenous features.
        transformations (list, optional): List of transformations to apply.
        step_size (int, optional): Step size for rolling window.
        validation_type (str, optional): Type of validation to use ("cv", "BIC", "AIC" or both "AIC_BIC"). if "AIC_BIC" are both selected, the model will be evaluated using both criteria.
        iterations (int, optional): Number of iterations for model fitting to update parameters.
        tol (float, optional): Tolerance for convergence.
        verbose (bool, optional): Whether to print progress messages.
    Returns:
        dict: Dictionary of best features
    """


    if lags_to_consider is not None:
        if isinstance(lags_to_consider, int):
            remaining_lags = list(range(1, lags_to_consider + 1))
        elif isinstance(lags_to_consider, list):
            remaining_lags = lags_to_consider
        model.lags = None
        if start_lag is not None:
            if not isinstance(start_lag, list):
                raise ValueError("start_lag should be a list of integers.")
            model.lags = start_lag
            remaining_lags = [x for x in remaining_lags if x not in start_lag]

    if candidate_features is not None:
        df = df.drop(columns=candidate_features)
        df_orig = df.copy() # Keep original for feature add-ba
    if transformations is not None:
        if start_transform is not None:
            if not isinstance(start_transform, list):
                raise ValueError("start_transform should be a list of transformation instances.")
            model.lag_transform = start_transform
            transformations = [x for x in transformations if x not in start_transform]
        else:
            model.lag_transform = None
            
    best_features = {
    "best_lags": list(start_lag) if start_lag is not None else [],
    "best_exogs": [],
    "best_transforms": list(start_transform) if start_transform is not None else []}

    if validation_type == "cv":
        if isinstance(metrics, list):
            best_score = [float('inf')] * len(metrics)
        else:
            best_score = float('inf')
    elif validation_type in ("BIC", "AIC"):
        best_score = float('inf')
    elif validation_type == "AIC_BIC":
        best_score = [float('inf')] * 2
    else:
        raise ValueError("Invalid validation_type. Choose from 'cv', 'BIC', 'AIC', or 'AIC_BIC'.")

    if isinstance(best_score, list):
        def is_elementwise_improvement(score, best_s):
            return all(s < b for s, b in zip(score, best_s))
    else:
        def is_elementwise_improvement(score, best_s):
            return score < best_s

# After each feature selection step iterate model to make sure parameters are updated like transition probabilities and stds
    def model_update(model_test, df_test, iterations=iterations, tol=tol):
        model_test.data_prep(df_test) # update data preparation because if new lags to be consistent with coefficients
        model_test.compute_coeffs() # update model coefficients because of new lags
        prev_ll = model_test.LL
        for _ in range(iterations):
            model_test.fit(df_test)
            ll = model_test.LL
            if abs(ll - prev_ll) < tol:
                # print iteration number
                # print(f"Converged after {_} iterations")
                break
            else:
                prev_ll = ll
        return model_test
    

    def validation(model_test, df_test):
        if validation_type == "cv":
            cv_result = hmm_cross_validate(model=model_test, df=df_test, cv_split=n_folds, test_size=H,
                                metrics=metrics, step_size=step_size)
    
            if isinstance(metrics, list):
                score = cv_result["score"].tolist()
            else:
                score = cv_result["score"].values[0]
        elif validation_type == "BIC":
            score = model_test.BIC
        elif validation_type == "AIC":
            score = model_test.AIC
        elif validation_type == "AIC_BIC":
            score = [model_test.AIC, model_test.BIC]
        else:
            raise ValueError("Invalid validation_type. Choose from 'cv', 'BIC', 'AIC', or 'AIC_BIC'.")

        return score


    while True:
        improvement = False
        candidate = {'type': None, 'name': None}
        scores = best_score

        # Test Lags
        if lags_to_consider is not None:
            for lag in remaining_lags:
                current_lags = sorted(best_features["best_lags"] + [lag])
                model_test = model.copy()
                model_test.lags = current_lags
                model_test = model_update(model_test, df)
                score = validation(model_test, df)
                if is_elementwise_improvement(score, scores):
                    scores = score
                    candidate = {'type': 'lag', 'name': lag, 'model': model_test}
                    improvement = True

        # Test Exogenous Features
        if candidate_features is not None:
            for feat in candidate_features:
                df_test = df.copy()
                df_test[feat] = df_orig[feat]
                model_test = model.copy()
                model_test = model_update(model_test, df_test)
                score = validation(model_test, df_test)
                if is_elementwise_improvement(score, scores):
                    scores = score
                    candidate = {'type': 'exog', 'name': feat, 'model': model_test}
                    improvement = True

        # Test Transformations
        if transformations is not None:
            for trans in transformations:
                model_test = model.copy()
                lag_transform = (model_test.lag_transform or []) + [trans]
                model_test.lag_transform = lag_transform
                model_test = model_update(model_test, df)
                score = validation(model_test, df)
                if is_elementwise_improvement(score, scores):
                    scores = score
                    candidate = {'type': 'transform', 'name': trans, 'model': model_test}
                    improvement = True

        # Update best features
        if improvement:
            best_score = scores
            if candidate['type'] == 'lag':
                best_features["best_lags"].append(candidate['name'])
                remaining_lags.remove(candidate['name'])
            elif candidate['type'] == 'exog':
                best_features["best_exogs"].append(candidate['name'])
                candidate_features.remove(candidate['name'])
                df[candidate['name']] = df_orig[candidate['name']]
            elif candidate['type'] == 'transform':
                best_features["best_transforms"].append(candidate['name'])
                transformations.remove(candidate['name'])
                if model.lag_transform is None:
                    model.lag_transform = [candidate['name']]
                else:
                    model.lag_transform.append(candidate['name'])
            # update transition probs and stds of states

            model.A = candidate['model'].A
            model.stds = candidate['model'].stds
            model.LL = candidate['model'].LL

            if verbose:
                print(f"Added {candidate['type']}: {candidate['name']} with score: {best_score} and loglik and BIC: {model.LL}, {model.BIC}")
        else:
            break  # No improvement

    # Finalize model with best features
    model_ = model.copy()
    if lags_to_consider is not None and best_features["best_lags"]:
        model_.lags = best_features["best_lags"]
    if transformations is not None and best_features["best_transforms"]:
        model_.lag_transform = best_features["best_transforms"]
    model_ = model_update(model_, df)


    if transformations is not None and best_features["best_transforms"]:
        best_features["best_transforms"] = [trans.get_name() for trans in best_features["best_transforms"]]
    
    return best_features, model_



def hmm_backward_feature_selection(df, n_folds = None, H = None, model = None, metrics = None,
                                  lags_to_consider = None, candidate_features = None, transformations = None, 
                                    step_size = None, validation_type = "cv", iterations = 10, tol = 1e-4, verbose = False):
    """
    Performs backward lag selection for Regression models.
    Parameters:
        df (pd.DataFrame): DataFrame containing the time series data.
        n_folds (int, optional): Number of cross-validation folds.
        H (int, optional): Forecast horizon.
        model: Model to be used for training and evaluation.
        metrics (list, optional): List of metrics to evaluate the model.
        lags_to_consider (list, optional): List of lags to consider for feature selection.
        candidate_features (list, optional): List of candidate exogenous features.
        transformations (list, optional): List of transformations to apply.
        step_size (int, optional): Step size for rolling window.
        validation_type (str, optional): Type of validation to use ("cv", "BIC", "AIC" or both "AIC_BIC"). if "AIC_BIC" are both selected, the model will be evaluated using both criteria.
        iterations (int, optional): Number of iterations for model fitting to update parameters.
        tol (float, optional): Tolerance for convergence.
        verbose (bool, optional): Whether to print progress messages.
    Returns:
        dict: Dictionary of best features
    """
    remaining_lags = list(range(1, lags_to_consider + 1)) if lags_to_consider is not None else []
    candidate_features = candidate_features.copy() if candidate_features is not None else []
    transformations = transformations.copy() if transformations is not None else []
    best_features = {"best_lags": remaining_lags, "best_exogs": candidate_features, "best_transforms": transformations}

    ## setting the full model
    # model_full = model.copy()
    if lags_to_consider is not None:
        model.lags = remaining_lags
    if transformations is not None:
        model.lag_transform = transformations

    # set validation
    if validation_type == "cv":
        if isinstance(metrics, list):
            best_score = [float('inf')] * len(metrics)
        else:
            best_score = float('inf')
    elif validation_type in ("BIC", "AIC"):
        best_score = float('inf')
    elif validation_type == "AIC_BIC":
        best_score = [float('inf')] * 2
    else:
        raise ValueError("Invalid validation_type. Choose from 'cv', 'BIC', 'AIC', or 'AIC_BIC'.")

    if isinstance(best_score, list):
        def is_elementwise_improvement(score, best_s):
            return all(s < b for s, b in zip(score, best_s))
    else:
        def is_elementwise_improvement(score, best_s):
            return score < best_s
        
# After each feature selection step iterate model to make sure parameters are updated like transition probabilities and stds
    def model_update(model_test, df_test, iterations=iterations, tol=tol):
        model_test.data_prep(df_test) # update data preparation because if new lags to be consistent with coefficients
        model_test.compute_coeffs() # update model coefficients because of new lags
        prev_ll = model_test.LL
        for _ in range(iterations):
            model_test.fit(df_test)
            ll = model_test.LL
            if abs(ll - prev_ll) < tol:
                # print iteration number
                # print(f"Converged after {_} iterations")
                break
            else:
                prev_ll = ll
        return model_test
    

    def validation(model_test, df_test):
        if validation_type == "cv":
            cv_result = hmm_cross_validate(model=model_test, df=df_test, cv_split=n_folds, test_size=H,
                                metrics=metrics, step_size=step_size)
    
            if isinstance(metrics, list):
                score = cv_result["score"].tolist()
            else:
                score = cv_result["score"].values[0]
        elif validation_type == "BIC":
            score = model_test.BIC
        elif validation_type == "AIC":
            score = model_test.AIC
        elif validation_type == "AIC_BIC":
            score = [model_test.AIC, model_test.BIC]
        else:
            raise ValueError("Invalid validation_type. Choose from 'cv', 'BIC', 'AIC', or 'AIC_BIC'.")

        return score

    # best_lags = None
    while True:
        improvement = False
        candidate = {'type': None, 'name': None}
        scores = best_score
        if best_features["best_lags"]:
            for lg in best_features["best_lags"]:
                lags_to_test = [x for x in best_features["best_lags"] if x != lg]
                lags_to_test.sort()
                model_test = model.copy()
                model_test.lags = lags_to_test
                model_test = model_update(model_test, df)
                score = validation(model_test, df)
                if is_elementwise_improvement(score, scores):
                    scores = score
                    candidate = {'type': 'lag', 'name': lg, 'model': model_test}
                    improvement = True
        if best_features["best_transforms"]:
            for trans in best_features["best_transforms"]:
                trans_to_test = [x for x in best_features["best_transforms"] if x != trans]
                model_test = model.copy()
                model_test.lag_transform = trans_to_test
                model_test = model_update(model_test, df)
                score = validation(model_test, df)
                if is_elementwise_improvement(score, scores):
                    scores = score
                    candidate = {'type': 'transform', 'name': trans, 'model': model_test}
                    improvement = True
        if best_features["best_exogs"]:
            for feat in best_features["best_exogs"]:
                # feat_to_test = [x for x in candidate_features if x != feat]
                df_test = df.drop(columns=feat)
                model_test = model.copy()
                model_test = model_update(model_test, df_test)
                score = validation(model_test, df_test)
                if is_elementwise_improvement(score, scores):
                    scores = score
                    candidate = {'type': 'exog', 'name': feat, 'model': model_test}
                    improvement = True

        # Update best features
        if improvement and candidate['type']:
            best_score = scores
            if candidate['type'] == 'lag':
                best_features["best_lags"].remove(candidate['name'])
            elif candidate['type'] == 'exog':
                best_features["best_exogs"].remove(candidate['name'])
                df = df.drop(columns=candidate['name'])
            elif candidate['type'] == 'transform':
                best_features["best_transforms"].remove(candidate['name'])
                if not best_features["best_transforms"]:
                    model.lag_transform = best_features["best_transforms"]
                else:
                    model.lag_transform = None
            model.A = candidate['model'].A
            model.stds = candidate['model'].stds
            model.LL = candidate['model'].LL

            if verbose:
                print(f"Removed {candidate['type']}: {candidate['name']} with score: {best_score} and loglik and BIC: {model.LL}, {model.BIC}")
        else:
            break  # No improvement

        # Finalize model with best features
        model_ = model.copy()
        if lags_to_consider is not None and best_features["best_lags"]:
            model_.lags = best_features["best_lags"]
        if transformations is not None and best_features["best_transforms"]:
            model_.lag_transform = best_features["best_transforms"]
        model_ = model_update(model_, df)


    if transformations is not None and best_features["best_transforms"]:
        best_features["best_transforms"] = [trans.get_name() for trans in best_features["best_transforms"]]

    return best_features, model_


def hmm_mv_forward_feature_selection(df, target_col, n_folds = None, H = None, model = None, metrics = None,
                                  lags_to_consider = None, candidate_features = None, transformations = None, 
                                    step_size = None, starting_lag = None, starting_transform = None,
                                    validation_type = "cv", iterations = 10, tol = 1e-4, verbose = False):
    """
    Performs forward lag selection for Vektor Autoregressive models and bidirectional ml models
    Parameters:
        df (pd.DataFrame): DataFrame containing the time series data.
        target_col (str): The target column for accuracy evaluation.
        n_folds (int): Number of folds for cross-validation.
        H (int): Forecast horizon.
        model: The forecasting model to be used.
        metrics (list): List of metrics to evaluate the model.
        lags_to_consider (dict): Dictionary of maximum lags for each variable.
        candidate_features (list): List of candidate exogenous features.
        transformations (list): List of transformations to consider.
        step_size (int, optional): Step size for lag selection. Defaults to None.
        verbose (bool, optional): Whether to print progress. Defaults to False.
    Returns:
        dict: Dictionary of best features for each variable.
    """

    # max_lag = sum(x for x in max_lags.values())
    
    # lags = list(range(1, max_lags+1))

    best_features = {"best_lags": {i: [] for i in lags_to_consider if lags_to_consider is not None}, "best_transforms": {i: [] for i in transformations if transformations is not None}, "best_exogs": []}
    remaining_lags = {i:list(range(1, j+1)) for i, j in lags_to_consider.items()}
    if starting_lag is not None:
        for k, v in starting_lag.items():
            all_lags = remaining_lags[k]
            remaining_lags[k] = [x for x in all_lags if x not in v]
            best_features["best_lags"][k].extend(v)

    if lags_to_consider is not None:
        model.lags = None # Start with no lags

    # Keep original for feature add-back
    df_orig = df.copy()

    # Drop candidate features initially
    if candidate_features:
        df = df.drop(columns=candidate_features) # Drop candidate features to start with feature selection
    if transformations is not None:
        if starting_transform is not None:
            for k, v in starting_transform.items():
                transformations[k] = [x for x in transformations if x not in v]
                best_features["best_transforms"][k].extend(v)
            model.lag_transform = starting_transform
        else:
            model.lag_transform = None # Start with no transformations


    if validation_type == "cv":
        if isinstance(metrics, list):
            best_score = [float('inf')] * len(metrics)
        else:
            best_score = float('inf')
    elif validation_type in ("BIC", "AIC"):
        best_score = float('inf')
    elif validation_type == "AIC_BIC":
        best_score = [float('inf')] * 2
    else:
        raise ValueError("Invalid validation_type. Choose from 'cv', 'BIC', 'AIC', or 'AIC_BIC'.")

    if isinstance(best_score, list):
        def is_elementwise_improvement(score, best_s):
            return all(s < b for s, b in zip(score, best_s))
    else:
        def is_elementwise_improvement(score, best_s):
            return score < best_s

# After each feature selection step iterate model to make sure parameters are updated like transition probabilities and stds
    def model_update(model_test, df_test, iterations=iterations, tol=tol):
        model_test.data_prep(df_test) # update data preparation because if new lags to be consistent with coefficients
        model_test.compute_coeffs() # update model coefficients because of new lags
        prev_ll = model_test.LL
        for _ in range(iterations):
            model_test.fit(df_test)
            ll = model_test.LL
            if abs(ll - prev_ll) < tol:
                # print iteration number
                # print(f"Converged after {_} iterations")
                break
            else:
                prev_ll = ll
        return model_test
    

    def validation(model_test, df_test):
        if validation_type == "cv":
            cv_result = hmm_mv_cross_validate(model = model_test, df=df_test, cv_split=n_folds, test_size=H,
                                        metrics = metrics, step_size= step_size)
    
            if isinstance(metrics, list):
                score = cv_result[target_col].tolist()
            else:
                score = cv_result[target_col].values[0]
        elif validation_type == "BIC":
            score = model_test.BIC
        elif validation_type == "AIC":
            score = model_test.AIC
        elif validation_type == "AIC_BIC":
            score = [model_test.AIC, model_test.BIC]
        else:
            raise ValueError("Invalid validation_type. Choose from 'cv', 'BIC', 'AIC', or 'AIC_BIC'.")

        return score

    
    # while max_lag>0:
    while True:
        improvement = False
        candidate = {'target': None, 'type': None, 'name': None}
        scores = best_score
        if lags_to_consider is not None:
            for k, lg in remaining_lags.items():
                for x in lg:
                    model_test = model.copy()
                    current_lag = {a:b for a, b in best_features['best_lags'].items()}
                    current_lag[k] = best_features['best_lags'][k] + [x]
                    current_lag[k].sort()
                    model_test.lags = current_lag

                    model_test = model_update(model_test, df)
                    score = validation(model_test, df)

                    if is_elementwise_improvement(score, scores):
                        scores = score
                        candidate = {'target': k, 'type': 'lag', 'name': x, 'model': model_test}
                        improvement = True

        # Test Exogenous Features
        if candidate_features is not None:
            for feat in candidate_features:
                df_test = df.copy()
                df_test[feat] = df_orig[feat]
                model_test = model.copy()
                model_test = model_update(model_test, df_test)
                score = validation(model_test, df_test)

                if is_elementwise_improvement(score, scores):
                    scores = score
                    candidate = {'target': None, 'type': 'exog', 'name': feat, 'model': model_test}
                    improvement = True

            # Test Transformations
        if transformations is not None:
            for k, trans in transformations.items():
                for t in trans:
                    model_test = model.copy()
                    lag_transform = (model_test.lag_transform[k] or []) + [t]
                    model_test.lag_transform[k] = lag_transform
                    model_test = model_update(model_test, df)
                    score = validation(model_test, df)
                    if is_elementwise_improvement(score, scores):
                        scores = score
                        candidate = {'target': k, 'type': 'transform', 'name': t, 'model': model_test}
                        improvement = True

        # Update best features
        if improvement:
            best_score = scores
            if candidate['type'] == 'lag':
                best_features["best_lags"][candidate['target']].append(candidate['name']) # store lags by target
                remaining_lags[candidate['target']].remove(candidate['name'])
            elif candidate['type'] == 'exog':
                best_features["best_exogs"].append(candidate['name'])
                candidate_features.remove(candidate['name'])
                df[candidate['name']] = df_orig[candidate['name']]
            elif candidate['type'] == 'transform':
                best_features["best_transforms"][candidate['target']].append(candidate['name'])
                transformations[candidate['target']].remove(candidate['name'])
                if model.lag_transform is None:
                    transform_dict = {candidate['target']: [candidate['name']]}
                    model.lag_transform = transform_dict
                else:
                    if candidate['target'] not in model.lag_transform:
                        model.lag_transform[candidate['target']] = [candidate['name']]
                    else:
                        model.lag_transform[candidate['target']].append(candidate['name'])

            model.A = candidate['model'].A
            model.covs = candidate['model'].covs
            model.LL = candidate['model'].LL

            if verbose:
                print(f"Added {candidate['type']}: {candidate['name']} with score: {best_score} and loglik and BIC: {model.LL}, {model.BIC}")
        else:
            break  # No improvement

    # Finalize model with best features
    model_ = model.copy()
    if lags_to_consider is not None and best_features["best_lags"]:
        model_.lags = best_features["best_lags"]
    if transformations is not None and best_features["best_transforms"]:
        model_.lag_transform = best_features["best_transforms"]

    model_ = model_update(model_, df)


    if transformations is not None:
        for key, trans in best_features["best_transforms"].items():
            if trans:  # only process non-empty lists
                best_features["best_transforms"][key] = [t.get_name() for t in trans]

    return best_features, model_


def hmm_mv_backward_feature_selection(df, target_col, n_folds = None, H = None, model = None, metrics = None,
                                  lags_to_consider = None, candidate_features = None, transformations = None, 
                                    step_size = None, validation_type = "cv", iterations = 10, tol = 1e-4, 
                                    verbose = False):
    """
    Performs backward lag selection for Regression models.
    Parameters:
        df (pd.DataFrame): DataFrame containing the time series data.
        target_col (str): The target column for accuracy evaluation.
        n_folds (int, optional): Number of cross-validation folds.
        H (int, optional): Forecast horizon.
        model: The forecasting model to be used.
        metrics (list): List of metrics to evaluate the model.
        step_size (int, optional): Step size for cross-validation. Defaults to None.
        verbose (bool, optional): Whether to print progress. Defaults to False.
    Returns:
        dict: Dictionary of best features for each variable.

    """

    # remaining_lags = {i:list(range(1, j+1)) for i, j in lags_to_consider.items()}
    # best_lags = {i:[] for i in max_lags}
    best_features = {
        "best_lags": {i: list(range(1, j+1)) for i, j in (lags_to_consider or {}).items()},
        "best_exogs": candidate_features.copy() if candidate_features is not None else [],
        "best_transforms": {i: j for i, j in (transformations or {}).items()}
}
    
    ## setting the full model
    if lags_to_consider is not None:
        model.lags = best_features["best_lags"]
    if transformations is not None:
        model.lag_transform = best_features["best_transforms"]
    # exogenous variables should be in df before passing df

    if validation_type == "cv":
        if isinstance(metrics, list):
            best_score = [float('inf')] * len(metrics)
        else:
            best_score = float('inf')
    elif validation_type in ("BIC", "AIC"):
        best_score = float('inf')
    elif validation_type == "AIC_BIC":
        best_score = [float('inf')] * 2
    else:
        raise ValueError("Invalid validation_type. Choose from 'cv', 'BIC', 'AIC', or 'AIC_BIC'.")

    if isinstance(best_score, list):
        def is_elementwise_improvement(score, best_s):
            return all(s < b for s, b in zip(score, best_s))
    else:
        def is_elementwise_improvement(score, best_s):
            return score < best_s

# After each feature selection step iterate model to make sure parameters are updated like transition probabilities and stds
    def model_update(model_test, df_test, iterations=iterations, tol=tol):
        model_test.data_prep(df_test) # update data preparation because if new lags to be consistent with coefficients
        model_test.compute_coeffs() # update model coefficients because of new lags
        prev_ll = model_test.LL
        for _ in range(iterations):
            model_test.fit(df_test)
            ll = model_test.LL
            if abs(ll - prev_ll) < tol:
                # print iteration number
                # print(f"Converged after {_} iterations")
                break
            else:
                prev_ll = ll
        return model_test
    

    def validation(model_test, df_test):
        if validation_type == "cv":
            cv_result = hmm_mv_cross_validate(model = model_test, df=df_test, cv_split=n_folds, test_size=H,
                                        metrics = metrics, step_size= step_size)
    
            if isinstance(metrics, list):
                score = cv_result[target_col].tolist()
            else:
                score = cv_result[target_col].values[0]
        elif validation_type == "BIC":
            score = model_test.BIC
        elif validation_type == "AIC":
            score = model_test.AIC
        elif validation_type == "AIC_BIC":
            score = [model_test.AIC, model_test.BIC]
        else:
            raise ValueError("Invalid validation_type. Choose from 'cv', 'BIC', 'AIC', or 'AIC_BIC'.")

        return score

    
    while True:
        improvement = False
        candidate = {'target': None, 'type': None, 'name': None}
        scores = best_score
        if lags_to_consider is not None:
            for targ_l, lags in best_features["best_lags"].items():
                for lg in lags:
                    lags_to_test = {a:b for a, b in lags.items()}
                    # Remove the current lag lg from current target
                    lags_to_test[targ_l] = [x for x in lags if x != lg]
                    lags_to_test[targ_l].sort()
                    model_test = model.copy()
                    model_test.lags = lags_to_test

                    model_test = model_update(model_test, df)
                    score = validation(model_test, df)
                    if is_elementwise_improvement(score, scores):
                        scores = score
                        candidate = {'target': targ_l, 'type': 'lag', 'name': lg, 'model': model_test}
                        improvement = True
        if transformations is not None:
            for targ_t, trans in best_features["best_transforms"].items():
                for tr in trans:
                    trans_to_test = {a:b for a, b in best_features["best_transforms"].items()}
                    trans_to_test[targ_t] = [x for x in trans if x != tr]
                    model_test = model.copy()
                    # model_test.lags = remaining_lags
                    model_test.lag_transform = trans_to_test
                    model_test = model_update(model_test, df)
                    score = validation(model_test, df)
                    if is_elementwise_improvement(score, scores):
                        scores = score
                        candidate = {'target': targ_t, 'type': 'transform', 'name': trans, 'model': model_test}
                        improvement = True
        if candidate_features is not None:
            for feat in best_features["best_exogs"]:
                # feat_to_test = [x for x in candidate_features if x != feat]
                df_test = df.drop(columns=feat)
                model_test = model.copy()
                model_test = model_update(model_test, df_test)
                score = validation(model_test, df_test)
                if is_elementwise_improvement(score, scores):
                    scores = score
                    candidate = {'target': None, 'type': 'exog', 'name': feat, 'model': model_test}
                    improvement = True

        # Update best features
        if improvement and candidate['type']:
            best_score = scores
            if candidate['type'] == 'lag':
                best_features["best_lags"][candidate['target']].remove(candidate['name'])
            elif candidate['type'] == 'exog':
                best_features["best_exogs"].remove(candidate['name'])
                df = df.drop(columns=candidate['name'])
            elif candidate['type'] == 'transform':
                best_features["best_transforms"][candidate['target']].remove(candidate['name'])
                if any(best_features["best_transforms"][key] for key in best_features["best_transforms"]):
                    best_features["best_transforms"] = {k: v for k, v in best_features["best_transforms"].items() if not len(v) == 0}
                    model.lag_transform = best_features["best_transforms"]
                else:
                    model.lag_transform = None

            model.A = candidate['model'].A
            model.covs = candidate['model'].covs
            model.LL = candidate['model'].LL

            if verbose:
                print(f"Removed {candidate['type']}: {candidate['name']} with score: {best_score} and loglik and BIC: {model.LL}, {model.BIC}")
        else:
            break  # No improvement

    # Finalize model with best features
    model_ = model.copy()
    if lags_to_consider is not None and best_features["best_lags"]:
        model_.lags = best_features["best_lags"]
    if transformations is not None and best_features["best_transforms"]:
        model_.lag_transform = best_features["best_transforms"]

    model_ = model_update(model_, df)

    # if transformations is not None and at least one key is not empty get their names
    if transformations is not None:
        for key, trans in best_features["best_transforms"].items():
            if trans:  # only process non-empty lists
                best_features["best_transforms"][key] = [t.get_name() for t in trans]

    return best_features, model_

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

def cv_hmm_lag_tune(
    model,
    df,
    cv_split,
    test_size,
    eval_metric,
    lag_space=None,
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
    lag_space : dict
        Hyperopt lag search space.
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
    
    max_lag = lag_space 
    space = {f"lag_{i}": hp.choice(f"lag_{i}", [0, 1]) for i in range(1, max_lag + 1)}
    def objective(params):

        selected_lags = [i for i in range(1, max_lag + 1) if params[f"lag_{i}"] == 1]
        model_ = model.copy()
        model_.lags = selected_lags

                # Optional: penalize too few lags
        if len(selected_lags) < 1:
            return {"loss": 1e6, "status": STATUS_OK}

        metrics = []
        for idx, (train_index, test_index) in enumerate(tscv.split(df)):
            train, test = df.iloc[train_index], df.iloc[test_index]
            x_test = test.drop(columns=[model.target_col])
            y_test = np.array(test[model.target_col])
            if idx == 0:
                model_.fit_em(train)
            else:
                model_.fit(train)
            
            y_pred = model_.forecast(len(y_test), x_test)

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
        space=space,
        algo=tpe.suggest,
        max_evals=eval_num,
        trials=trials,
    )

    # Extract and sort lag values
    best_lag_indexes = [value for key, value in sorted(((k, v) for k, v in best_hyperparams.items() if k.startswith("lag_")),
                                                          key=lambda x: int(x[0].split("_")[1]))]
    best_lag_values = [i for i in range(1, lag_space + 1) if best_lag_indexes[i-1]==1]
    return best_lag_values

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


def mv_cross_validate(model, df, cv_split, test_size, metrics, step_size=None):
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
    lag_space=None,
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

    def _get_model_params_for_fit(params):
        # Exclude special parameters that should not be passed to the model constructor
        skip_keys = {
            "box_cox", "n_lag", "box_cox_lmda", "box_cox_biasadj",
            "trend", "damped_trend", "seasonal", "seasonal_periods",
            "smoothing_level", "smoothing_trend", "smoothing_seasonal", "damping_trend",
            "differencing_number"
        }
        if lag_space is not None:
            skip_keys.update([f"lag_{i}" for i in range(1, lag_space+1)])

        return {k: v for k, v in params.items() if k not in skip_keys}
    
    if lag_space is not None:
        lag_postions = {f"lag_{i}": hp.choice(f"lag_{i}", [0, 1]) for i in range(1, lag_space + 1)}
        search_space = {**lag_postions, **param_space}
    else:
        search_space = {**param_space}

    def objective(params):
        _set_model_params(params)
        if isinstance(model.model, LinearRegression):
            # For LinearRegression, we don't need to set model_params
            model_params = None
        else:
            # For other models, get the parameters to set
            model_params = _get_model_params_for_fit(params)

        if lag_space is not None:
            selected_lags = [i for i in range(1, lag_space+1) if params[f"lag_{i}"] == 1]
            model.n_lag = selected_lags

                # Optional: penalize too few lags
        if len(selected_lags) < 1:
            return {"loss": 1e6, "status": STATUS_OK}

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
        space=search_space,
        algo=tpe.suggest,
        max_evals=eval_num,
        trials=trials,
    )

    model.tuned_params = [
        space_eval(param_space, {k: v[0] for k, v in t["misc"]["vals"].items()})
        for t in trials.trials
    ]

    # Extract and sort lag values
    if lag_space is not None:
        best_lag_indexes = [value for key, value in sorted(((k, v) for k, v in best_hyperparams.items() if k.startswith("lag_")),
                                                          key=lambda x: int(x[0].split("_")[1]))]
        best_lag_values = [i for i in range(1, lag_space + 1) if best_lag_indexes[i-1]==1]
        return space_eval(param_space, best_hyperparams), best_lag_values
    else:
        return space_eval(param_space, best_hyperparams)

def mv_cv_tune(
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

def cv_lag_tune(
    model,
    df,
    cv_split,
    test_size,
    eval_metric,
    lag_space=None,
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
    lag_space : dict
        Hyperopt lag search space.
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
    max_lag = lag_space
    space = {f"lag_{i}": hp.choice(f"lag_{i}", [0, 1]) for i in range(1, max_lag + 1)}
    def objective(params):

        selected_lags = [i for i in range(1, max_lag + 1) if params[f"lag_{i}"] == 1]
        model_ = model.copy()
        model_.n_lag = selected_lags

                # Optional: penalize too few lags
        if len(selected_lags) < 1:
            return {"loss": 1e6, "status": STATUS_OK}

        metrics = []
        for train_index, test_index in tscv.split(df):
            train, test = df.iloc[train_index], df.iloc[test_index]
            x_test = test.drop(columns=[model.target_col])
            y_test = np.array(test[model.target_col])
            model_.fit(train)

            y_pred = model_.forecast(len(y_test), x_test)

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
        space=space,
        algo=tpe.suggest,
        max_evals=eval_num,
        trials=trials,
    )

    # Extract and sort lag values
    best_lag_indexes = [value for key, value in sorted(((k, v) for k, v in best_hyperparams.items() if k.startswith("lag_")),
                                                          key=lambda x: int(x[0].split("_")[1]))]
    best_lag_values = [i for i in range(1, lag_space + 1) if best_lag_indexes[i-1]==1]
    return best_lag_values

#------------------------------------------------------------------------------
# HMM CV utility function
#------------------------------------------------------------------------------

def hmm_cross_validate(model, df, cv_split, test_size, metrics, learn_per_fold = None, step_size= None):
    """
    Run cross-validation using time series splits.

    Args:
        model (class): Machine learning model class (e.g., CatBoostRegressor, LGBMRegressor).
        df (pd.DataFrame): Input data.
        cv_split (int): Number of splits in TimeSeriesSplit.
        test_size (int): Size of test window.
        metrics (list): List of metric functions.
        learn (bool): If True, learn parameters on the entire dataset, otherwise fit the model on each fold.
        learn_per_fold (str): If "first", learns parameters on the first fold and fits on the rest,
        if "all", learns on all folds, if "None", do not learn, just fit the model.
        step_size (int): Step size for time series cross-validation.
    
    Returns:
        pd.DataFrame: Performance metrics for CV.
    """
    tscv = ParametricTimeSeriesSplit(n_splits=cv_split, test_size=test_size, step_size=step_size)
    metrics_dict = {m.__name__: [] for m in metrics}
    for idx, (train_index, test_index) in enumerate(tscv.split(df)):
        train, test = df.iloc[train_index], df.iloc[test_index]
        x_test = test.drop(columns=[model.target_col])
        y_test = np.array(test[model.target_col])

        # If it is first fold, fit the model
        model_ = model.copy()
        if (idx == 0) and (learn_per_fold in ["first", "all"]):
            model_.fit_em(train)
        # If learning per fold, learn the model on each fold
        elif (learn_per_fold == "all") and (idx != 0):
            model_.fit_em(train)
        # If not learning per fold, fit the model on the first fold
        else: # learn_per_fold == "None" or learn_per_fold == "first" for remaining folds
            model_.fit(train)

        # Forecast using the model
        bb_forecast = model_.forecast(test_size, exog=x_test)
        # Evaluate each metric
        for m in metrics:
            if m.__name__ == 'MASE':
                eval_val = m(y_test, bb_forecast)
            else:
                eval_val = m(y_test, bb_forecast)
            metrics_dict[m.__name__].append(eval_val)
    overall_performance = [[m.__name__, np.mean(metrics_dict[m.__name__])] for m in metrics]
    return pd.DataFrame(overall_performance).rename(columns={0: "eval_metric", 1: "score"})

def hmm_mv_cross_validate(model, df, cv_split, test_size, metrics, learn_per_fold=None, step_size=None):
    """
    Cross-validate the bidirectional Vector Autoregressive Hidden Markov model.
    Args:
        df (pd.DataFrame): Input dataframe.
        cv_split (int): Number of folds.
        test_size (int): Forecast window for each split.
        metrics (list): List of evaluation metric functions.
        learn_per_fold (str, optional): Learning strategy for each fold. If "first", learns on the first fold only. If "all", learns on all folds. If "None", does not learn, just fits the model.
        step_size (int, optional): Step size for time series cross-validation.
    Returns:
        pd.DataFrame: CV performance metrics for each target variable.
    """
    tscv = ParametricTimeSeriesSplit(n_splits=cv_split, test_size=test_size, step_size=step_size)
    metrics_dict = {m.__name__: [] for m in metrics}

    for idx, (train_index, test_index) in enumerate(tscv.split(df)):
        train, test = df.iloc[train_index], df.iloc[test_index]
        x_test = test.drop(columns=model.target_col)
        # y_test1 = np.array(test[model.target_cols[0]])
        # y_test2 = np.array(test[model.target_cols[1]])

        # If it is first fold, fit the model
        if (idx == 0) and (learn_per_fold in ["first", "all"]):
            model.fit_em(train)
        # If learning per fold, learn the model on each fold
        elif (learn_per_fold == "all") and (idx != 0):
            model.fit_em(train)
        # If not learning per fold, fit the model on the first fold
        else: # learn_per_fold == "None" or learn_per_fold == "first" for remaining folds
            model.fit(train)

        forecasts = model.forecast(test_size, exog=x_test) # dictionary of forecasts for all target columns
        for m in metrics:
            if m.__name__ == 'MASE':
                val = [m(test[target_col], forecasts[target_col], train[target_col]) for target_col in model.target_col]
            else:
                val = [m(test[target_col], forecasts[target_col]) for target_col in model.target_col]
            # Append the list of metric value for each target column
            metrics_dict[m.__name__].append(val)

    # Compute average metric for each target_col
    results = []
    for m in metrics:
        metric_name = m.__name__
        vals = np.array(metrics_dict[metric_name])  # shape: (n_folds, n_targets)
        avg_vals = np.mean(vals, axis=0)
        results.append([metric_name] + list(avg_vals))

    # Create dynamic column names
    columns = ["eval_metric"] + [col for col in model.target_col]
    return pd.DataFrame(results, columns=columns)

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