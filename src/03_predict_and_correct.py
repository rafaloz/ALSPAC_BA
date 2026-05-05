"""
03_predict_and_correct.py
=========================
Generates brain-age predictions for the held-out and external cohorts and
applies the Zhang age-level bias correction on top of Cole's post-hoc
linear correction (no leakage: both corrections are calibrated on the
training set only).

Run from the ``src/`` directory.

Inputs
------
* ../model/train_sample.csv                : training reference (for outlier
                                             flattening + min-max scaling).
* ../model/test_sample.csv                 : internal hold-out test set.
* ../model/bias_correction_reference.csv   : training-set predictions used to
                                             fit Cole's linear bias model and
                                             to build the Zhang age-bin table.
* ../model/selected_features.csv           : list of 621 selected features.
* ../model/MLP_brain_age.pkl               : trained MLP regressor.
* ../data/AgeRisk_harmonized.csv           : harmonized AgeRisk cohort.
* ../data/ALSPAC_20_MRI_I_harmonized.csv   : harmonized ALSPAC-20 (MRI-I).
* ../data/ALSPAC_30_MRI_II_harmonized.csv  : harmonized ALSPAC-30 (MRI-II).

Outputs
-------
* ../data/predictions_TestSet.csv
* ../data/predictions_AgeRisk.csv
* ../data/predictions_ALSPAC_20_MRI_I.csv
* ../data/predictions_ALSPAC_30_MRI_II.csv
"""

import ast
import pickle

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.utils import resample

# Required so pickle.load can resolve the MLP class definitions.
from MultilayerPerceptron.MLP_1_layer import Perceptron, PerceptronFunnel  # noqa: F401

from utils.train_utils import (
    outlier_flattening_2_entries,
    normalize_data_min_max_II,
)


# ---------------------------------------------------------------------------
# Bootstrap CI helpers
# ---------------------------------------------------------------------------
def bootstrap_mae(y_true, y_pred, n_boot=5000):
    maes = []
    for _ in range(n_boot):
        idx = resample(range(len(y_true)))
        maes.append(np.mean(np.abs(y_true[idx] - y_pred[idx])))
    return np.percentile(maes, [2.5, 97.5])


def bootstrap_r2(y_true, y_pred, n_boot=5000, ci=95, random_state=None):
    """Bootstrap CI for R^2."""
    rng = np.random.default_rng(random_state)
    stats_ = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        stats_.append(r2_score(y_true[idx], y_pred[idx]))
    alpha = (100 - ci) / 2
    return np.percentile(stats_, [alpha, 100 - alpha])


def bootstrap_pearsonr(y_true, y_pred, n_boot=5000, ci=95, random_state=None):
    """Bootstrap CI for Pearson r."""
    rng = np.random.default_rng(random_state)
    stats_ = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        stats_.append(np.corrcoef(y_true[idx], y_pred[idx])[0, 1])
    alpha = (100 - ci) / 2
    return np.percentile(stats_, [alpha, 100 - alpha])


# ---------------------------------------------------------------------------
# Prediction + bias correction
# ---------------------------------------------------------------------------
def predict_with_bias_correction(X_train, X_test, results, features):
    """
    Predicts brain age for ``X_test`` and applies Cole's post-hoc linear
    bias correction followed by Zhang's age-level centring.  Both
    corrections are calibrated on the *training* set only (no leakage).

    Parameters
    ----------
    X_train  : DataFrame with the training features and an 'Edad' column.
    X_test   : DataFrame with the test features (same columns as X_train).
    results  : DataFrame with the test metadata (Edad, sexo, eTIV, ...).
    features : list of feature names to use for the model.

    Returns
    -------
    DataFrame with the per-subject metadata plus the columns
    ``pred_Edad``, ``pred_Edad_c``, ``BrainPAD`` and ``BrainPAD_c``.
    """
    etiv = results['eTIV'].values.tolist()

    edades_train = X_train['Edad'].values
    edades_test = results['Edad'].values

    X_train = X_train[features]
    X_test = X_test[features]

    # Outlier flattening using training-set quantiles.
    X_train, X_test = outlier_flattening_2_entries(X_train, X_test)

    # Min-max scaling to [-1, 1] using training-set bounds.
    # NOTE: choice of normalisation (min-max vs z-score) measurably affects
    # the final accuracy.
    X_train, X_test = normalize_data_min_max_II(X_train, X_test, (-1, 1))

    # Load the trained MLP and predict.
    file_path = '../model/MLP_brain_age.pkl'
    with open(file_path, 'rb') as f:
        regresor = pickle.load(f)

    results['pred_Edad'] = regresor.predict(X_test)
    results['BrainPAD'] = results['pred_Edad'] - edades_test
    results['eTIV'] = etiv

    # ----- Cole's post-hoc linear bias correction -------------------------
    # Fit pred_train ~ a + b * age_train on the *training* predictions and
    # apply the inverse to the test predictions.
    df_bias_correction = pd.read_csv('../model/bias_correction_reference.csv')

    model = LinearRegression()
    model.fit(df_bias_correction[['edades_train']],
              df_bias_correction['pred_train'])

    slope = model.coef_[0]
    intercept = model.intercept_

    results['pred_Edad_c'] = (results['pred_Edad'] - intercept) / slope
    results['BrainPAD_c'] = results['pred_Edad_c'] - edades_test

    # ----- Zhang's age-level centring (calibrated on training set) -------
    # Build a lookup table {integer age -> mean BrainPAD_c on the training
    # set} and subtract it from the corresponding test BrainPAD_c.
    df_bias_correction['BrainPAD_c'] = (
        (df_bias_correction['pred_train'] - intercept) / slope
        - df_bias_correction['edades_train']
    )
    df_bias_correction['edad_int'] = (
        df_bias_correction['edades_train'].round().astype(int)
    )
    bias_table = (
        df_bias_correction.groupby('edad_int')['BrainPAD_c'].mean().to_dict()
    )

    results['edad_int'] = edades_test.round().astype(int)
    results['BrainPAD_c'] = (
        results['BrainPAD_c'] - results['edad_int'].map(bias_table)
    )

    # ----- Diagnostics ---------------------------------------------------
    r_before, _ = pearsonr(results['pred_Edad'] - results['Edad'],
                           results['Edad'])
    print('BrainPAD-Age correlation before correction: ' + str(r_before))

    r_after, _ = pearsonr(results['pred_Edad_c'] - results['Edad'],
                          results['Edad'])
    print('BrainPAD-Age correlation after correction:  ' + str(r_after))

    # eTIV converted from mm^3 to litres for the regression read-out.
    results['eTIV'] = results['eTIV'] / 1_000_000

    results.rename(columns={'sexo(M=1;F=0)': 'sexo'}, inplace=True)
    rlm_model = smf.rlm('BrainPAD ~ C(sexo) + eTIV', data=results).fit()
    print(rlm_model.summary())

    return results


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
if __name__ == '__main__':

    # Training reference (features + Edad).
    X_train = pd.read_csv('../model/train_sample.csv')

    # Internal hold-out test set.
    Test_Set = pd.read_csv('../model/test_sample.csv')

    # External / longitudinal harmonized cohorts.
    AgeRisk_harmo = pd.read_csv('../data/AgeRisk_harmonized.csv')
    X_test_Cardiff_I_harmo = pd.read_csv('../data/ALSPAC_20_MRI_I_harmonized.csv')
    X_test_Cardiff_II_harmo = pd.read_csv('../data/ALSPAC_30_MRI_II_harmonized.csv')

    # Selected feature list (621 morphometric features).
    features = pd.read_csv('../model/selected_features.csv')
    features = ast.literal_eval(features.iloc[0, 2])

    # ---------- Internal hold-out --------------------------------------- #
    # The last 7 columns of Test_Set are the per-subject metadata.
    X_test_results = Test_Set.iloc[:, -7:].copy()
    X_test_results['eTIV'] = Test_Set['eTIV'].values
    X_test = Test_Set.iloc[:, :-7].drop(columns=['eTIV'])
    result_TestSet = predict_with_bias_correction(
        X_train, X_test, X_test_results, features
    )
    result_TestSet.to_csv('../data/predictions_TestSet.csv', index=False)

    # ---------- AgeRisk -------------------------------------------------- #
    X_test_results = AgeRisk_harmo.iloc[:, -7:].copy()
    X_test_results['eTIV'] = AgeRisk_harmo['eTIV'].values
    X_test = AgeRisk_harmo.iloc[:, :-7].drop(columns=['eTIV'])
    result_AgeRisk = predict_with_bias_correction(
        X_train, X_test, X_test_results, features
    )
    result_AgeRisk.to_csv('../data/predictions_AgeRisk.csv', index=False)

    # ---------- ALSPAC-20 MRI-I ----------------------------------------- #
    # Cardiff cohorts carry an extra metadata column (8 instead of 7).
    X_test_results = X_test_Cardiff_I_harmo.iloc[:, -8:].copy()
    X_test_results['eTIV'] = X_test_Cardiff_I_harmo['eTIV'].values
    X_test = X_test_Cardiff_I_harmo.iloc[:, :-8].drop(columns=['eTIV'])
    result_ALSPAC_I = predict_with_bias_correction(
        X_train, X_test, X_test_results, features
    )
    result_ALSPAC_I.to_csv('../data/predictions_ALSPAC_20_MRI_I.csv', index=False)

    # ---------- ALSPAC-30 MRI-II ---------------------------------------- #
    X_test_results = X_test_Cardiff_II_harmo.iloc[:, -8:].copy()
    X_test_results['eTIV'] = X_test_Cardiff_II_harmo['eTIV'].values
    X_test = X_test_Cardiff_II_harmo.iloc[:, :-8].drop(columns=['eTIV'])
    result_ALSPAC_II = predict_with_bias_correction(
        X_train, X_test, X_test_results, features
    )
    result_ALSPAC_II.to_csv('../data/predictions_ALSPAC_30_MRI_II.csv', index=False)
