"""
Training-time utilities for the ALSPAC brain-age pipeline.

Functions:
  - outlier_flattening
  - outlier_flattening_2_entries
  - normalize_data_min_max
  - normalize_data_min_max_II
  - define_lists_cnn
  - execute_in_val_and_test_NN
"""

import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)
from scipy import stats


def outlier_flattening(data_train, data_val, data_test):
    """Clip every numeric column to its 2.5%/97.5% percentiles in train, val and test."""
    data_train_flat = data_train.copy()
    data_val_flat = data_val.copy()
    data_test_flat = data_test.copy()

    for col in data_train.columns:
        if col == 'sexo':
            continue
        percentiles = data_train[col].quantile([0.025, 0.975]).values
        data_train_flat[col] = np.clip(data_train[col], percentiles[0], percentiles[1])
        data_val_flat[col] = np.clip(data_val[col], percentiles[0], percentiles[1])
        data_test_flat[col] = np.clip(data_test[col], percentiles[0], percentiles[1])

    return data_train_flat, data_val_flat, data_test_flat


def outlier_flattening_2_entries(data_train, data_test):
    """Clip every numeric column to its 2.5%/97.5% percentiles (train + test only)."""
    data_train_flat = data_train.copy()
    data_test_flat = data_test.copy()

    for col in data_train.columns:
        if col == 'sexo':
            continue
        percentiles = data_train[col].quantile([0.025, 0.975]).values
        data_train_flat[col] = np.clip(data_train[col], percentiles[0], percentiles[1])
        data_test_flat[col] = np.clip(data_test[col], percentiles[0], percentiles[1])

    return data_train_flat, data_test_flat


def normalize_data_min_max(data_train, data_val, data_test, range):
    """MinMax-scale train/val/test, fitting the scaler on train only."""
    scaler = MinMaxScaler(feature_range=range)
    data_train = scaler.fit_transform(data_train)
    data_val = scaler.transform(data_val)
    data_test = scaler.transform(data_test)
    return data_train, data_val, data_test


def normalize_data_min_max_II(data_train, data_test, range):
    """MinMax-scale train and test, fitting the scaler on train only."""
    scaler = MinMaxScaler(feature_range=range)
    data_train = scaler.fit_transform(data_train)
    data_test = scaler.transform(data_test)
    return data_train, data_test


def define_lists_cnn():
    """Initialise the result-collection lists used by the CNN-style training loop."""
    (
        MAE_list_train_tab_CNN,
        MAE_list_train_unbiased_tab_CNN,
        r_list_train_tab_CNN,
        r_list_train_unbiased_tab_CNN,
        rs_BAG_train_tab_CNN,
        rs_BAG_train_unbiased_tab_CNN,
        alphas_tab_CNN,
        betas_tab_CNN,
    ) = [], [], [], [], [], [], [], []
    BAG_ChronoAge_df_tab_CNN = pd.DataFrame()

    return [
        MAE_list_train_tab_CNN,
        MAE_list_train_unbiased_tab_CNN,
        r_list_train_tab_CNN,
        r_list_train_unbiased_tab_CNN,
        rs_BAG_train_tab_CNN,
        rs_BAG_train_unbiased_tab_CNN,
        alphas_tab_CNN,
        betas_tab_CNN,
        BAG_ChronoAge_df_tab_CNN,
        'tab_CNN',
    ]


def execute_in_val_and_test_NN(
    data_train_filtered,
    ages_train,
    data_val_filtered,
    ages_val,
    data_test_filtered,
    ages_test,
    lista,
    regressor,
    n_features,
    save_dir,
    fold,
):
    """Train the NN regressor + an ElasticNet baseline, evaluate on test and persist artefacts.

    Hyper-parameters baked in below match the supplement (Methods, Machine Learning):
        621 input features, 16 hidden units, lr = 2e-4, weight_decay = 1e-3,
        dropout = 0.35, patience = 10, batch_size = 128, Smooth-L1 (Huber, beta=3).
    """
    regressor_used = lista[9]

    # Train the network with validation-based early stopping
    regressor.fit(
        data_train_filtered, ages_train,
        data_val_filtered, ages_val,
        fold, 621, 16,
        lr=2 * 1e-4, weight_decay=1e-3, dropout=0.35,
        patience=10, batch_size=128,
    )

    # ElasticNet baseline (used downstream as a sanity reference)
    elnet = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)
    elnet.fit(data_train_filtered, ages_train)

    # Predictions on validation set (for fitting the bias-correction model)
    pred_val = regressor.predict(data_val_filtered)

    pd.DataFrame({'edades_train': ages_val, 'pred_train': pred_val}) \
        .to_csv(os.path.join(save_dir, 'bias_correction_reference.csv'), index=False)

    # Predictions on test set
    pred_test = regressor.predict(data_test_filtered)
    pred_test_elnet = elnet.predict(data_test_filtered)

    BAG_test = pred_test - ages_test
    BAG_test_elnet = pred_test_elnet - ages_test

    # NN test metrics
    huber = nn.SmoothL1Loss(beta=3)
    huber_test = huber(torch.tensor(ages_test).float(), torch.tensor(pred_test).float())
    mae_test = mean_absolute_error(ages_test, pred_test)
    mape_test = mean_absolute_percentage_error(ages_test, pred_test)
    r2_test = r2_score(ages_test, pred_test)
    r_test = stats.pearsonr(ages_test, pred_test)[0]
    r_bag_test = stats.pearsonr(BAG_test, ages_test)[0]

    # ElasticNet baseline metrics
    mae_test_elnet = mean_absolute_error(ages_test, pred_test_elnet)
    mape_test_elnet = mean_absolute_percentage_error(ages_test, pred_test_elnet)
    r2_test_elnet = r2_score(ages_test, pred_test_elnet)
    r_test_elnet = stats.pearsonr(ages_test, pred_test_elnet)[0]
    r_bag_test_elnet = stats.pearsonr(BAG_test_elnet, ages_test)[0]

    print('----------- ' + regressor_used + ' r & MAE test (biased) -------------')
    print('MAE test:   ' + str(mae_test))
    print('Huber test: ' + str(huber_test))
    print('MAPE test:  ' + str(mape_test))
    print('r test:     ' + str(r_test))
    print('R2 test:    ' + str(r2_test))
    print('--------- ' + regressor_used + ' BAG vs real-age correlation, test -------------')
    print('r BAG-real-age test (biased): ' + str(r_bag_test))
    print('')
    print('----------- ElasticNet r & MAE test (biased) -------------')
    print('MAE test:  ' + str(mae_test_elnet))
    print('MAPE test: ' + str(mape_test_elnet))
    print('r test:    ' + str(r_test_elnet))
    print('R2 test:   ' + str(r2_test_elnet))
    print('--------- ElasticNet BAG vs real-age correlation, test -------------')
    print('r BAG-real-age test (biased): ' + str(r_bag_test_elnet))
    print('')

    # Predicted-vs-real scatter
    plt.figure(figsize=(8, 6))
    plt.scatter(ages_test, pred_test, color='blue', label='Predictions')
    plt.plot([ages_test.min(), ages_test.max()],
             [ages_test.min(), ages_test.max()],
             'k--', lw=2, label='Ideal Fit')
    plt.xlabel('Real Age')
    plt.ylabel('Predicted Age')
    plt.title('Predicted Age vs. Real Age')

    textstr = '\n'.join((
        f'MAE: {mae_test:.2f}',
        f'Pearson r: {r_test:.2f}',
        f'R²: {r2_test:.2f}',
    ))
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.savefig(os.path.join(save_dir, f'Model_PredAge_vs_Age_fold_{fold}.svg'))

    metrics_test = pd.DataFrame(
        list(zip([mae_test], [r_test], [r_bag_test])),
        columns=['MAE_biased_test', 'r_biased_test', 'r_bag_real_biased_test'],
    )

    pickle.dump(
        regressor,
        open(os.path.join(save_dir, 'MLP_brain_age.pkl'), 'wb'),
    )

    return metrics_test
