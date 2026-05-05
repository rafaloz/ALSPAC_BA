"""
02_train_MLP.py
================

Train the MLP brain-age regressor on the (already harmonized) multi-site
training dataset. Splits 60% / 30% / 10% (train / val / hold-out test),
fits the MLP via :func:`utils.train_utils.execute_in_val_and_test_NN`, and
persists the model, the train/val/test splits, the bias-correction CSVs
and the retained-feature list under ``../model/``.

Run from ``src/``.
"""

import os
from datetime import datetime

import pandas as pd

from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split

from utils.train_utils import (
    outlier_flattening,
    normalize_data_min_max,
    define_lists_cnn,
    execute_in_val_and_test_NN,
)
from MultilayerPerceptron.MLP_1_layer import Perceptron


# Paths
DATA_DIR = "../data"
MODEL_DIR = "../model"
os.makedirs(MODEL_DIR, exist_ok=True)

TRAINING_DATA_CSV = os.path.join(DATA_DIR, "training_data_harmonized.csv")
SAVE_DIR = MODEL_DIR


# Load the multi-site training dataset (post-harmonization)
data_all = pd.read_csv(TRAINING_DATA_CSV)

# Shuffle deterministically and split 90/10 (train+val vs hold-out test)
data_all = data_all.sample(frac=1, random_state=42).reset_index(drop=True)
data_train_all, data_test_all = train_test_split(data_all, test_size=0.1, random_state=42)


ks_result, features, features_tag = [], [], []
results_perceptron_test = pd.DataFrame()

for j in [621]:
    fold = 0

    data_train, data_test = data_train_all, data_test_all
    data_train, data_val = train_test_split(data_train, test_size=0.3333333,
                                            random_state=42)

    data_train.to_csv(os.path.join(SAVE_DIR, 'train_sample.csv'), index=False)
    data_test.to_csv(os.path.join(SAVE_DIR, 'test_sample.csv'),  index=False)
    data_val.to_csv(os.path.join(SAVE_DIR, 'val_sample.csv'),   index=False)

    ages_train = data_train['Edad'].values
    ages_val   = data_val['Edad'].values
    ages_test  = data_test['Edad'].values

    # Drop metadata + eTIV (the latter is a ComBat-GAM covariate, not an MLP input)
    drop_cols = ['ID', 'Bo', 'Escaner', 'DataBase', 'Edad', 'Patologia',
                 'sexo(M=1;F=0)', 'eTIV']
    data_train = data_train.drop(drop_cols, axis=1)
    data_val   = data_val.drop(drop_cols,   axis=1)
    data_test  = data_test.drop(drop_cols,  axis=1)

    # Sanity-check: train/val/test age distributions overlap (Kolmogorov-Smirnov)
    p_train_test = ks_2samp(ages_train, ages_test)[1]
    p_train_val  = ks_2samp(ages_train, ages_val)[1]
    print('KS p-value train vs test:', p_train_test,
          '(>0.05 means we cannot reject same distribution)')
    print('KS p-value train vs val: ', p_train_val)
    ks_result.extend([p_train_test, p_train_val])

    morphological_features = data_train.columns.tolist()

    # 1) Outlier flattening (clip to 2.5/97.5 percentiles)
    print('[INFO] ##### Outliers #####')
    t0 = datetime.now()
    X_train, X_val, X_test = outlier_flattening(data_train, data_val, data_test)
    print(f"  done in {(datetime.now() - t0).total_seconds():.1f} s")

    X_train, X_val, X_test = X_train.values, X_val.values, X_test.values

    # 2) Min-max normalisation to (-1, 1)
    print('[INFO] ##### Normalization #####')
    t0 = datetime.now()
    X_train, X_val, X_test = normalize_data_min_max(X_train, X_val, X_test, (-1, 1))
    print(f"  done in {(datetime.now() - t0).total_seconds():.1f} s")

    print('[INFO] Number of features:', j)
    features.append(morphological_features)
    features_tag.append(f'features_nfeats_{j}')

    # 3) Train the MLP and the ElasticNet bias-correction baseline
    print('[INFO] ##### Training #####')
    listas_perceptron = define_lists_cnn()
    model = Perceptron()
    metrics_perceptron_test = execute_in_val_and_test_NN(
        X_train, ages_train,
        X_val,   ages_val,
        X_test,  ages_test,
        listas_perceptron, model, j,
        SAVE_DIR, fold,
    )
    results_perceptron_test = pd.concat(
        [metrics_perceptron_test, results_perceptron_test], axis=0,
    )


df_features = pd.DataFrame(
    list(zip(features_tag, features)),
    columns=['features_tag', 'features'],
)
df_features.to_csv(os.path.join(SAVE_DIR, 'selected_features.csv'))

print('Training complete. KS p-values:', ks_result)
print('Test metrics:')
print(results_perceptron_test.to_string(index=False))
