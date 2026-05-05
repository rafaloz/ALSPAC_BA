"""
ComBat-GAM helpers used by the ALSPAC brain-age pipeline.

Public API:
  - learn_harmonization(reference_data, source_data, harmo_name, model_save_dir='.')
  - apply_harmonization(data_to_apply, reference_data, ref_level, save_dir, name)

Both wrap the locally-vendored fork of neuroHarmonize (``./neuroharmonize/``),
which adds the batch-level diagnostics used to produce supplement Table S4.
"""

import os

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from neuroharmonize.neuroharmonize import harmonizationLearn, harmonizationApply
from neuroharmonize.neuroharmonize.harmonizationLearn import saveHarmonizationModel
from neuroharmonize.neuroharmonize.harmonizationApply import loadHarmonizationModel


def learn_harmonization(reference_data, source_data, harmo_name, model_save_dir='.'):
    """Fit and save a ComBat-GAM harmonization for ``source_data`` against ``reference_data``.

    The reference scanner is renamed to a sentinel value (``'zarmonization_1'``)
    so it lands first in alphabetical order and is picked as the reference batch.
    """
    pliks_flag = False

    reference_data['Escaner'] = 'zarmonization_1'

    if 'pliks' in source_data.columns.tolist()[4]:
        pliks_flag = True
        pliks = source_data['pliks18TH'].values
        source_data = source_data.drop(['pliks18TH', 'pliks20TH'], axis='columns')
        source_data['Escaner'] = 'CARDIFF_ESCANER'
        source_data = source_data.rename(columns={'sexo': 'sexo(M=1;F=0)'})
        source_data['Bo'] = '3.0T'
        source_data = source_data[reference_data.columns.tolist()]

    source_data = source_data.loc[:, ~source_data.columns.duplicated()]
    source_data = source_data[reference_data.columns.tolist()]
    data_combined = pd.concat([source_data, reference_data])

    scanner_list = data_combined['Escaner'].values
    etiv_all = data_combined['eTIV'].values
    ages_all = data_combined['Edad']
    bo_all = data_combined['Bo'].values
    sex_all = data_combined['sexo(M=1;F=0)'].values
    ids = data_combined['ID'].values
    db_all = data_combined['DataBase'].values
    pathology_all = data_combined['Patologia'].values
    data_combined = data_combined.drop(
        ['ID', 'Bo', 'sexo(M=1;F=0)', 'Escaner', 'DataBase', 'Patologia', 'Edad'],
        axis=1,
    )

    feature_names = data_combined.columns.tolist()

    le_scanner = LabelEncoder()
    scanners_num = pd.DataFrame(le_scanner.fit_transform(scanner_list))
    sex_num = pd.DataFrame(sex_all)

    covars = pd.DataFrame({
        'SITE': scanners_num.values.ravel().tolist(),
        'SEX': np.squeeze(sex_num.values).tolist(),
        'ETIV': etiv_all.tolist(),
        'AGE': ages_all.values.tolist(),
    })

    my_model, harmonized = harmonizationLearn(
        data_combined.values, covars,
        smooth_terms=['ETIV', 'AGE', 'SEX'],
        ref_batch=1,
    )

    model_path = os.path.join(model_save_dir, harmo_name)
    if not os.path.isfile(model_path):
        saveHarmonizationModel(my_model, harmo_name)
        print('Model saved.')
    else:
        print('Model already exists. Not saved again.')

    harmonized_df = pd.DataFrame(harmonized, columns=feature_names)
    harmonized_df['Edad'] = ages_all.values
    harmonized_df['Escaner'] = scanner_list
    harmonized_df['Bo'] = bo_all
    harmonized_df['sexo(M=1;F=0)'] = sex_all
    harmonized_df['eTIV'] = etiv_all
    harmonized_df['ID'] = ids
    harmonized_df['DataBase'] = db_all
    harmonized_df['Patologia'] = pathology_all

    if pliks_flag:
        harmonized_df = harmonized_df[harmonized_df['Escaner'] != 'zarmonization_1']
        harmonized_df['pliks18TH'] = pliks

    return harmonized_df, my_model


def apply_harmonization(data_to_apply, reference_data, ref_level, save_dir, name):
    """Load a saved harmonization model and apply it to new data."""
    pliks_flag = False
    if 'pliks' in data_to_apply.columns.tolist()[4]:
        pliks_flag = True
        pliks = data_to_apply['pliks18TH'].values
        data_to_apply = data_to_apply.drop(['pliks18TH', 'pliks20TH'], axis='columns')
        data_to_apply['Bo'] = '3.0T'
        data_to_apply['Escaner'] = 'CARDIFF_ESCANER'
        data_to_apply = data_to_apply.rename(columns={'sexo': 'sexo(M=1;F=0)'})

    data_to_apply = data_to_apply[reference_data.columns.tolist()]

    scanner_list = data_to_apply['Escaner'].values
    etiv_all = data_to_apply['eTIV'].values
    ages_all = data_to_apply['Edad']
    bo_all = data_to_apply['Bo'].values
    sex_all = data_to_apply['sexo(M=1;F=0)'].values
    ids = data_to_apply['ID'].values
    db_all = data_to_apply['DataBase'].values
    pathology_all = data_to_apply['Patologia'].values
    data_to_apply = data_to_apply.drop(
        ['ID', 'Bo', 'sexo(M=1;F=0)', 'Escaner', 'DataBase', 'Patologia', 'Edad'],
        axis=1,
    )

    feature_names = data_to_apply.columns.tolist()

    le_scanner = LabelEncoder()
    scanners_num = pd.DataFrame(le_scanner.fit_transform(scanner_list))
    le_sex = LabelEncoder()
    sex_num = pd.DataFrame(le_sex.fit_transform(sex_all))

    covars = pd.DataFrame({
        'SITE': scanners_num.values.ravel().tolist(),
        'SEX': np.squeeze(sex_num.values).tolist(),
        'ETIV': etiv_all.tolist(),
        'AGE': ages_all.values.tolist(),
    })

    my_model = loadHarmonizationModel(os.path.join(save_dir, name))
    harmonized_array = harmonizationApply(data_to_apply.values, covars, my_model, ref_level)

    harmonized_df = pd.DataFrame(harmonized_array, columns=feature_names)
    harmonized_df['ID'] = ids
    harmonized_df['DataBase'] = db_all
    harmonized_df['Edad'] = ages_all.values
    harmonized_df['sexo(M=1;F=0)'] = sex_all
    harmonized_df['Escaner'] = scanner_list
    harmonized_df['Patologia'] = pathology_all
    harmonized_df['Bo'] = bo_all
    harmonized_df['eTIV'] = etiv_all

    if pliks_flag:
        harmonized_df['pliks18TH'] = pliks

    return harmonized_df
