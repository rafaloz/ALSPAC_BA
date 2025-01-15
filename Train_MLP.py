# -----------------------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------------------
from utils import *                             # Utility functions (DO NOT change)
import pickle
from scipy.stats import ks_2samp
from MultilayerPerceptron.MLP_1_layer import *  # Perceptron model (DO NOT change)
from sklearn.model_selection import train_test_split
from datetime import datetime

# -----------------------------------------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------------------------------------
df_all = pd.read_csv('....data.csv')
save_dir = '.../folder_model/'

# -----------------------------------------------------------------------------------------
# DATA PREPARATION
# -----------------------------------------------------------------------------------------
# Shuffle rows and reset indexes
df_all = df_all.sample(frac=1, random_state=10).reset_index(drop=True)

# Split into training and testing sets
df_train_all, df_test_all = train_test_split(df_all, test_size=0.1, random_state=10)

# -----------------------------------------------------------------------------------------
# GLOBAL VARIABLES TO STORE RESULTS
# -----------------------------------------------------------------------------------------
ks_result = []
features = []
features_tag = []

Results_dataframe_SVR_test = pd.DataFrame()
Results_dataframe_perceptron_test = pd.DataFrame()
Results_dataframe_RF_test = pd.DataFrame()

Results_dataframe_SVR_val = pd.DataFrame()
Results_dataframe_perceptron_val = pd.DataFrame()
Results_dataframe_RF_val = pd.DataFrame()

MAE_val = []
MAE_test = []
r_test = []

prediction_SVR_saved_test = []
prediction_perceptron_saved_test = []
prediction_RandomForest_saved_test = []

prediction_SVR_saved_val = []
prediction_perceptron_saved_val = []
prediction_RandomForest_saved_val = []

ages_val = []
ages_test = []

# -----------------------------------------------------------------------------------------
# MAIN PROCESS
# -----------------------------------------------------------------------------------------
for j in [100]:
    i = 0

    # Split the data again from the previously separated train/test subsets
    df_train, df_test = df_train_all, df_test_all

    # Further split df_train into df_train and df_val
    df_train, df_val = train_test_split(df_train, test_size=0.11, random_state=10)

    # Save the resulting sets for reference
    df_train.to_csv(os.path.join(save_dir, 'Datos_train_sample.csv'), index=False)
    df_test.to_csv(os.path.join(save_dir, 'Datos_Test_sample.csv'), index=False)
    df_val.to_csv(os.path.join(save_dir, 'Datos_val_sample.csv'), index=False)

    # Extract age information
    ages_train = df_train['Edad'].values
    ages_val_ = df_val['Edad'].values
    ages_test_ = df_test['Edad'].values

    ages_test.append(ages_test_)

    # Extract other variables (scanner, Bo, sex, etc.)
    scanners_train = df_train['Escaner'].values
    scanners_val = df_val['Escaner'].values
    scanners_test = df_test['Escaner'].values

    Bo_train = df_train['Bo'].values
    Bo_val = df_val['Bo'].values
    Bo_test = df_test['Bo'].values

    sex_train = df_train['sexo(M=1;F=0)'].values
    sex_val = df_val['sexo(M=1;F=0)'].values
    sex_test = df_test['sexo(M=1;F=0)'].values

    # -------------------------------------------------------------------------------------
    # PREPROCESSING: DROPPING IRRELEVANT COLUMNS
    # -------------------------------------------------------------------------------------
    print('[INFO] ##### Standardization #####')
    df_train = df_train.drop(['ID', 'Bo', 'Escaner', 'DataBase', 'Age', 'Pathology'], axis=1)
    df_val = df_val.drop(['ID', 'Bo', 'Escaner', 'DataBase', 'Age', 'Pathology'], axis=1)
    df_test = df_test.drop(['ID', 'Bo', 'Escaner', 'DataBase', 'Age', 'Pathology'], axis=1)

    # -------------------------------------------------------------------------------------
    # KOLMOGOROV-SMIRNOV TEST FOR AGES (TO CHECK SIMILAR DISTRIBUTIONS)
    # -------------------------------------------------------------------------------------
    ks_test = ks_2samp(ages_train, ages_test_)
    print('Kolmogorov-Smirnov test for ages (train vs test)')
    print('If p-value > 0.05, we cannot reject that they are from the same distribution: ' + str(ks_test[1]))
    ks_result.append(ks_test[1])

    ks_test = ks_2samp(ages_train, ages_val_)
    print('Kolmogorov-Smirnov test for ages (train vs val)')
    print('If p-value > 0.05, we cannot reject that they are from the same distribution: ' + str(ks_test[1]))
    ks_result.append(ks_test[1])

    # -------------------------------------------------------------------------------------
    # MORPHOLOGICAL FEATURE SELECTION
    # -------------------------------------------------------------------------------------
    morphological_features = df_train.columns.tolist()

    print('[INFO] ##### Handling Outliers #####')
    # Remove or flatten outliers using the corresponding function
    start_time = datetime.now()
    X_train, X_val, X_test = outlier_flattening(df_train, df_val, df_test)  # DO NOT change
    end_time = datetime.now()
    print(f"Function execution time: {(end_time - start_time).total_seconds()} seconds")

    # Convert to NumPy arrays
    X_train = X_train.values
    X_val = X_val.values
    X_test = X_test.values

    # -------------------------------------------------------------------------------------
    # NORMALIZATION
    # -------------------------------------------------------------------------------------
    print('[INFO] ##### Normalization #####')
    start_time = datetime.now()
    X_train, X_val, X_test = normalize_data_min_max(X_train, X_val, X_test, (-1, 1))  # DO NOT change
    end_time = datetime.now()
    print(f"Function execution time: {(end_time - start_time).total_seconds()} seconds")

    # -------------------------------------------------------------------------------------
    # FEATURE SELECTION
    # -------------------------------------------------------------------------------------
    print('[INFO] ##### Feature Selection #####')

    # Reconstruct DataFrames with original column names
    X_train = pd.DataFrame(X_train, columns=morphological_features)
    X_val = pd.DataFrame(X_val, columns=morphological_features)
    X_test = pd.DataFrame(X_test, columns=morphological_features)

    start_time = datetime.now()
    X_train, X_val, X_test, features_names_SFS = feature_selection(
        X_train, X_val, X_test, ages_train, n_features=j, n_features_to_try=30
    )  # DO NOT change
    end_time = datetime.now()
    print(f"Function execution time: {(end_time - start_time).total_seconds()} seconds")

    print('[INFO] Number of selected features: ' + str(j))
    print('[INFO] Selected features:')
    print(features_names_SFS)
    print('Shape of the training data: ' + str(X_train.shape) + '\n')

    features.append(features_names_SFS)
    features_tag.append('features_nfeats_' + str(j))

    # -------------------------------------------------------------------------------------
    # DEFINITION OF LISTS AND PERCEPTRON MODEL TRAINING
    # -------------------------------------------------------------------------------------
    # Define auxiliary lists
    perceptron_lists = define_lists_cnn()  # DO NOT change

    print('[INFO] ##### Training Perceptron Model #####')
    model = Perceptron()  # DO NOT change

    MAEs_and_rs_perceptron_test = execute_in_val_and_test_NN(
        X_train,
        ages_train,
        X_val,
        ages_val_,
        X_test,
        ages_test_,
        perceptron_lists,
        model,
        j,
        save_dir,
        i
    )  # DO NOT change

    Results_dataframe_perceptron_test = pd.concat(
        [MAEs_and_rs_perceptron_test, Results_dataframe_perceptron_test],
        axis=0
    )

# -----------------------------------------------------------------------------------------
# SAVING RESULTS
# -----------------------------------------------------------------------------------------
Results_dataframe_perceptron_test.to_csv(os.path.join(save_dir, 'results_perceptron_test.csv'))

with open(os.path.join(save_dir, 'perceptron_test_list.pkl'), 'wb') as f:
    pickle.dump(prediction_perceptron_saved_test, f)

with open(os.path.join(save_dir, 'ages_test.pkl'), 'wb') as f:
    pickle.dump(ages_test, f)

df_features = pd.DataFrame(list(zip(features_tag, features)), columns=['features_tag', 'features'])
df_features.to_csv(os.path.join(save_dir, 'stored_features.csv'))

# -----------------------------------------------------------------------------------------
# PRINT RESULTS
# -----------------------------------------------------------------------------------------
print(features)
print(ks_result)
