import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

from utils_Train import *
import pickle
import ast

from itertools import islice
from sklearn.utils import shuffle

from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from scipy.stats import kruskal
from scipy.stats import f

from utils_Harmonization import *

import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.formula.api import ols
from statsmodels.formula.api import wls
from statsmodels.stats.multicomp import MultiComparison
import pingouin as pg

def calculate_cohen_d(group1, group2):
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    return (mean2 - mean1) / pooled_std

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def figura_edad_y_edad_predicha(edades_test, pred_test):

    # calculo MAE, MAPE y r test
    MAE_biased_test = mean_absolute_error(edades_test, pred_test)
    r_squared = r2_score(edades_test, pred_test)
    r_biased_test = stats.pearsonr(edades_test, pred_test)[0]

    # Figura concordancia entre predichas y reales con reg lineal
    plt.figure(figsize=(8, 8))
    plt.scatter(edades_test, pred_test, color='blue', label='Predictions')
    plt.plot([10, 100], [10, 100], 'k--', lw=2, label='Ideal Fit')
    plt.xlabel('Real Age')
    plt.ylabel('Predicted Age')
    plt.title('Predicted Age vs. Real Age')

    # Set x and y axis limits
    plt.xlim(5, 100)
    plt.ylim(5, 100)

    # Ensure x and y axes have the same scale
    plt.gca().set_aspect('equal', adjustable='box')

    # Annotate MAE, Pearson correlation r, and R² in the plot
    textstr = '\n'.join((
        f'MAE: {MAE_biased_test:.2f}',
        f'Pearson r: {r_biased_test:.2f}',
        f'R²: {r_squared:.2f}'))
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.show()

def model_evaluation_MCCQR_MLP(X_train, X_test, results, features):
    etiv = X_test['eTIV'].values.tolist()

    edades_train = X_train['Edad'].values
    edades_test = results['Edad'].values

    X_train = X_train[features]
    X_test = X_test[features]

    # aplico la eliminación de outliers
    X_train, X_test = outlier_flattening(X_train, X_test)

    # 3.- normalizo los datos OJO LA NORMALIZACION QUE CON Z NORM O CON 0-1 PUEDE VARIAR EL RESULTADO BASTANTE!
    X_train, X_test = normalize_data_min_max(X_train, X_test, (-1, 1))

    X_train_df = pd.DataFrame(X_train)
    X_train_df.columns = features
    X_test_df = pd.DataFrame(X_test)
    X_test_df.columns = features
    X_test_df['Edad'] = edades_test
    X_train_df['Edad'] = edades_train
    X_train_df.to_csv('X_train_df_norm.csv', index=False)
    X_test_df.to_csv('X_test_df_norm.csv', index=False)

    file_path = ('/home/rafa/PycharmProjects/Cardiff_ALSPAC/modelos/modelo_morfo_100_con_WAND/SimpleMLP_nfeats_100_fold_0.pkl')
    with open(file_path, 'rb') as file:
        regresor = pickle.load(file)

    pred_test_median_all = regresor.predict(X_test, apply_calibration=True)
    aleatory_noEpistemic_pred = pd.DataFrame(dict(islice(pred_test_median_all.items(), 4, 104)))
    df_colsest_to_zero = ((aleatory_noEpistemic_pred.T - edades_test).T).abs()
    closest_to_zero = df_colsest_to_zero.idxmin(axis=1)

    indexes = []
    for value in closest_to_zero:
        indexes.append(float(value[0:5])-0.50)

    plt.figure(figsize=(10, 5))
    sns.kdeplot(indexes, shade=True)
    plt.title('Density Plot')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.show()

    pred_test_median = pred_test_median_all['median_aleatory_noEpistemic']

    # bias correction
    df_bias_correction = pd.read_csv('/home/rafa/PycharmProjects/Cardiff_ALSPAC/modelos/modelo_morfo_100_con_WAND/DataFrame_bias_correction.csv')

    model = LinearRegression()
    model.fit(df_bias_correction[['edades_train']], df_bias_correction['pred_train'])

    slope = model.coef_[0]
    intercept = model.intercept_

    results['pred_Edad_c'] = (pred_test_median - intercept) / slope
    results['pred_Edad'] = pred_test_median
    results['BrainPAD'] = results['pred_Edad_c'] - edades_test
    results['eTIV'] = etiv

    r_antes, _ = pearsonr(results['pred_Edad'] - results['Edad'], results['Edad'])
    print('Correlacion Edad predicha - Edad antes de la corrección: '+str(r_antes))

    r_tras, _ = pearsonr(results['pred_Edad_c'] - results['Edad'], results['Edad'])
    print('Correlacion Edad predicha - Edad tras la corrección: '+str(r_tras))

    ancova_results = pg.ancova(data=results, dv='BrainPAD', between='sexo(M=1;F=0)', covar=['eTIV'])
    print(ancova_results)

    return results

def model_evaluation(X_train, X_test, results, features):
    etiv = X_test['eTIV'].values.tolist()

    edades_train = X_train['Edad'].values
    edades_test = results['Edad'].values

    X_train = X_train[features]
    X_test = X_test[features]

    # aplico la eliminación de outliers
    X_train, X_test = outlier_flattening(X_train, X_test)

    # 3.- normalizo los datos OJO LA NORMALIZACION QUE CON Z NORM O CON 0-1 PUEDE VARIAR EL RESULTADO BASTANTE!
    X_train, X_test = normalize_data_min_max(X_train, X_test, (-1, 1))

    X_train_df = pd.DataFrame(X_train)
    X_train_df.columns = features
    X_test_df = pd.DataFrame(X_test)
    X_test_df.columns = features
    X_test_df['Edad'] = edades_test
    X_train_df['Edad'] = edades_train
    X_train_df.to_csv('X_train_df_norm.csv', index=False)
    X_test_df.to_csv('X_test_df_norm.csv', index=False)

    file_path = ('/home/rafa/PycharmProjects/Cardiff_ALSPAC/modelos/modelo_morfo_100_Cardiff_balanced_WAND/SimpleMLP_nfeats_100_fold_0.pkl')
    with open(file_path, 'rb') as file:
        regresor = pickle.load(file)

    pred_test_median = regresor.predict(X_test)

    # bias correction
    df_bias_correction = pd.read_csv('/home/rafa/PycharmProjects/Cardiff_ALSPAC/modelos/modelo_morfo_100_Cardiff_balanced_WAND/DataFrame_bias_correction.csv')

    model = LinearRegression()
    model.fit(df_bias_correction[['edades_train']], df_bias_correction['pred_train'])

    slope = model.coef_[0]
    intercept = model.intercept_

    results['pred_Edad_c'] = (pred_test_median - intercept) / slope
    results['pred_Edad'] = pred_test_median
    results['BrainPAD'] = results['pred_Edad_c'] - edades_test
    results['eTIV'] = etiv

    # if'pliks18TH' in results.columns:
    # results = results.groupby('pliks18TH').apply(remove_outliers, column='BrainPAD').reset_index(drop=True)

    r_antes, _ = pearsonr(results['pred_Edad'] - results['Edad'], results['Edad'])
    print('Correlacion Edad predicha - Edad antes de la corrección: '+str(r_antes))

    r_tras, _ = pearsonr(results['pred_Edad_c'] - results['Edad'], results['Edad'])
    print('Correlacion Edad predicha - Edad tras la corrección: '+str(r_tras))

    results['eTIV_std'] = (results['eTIV'] - results['eTIV'].mean()) / results['eTIV'].std()

    results.rename(columns={'sexo(M=1;F=0)': 'sexo'}, inplace=True)
    model = smf.rlm('BrainPAD ~ C(sexo) + eTIV_std', data=results).fit()

    print(model.summary())

    return results

def benjamini_hochberg_correction(p_values):
    n = len(p_values)
    sorted_p_values = np.array(sorted(p_values))
    ranks = np.arange(1, n+1)

    # Calculate the cumulative minimum of the adjusted p-values in reverse
    adjusted_p_values = np.minimum.accumulate((sorted_p_values * n) / ranks)[::-1]

    # Reverse back to original order
    reverse_indices = np.argsort(p_values)
    return adjusted_p_values[reverse_indices]

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    r, _ = pearsonr(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, r, r2

X_train = pd.read_csv('/home/rafa/PycharmProjects/Cardiff_ALSPAC/modelos/modelo_morfo_100_Cardiff_balanced_WAND/Datos_train_sample.csv')
x_test = pd.read_csv('/home/rafa/PycharmProjects/Cardiff_ALSPAC/modelos/modelo_morfo_100_Cardiff_balanced_WAND/Datos_Test_sample.csv')
X_test_OutSample = pd.read_csv('/home/rafa/PycharmProjects/Cardiff_ALSPAC/modelos/modelo_morfo_100_Cardiff_balanced_WAND/Datos_AgeRisk_To_Test.csv')
X_test_Cardiff = pd.read_csv('/home/rafa/PycharmProjects/JoinData_FastSurfer_V2/Scripts_Join_data/Datos_per_DB/CARDIFF_PE_II_FastSurfer_V2_data.csv')
# casos_seleccioandos = pd.read_csv('/home/rafa/PycharmProjects/Cardiff_ALSPAC/datos/subjects_ALSPAC_II_adq/Datos_ALSPAC_II_Seleccionados.csv')
# X_test_Cardiff = X_test_Cardiff[X_test_Cardiff['ID'].isin(casos_seleccioandos['ID'])]

# List of IDs to remove
ids_to_remove = ['15382A_brain', '19674A_brain', '20606A_brain', '13123A_brain', '17524A_brain']

# Filter out the rows with these IDs
# X_test_Cardiff = X_test_Cardiff[~X_test_Cardiff['ID'].isin(ids_to_remove)]

print('nº cases total: '+str(X_test_Cardiff.shape[0]))
print('Age average: '+str(X_test_Cardiff['Edad'].mean()))
print('Age std: '+str(X_test_Cardiff['Edad'].std()))
print('Age max: '+str(X_test_Cardiff['Edad'].max()))
print('Age min: '+str(X_test_Cardiff['Edad'].min()))
print('Males: '+str(X_test_Cardiff['sexo'].sum()))
print('Females: '+str(X_test_Cardiff.shape[0]-X_test_Cardiff['sexo'].sum()))

features = pd.read_csv('/home/rafa/PycharmProjects/Cardiff_ALSPAC/modelos/modelo_morfo_100_Cardiff_balanced_WAND/df_features_con_CoRR.csv')
features = ast.literal_eval(features.iloc[0, 2])

# Evaluo test out of sample (Age Risk)
test_Age = x_test['Edad'].values
all_data_test = x_test.copy()
X_test_results = x_test.iloc[:, 0:7]
X_test = pd.concat([x_test.iloc[:, [2]], x_test.iloc[:, 7:]], axis=1)
features_to_armo = X_test.columns.tolist()

result_x_test = model_evaluation(X_train, X_test, X_test_results, features)
figura_edad_y_edad_predicha(result_x_test['Edad'].values, result_x_test['pred_Edad'].values)

# Evaluo test out of sample (Age Risk)
test_Age = X_test_OutSample['Edad'].values
all_data_test = X_test_OutSample.copy()
Age_Risk_results = X_test_OutSample.iloc[:, 0:7]
X_test = pd.concat([X_test_OutSample.iloc[:, [2]], X_test_OutSample.iloc[:, 7:]], axis=1)
features_to_armo = X_test.columns.tolist()

result_AgeRisk = model_evaluation(X_train, X_test, Age_Risk_results, features)
figura_edad_y_edad_predicha(result_AgeRisk['Edad'].values, result_AgeRisk['pred_Edad'].values)

# Evaluo los casos de Cardiff; aplico armonización; Armonizo test a los datos de entrenamiento
# Elimino las características que no me dejan armonizar bien
X_test_Cardiff.rename(columns={'sexo': 'sexo(M=1;F=0)'}, inplace=True)
X_test_Cardiff = pd.concat([X_test_Cardiff.iloc[:, 0:9], X_test_Cardiff[features_to_armo[1:]]], axis=1) # [:, 0:7]

# Separo sanos y enfermos pliks18 0 y pliks18 1-3
X_test_pliks_0 = X_test_Cardiff[X_test_Cardiff['pliks18TH'] == 0]
X_test_pliks_1_3 = X_test_Cardiff[(X_test_Cardiff['pliks18TH'] == 1) |
                                  (X_test_Cardiff['pliks18TH'] == 2) |
                                  (X_test_Cardiff['pliks18TH'] == 3)]

# aplico la eliminación de outliers
Cardiff_results = X_test_Cardiff.iloc[:, 0:9] # [:, 0:7]
Cardiff_results.rename(columns={'sexo': 'sexo(M=1;F=0)'}, inplace=True)
results_Cardiff = model_evaluation(X_train, X_test_Cardiff, Cardiff_results, features)
figura_edad_y_edad_predicha(results_Cardiff['Edad'], results_Cardiff['pred_Edad'])
pliks_info = X_test_Cardiff[['ID', 'pliks18TH']]

# Explicitly copy each slice to ensure they are independent DataFrames
healthyALSPAC = results_Cardiff[results_Cardiff['pliks18TH'] == 0].copy()
crazyALSPAC_1 = results_Cardiff[results_Cardiff['pliks18TH'] == 1].copy()
crazyALSPAC_2 = results_Cardiff[results_Cardiff['pliks18TH'] == 2].copy()
crazyALSPAC_3 = results_Cardiff[results_Cardiff['pliks18TH'] == 3].copy()

scaler_men = StandardScaler()
scaler_woman = StandardScaler()

healthyALSPAC_men = healthyALSPAC[healthyALSPAC['sexo']==1]
healthyALSPAC_woman = healthyALSPAC[healthyALSPAC['sexo']==0]
healthyALSPAC_men['brainPAD_standardized'] = scaler_men.fit_transform(healthyALSPAC_men[['BrainPAD']])
healthyALSPAC_woman['brainPAD_standardized'] = scaler_woman.fit_transform(healthyALSPAC_woman[['BrainPAD']])
healthyALSPAC = pd.concat([healthyALSPAC_men, healthyALSPAC_woman], axis=0)

crazyALSPAC_1_men = crazyALSPAC_1[crazyALSPAC_1['sexo']==1]
crazyALSPAC_1_woman = crazyALSPAC_1[crazyALSPAC_1['sexo']==0]
crazyALSPAC_1_men['brainPAD_standardized'] = scaler_men.transform(crazyALSPAC_1_men[['BrainPAD']])
crazyALSPAC_1_woman['brainPAD_standardized'] = scaler_woman.transform(crazyALSPAC_1_woman[['BrainPAD']])
crazyALSPAC_1 = pd.concat([crazyALSPAC_1_men, crazyALSPAC_1_woman], axis=0)

crazyALSPAC_2_men = crazyALSPAC_2[crazyALSPAC_2['sexo']==1]
crazyALSPAC_2_woman = crazyALSPAC_2[crazyALSPAC_2['sexo']==0]
crazyALSPAC_2_men['brainPAD_standardized'] = scaler_men.transform(crazyALSPAC_2_men[['BrainPAD']])
crazyALSPAC_2_woman['brainPAD_standardized'] = scaler_woman.transform(crazyALSPAC_2_woman[['BrainPAD']])
crazyALSPAC_2 = pd.concat([crazyALSPAC_2_men, crazyALSPAC_2_woman], axis=0)

crazyALSPAC_3_men = crazyALSPAC_3[crazyALSPAC_3['sexo']==1]
crazyALSPAC_3_woman = crazyALSPAC_3[crazyALSPAC_3['sexo']==0]
crazyALSPAC_3_men['brainPAD_standardized'] = scaler_men.transform(crazyALSPAC_3_men[['BrainPAD']])
crazyALSPAC_3_woman['brainPAD_standardized'] = scaler_woman.transform(crazyALSPAC_3_woman[['BrainPAD']])
crazyALSPAC_3 = pd.concat([crazyALSPAC_3_men, crazyALSPAC_3_woman], axis=0)

# Printing the number of individuals in each dataframe
print("Number of individuals in healthyALSPAC:", healthyALSPAC.shape[0])
print("Number of individuals in crazyALSPAC_1:", crazyALSPAC_1.shape[0])
print("Number of individuals in crazyALSPAC_2:", crazyALSPAC_2.shape[0])
print("Number of individuals in crazyALSPAC_3:", crazyALSPAC_3.shape[0])

healthyALSPAC.loc[:, 'Group'] = 'Healthy'
crazyALSPAC_1.loc[:, 'Group'] = 'Crazy_1'
crazyALSPAC_2.loc[:, 'Group'] = 'Crazy_2'
crazyALSPAC_3.loc[:, 'Group'] = 'Crazy_3'

crazyALSPAC = pd.concat([crazyALSPAC_1, crazyALSPAC_2, crazyALSPAC_3], axis=0)
crazyALSPAC['Group'] = 'PE_1-3'
crazyALSPAC = crazyALSPAC.rename(columns={'sexo': 'sexo(M=1;F=0)'})
healthyALSPAC = healthyALSPAC.rename(columns={'sexo': 'sexo(M=1;F=0)'})

# Calculate Cohen's d for each pair of levels
cohen_d_0_123 = calculate_cohen_d(healthyALSPAC['brainPAD_standardized'].values, crazyALSPAC['brainPAD_standardized'].values)
cohen_d_0_1 = calculate_cohen_d(healthyALSPAC['brainPAD_standardized'].values, crazyALSPAC_1['brainPAD_standardized'].values)
cohen_d_0_2 = calculate_cohen_d(healthyALSPAC['brainPAD_standardized'].values, crazyALSPAC_2['brainPAD_standardized'].values)
cohen_d_0_3 = calculate_cohen_d(healthyALSPAC['brainPAD_standardized'].values, crazyALSPAC_3['brainPAD_standardized'].values)
cohen_d_1_2 = calculate_cohen_d(crazyALSPAC_1['brainPAD_standardized'].values, crazyALSPAC_2['brainPAD_standardized'].values)
cohen_d_1_3 = calculate_cohen_d(crazyALSPAC_1['brainPAD_standardized'].values, crazyALSPAC_3['brainPAD_standardized'].values)
cohen_d_2_3 = calculate_cohen_d(crazyALSPAC_2['brainPAD_standardized'].values, crazyALSPAC_3['brainPAD_standardized'].values)

print("Cohen's D 0_123: "+str(cohen_d_0_123))
print("Cohen's D 0_1: "+str(cohen_d_0_1))
print("Cohen's D 0_2: "+str(cohen_d_0_2))
print("Cohen's D 0_3: "+str(cohen_d_0_3))
print("Cohen's D 1_2: "+str(cohen_d_1_2))
print("Cohen's D 1_3: "+str(cohen_d_1_3))
print("Cohen's D 2_3: "+str(cohen_d_2_3))

# Step 1: Calculate Q1 and Q3 and then IQR for 'BrainPAD'
Q1 = healthyALSPAC['brainPAD_standardized'].quantile(0.25)
Q3 = healthyALSPAC['brainPAD_standardized'].quantile(0.75)
IQR = Q3 - Q1

# Step 2: Define the bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Step 3: Filter outliers
outliers_df_healthy = healthyALSPAC[(healthyALSPAC['brainPAD_standardized'] < lower_bound) | (healthyALSPAC['brainPAD_standardized'] > upper_bound)]

# Step 4: Print the IDs of the outliers
outlier_ids_healthy = outliers_df_healthy['ID']
print("Outlier IDs:", outlier_ids_healthy.tolist())

indexes_to_drop = healthyALSPAC[healthyALSPAC['ID'].isin(outlier_ids_healthy.tolist())].index
# healthyALSPAC = healthyALSPAC.drop(indexes_to_drop)

# Step 1: Calculate Q1 and Q3 and then IQR for 'BrainPAD'
Q1 = crazyALSPAC['brainPAD_standardized'].quantile(0.25)
Q3 = crazyALSPAC['brainPAD_standardized'].quantile(0.75)
IQR = Q3 - Q1

# Step 2: Define the bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Step 3: Filter outliers
outliers_df_crazy = crazyALSPAC[(crazyALSPAC['brainPAD_standardized'] < lower_bound) | (crazyALSPAC['brainPAD_standardized'] > upper_bound)]

# Step 4: Print the IDs of the outliers
outlier_ids_crazy = outliers_df_crazy['ID']
print("Outlier IDs:", outlier_ids_crazy.tolist())

indexes_to_drop = crazyALSPAC[crazyALSPAC['ID'].isin(outlier_ids_crazy.tolist())].index
# crazyALSPAC = crazyALSPAC.drop(indexes_to_drop)

Results_ALSPAC = pd.concat([healthyALSPAC, crazyALSPAC], axis=0)

# Assuming datos_table_1 is your DataFrame
summary_stats = Results_ALSPAC.groupby('Group').agg(
    num_cases=('ID', 'count'),
    mean_age=('Edad', 'mean'),
    std_age=('Edad', 'std'),
    age_range_min=('Edad', 'min'),
    age_range_max=('Edad', 'max'),
    num_males=('sexo(M=1;F=0)', lambda x: (x == 1).sum()),
    num_females=('sexo(M=1;F=0)', lambda x: (x == 0).sum())
)

summary_stats = summary_stats.sort_index()

pd.set_option('display.max_rows', 1000)  # Set to a high number of rows you want to see
pd.set_option('display.max_columns', 100)  # Set to a high number of columns

print(summary_stats)

print('\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
print('^^^^^^^^^^^^^^^^^ ARE THE GROUPS COMPARABLE IN AGE AND SEX (YES THEY ARE) ^^^^^^^^^^^^^^')
print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n')
# Check Normality for Age in both groups using the Shapiro-Wilk test
print("Checking Normality of Age Distribution:")
for group, name in [(healthyALSPAC, "Healthy"), (crazyALSPAC, "Crazy")]:
    # Apply Shapiro-Wilk
    stat, p_value = stats.shapiro(group['Edad'])
    print(f"{name} Group - Shapiro-Wilk Test: Stat={stat:.4f}, P-value={p_value:.4f}")
    if p_value < 0.05:
        print(f"The age distribution for the {name} group does not appear to be normal.\n")
    else:
        print(f"The age distribution for the {name} group appears to be normal.\n")

# Check Equality of Variances using Levene's test
stat, p_value = stats.levene(healthyALSPAC['Edad'], crazyALSPAC['Edad'])
print(f"Levene’s Test for Equality of Variances: Stat={stat}, P-value={p_value}")
if p_value < 0.05:
    print("Variances of the age distributions between the groups are significantly different.\n")
else:
    print("No significant difference in variances of the age distributions between the groups.\n")

# Welch's mann-whitney for Age assuming unequal variances
t_stat, p_value_age = stats.mannwhitneyu(healthyALSPAC['Edad'], crazyALSPAC['Edad'], alternative='two-sided')
print("Mann-Whiteney U test for Age:")
print(f"Mann-Whiteney U: {t_stat}, P-value: {p_value_age}")
if p_value_age < 0.05:
    print("Significant differences in age distributions between the groups.")
else:
    print("No significant differences in age distributions between the groups.")

# Check for Sex Comparability (Categorical Data) using a Chi-Square test
# First, create a contingency table for the 'Sex' column
# Count the occurrences of each 'Sex' category within each group
healthy_sex_counts = healthyALSPAC['sexo(M=1;F=0)'].value_counts()
crazy_sex_counts = crazyALSPAC['sexo(M=1;F=0)'].value_counts()

# Create a DataFrame to represent the contingency table
contingency_table = pd.DataFrame({'Healthy': healthy_sex_counts, 'Crazy': crazy_sex_counts})
chi2_stat, p_value_sex, dof, expected = stats.chi2_contingency(contingency_table)

print("\nChi-Square Test for Sex:")
print(f"Chi-square statistic: {chi2_stat}, P-value: {p_value_sex}")
if p_value_sex < 0.05:
    print("Significant differences in sex distribution between the groups.")
else:
    print("No significant differences in sex distribution between the groups.")


print('\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
print('^^^^^^^^^^^^^^^^^^^^^ ARE THE BrainPAD COMPARABLE  ^^^^^^^^^^^^^^^^^^^^^^')
print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n')

# Check Normality for BrainPAD in both groups using the Shapiro-Wilk test
p_val_norm = []
print("Checking Normality of BrainPAD Distribution:")
for group, name in [(healthyALSPAC, "Healthy"), (crazyALSPAC, "Crazy")]:
    # Apply Shapiro-Wilk test
    stat, p_value = stats.shapiro(group['brainPAD_standardized'])
    print(f"{name} Group - Shapiro-Wilk Test: Stat={stat:.4f}, P-value={p_value:.4f}")
    if p_value < 0.05:
        print(f"The BrainPAD distribution for the {name} group does not appear to be normal.\n")
    else:
        print(f"The BrainPAD distribution for the {name} group appears to be normal.\n")
    p_val_norm.append(p_value)

# Check Equality of Variances using Levene's test
stat, p_value_lev = stats.levene(healthyALSPAC['brainPAD_standardized'], crazyALSPAC['brainPAD_standardized'])
print(f"levene’s Test for Equality of Variances: Stat={stat}, P-value={p_value_lev}")
if p_value_lev < 0.05:
    print("Variances of the BrainPAD distributions between the groups are significantly different.\n")
else:
    print("No significant difference in variances of the BrainPAD distributions between the groups.\n")

# Welch's mann-whitney for Age assuming unequal variances
t_stat, p_value_age = stats.ttest_ind(healthyALSPAC['brainPAD_standardized'], crazyALSPAC['brainPAD_standardized'], equal_var=False)
print("t_stat test for brainPAD_standardized:")
print(f"t_stat: {t_stat}, P-value: {p_value_age}")
if p_value_age < 0.05:
    print("Significant differences in brainPAD distributions between the groups.")
else:
    print("No significant differences in brainPAD distributions between the groups.")


print('\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
print('^^^^^^^^^^^^^^^^^^^^^ MAE r R2 regressor predictions ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n')

mae = mean_absolute_error(healthyALSPAC['Edad'], healthyALSPAC['pred_Edad'])
r, _ = pearsonr(healthyALSPAC['Edad'], healthyALSPAC['pred_Edad'])
r2 = r2_score(healthyALSPAC['Edad'], healthyALSPAC['pred_Edad'])

print(f"######################## Healthy group ##############################")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Pearson Correlation Coefficient (r): {r}")
print(f"Coefficient of Determination (R2): {r2}")

mae = mean_absolute_error(crazyALSPAC['Edad'], crazyALSPAC['pred_Edad'])
r, _ = pearsonr(crazyALSPAC['Edad'], crazyALSPAC['pred_Edad'])
r2 = r2_score(crazyALSPAC['Edad'], crazyALSPAC['pred_Edad'])

print(f"######################## Crazy group ##############################")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Pearson Correlation Coefficient (r): {r}")
print(f"Coefficient of Determination (R2): {r2}")

print('\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ T-test ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n')

merged_df = pd.concat([healthyALSPAC, crazyALSPAC], axis=0)
df = merged_df[['ID', 'BrainPAD', 'brainPAD_standardized', 'Edad', 'sexo(M=1;F=0)', 'Group', 'pliks18TH', 'eTIV']]
df.columns = ['ID', 'BrainPAD', 'brainPAD_standardized', 'Edad', 'sexo', 'Group', 'pliks18TH', 'eTIV']

merged_df.to_csv('ALSPAC_II_merged_II.csv', index=False)

# ttestTest
U1, p_value = stats.ttest_ind(healthyALSPAC['brainPAD_standardized'], crazyALSPAC['brainPAD_standardized'], equal_var=True)
print('T-test p values:'+str(p_value))

print('--------------------------------')
print('====== PairWise ANALYSIS =======')
print('--------------------------------')

# 1. Perform ANCOVA
print("=== ANCOVA ===")
formula = 'brainPAD_standardized ~ Group + n_euler + Edad'
ancova_model = ols(formula, data=merged_df).fit()
anova_table = sm.stats.anova_lm(ancova_model, typ=2)
print(anova_table)

# Assumption Checks for ANCOVA
print("\n--- Assumption Checks for ANCOVA ---")
residuals = ancova_model.resid

# 1. Normality of Residuals (Shapiro-Wilk Test)
shapiro_trend = stats.shapiro(residuals)
print(f"Shapiro-Wilk Test: Statistics={shapiro_trend.statistic:.3f}, p-value={shapiro_trend.pvalue:.3f}")

# Q-Q Plot
sm.qqplot(residuals, line='45')
plt.title('Q-Q Plot of ANCOVA Residuals')
plt.show()

# 2. Homogeneity of Regression Slopes
formula_interaction = 'brainPAD_standardized ~ Group * n_euler + Group * Edad'
model_interaction = ols(formula_interaction, data=merged_df).fit()
anova_interaction = sm.stats.anova_lm(model_interaction, typ=2)
print("\nANCOVA with Interaction Terms:")
print(anova_interaction)

# 3. Homogeneity of Variances (Levene's Test)
group_data = [merged_df[merged_df['Group'] == grp]['brainPAD_standardized'] for grp in merged_df['Group'].unique()]
levene_test = stats.levene(*group_data)
print(f"\nLevene's Test for Homogeneity of Variances: Statistics={levene_test.statistic:.3f}, p-value={levene_test.pvalue:.3f}")

# 4. Homoscedasticity (Breusch-Pagan Test)
bp_test = het_breuschpagan(residuals, ancova_model.model.exog)
labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
print("\nBreusch-Pagan Test for Homoscedasticity:")
print(dict(zip(labels, bp_test)))

# Decision tree based on assumptions
if shapiro_trend.pvalue > 0.6:  # Normality assumption check
    if levene_test.pvalue > 0.05:  # Homogeneity of variances check
        if anova_interaction['PR(>F)'][1] > 0.05 and bp_test[3] > 0.05:  # Other ANCOVA assumptions
            print("\nAll ANCOVA assumptions are met. Proceeding with ANCOVA results.")
            # Perform ANCOVA
            model_ancova = ols(formula, data=merged_df).fit()
            print(sm.stats.anova_lm(model_ancova, typ=2))
        else:
            print("\nANCOVA assumptions violated. Proceeding to Welch-ANCOVA due to issues with interaction terms or heteroscedasticity.")
            # Perform Welch-ANCOVA using WLS
            group_var = 'Group'  # Replace with your actual grouping variable name
            dependent_var = 'brainPAD_standardized'

            # Calcular la varianza de brainPAD_standardized dentro de cada grupo
            group_variances = merged_df.groupby(group_var)[dependent_var].var().reset_index()

            # Renombrar la columna de varianza para claridad
            group_variances.rename(columns={dependent_var: 'Variance'}, inplace=True)

            # Merge group variances back to the original DataFrame
            merged_df = merged_df.merge(group_variances, on=group_var, how='left')

            # Add a small constant to variances to prevent division by zero
            merged_df['Variance'] = merged_df['Variance'].replace(0, 1e-6)

            # Calculate weights as the inverse of variances
            merged_df['weights'] = 1 / merged_df['Variance']

            model_wls = wls(formula, data=merged_df, weights=merged_df['weights']).fit()
            print(model_wls.summary())
    else:
        print("\nHomogeneity of variances violated. Proceeding to ANCOVA robust standard errors.")
        # Define the formula with an interaction term
        formula = 'brainPAD_standardized ~ Group + n_euler + Edad'

        # Fit the model with robust standard errors
        model = smf.ols(formula, data=merged_df).fit(cov_type='HC3')

        # Print the summary
        print(model.summary())

if shapiro_trend.pvalue < 1:
    print("\nNormality assumption violated. Considering non-parametric alternatives.")
    if levene_test.pvalue > 0.05:  # If variances are still equal
        print("\nProceeding to Quade test as a non-parametric alternative.")
        # Assuming merged_df contains your data with dependent variable 'brainPAD_standardized', groups in 'Group', and covariates 'age' and 'sex'
        data = merged_df.copy()
        # Step 1: Regress the dependent variable on covariates
        covariates_model = ols('brainPAD_standardized ~ n_euler + Edad', data=data).fit()

        # Print covariate results: coefficients, t-statistics, p-values
        print("\n=== Covariates Regression Results ===")
        print(covariates_model.summary())

        # Step 2: Extract residuals and rank them
        data['residuals'] = covariates_model.resid
        data['rank_residuals'] = data['residuals'].rank()

        # Step 3: Rank the original dependent variable
        data['rank_brainPAD'] = data['brainPAD_standardized'].rank()

        # Step 4: Calculate the "Quade scores" (difference between original rank and residual rank)
        data['quade_scores'] = data['rank_brainPAD'] - data['rank_residuals']

        # Step 5: Group by 'Group' and calculate the sum of Quade scores per group
        grouped_scores = data.groupby('Group')['quade_scores'].sum()

        # Step 6: Calculate the F-statistic for group differences on the Quade scores
        n_groups = len(data['Group'].unique())
        n_samples = len(data)

        # Variance of the Quade scores
        quade_score_var = np.var(data['quade_scores'], ddof=1)

        # Between-group sum of squares (BSS) and within-group sum of squares (WSS)
        bss = np.sum((grouped_scores ** 2) / data.groupby('Group').size()) - (
                    data['quade_scores'].sum() ** 2 / n_samples)
        wss = quade_score_var * (n_samples - n_groups)

        # F-statistic
        f_statistic = (bss / (n_groups - 1)) / (wss / (n_samples - n_groups))

        # Calculate p-value using F-distribution
        p_value = 1 - f.cdf(f_statistic, n_groups - 1, n_samples - n_groups)

        # Print the Quade test results
        print("\n=== Quade Test Results ===")
        print(f"Quade Test F-Statistic: {f_statistic:.3f}")
        print(f"Quade Test p-value: {p_value:.3f}")

        # Decision: If p-value < 0.05, reject the null hypothesis of no difference between groups
        if p_value < 0.05:
            print("Significant differences between groups.")
        else:
            print("No significant differences between groups.")
    else:
        print("\nNormality and homogeneity of variances both violated. Proceeding to Permutation Test.")
        # You can implement a permutation test here for more robust analysis
        print('===== Permutation test =====')

        # Merge the DataFrames on 'ID'
        n_permutations = 5000
        formula = 'brainPAD_standardized ~ Group + n_euler + Edad'

        # Coefficients of interest
        covariates = ['Group[T.PE_1-3]', 'n_euler', 'Edad']

        # Step 1: Fit GLM on the original data to get observed coefficients
        observed_coefs = fit_glm_and_get_all_coef(merged_df, formula)

        # Step 2: Initialize a dictionary to store permutation coefficients for each covariate
        perm_coefs = {covariate: [] for covariate in covariates}

        # Create a copy of the DataFrame for permutations
        df_perm = merged_df.copy()

        # Step 3: Perform permutations
        for _ in range(n_permutations):
            # Permute the dependent variable
            df_perm['brainPAD_standardized'] = np.random.permutation(merged_df['brainPAD_standardized'])

            # Get the permuted coefficients for all covariates
            permuted_coefs = fit_glm_and_get_all_coef(df_perm, formula)

            # Store the permuted coefficients for each covariate
            for covariate in covariates:
                perm_coefs[covariate].append(permuted_coefs[covariate])

        # Convert the permutation coefficients to numpy arrays
        perm_coefs = {covariate: np.array(coefs) for covariate, coefs in perm_coefs.items()}

        # Step 4: Calculate p-values for each covariate
        p_values = {}
        for covariate in covariates:
            observed_coef = observed_coefs[covariate]
            permuted_coef_distribution = perm_coefs[covariate]

            # Calculate p-value by comparing the observed coefficient to the permuted distribution
            p_value = np.mean(np.abs(permuted_coef_distribution) >= np.abs(observed_coef))
            p_values[covariate] = p_value

            # Print the observed coefficient and the p-value for each covariate
            print(f"Coeficiente observado ({covariate}): {observed_coef}")
            print(f"p-valor de la prueba de permutación ({covariate}): {p_value}")


print('-----------------------------')
print('====== TREND ANALYSIS =======')
print('-----------------------------')

print('--- check Age ---')

# Shapiro-Wilk Test for Normality within each group
print("Shapiro-Wilk Test Results for Normality (Age): ")
for group in sorted(merged_df['pliks18TH'].unique()):
    sample = merged_df[merged_df['pliks18TH'] == group]['Edad']
    # Apply Shapiro-Wilk test for each group
    stat, p_value = stats.shapiro(sample)
    print(f"Group {group}: Shapiro-Wilk Statistic={stat:.4f}, P-value={p_value:.4f}")
    if p_value < 0.05:
        print(f"The age distribution for Group {group} does not appear to be normal.\n")
    else:
        print(f"The age distribution for Group {group} appears to be normal.\n")

# Levene's Test for Homogeneity of Variances
w, p_value = stats.levene(*[merged_df[merged_df['pliks18TH'] == group]['Edad'] for group in sorted(merged_df['pliks18TH'].unique())])
print(f"\nLevene's Test (Age): W-statistic={w:.4f}, p-value={p_value:.4f}")

# Extract unique groups
groups = merged_df['pliks18TH'].unique()

# Prepare age data for each group
age_data = [merged_df[merged_df['pliks18TH'] == group]['Edad'].values for group in groups]

# Perform Kruskal-Wallis H Test
statistic, p_value = kruskal(*age_data)

print(f"\nKruskal-Wallis H Test Statistic (Age): {statistic:.4f}")
print(f"P-Value: {p_value:.4f}")


# Contingency table (replace with actual counts)
data = np.array([[26, 43], [14, 15], [9, 17], [3, 11]])

# Chi-square test
chi2, p, dof, expected = chi2_contingency(data)

print(f"\nChi-square statistic: {chi2}")
print(f"p-value: {p}")

print('\n--- check std brainPAD ---')

# Shapiro-Wilk Test for Normality within each group
print("Shapiro-Wilk Test Results for Normality (std brainPAD):")
for group in sorted(merged_df['pliks18TH'].unique()):
    sample = merged_df[merged_df['pliks18TH'] == group]['brainPAD_standardized']
    # Apply Shapiro-Wilk test for each group
    stat, p_value = stats.shapiro(sample)
    print(f"Group {group}: Shapiro-Wilk Statistic={stat:.4f}, P-value={p_value:.4f}")
    if p_value < 0.05:
        print(f"The brainPAD distribution for Group {group} does not appear to be normal.\n")
    else:
        print(f"The brainPAD distribution for Group {group} appears to be normal.\n")

# Levene's Test for Homogeneity of Variances
w, p_value = stats.levene(*[merged_df[merged_df['pliks18TH'] == group]['brainPAD_standardized'] for group in sorted(merged_df['pliks18TH'].unique())])
print(f"\nLevene's Test (std brainPAD): W-statistic={w:.4f}, p-value={p_value:.4f}")

print('===== Permutation test =====')

# Merge the DataFrames on 'ID'
n_permutations = 5000
formula = 'brainPAD_standardized ~ pliks18TH + n_euler + Edad'

# Coefficients of interest
covariates = ['pliks18TH', 'n_euler', 'Edad']

# Step 1: Fit GLM on the original data to get observed coefficients
observed_coefs = fit_glm_and_get_all_coef(merged_df, formula)

# Step 2: Initialize a dictionary to store permutation coefficients for each covariate
perm_coefs = {covariate: [] for covariate in covariates}

# Create a copy of the DataFrame for permutations
df_perm = merged_df.copy()

# Step 3: Perform permutations
for _ in range(n_permutations):
    # Permute the dependent variable
    df_perm['brainPAD_standardized'] = np.random.permutation(merged_df['brainPAD_standardized'])

    # Get the permuted coefficients for all covariates
    permuted_coefs = fit_glm_and_get_all_coef(df_perm, formula)

    # Store the permuted coefficients for each covariate
    for covariate in covariates:
        perm_coefs[covariate].append(permuted_coefs[covariate])

# Convert the permutation coefficients to numpy arrays
perm_coefs = {covariate: np.array(coefs) for covariate, coefs in perm_coefs.items()}

# Step 4: Calculate p-values for each covariate
p_values = {}
for covariate in covariates:
    observed_coef = observed_coefs[covariate]
    permuted_coef_distribution = perm_coefs[covariate]

    # Calculate p-value by comparing the observed coefficient to the permuted distribution
    p_value = np.mean(np.abs(permuted_coef_distribution) >= np.abs(observed_coef))
    p_values[covariate] = p_value

    # Print the observed coefficient and the p-value for each covariate
    print(f"Coeficiente observado ({covariate}): {observed_coef}")
    print(f"p-valor de la prueba de permutación ({covariate}): {p_value}")

rain_cloud_plot_III(merged_df)
rain_cloud_plot_V(merged_df)

