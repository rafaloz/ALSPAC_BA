import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

from utils_Train import *
import pickle
import ast

from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from scipy.stats import kruskal
from scipy.stats import f

import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.formula.api import ols
from statsmodels.formula.api import wls

X_train = pd.read_csv('/home/rafa/PycharmProjects/Cardiff_ALSPAC/modelos/modelo_morfo_100_Cardiff_balanced_WAND/Datos_train_sample.csv')
x_test = pd.read_csv('/home/rafa/PycharmProjects/Cardiff_ALSPAC/modelos/modelo_morfo_100_Cardiff_balanced_WAND/Datos_Test_sample.csv')
X_test_OutSample = pd.read_csv('/home/rafa/PycharmProjects/Cardiff_ALSPAC/modelos/modelo_morfo_100_Cardiff_balanced_WAND/Datos_AgeRisk_To_Test.csv')
X_test_Cardiff = pd.read_csv('/home/rafa/PycharmProjects/JoinData_FastSurfer_V2/Scripts_Join_data/Datos_per_DB/CARDIFF_PE_II_FastSurfer_V2_data.csv')
# casos_seleccioandos = pd.read_csv('/home/rafa/PycharmProjects/Cardiff_ALSPAC/datos/subjects_ALSPAC_II_adq/Datos_ALSPAC_II_Seleccionados.csv')
# X_test_Cardiff = X_test_Cardiff[X_test_Cardiff['ID'].isin(casos_seleccioandos['ID'])]

# List of IDs to remove
ids_to_remove = ['15382A_brain', '19674A_brain', '20606A_brain', '13123A_brain', '17524A_brain']

# Filter out the rows with these IDs
X_test_Cardiff = X_test_Cardiff[~X_test_Cardiff['ID'].isin(ids_to_remove)]

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

merged_df = pd.concat([healthyALSPAC, crazyALSPAC], axis=0)
df = merged_df[['ID', 'BrainPAD', 'brainPAD_standardized', 'Edad', 'sexo(M=1;F=0)', 'Group', 'pliks18TH', 'eTIV']]
df.columns = ['ID', 'BrainPAD', 'brainPAD_standardized', 'Edad', 'sexo', 'Group', 'pliks18TH', 'eTIV']

merged_df.to_csv('ALSPAC_II_merged_II.csv', index=False)

merged_df['n_euler'] = 2

print('--------------------------------')
print('====== PairWise ANALYSIS =======')
print('--------------------------------')

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
chi2, p, dof, expected = stats.chi2_contingency(data)

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

