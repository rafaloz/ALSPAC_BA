import warnings
warnings.filterwarnings("ignore")

from utils_Train import *

from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr
from scipy.stats import kruskal

X_test = '/home/rafa/PycharmProjects/ALSPAC_BA/Data/data_test_ALSPAC_I.csv'
X_Cardiff = '/home/rafa/PycharmProjects/ALSPAC_BA/Data/ALSPAC_PE_I.csv'

# Obtain predictions from ALSPAC data
ALSPAC_results = pd.read_csv(X_Cardiff)
results_Cardiff = model_evaluation_modified(X_test, ALSPAC_results)
pliks_info = ALSPAC_results[['ID', 'pliks18TH']]

# Explicitly copy each slice to ensure they are independent DataFrames
controlALSPAC = results_Cardiff[results_Cardiff['pliks18TH'] == 0].copy()
PEs_ALSPAC_1 = results_Cardiff[results_Cardiff['pliks18TH'] == 1].copy()
PEs_ALSPAC_2 = results_Cardiff[results_Cardiff['pliks18TH'] == 2].copy()
PEs_ALSPAC_3 = results_Cardiff[results_Cardiff['pliks18TH'] == 3].copy()

scaler_men = StandardScaler()
scaler_woman = StandardScaler()

controlALSPAC_men = controlALSPAC[controlALSPAC['sexo']==1]
controlALSPAC_woman = controlALSPAC[controlALSPAC['sexo']==0]
controlALSPAC_men['brainPAD_standardized'] = scaler_men.fit_transform(controlALSPAC_men[['BrainPAD']])
controlALSPAC_woman['brainPAD_standardized'] = scaler_woman.fit_transform(controlALSPAC_woman[['BrainPAD']])
controlALSPAC = pd.concat([controlALSPAC_men, controlALSPAC_woman], axis=0)

PEs_ALSPAC_1_men = PEs_ALSPAC_1[PEs_ALSPAC_1['sexo']==1]
PEs_ALSPAC_1_woman = PEs_ALSPAC_1[PEs_ALSPAC_1['sexo']==0]
PEs_ALSPAC_1_men['brainPAD_standardized'] = scaler_men.transform(PEs_ALSPAC_1_men[['BrainPAD']])
PEs_ALSPAC_1_woman['brainPAD_standardized'] = scaler_woman.transform(PEs_ALSPAC_1_woman[['BrainPAD']])
PEs_ALSPAC_1 = pd.concat([PEs_ALSPAC_1_men, PEs_ALSPAC_1_woman], axis=0)

PEs_ALSPAC_2_men = PEs_ALSPAC_2[PEs_ALSPAC_2['sexo']==1]
PEs_ALSPAC_2_woman = PEs_ALSPAC_2[PEs_ALSPAC_2['sexo']==0]
PEs_ALSPAC_2_men['brainPAD_standardized'] = scaler_men.transform(PEs_ALSPAC_2_men[['BrainPAD']])
PEs_ALSPAC_2_woman['brainPAD_standardized'] = scaler_woman.transform(PEs_ALSPAC_2_woman[['BrainPAD']])
PEs_ALSPAC_2 = pd.concat([PEs_ALSPAC_2_men, PEs_ALSPAC_2_woman], axis=0)

PEs_ALSPAC_3_men = PEs_ALSPAC_3[PEs_ALSPAC_3['sexo']==1]
PEs_ALSPAC_3_woman = PEs_ALSPAC_3[PEs_ALSPAC_3['sexo']==0]
PEs_ALSPAC_3_men['brainPAD_standardized'] = scaler_men.transform(PEs_ALSPAC_3_men[['BrainPAD']])
PEs_ALSPAC_3_woman['brainPAD_standardized'] = scaler_woman.transform(PEs_ALSPAC_3_woman[['BrainPAD']])
PEs_ALSPAC_3 = pd.concat([PEs_ALSPAC_3_men, PEs_ALSPAC_3_woman], axis=0)

controlALSPAC.loc[:, 'Group'] = 'Controls'
PEs_ALSPAC_1.loc[:, 'Group'] = 'PEs_1'
PEs_ALSPAC_2.loc[:, 'Group'] = 'PEs_2'
PEs_ALSPAC_3.loc[:, 'Group'] = 'PEs_3'

PEs_ALSPAC = pd.concat([PEs_ALSPAC_1, PEs_ALSPAC_2, PEs_ALSPAC_3], axis=0)

Results_ALSPAC = pd.concat([controlALSPAC, PEs_ALSPAC], axis=0)

# Assuming datos_table_1 is your DataFrame
summary_stats = Results_ALSPAC.groupby('Group').agg(
    num_cases=('ID', 'count'),
    mean_age=('Edad', 'mean'),
    std_age=('Edad', 'std'),
    age_range_min=('Edad', 'min'),
    age_range_max=('Edad', 'max'),
    num_males=('sexo', lambda x: (x == 1).sum()),
    num_females=('sexo', lambda x: (x == 0).sum())
)

summary_stats = summary_stats.sort_index()
summary_stats = summary_stats.round(2)

pd.set_option('display.max_rows', 1000)  # Set to a high number of rows you want to see
pd.set_option('display.max_columns', 100)  # Set to a high number of columns

PEs_ALSPAC['Group'] = 'PE_1-3'

print(summary_stats)

# List of groups and their labels
groups = {
    "0": controlALSPAC['brainPAD_standardized'].values,
    "1": PEs_ALSPAC_1['brainPAD_standardized'].values,
    "2": PEs_ALSPAC_2['brainPAD_standardized'].values,
    "3": PEs_ALSPAC_3['brainPAD_standardized'].values,
    "123": PEs_ALSPAC['brainPAD_standardized'].values,
}

# Calculate Cohen's d for each pair of levels
results = {}
pairs = [("0", "123"), ("0", "1"), ("0", "2"), ("0", "3"), ("1", "2"), ("1", "3"), ("2", "3")]

for pair in pairs:
    key = f"{pair[0]}_{pair[1]}"
    results[key] = calculate_cohen_d(groups[pair[0]], groups[pair[1]])

# Print results with a beautiful title and separators
print("=" * 65)
print("       Brain Age Gap Cohen's D Calculation Among Groups         ")
print("=" * 65)

for key, value in results.items():
    print(f"Cohen's D {key}: {round(value, 2)}")

print("=" * 65)

print('\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
print('^^^^^^^^^^^^^^^^^^^ MAE r R2 regressor predictions ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n')

mae = mean_absolute_error(controlALSPAC['Edad'], controlALSPAC['pred_Edad'])
r, _ = pearsonr(controlALSPAC['Edad'], controlALSPAC['pred_Edad'])
r2 = r2_score(controlALSPAC['Edad'], controlALSPAC['pred_Edad'])

print(f"######################## Control group ###########################")
print(f"Mean Absolute Error (MAE): {mae:.3}")
print(f"Pearson Correlation Coefficient (r): {r:.2}")
print(f"Coefficient of Determination (R2): {r2:.2}")

mae = mean_absolute_error(PEs_ALSPAC['Edad'], PEs_ALSPAC['pred_Edad'])
r, _ = pearsonr(PEs_ALSPAC['Edad'], PEs_ALSPAC['pred_Edad'])
r2 = r2_score(PEs_ALSPAC['Edad'], PEs_ALSPAC['pred_Edad'])

print(f"######################## PEs group ##############################")
print(f"Mean Absolute Error (MAE): {mae:.3}")
print(f"Pearson Correlation Coefficient (r): {r:.2}")
print(f"Coefficient of Determination (R2): {r2:.2}")

print('\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
print('^^^^^^^^^^^^^^^^^ ASSESING GROUP COMPARABILITY FOR AGE AND SEX ^^^^^^^^^^^^^^^^^^^^^^^^')
print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n')
# Check Normality for Age in both groups using the kstest test
print("Checking Normality of Age Distribution:")
for group, name in [(controlALSPAC, "Controls"), (PEs_ALSPAC, "PEs")]:
    # Use Shapiro-Wilk test instead of KS test
    stat, p_value = stats.shapiro(group['Edad'])
    print(f"{name} Group - Shapiro-Wilk Test: Stat={round(value, 3)}, P-value={p_value:.2e}")
    if p_value < 0.05:
        print(f"The age distribution for the {name} group does not appear to be normal.\n")
    else:
        print(f"The age distribution for the {name} group appears to be normal.\n")

# Check Equality of Variances using Levene's test
stat, p_value = stats.levene(controlALSPAC['Edad'], PEs_ALSPAC['Edad'])
print(f"Levene’s Test for Equality of Variances: Stat={round(stat, 3)}, P-value={p_value:.2e}")
if p_value < 0.05:
    print("Variances of the age distributions between the groups are significantly different.\n")
else:
    print("No significant difference in variances of the age distributions between the groups.\n")

# Welch's mann-whitney for Age assuming unequal variances
t_stat, p_value_age = stats.mannwhitneyu(controlALSPAC['Edad'], PEs_ALSPAC['Edad'], alternative='two-sided')
print("Mann-Whiteney U test for Age:")
print(f"Mann-Whiteney U: {t_stat}, P-value: {p_value_age:.2e}")
if p_value_age < 0.05:
    print("Significant differences in age distributions between the groups.")
else:
    print("No significant differences in age distributions between the groups.")

# Check for Sex Comparability (Categorical Data) using a Chi-Square test
control_sex_counts = controlALSPAC['sexo(M=1;F=0)'].value_counts()
PEs_sex_counts = PEs_ALSPAC['sexo(M=1;F=0)'].value_counts()

# Create a DataFrame to represent the contingency table
contingency_table = pd.DataFrame({'Control': control_sex_counts, 'PEs': PEs_sex_counts})
chi2_stat, p_value_sex, dof, expected = stats.chi2_contingency(contingency_table)

print("\nChi-Square Test for Sex:")
print(f"Chi-square statistic: {round(chi2_stat, 2)}, P-value: {p_value_sex:.2e}")
if p_value_sex < 0.05:
    print("Significant differences in sex distribution between the groups.")
else:
    print("No significant differences in sex distribution between the groups.")


print('\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
print('^^^^^^^^^^^^^^^^^^^ IS THE BRAIN AGE GAP COMPARABLE  ^^^^^^^^^^^^^^^^^^^^^')
print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n')

# Check Normality for BrainPAD in both groups using the Shapiro-Wilk test
p_val_norm = []
print("Checking Normality of BrainPAD Distribution:")
for group, name in [(controlALSPAC, "Controls"), (PEs_ALSPAC, "PEs")]:
    # Use Shapiro-Wilk test instead of KS test
    stat, p_value = stats.shapiro(group['brainPAD_standardized'])
    print(f"{name} Group - Shapiro-Wilk Test: Stat={round(stat, 3)}, P-value={p_value:.2e}")
    if p_value < 0.05:
        print(f"The BrainPAD distribution for the {name} group does not appear to be normal.\n")
    else:
        print(f"The BrainPAD distribution for the {name} group appears to be normal.\n")
    p_val_norm.append(p_value)

# Check Equality of Variances using Levene's test
stat, p_value_lev = stats.levene(controlALSPAC['brainPAD_standardized'], PEs_ALSPAC['brainPAD_standardized'])
print(f"levene’s Test for Equality of Variances: Stat={round(stat, 3)}, P-value={p_value_lev:.2e}")
if p_value_lev < 0.05:
    print("Variances of the BrainPAD distributions between the groups are significantly different.\n")
else:
    print("No significant difference in variances of the BrainPAD distributions between the groups.\n")


merged_df = pd.concat([controlALSPAC, PEs_ALSPAC], axis=0)
df = merged_df[['ID', 'BrainPAD', 'brainPAD_standardized', 'Edad', 'sexo(M=1;F=0)', 'Group', 'pliks18TH', 'eTIV']]
df.columns = ['ID', 'BrainPAD', 'brainPAD_standardized', 'Edad', 'sexo', 'Group', 'pliks18TH', 'eTIV']

merged_df.to_csv('ALSPAC_I_merged.csv', index=False)

merged_df['n_euler'] = 2

print('--------------------------------')
print('======= PAIRWISE ANALYSIS ======')
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
    print(f"Coeficiente observado ({covariate}): {observed_coef:.3}")
    print(f"p-valor de la prueba de permutación ({covariate}): {p_value:.2e}")

print('-----------------------------')
print('====== TREND ANALYSIS =======')
print('-----------------------------')

print('--- check Age ---')

# Shapiro-Wilk Test for Normality within each group
print("Shapiro-Wilk Test Results for Normality (Age): ")
for group in sorted(merged_df['pliks18TH'].unique()):
    sample = merged_df[merged_df['pliks18TH'] == group]['Edad']
    # Apply the Shapiro-Wilk test for normality
    stat, p_value = stats.shapiro(sample)
    print(f"Group {group}: Shapiro-Wilk Statistic={stat:.3}, p-value={p_value:.2e}")

# Levene's Test for Homogeneity of Variances
w, p_value = stats.levene(*[merged_df[merged_df['pliks18TH'] == group]['Edad'] for group in sorted(merged_df['pliks18TH'].unique())])
print(f"\nLevene's Test (Age): W-statistic={w:.3}, p-value={p_value:.2e}")

# Extract unique groups
groups = merged_df['pliks18TH'].unique()

# Prepare age data for each group
age_data = [merged_df[merged_df['pliks18TH'] == group]['Edad'].values for group in groups]

# Perform Kruskal-Wallis H Test
statistic, p_value = kruskal(*age_data)

print(f"\nKruskal-Wallis H Test Statistic (Age): {statistic:.3}")
print(f"P-Value: {p_value:.2e}")

# Contingency table (replace with actual counts)
data = np.array([[75, 49], [27, 14], [29, 16], [27, 8]])

# Chi-square test
chi2, p, dof, expected = stats.chi2_contingency(data)

print(f"\nChi-square statistic: {chi2:.3}")
print(f"p-value: {p:.2e}\n")

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
    print(f"Observed coeficient ({covariate}): {observed_coef:.3}")
    print(f"p-value of the permutation test ({covariate:}): {p_value:.2e}")

rain_cloud_plot_III(merged_df)
rain_cloud_plot_V(merged_df)
