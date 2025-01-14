import warnings
warnings.filterwarnings("ignore")

from utils_Train import *

from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr
from scipy.stats import kruskal

X_test = '/home/rafa/PycharmProjects/ALSPAC_BA/Data/data_test_ALSPAC_II.csv'
X_Cardiff = '/home/rafa/PycharmProjects/ALSPAC_BA/Data/ALSPAC_PE_II.csv'

# Obtain predictions from ALSPAC data
ALSPAC_results = pd.read_csv(X_Cardiff)
results_Cardiff = model_evaluation_modified(X_test, ALSPAC_results)
pliks_info = ALSPAC_results[['ID', 'pliks18TH']]

# Explicitly copy each slice to ensure they are independent DataFrames
controlALSPAC = results_Cardiff[results_Cardiff['pliks18TH'] == 0].copy()
PEs_ALSPAC_1 = results_Cardiff[results_Cardiff['pliks18TH'] == 1].copy()
PEs_ALSPAC_2 = results_Cardiff[results_Cardiff['pliks18TH'] == 2].copy()
PEs_ALSPAC_3 = results_Cardiff[results_Cardiff['pliks18TH'] == 3].copy()

# Initialize scalers
scaler_men = StandardScaler()
scaler_women = StandardScaler()

# Train the scalers on the control group and apply them to other groups
controlALSPAC = standardize_with_control(controlALSPAC, controlALSPAC, 'sex', 'BrainPAD', scaler_men, scaler_women)
PEs_ALSPAC_1 = standardize_with_control(controlALSPAC, PEs_ALSPAC_1, 'sex', 'BrainPAD', scaler_men, scaler_women)
PEs_ALSPAC_2 = standardize_with_control(controlALSPAC, PEs_ALSPAC_2, 'sex', 'BrainPAD', scaler_men, scaler_women)
PEs_ALSPAC_3 = standardize_with_control(controlALSPAC, PEs_ALSPAC_3, 'sex', 'BrainPAD', scaler_men, scaler_women)

controlALSPAC.loc[:, 'Group'] = 'Controls'
PEs_ALSPAC_1.loc[:, 'Group'] = 'PEs_1'
PEs_ALSPAC_2.loc[:, 'Group'] = 'PEs_2'
PEs_ALSPAC_3.loc[:, 'Group'] = 'PEs_3'

PEs_ALSPAC = pd.concat([PEs_ALSPAC_1, PEs_ALSPAC_2, PEs_ALSPAC_3], axis=0)
Results_ALSPAC = pd.concat([controlALSPAC, PEs_ALSPAC], axis=0)

# Assuming datos_table_1 is your DataFrame
summary_stats = Results_ALSPAC.groupby('Group').agg(
    num_cases=('ID', 'count'),
    mean_age=('Age', 'mean'),
    std_age=('Age', 'std'),
    age_range_min=('Age', 'min'),
    age_range_max=('Age', 'max'),
    num_males=('sex', lambda x: (x == 1).sum()),
    num_females=('sex', lambda x: (x == 0).sum())
)

summary_stats = summary_stats.sort_index()
summary_stats = summary_stats.round(2)

print(summary_stats.to_string(), '\n')

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
    print(f"Cohen's D {key}: {value:.3}")

print("=" * 65)

print('^' *74)
print('^^^^^^^^^^^^^^^^^^^ MAE r R2 regressor predictions ^^^^^^^^^^^^^^^^^^^^^^^^')
print('^' *74)

mae = mean_absolute_error(controlALSPAC['Age'], controlALSPAC['pred_Age'])
r, _ = pearsonr(controlALSPAC['Age'], controlALSPAC['pred_Age'])
r2 = r2_score(controlALSPAC['Age'], controlALSPAC['pred_Age'])

print(f"######################## Control group ##############################")
print(f"Mean Absolute Error (MAE): {mae:.3}")
print(f"Pearson Correlation Coefficient (r): {r:.2}")
print(f"Coefficient of Determination (R2): {r2:.2}")

mae = mean_absolute_error(PEs_ALSPAC['Age'], PEs_ALSPAC['pred_Age'])
r, _ = pearsonr(PEs_ALSPAC['Age'], PEs_ALSPAC['pred_Age'])
r2 = r2_score(PEs_ALSPAC['Age'], PEs_ALSPAC['pred_Age'])

print(f"######################## PEs group ##############################")
print(f"Mean Absolute Error (MAE): {mae:.3}")
print(f"Pearson Correlation Coefficient (r): {r:.2}")
print(f"Coefficient of Determination (R2): {r2:.2}")

print('\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
print('^^^^^^^^^^^^^^^^^ ASSESING GROUP COMPARABILITY FOR AGE AND SEX ^^^^^^^^^^^^^^^^^^^^^^^^')
print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n')

# Check Normality for Age in both groups using the Shapiro-Wilk test
print("Checking Normality of Age Distribution:")
for group, name in [(controlALSPAC, "Controls"), (PEs_ALSPAC, "PEs")]:
    # Apply Shapiro-Wilk
    stat, p_value = stats.shapiro(group['Age'])
    print(f"{name} Group - Shapiro-Wilk Test: Stat={stat:.3}, P-value={p_value:.2e}")
    if p_value < 0.05:
        print(f"The age distribution for the {name} group does not appear to be normal.\n")
    else:
        print(f"The age distribution for the {name} group appears to be normal.\n")

# Check Equality of Variances using Levene's test
stat, p_value = stats.levene(controlALSPAC['Age'], PEs_ALSPAC['Age'])
print(f"Levene’s Test for Equality of Variances: Stat={stat:.3}, P-value={p_value:.2e}")
if p_value < 0.05:
    print("Variances of the age distributions between the groups are significantly different.\n")
else:
    print("No significant difference in variances of the age distributions between the groups.\n")

# Welch's mann-whitney for Age assuming unequal variances
t_stat, p_value_age = stats.mannwhitneyu(controlALSPAC['Age'], PEs_ALSPAC['Age'], alternative='two-sided')
print("Mann-Whiteney U test for Age:")
print(f"Mann-Whiteney U: {t_stat:.3}, P-value: {p_value_age:.2e}")
if p_value_age < 0.05:
    print("Significant differences in age distributions between the groups.")
else:
    print("No significant differences in age distributions between the groups.")

# Check for Sex Comparability (Categorical Data) using a Chi-Square test
control_sex_counts = controlALSPAC['sex'].value_counts()
PEs_sex_counts = PEs_ALSPAC['sex'].value_counts()

# Create a DataFrame to represent the contingency table
contingency_table = pd.DataFrame({'Control': control_sex_counts, 'PEs': PEs_sex_counts})
chi2_stat, p_value_sex, dof, expected = stats.chi2_contingency(contingency_table)

print("\nChi-Square Test for Sex:")
print(f"Chi-square statistic: {chi2_stat:.3}, P-value: {p_value_sex:.2e}")
if p_value_sex < 0.05:
    print("Significant differences in sex distribution between the groups.")
else:
    print("No significant differences in sex distribution between the groups.")


print('\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
print('^^^^^^^^^^^^^^^^^^^^^ IS THE BRAIN AGE GAP COMPARABLE  ^^^^^^^^^^^^^^^^^^^^^^')
print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n')

# Check Normality for BrainPAD in both groups using the Shapiro-Wilk test
p_val_norm = []
print("Checking Normality of BrainPAD Distribution:")
for group, name in [(controlALSPAC, "Controls"), (PEs_ALSPAC, "PEs")]:
    # Apply Shapiro-Wilk test
    stat, p_value = stats.shapiro(group['brainPAD_standardized'])
    print(f"{name} Group - Shapiro-Wilk Test: Stat={stat:.3}, P-value={p_value:.2e}")
    if p_value < 0.05:
        print(f"The BrainPAD distribution for the {name} group does not appear to be normal.\n")
    else:
        print(f"The BrainPAD distribution for the {name} group appears to be normal.\n")
    p_val_norm.append(p_value)

# Check Equality of Variances using Levene's test
stat, p_value_lev = stats.levene(controlALSPAC['brainPAD_standardized'], PEs_ALSPAC['brainPAD_standardized'])
print(f"levene’s Test for Equality of Variances: Stat={stat:.3}, P-value={p_value_lev:.2e}")
if p_value_lev < 0.05:
    print("Variances of the BrainPAD distributions between the groups are significantly different.\n")
else:
    print("No significant difference in variances of the BrainPAD distributions between the groups.\n")

PEs_ALSPAC['Group'] = 'PE_1-3'
merged_df = pd.concat([controlALSPAC, PEs_ALSPAC], axis=0)
df = merged_df[['ID', 'BrainPAD', 'brainPAD_standardized', 'Age', 'sex', 'Group', 'pliks18TH', 'eTIV']]

merged_df['n_euler'] = 2

print('--------------------------------')
print('====== PAIRWISE ANALYSIS =======')
print('--------------------------------')

print('===== Permutation test =====')

# Merge the DataFrames on 'ID'
n_permutations = 5000
formula = 'brainPAD_standardized ~ Group + n_euler + Age'

# Coefficients of interest
covariates = ['Group[T.PE_1-3]', 'n_euler', 'Age']

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
    sample = merged_df[merged_df['pliks18TH'] == group]['Age']
    # Apply Shapiro-Wilk test for each group
    stat, p_value = stats.shapiro(sample)
    print(f"Group {group}: Shapiro-Wilk Statistic={stat:.3}, P-value={p_value:.2e}")

# Levene's Test for Homogeneity of Variances
w, p_value = stats.levene(*[merged_df[merged_df['pliks18TH'] == group]['Age'] for group in sorted(merged_df['pliks18TH'].unique())])
print(f"\nLevene's Test (Age): W-statistic={w:.3}, p-value={p_value:.2e}")

# Extract unique groups
groups = merged_df['pliks18TH'].unique()

# Prepare age data for each group
age_data = [merged_df[merged_df['pliks18TH'] == group]['Age'].values for group in groups]

# Perform Kruskal-Wallis H Test
statistic, p_value = kruskal(*age_data)

print(f"\nKruskal-Wallis H Test Statistic (Age): {statistic:.3}")
print(f"P-Value: {p_value:.2e}")

# Contingency table (replace with actual counts)
data = np.array([[26, 43], [14, 15], [9, 17], [3, 11]])

# Chi-square test
chi2, p, dof, expected = stats.chi2_contingency(data)

print(f"\nChi-square statistic: {chi2:.3}")
print(f"p-value: {p:.2e}")

print('===== Permutation test =====')

# Merge the DataFrames on 'ID'
n_permutations = 5000
formula = 'brainPAD_standardized ~ pliks18TH + n_euler + Age'

# Coefficients of interest
covariates = ['pliks18TH', 'n_euler', 'Age']

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
    print(f"p-value of the permutation test ({covariate}): {p_value:.2e}")

rain_cloud_plot_III(merged_df)
rain_cloud_plot_V(merged_df)

