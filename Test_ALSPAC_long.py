import warnings
warnings.filterwarnings("ignore")

from utils import *

from scipy.stats import kruskal, chi2_contingency
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

ALSPAC_I_merged = pd.read_csv('/home/rafa/PycharmProjects/ALSPAC_BA/Data/ALSPAC_I_merged.csv')
ALSPAC_II_merged = pd.read_csv('/home/rafa/PycharmProjects/ALSPAC_BA/Data/ALSPAC_II_merged.csv')

new_values = [value[4:] + '_brain' for value in ALSPAC_II_merged['ID']]
ALSPAC_II_merged['ID'] = new_values

X_test_Cardiff_I = ALSPAC_I_merged[ALSPAC_I_merged['ID'].isin(ALSPAC_II_merged['ID'])]
X_test_Cardiff_II = ALSPAC_II_merged[ALSPAC_II_merged['ID'].isin(ALSPAC_I_merged['ID'])]

group_conditions = {
    lambda df: (df['pliks18TH'] >= 1) & (df['pliks30TH'] >= 1): 'Persistent',
    lambda df: (df['pliks18TH'] >= 1) & (df['pliks30TH'] == 0): 'Remitted',
    lambda df: (df['pliks18TH'] == 0) & (df['pliks30TH'] == 0): 'LControl',
    lambda df: (df['pliks18TH'] == 0) & (df['pliks30TH'] >= 1): 'Incident',
}

# Assign groups to X_test_Cardiff_II
X_test_Cardiff_II = assign_groups(X_test_Cardiff_II, group_conditions)

# Add pliks30TH from X_test_Cardiff_II to X_test_Cardiff_I
X_test_Cardiff_I['pliks30TH'] = X_test_Cardiff_II['pliks30TH'].values

# Assign groups to X_test_Cardiff_I
X_test_Cardiff_I = assign_groups(X_test_Cardiff_I, group_conditions)

# Rename columns for timepoints
Longi_I_cardiff = X_test_Cardiff_I[['ID', 'Age', 'sex', 'pliks18TH', 'pliks20TH', 'pliks30TH', 'brainPAD_standardized', 'group']]
Longi_II_cardiff = X_test_Cardiff_II[['ID', 'Age', 'sex', 'pliks18TH', 'pliks20TH', 'pliks30TH', 'brainPAD_standardized', 'group']]
Longi_I_cardiff = Longi_I_cardiff.rename(columns={'brainPAD_standardized': 'brainPAD_standardized_I', 'group': 'group_I', 'Age': 'Age_I', 'sex': 'sex_I'})
Longi_II_cardiff = Longi_II_cardiff.rename(columns={'brainPAD_standardized': 'brainPAD_standardized_II', 'group': 'group_II', 'Age': 'Age_II', 'sex': 'sex_II'})

# Merge longitudinal data
longAll = pd.merge(Longi_I_cardiff, Longi_II_cardiff, on='ID', how='inner')

# Calculate DeltaBrainPAD
longAll['DeltaBrainPAD'] = longAll['brainPAD_standardized_II'] - longAll['brainPAD_standardized_I']

LControl_ALSPAC = longAll[longAll['group_II']=='LControl']
Remitted_ALSPAC = longAll[longAll['group_II']=='Remitted']
Persistent_ALSPAC = longAll[longAll['group_II']=='Persistent']
Incident_ALSPAC = longAll[longAll['group_II']=='Incident']

# Aggregating statistics by groups
summary_stats = longAll.groupby('group_II').agg(
    num_cases=('ID', 'count'),
    mean_age=('Age_II', 'mean'),
    std_age=('Age_II', 'std'),
    age_range_min=('Age_II', 'min'),
    age_range_max=('Age_II', 'max'),
    num_males=('sex_II', lambda x: (x == 1).sum()),
    num_females=('sex_II', lambda x: (x == 0).sum()),
    mean_delta_brainPAD=('DeltaBrainPAD', 'mean')
)

LongPE = pd.concat([Persistent_ALSPAC, Incident_ALSPAC, Remitted_ALSPAC], axis=0, ignore_index=True)

total_stats_pe = pd.DataFrame({
    'num_cases': [LongPE.shape[0]],
    'mean_age': [LongPE['Age_II'].mean()],
    'std_age': [LongPE['Age_II'].std()],
    'age_range_min': [LongPE['Age_II'].min()],
    'age_range_max': [LongPE['Age_II'].max()],
    'num_males': [LongPE['sex_II'].sum()],
    'num_females': [LongPE.shape[0] - LongPE['sex_II'].sum()],
    'mean_delta_brainPAD': [LongPE['DeltaBrainPAD'].mean()],
}, index=['PE'])

# Adding the total PE stats to summary
summary_stats = pd.concat([summary_stats, total_stats_pe])

# Define the desired order for the rows
desired_order = ['Incident', 'Persistent', 'Remitted', 'LControl', 'PE']

# Reorder the rows
summary_stats = summary_stats.reindex(desired_order)

print("\n====== Summary Stats ======")
# Rounding for cleaner output
summary_stats = summary_stats.round(2)

# Print the summary statistics
print(summary_stats.to_string(), '\n')

# Define group data
groups = {
    "0": LControl_ALSPAC['DeltaBrainPAD'].values,
    "1": Remitted_ALSPAC['DeltaBrainPAD'].values,
    "2": Persistent_ALSPAC['DeltaBrainPAD'].values,
    "3": Incident_ALSPAC['DeltaBrainPAD'].values,
    "123": LongPE['DeltaBrainPAD'].values
}

# Define pairs of groups for Cohen's D calculation
pairs = [
    ("0", "123"),
    ("0", "1"),
    ("0", "2"),
    ("0", "3"),
    ("1", "2"),
    ("1", "3"),
    ("2", "3")
]

# Calculate Cohen's D for each pair
results = {}
for pair in pairs:
    key = f"{pair[0]}_{pair[1]}"
    results[key] = calculate_cohen_d(groups[pair[0]], groups[pair[1]])

# Print results
print("=" * 60)
print("       Cohen's D Calculations for DeltaBrainPAD       ")
print("=" * 60)
for key, value in results.items():
    print(f"Cohen's D {key}: {value:.2f}")
print("=" * 60)

# Create a new column based on the old column values
mapping = {'LControl': 1, 'Remitted': 0, 'Persistent': 2, 'Incident': 3}
longAll['group_ordinal'] = longAll['group_II'].map(mapping).astype(int)
longAll['n_euler'] = 2

# Reshape data into long format
timepoint_I = longAll[['ID', 'Age_I', 'sex_I', 'brainPAD_standardized_I', 'group_I', 'n_euler']].copy()
timepoint_I['Time'] = 'I'
timepoint_I.rename(columns={
    'Age_I': 'Age',
    'sex_I': 'sex',
    'brainPAD_standardized_I': 'brainPAD_standardized',
    'group_I': 'group'
}, inplace=True)

timepoint_II = longAll[['ID', 'Age_II', 'sex_II', 'brainPAD_standardized_II', 'group_II', 'n_euler']].copy()
timepoint_II['Time'] = 'II'
timepoint_II.rename(columns={
    'Age_II': 'Age',
    'sex_II': 'sex',
    'brainPAD_standardized_II': 'brainPAD_standardized',
    'group_II': 'group'
}, inplace=True)

long_data = pd.concat([timepoint_I, timepoint_II], ignore_index=True)

hue_order = ['Remitted', 'LControl', 'Persistent', 'Incident']
labels = ['Longitudinal Controls', 'Incident', 'Remittent', 'Persistent']

# Plot observed means by group at each timepoint
sns.lineplot(data=long_data, x='Time', y='brainPAD_standardized', hue='group', estimator='mean', marker='o')

plt.title('Brain Age Predictions Timepoints')
plt.xlabel('Time')
plt.ylabel('Brain Age (Predicted)')

# Get current legend and change labels
handles, _ = plt.gca().get_legend_handles_labels()
plt.legend(handles=handles, labels=labels, title='Group')

plt.show()

longAll['Age_dif'] = longAll['Age_II'] - longAll['Age_I']

print('--------------------------------')
print('====== PAIRWISE ANALYSIS =======')
print('--------------------------------')

longAll.loc[:, 'group_cat'] = 'NoDefinido'
longAll.loc[(longAll['group_ordinal'] != 1), 'group_cat'] = 1
longAll.loc[(longAll['group_ordinal'] == 1), 'group_cat'] = 0

print('--- check Age & sex ---')

# Shapiro-Wilk Test for Normality within each group
print("Shapiro-Wilk Test Results for Normality (Age): ")
for group in sorted(longAll['group_cat'].unique()):
    sample = longAll[longAll['group_cat'] == group]['Age_II']
    stat, p_value = stats.shapiro(sample)
    print(f"Group {group}: W-statistic={stat:.3}, p-value={p_value:.2e}")

# Levene's Test for Homogeneity of Variances
w, p_value = stats.levene(*[longAll[longAll['group_cat'] == group]['Age_II'] for group in sorted(longAll['group_II'].unique())])
print(f"\nLevene's Test (Age): W-statistic={w:.3}, p-value={p_value:.2e}")

# Extract unique groups
groups = longAll['group_cat'].unique()

# Perform Welch's t-test (both groups normal, variances not equal)
t_stat, t_p_value = stats.ttest_ind(longAll[longAll['group_cat']==0]['Age_II'], longAll[longAll['group_cat']==1]['Age_II'], equal_var=False)
print(f"Welch's T-test: Statistic={t_stat:.3}, P-value={t_p_value:.2e}")

# Contingency table (replace with actual counts)
data = np.array([[37, 19], [37, 20]])

# Chi-square test
chi2, p, dof, expected = chi2_contingency(data)

print(f"\nChi-square statistic (Sex): {chi2:.3}")
print(f"p-value: {p:.2e}")

print('===== Permutation test =====')

# Merge the DataFrames on 'ID'
n_permutations = 5000
formula = 'DeltaBrainPAD ~ group_cat + n_euler + Age_dif'

# Coefficients of interest
covariates = ['group_cat[T.1]', 'n_euler', 'Age_dif']

# Step 1: Fit GLM on the original data to get observed coefficients
observed_coefs = fit_glm_and_get_all_coef(longAll, formula)

# Step 2: Initialize a dictionary to store permutation coefficients for each covariate
perm_coefs = {covariate: [] for covariate in covariates}

# Create a copy of the DataFrame for permutations
df_perm = longAll.copy()

# Step 3: Perform permutations
for _ in range(n_permutations):
    # Permute the dependent variable
    df_perm['DeltaBrainPAD'] = np.random.permutation(longAll['DeltaBrainPAD'])

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

print('-----------------------------')
print('====== TREND ANALYSIS =======')
print('-----------------------------')

print('--- check Age & sex ---')

# Shapiro-Wilk Test for Normality within each group
print("Shapiro-Wilk Test Results for Normality (Age): ")
for group in sorted(longAll['group_II'].unique()):
    sample = longAll[longAll['group_II'] == group]['Age_II']
    # Perform the Shapiro-Wilk test on the sample
    stat, p_value = stats.shapiro(sample)
    print(f"Group {group}: W-statistic={stat:.3}, p-value={p_value:.2e}")

# Levene's Test for Homogeneity of Variances
w, p_value = stats.levene(*[longAll[longAll['group_II'] == group]['Age_II'] for group in sorted(longAll['group_II'].unique())])
print(f"\nLevene's Test (Age): W-statistic={w:.3}, p-value={p_value:.2e}")

# Extract unique groups
groups = longAll['group_II'].unique()

# Prepare age data for each group
age_data = [longAll[longAll['group_II'] == group]['Age_II'].values for group in groups]

# Perform Kruskal-Wallis H Test
statistic, p_value = kruskal(*age_data)

print(f"\nKruskal-Wallis H Test Statistic (Age): {statistic:.3}")
print(f"P-Value: {p_value:.2e}")

# Contingency table (replace with actual counts)
data = np.array([[37, 19], [22, 13], [3, 2], [12, 5]])

# Chi-square test
chi2, p, dof, expected = chi2_contingency(data)

print(f"\nChi-square statistic (Sex): {chi2:.3}")
print(f"p-value: {p:.2e}")

print('===== Permutation test =====')

# Merge the DataFrames on 'ID'
n_permutations = 5000
formula = 'DeltaBrainPAD ~ group_ordinal + n_euler + Age_dif'

# Coefficients of interest
covariates = ['group_ordinal', 'n_euler', 'Age_dif']

# Step 1: Fit GLM on the original data to get observed coefficients
observed_coefs = fit_glm_and_get_all_coef(longAll, formula)

# Step 2: Initialize a dictionary to store permutation coefficients for each covariate
perm_coefs = {covariate: [] for covariate in covariates}

# Create a copy of the DataFrame for permutations
df_perm = longAll.copy()

# Step 3: Perform permutations
for _ in range(n_permutations):
    # Permute the dependent variable
    df_perm['DeltaBrainPAD'] = np.random.permutation(longAll['DeltaBrainPAD'])

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

rain_cloud_plot_III(longAll)
rain_cloud_plot_IV(longAll)
