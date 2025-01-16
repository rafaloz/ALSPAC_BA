import warnings
warnings.filterwarnings("ignore")

from utils import *

from scipy.stats import shapiro, chi2_contingency, kruskal

X_test_ALSPAC_I = pd.read_csv('/home/rafa/PycharmProjects/ALSPAC_BA/Data/ALSPAC_I_merged.csv')
X_test_ALSPAC_II = pd.read_csv('/home/rafa/PycharmProjects/ALSPAC_BA/Data/ALSPAC_II_merged.csv')

# For X_test_ALSPAC_II
X_test_ALSPAC_II.loc[X_test_ALSPAC_II['pliks18TH'] == 0, 'group'] = 'Control'
X_test_ALSPAC_II.loc[X_test_ALSPAC_II['pliks18TH'] == 1, 'group'] = 'Suspected'
X_test_ALSPAC_II.loc[X_test_ALSPAC_II['pliks18TH'] == 2, 'group'] = 'Definite'
X_test_ALSPAC_II.loc[X_test_ALSPAC_II['pliks18TH'] == 3, 'group'] = 'Clinical Disorder'

# Copy pliks30TH from II into I
X_test_ALSPAC_I['pliks30TH'] = X_test_ALSPAC_II['pliks30TH'].values

# For X_test_ALSPAC_I
X_test_ALSPAC_I.loc[X_test_ALSPAC_I['pliks18TH'] == 0, 'group'] = 'Control'
X_test_ALSPAC_I.loc[X_test_ALSPAC_I['pliks18TH'] == 1, 'group'] = 'Suspected'
X_test_ALSPAC_I.loc[X_test_ALSPAC_I['pliks18TH'] == 2, 'group'] = 'Definite'
X_test_ALSPAC_I.loc[X_test_ALSPAC_I['pliks18TH'] == 3, 'group'] = 'Clinical Disorder'

Long_I_cardiff = X_test_ALSPAC_I[['ID', 'Age', 'sex', 'pliks18TH', 'pliks20TH', 'pliks30TH', 'brainPAD_standardized', 'group']]
Long_I_cardiff['Time'] = 'I'
Long_I_cardiff = Long_I_cardiff.rename(columns={'brainPAD_standardized': 'brainPAD_standardized_I', 'group': 'group_I','Age': 'Age_I','sex': 'sex_I'})

Long_II_cardiff = X_test_ALSPAC_II[['ID', 'Age', 'sex', 'pliks18TH', 'pliks20TH', 'pliks30TH', 'brainPAD_standardized', 'group', 'n_euler']]
Long_I_cardiff['Time'] = 'II'
Long_II_cardiff = Long_II_cardiff.rename(columns={'brainPAD_standardized': 'brainPAD_standardized_II', 'group': 'group_II', 'Age': 'Age_II', 'sex': 'sex_II'})

# Merge both timepoints
longAll = pd.merge(Long_I_cardiff, Long_II_cardiff, on='ID', how='inner')

# Compute DeltaBrainPAD
longAll['DeltaBrainPAD'] = longAll['brainPAD_standardized_II'] - longAll['brainPAD_standardized_I']
longAll['Age_dif'] = longAll['Age_II'] - longAll['Age_I']

Control_Cardiff = longAll[longAll['group_II'] == 'Control']
Suspected_Cardiff = longAll[longAll['group_II'] == 'Suspected']
Definite_Cardiff = longAll[longAll['group_II'] == 'Definite']
CD_Cardiff = longAll[longAll['group_II'] == 'Clinical Disorder']

# Example: combine subgroups "1", "2", "3" if needed
LongPE = pd.concat([Suspected_Cardiff, Definite_Cardiff, CD_Cardiff], ignore_index=True)

# Example aggregated stats
summary_stats = longAll.groupby('group_II').agg(
    N=('ID', 'count'),
    mean_age=('Age_II', 'mean'),
    std_age=('Age_II', 'std'),
    min_age=('Age_II', 'min'),
    max_age=('Age_II', 'max'),
    num_males=('sex_II', 'sum'),  # assuming sex_II==1 => male
    num_females=('sex_II', lambda x: (x==0).sum()),
    mean_delta=('DeltaBrainPAD', 'mean')
).round(2)

print("\n====== Summary Stats ======")
# Rounding for cleaner output
summary_stats = summary_stats.round(2)

# Print the summary statistics
print(summary_stats.to_string(), '\n')

groups = {
    "0": Control_Cardiff['DeltaBrainPAD'].values,
    "1": Suspected_Cardiff['DeltaBrainPAD'].values,
    "2": Definite_Cardiff['DeltaBrainPAD'].values,
    "3": CD_Cardiff['DeltaBrainPAD'].values,
    "123": LongPE['DeltaBrainPAD'].values
}

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
mapping = { 'Control': 0, 'Suspected': 1, 'Definite': 2, 'Clinical Disorder': 3}
longAll['group_ordinal'] = longAll['group_II'].map(mapping).astype(int)

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

hue_order = ['Control', 'Suspected', 'Definite', 'Clinical Disorder']
labels = ['Controls', 'Suspected', 'Definite', 'Clinical Disorder']

# Plot observed means by group at each timepoint
sns.lineplot(data=long_data, x='Time', y='brainPAD_standardized', hue='group', estimator='mean', marker='o')

plt.title('Brain Age Predictions Timepoints')
plt.xlabel('Time')
plt.ylabel('Brain Age (Predicted)')

# Get current legend and change labels
handles, _ = plt.gca().get_legend_handles_labels()
plt.legend(handles=handles, labels=labels, title='Group')

plt.show()

print('--------------------------------')
print('====== PAIRWISE ANALYSIS =======')
print('--------------------------------')

print('--- check Age & sex ---')

# Example: Shapiro-Wilk test for Age_II in "Control" vs "LongPE"
for subset, label in [(Control_Cardiff, 'Control'), (LongPE, 'LongPE')]:
    stat, pval = shapiro(subset['Age_II'])
    print(f"{label}: W={stat:.3}, p={pval:.2e}")

# Levene test (Age_II) for the same two groups
lev_stat, lev_p = stats.levene(Control_Cardiff['Age_II'], LongPE['Age_II'])
print(f"\nLevene's Test (Age_II) Control vs LongPE: stat={lev_stat:.3}, p={lev_p:.2e}")

u_stat, u_p_val = stats.mannwhitneyu(Control_Cardiff['Age_II'], LongPE['Age_II'], alternative='two-sided')

print(f"Mann-Whitney U: stat={u_stat:.3}, p={u_p_val:.2e}\n")

# Example: Chi-square on sex
data_2x2 = np.array([
    [Control_Cardiff['sex_II'].sum(), LongPE['sex_II'].sum()],
    [(Control_Cardiff['sex_II']==0).sum(), (LongPE['sex_II']==0).sum()]
])
chi2_stat, chi2_p, dof, exp = chi2_contingency(data_2x2)
print("Chi-square (sex) Control vs LongPE:")
print(f" chi2={chi2_stat:.3}, p={chi2_p:.2e}, dof={dof}")

# Convert to "long" format
timepoint_I = longAll[[
    'ID', 'Age_I', 'sex_I', 'brainPAD_standardized_I', 'group_I'
]].copy()
timepoint_I['Time'] = 'I'
timepoint_I.rename(columns={
    'Age_I': 'Age',
    'sex_I': 'sex',
    'brainPAD_standardized_I': 'brainPAD_standardized',
    'group_I': 'group'
}, inplace=True)

timepoint_II = longAll[[
    'ID', 'Age_II', 'sex_II', 'brainPAD_standardized_II', 'group_II'
]].copy()
timepoint_II['Time'] = 'II'
timepoint_II.rename(columns={
    'Age_II': 'Age',
    'sex_II': 'sex',
    'brainPAD_standardized_II': 'brainPAD_standardized',
    'group_II': 'group'
}, inplace=True)

long_data = pd.concat([timepoint_I, timepoint_II], ignore_index=True)

print('===== Permutation test =====')

longAll['group_cat'] = np.where(longAll['group_II'] == '0', 0, 1)  # e.g. '0' is control vs everything else

n_permutations = 100000
formula_perm = "DeltaBrainPAD ~ group_cat + n_euler + Age_dif"
covariates = ['group_cat', 'n_euler', 'Age_dif']

# 1) Fit original
obs_coefs = fit_glm_and_get_all_coef(longAll, formula_perm)

# 2) Permute
df_perm = longAll.copy()
perm_coefs = {c: [] for c in covariates}

for _ in range(n_permutations):
    df_perm['DeltaBrainPAD'] = np.random.permutation(longAll['DeltaBrainPAD'])
    perm_res = fit_glm_and_get_all_coef(df_perm, formula_perm)
    for c in covariates:
        perm_coefs[c].append(perm_res[c])

# 3) Convert to arrays
for c in covariates:
    perm_coefs[c] = np.array(perm_coefs[c])

# 4) Compare to observed
for c in covariates:
    observed = obs_coefs[c]
    distribution = perm_coefs[c]
    p_value = np.mean(np.abs(distribution) >= np.abs(observed))
    print(f"Covariate: {c}, Obs={observed:.3}, p={p_value:.2e}")
print()

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
data = np.array([[40, 21], [10, 8], [15, 7], [9, 3]])

# Chi-square test
chi2, p, dof, expected = chi2_contingency(data)

print(f"\nChi-square statistic (Sex): {chi2:.3}")
print(f"p-value: {p:.2e}")

print('===== Permutation test =====')

# Merge the DataFrames on 'ID'
n_permutations = 100000
formula = 'DeltaBrainPAD ~ pliks18TH_x + n_euler + Age_dif'

# Coefficients of interest
covariates = ['pliks18TH_x', 'n_euler', 'Age_dif']

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

# Custom Rain Cloud plots
rain_cloud_plot_III(longAll)
rain_cloud_plot_V(longAll)
