import warnings
warnings.filterwarnings("ignore")
from statsmodels.formula.api import glm
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm

def calculate_cohen_d(group1, group2):
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    return (mean2 - mean1) / pooled_std

# Función para ajustar el GLM y devolver todos los coefs
def fit_glm_and_get_all_coef(data, formula):
    model = glm(formula=formula, data=data, family=sm.families.Gaussian()).fit()
    return model.params

ALSPAC_I_merged = pd.read_csv('/home/rafa/PycharmProjects/Cardiff_ALSPAC/ALSPAC_I_merged.csv')
ALSPAC_II_merged = pd.read_csv('/home/rafa/PycharmProjects/Cardiff_ALSPAC/ALSPAC_II_merged_II.csv')

new_values = [value[4:] + '_brain' for value in ALSPAC_II_merged['ID']]
ALSPAC_II_merged['ID'] = new_values

X_test_Cardiff_I = ALSPAC_I_merged[ALSPAC_I_merged['ID'].isin(ALSPAC_II_merged['ID'])]
X_test_Cardiff_II = ALSPAC_II_merged[ALSPAC_II_merged['ID'].isin(ALSPAC_I_merged['ID'])]

# Crear una nueva columna 'grupo' con un valor por defecto
X_test_Cardiff_II.loc[:, 'grupo'] = 'NoDefinido'

# Asignar el grupo correspondiente según las condiciones
X_test_Cardiff_II.loc[(X_test_Cardiff_II['pliks18TH'] == 0), 'grupo'] = '0'
X_test_Cardiff_II.loc[(X_test_Cardiff_II['pliks18TH'] == 1), 'grupo'] = '1'
X_test_Cardiff_II.loc[(X_test_Cardiff_II['pliks18TH'] == 2), 'grupo'] = '2'
X_test_Cardiff_II.loc[(X_test_Cardiff_II['pliks18TH'] == 3), 'grupo'] = '3'

X_test_Cardiff_I['pliks30TH'] = X_test_Cardiff_II['pliks30TH'].values
X_test_Cardiff_I.loc[(X_test_Cardiff_I['pliks18TH'] == 0), 'grupo'] = '0'
X_test_Cardiff_I.loc[(X_test_Cardiff_I['pliks18TH'] == 1), 'grupo'] = '1'
X_test_Cardiff_I.loc[(X_test_Cardiff_I['pliks18TH'] == 2), 'grupo'] = '2'
X_test_Cardiff_I.loc[(X_test_Cardiff_I['pliks18TH'] == 3), 'grupo'] = '3'

Longi_I_cardiff = X_test_Cardiff_I[['ID', 'Edad', 'sexo(M=1;F=0)', 'pliks18TH', 'pliks20TH', 'pliks30TH', 'brainPAD_standardized', 'grupo']]
Longi_I_cardiff.columns = ['ID', 'Edad_I', 'sexo(M=1;F=0)_I', 'pliks18TH_I', 'pliks20TH_I', 'pliks30TH_I', 'brainPAD_standardized_I', 'grupo_I']
Longi_II_cardiff = X_test_Cardiff_II[['ID', 'Edad', 'sexo(M=1;F=0)', 'pliks18TH', 'pliks20TH', 'pliks30TH', 'brainPAD_standardized', 'grupo']]
Longi_II_cardiff.columns = ['ID', 'Edad_II', 'sexo(M=1;F=0)_II', 'pliks18TH_II', 'pliks20TH_II', 'pliks30TH_II', 'brainPAD_standardized_II', 'grupo_II']

longi_todos = pd.merge(Longi_I_cardiff, Longi_II_cardiff, on='ID', how='inner')
longi_todos.reset_index(drop=True, inplace=True)
longi_todos['DeltaBrainPAD'] = Longi_II_cardiff['brainPAD_standardized_II'].values-Longi_I_cardiff['brainPAD_standardized_I'].values

Control_Cardiff_Longi_control = longi_todos[longi_todos['grupo_II']=='0']
Control_Cardiff_Longi_NoPersi = longi_todos[longi_todos['grupo_II']=='1']
Control_Cardiff_Longi_Persi = longi_todos[longi_todos['grupo_II']=='2']
Control_Cardiff_Longi_Inci = longi_todos[longi_todos['grupo_II']=='3']

print('\nGrupo Control')
print('nº cases total: '+str(Control_Cardiff_Longi_control.shape[0]))
print('Age average: '+str(Control_Cardiff_Longi_control['Edad_II'].mean()))
print('Age std: '+str(Control_Cardiff_Longi_control['Edad_II'].std()))
print('Age max: '+str(Control_Cardiff_Longi_control['Edad_II'].max()))
print('Age min: '+str(Control_Cardiff_Longi_control['Edad_II'].min()))
print('Males: '+str(Control_Cardiff_Longi_control['sexo(M=1;F=0)_II'].sum()))
print('Females: '+str(Control_Cardiff_Longi_control.shape[0]-Control_Cardiff_Longi_control['sexo(M=1;F=0)_II'].sum()))

print('\nGrupo Disminuido')
print('nº cases total: '+str(Control_Cardiff_Longi_NoPersi.shape[0]))
print('Age average: '+str(Control_Cardiff_Longi_NoPersi['Edad_II'].mean()))
print('Age std: '+str(Control_Cardiff_Longi_NoPersi['Edad_II'].std()))
print('Age max: '+str(Control_Cardiff_Longi_NoPersi['Edad_II'].max()))
print('Age min: '+str(Control_Cardiff_Longi_NoPersi['Edad_II'].min()))
print('Males: '+str(Control_Cardiff_Longi_NoPersi['sexo(M=1;F=0)_II'].sum()))
print('Females: '+str(Control_Cardiff_Longi_NoPersi.shape[0]-Control_Cardiff_Longi_NoPersi['sexo(M=1;F=0)_II'].sum()))

print('\nGrupo Persistente')
print('nº cases total: '+str(Control_Cardiff_Longi_Persi.shape[0]))
print('Age average: '+str(Control_Cardiff_Longi_Persi['Edad_II'].mean()))
print('Age std: '+str(Control_Cardiff_Longi_Persi['Edad_II'].std()))
print('Age max: '+str(Control_Cardiff_Longi_Persi['Edad_II'].max()))
print('Age min: '+str(Control_Cardiff_Longi_Persi['Edad_II'].min()))
print('Males: '+str(Control_Cardiff_Longi_Persi['sexo(M=1;F=0)_II'].sum()))
print('Females: '+str(Control_Cardiff_Longi_Persi.shape[0]-Control_Cardiff_Longi_Persi['sexo(M=1;F=0)_II'].sum()))

print('\nGrupo Aumentado')
print('nº cases total: '+str(Control_Cardiff_Longi_Inci.shape[0]))
print('Age average: '+str(Control_Cardiff_Longi_Inci['Edad_II'].mean()))
print('Age std: '+str(Control_Cardiff_Longi_Inci['Edad_II'].std()))
print('Age max: '+str(Control_Cardiff_Longi_Inci['Edad_II'].max()))
print('Age min: '+str(Control_Cardiff_Longi_Inci['Edad_II'].min()))
print('Males: '+str(Control_Cardiff_Longi_Inci['sexo(M=1;F=0)_II'].sum()))
print('Females: '+str(Control_Cardiff_Longi_Inci.shape[0]-Control_Cardiff_Longi_Inci['sexo(M=1;F=0)_II'].sum()))

# Printing the number of individuals in each dataframe
print("Number of individuals:", longi_todos.shape[0])
print("Number of individuals in longi control:", Control_Cardiff_Longi_control.shape[0])
print("Number of individuals in longi noPersi:", Control_Cardiff_Longi_NoPersi.shape[0])
print("Number of individuals in longi Persi:", Control_Cardiff_Longi_Persi.shape[0])
print("Number of individuals in longi Inci:", Control_Cardiff_Longi_Inci.shape[0])

print("longi Disminuido:", Control_Cardiff_Longi_NoPersi['DeltaBrainPAD'].mean())
print("longi Persi:", Control_Cardiff_Longi_Persi['DeltaBrainPAD'].mean())
print("longi control:", Control_Cardiff_Longi_control['DeltaBrainPAD'].mean())
print("longi Aumentado:", Control_Cardiff_Longi_Inci['DeltaBrainPAD'].mean())

LongiConPE = pd.concat([Control_Cardiff_Longi_Persi, Control_Cardiff_Longi_Inci, Control_Cardiff_Longi_NoPersi], axis=0, ignore_index=True)

print('\nGrupo PE todo')
print('nº cases total: '+str(LongiConPE.shape[0]))
print('Age average: '+str(LongiConPE['Edad_II'].mean()))
print('Age std: '+str(LongiConPE['Edad_II'].std()))
print('Age max: '+str(LongiConPE['Edad_II'].max()))
print('Age min: '+str(LongiConPE['Edad_II'].min()))
print('Males: '+str(LongiConPE['sexo(M=1;F=0)_II'].sum()))
print('Females: '+str(LongiConPE.shape[0]-LongiConPE['sexo(M=1;F=0)_II'].sum()))

LongiALl = pd.concat([Control_Cardiff_Longi_control, LongiConPE], axis=0, ignore_index=True)

print('\nGrupo Longi Entero todo')
print('nº cases total: '+str(LongiALl.shape[0]))
print('Age average: '+str(LongiALl['Edad_II'].mean()))
print('Age std: '+str(LongiALl['Edad_II'].std()))
print('Age max: '+str(LongiALl['Edad_II'].max()))
print('Age min: '+str(LongiALl['Edad_II'].min()))
print('Males: '+str(LongiALl['sexo(M=1;F=0)_II'].sum()))
print('Females: '+str(LongiALl.shape[0]-LongiALl['sexo(M=1;F=0)_II'].sum()))

# Calculate Cohen's d for each pair of levels
cohen_d_0_123 = calculate_cohen_d(Control_Cardiff_Longi_control['DeltaBrainPAD'].values, LongiConPE['DeltaBrainPAD'].values)
cohen_d_0_1 = calculate_cohen_d(Control_Cardiff_Longi_control['DeltaBrainPAD'].values, Control_Cardiff_Longi_NoPersi['DeltaBrainPAD'].values)
cohen_d_0_2 = calculate_cohen_d(Control_Cardiff_Longi_control['DeltaBrainPAD'].values, Control_Cardiff_Longi_Persi['DeltaBrainPAD'].values)
cohen_d_0_3 = calculate_cohen_d(Control_Cardiff_Longi_control['DeltaBrainPAD'].values, Control_Cardiff_Longi_Inci['DeltaBrainPAD'].values)
cohen_d_1_2 = calculate_cohen_d(Control_Cardiff_Longi_NoPersi['DeltaBrainPAD'].values, Control_Cardiff_Longi_Persi['DeltaBrainPAD'].values)
cohen_d_1_3 = calculate_cohen_d(Control_Cardiff_Longi_NoPersi['DeltaBrainPAD'].values, Control_Cardiff_Longi_Inci['DeltaBrainPAD'].values)
cohen_d_2_3 = calculate_cohen_d(Control_Cardiff_Longi_Persi['DeltaBrainPAD'].values, Control_Cardiff_Longi_Inci['DeltaBrainPAD'].values)

print("Cohen's D 0_123: "+str(cohen_d_0_123))
print("Cohen's D 0_1: "+str(cohen_d_0_1))
print("Cohen's D 0_2: "+str(cohen_d_0_2))
print("Cohen's D 0_3: "+str(cohen_d_0_3))
print("Cohen's D 1_2: "+str(cohen_d_1_2))
print("Cohen's D 1_3: "+str(cohen_d_1_3))
print("Cohen's D 2_3: "+str(cohen_d_2_3))

# Shapiro-Wilk tests for normality
for group, name in [(LongiConPE['Edad_II'], "LongiConPE"), (Control_Cardiff_Longi_control['Edad_II'], "Control_longi")]:
    stat, p_value = stats.shapiro(group)
    print(f"Shapiro-Wilk Test Statistic {name}: {stat:.4f}, P-value: {p_value:.4f}")
    print(f"{name} follows a normal distribution." if p_value >= 0.05 else f"{name} does not follow a normal distribution.")

# Levene's test for homogeneity of variances (homoscedasticity)
levene_stat, levene_p = stats.levene(LongiConPE['Edad_II'], Control_Cardiff_Longi_control['Edad_II'])
print(f"Levene's Test: Statistic={levene_stat:.4f}, P-value={levene_p:.4f}")
print("Homoscedasticity assumption holds." if levene_p >= 0.05 else "Homoscedasticity assumption does not hold.")

# Choose the appropriate test for comparing group means based on normality and homoscedasticity
if p_value >= 0.05 and levene_p >= 0.05:
    # Perform t-test (both groups are normal, variances are equal)
    t_stat, t_p_value = stats.ttest_ind(LongiConPE['Edad_II'], Control_Cardiff_Longi_control['Edad_II'])
    print(f"T-test: Statistic={t_stat:.4f}, P-value={t_p_value:.4f}")
elif p_value < 0.05 and levene_p >= 0.05:
    # Perform Welch's t-test (both groups normal, variances not equal)
    t_stat, t_p_value = stats.ttest_ind(LongiConPE['Edad_II'], Control_Cardiff_Longi_control['Edad_II'], equal_var=False)
    print(f"Welch's T-test: Statistic={t_stat:.4f}, P-value={t_p_value:.4f}")
else:
    # Perform Mann-Whitney U test (non-normal distribution)
    u_stat, u_p_value = stats.mannwhitneyu(LongiConPE['Edad_II'], Control_Cardiff_Longi_control['Edad_II'])
    print(f"Mann-Whitney U Test: Statistic={u_stat:.4f}, P-value={u_p_value:.4f}")

from scipy.stats import chi2_contingency

data_2x4 = np.array([
    [19, 37],  # Group 1
    [20, 37]  # Group 2
])

# Perform Chi-Square Test
chi2, p, dof, expected = chi2_contingency(data_2x4)

print("Chi2 Statistic:", chi2)
print("P-value:", p)
print("Degrees of Freedom:", dof)
print("Expected Frequencies:\n", expected)

longi_todos['n_euler'] = 2
longi_todos.loc[:, 'grupo_cat'] = 'NoDefinido'
longi_todos.loc[(longi_todos['pliks18TH_I'] != 0), 'grupo_cat'] = 'Enfermos'
longi_todos.loc[(longi_todos['pliks18TH_I'] == 0), 'grupo_cat'] = 'Controles'

# Map 'grupo' to 'Grupo_ordinal' if needed
mapping = {'1': 1, '0': 0, '2': 2, '3': 3}
longi_todos['Grupo_ordinal'] = longi_todos['grupo_II'].map(mapping).astype(int)

print("longi Control:", Control_Cardiff_Longi_control['DeltaBrainPAD'].mean())
print("longi NoPersistente:", Control_Cardiff_Longi_NoPersi['DeltaBrainPAD'].mean())
print("longi Persistente:", Control_Cardiff_Longi_Persi['DeltaBrainPAD'].mean())
print("longi Incidente:", Control_Cardiff_Longi_Inci['DeltaBrainPAD'].mean())

# Shapiro-Wilk test for normality within each group
print("Shapiro-Wilk Test Results for Normality (Edad_II):")
normality_results = []
for group in sorted(longi_todos['Grupo_ordinal'].unique()):
    sample = longi_todos[longi_todos['Grupo_ordinal'] == group]['Edad_II']
    stat, p_value = stats.shapiro(sample)
    print(f"Group {group}: Shapiro-Wilk Statistic={stat:.4f}, P-value={p_value:.4f}")
    normality_results.append(p_value >= 0.05)  # True if the group is normally distributed

# Levene's test for homogeneity of variances (homoscedasticity)
levene_stat, levene_p = stats.levene(*[longi_todos[longi_todos['Grupo_ordinal'] == group]['Edad_II'] for group in sorted(longi_todos['Grupo_ordinal'].unique())])
print(f"\nLevene's Test for Homogeneity of Variances: Statistic={levene_stat:.4f}, P-value={levene_p:.4f}")

# Choose the appropriate test for comparing group means based on normality and homoscedasticity
if all(normality_results) and levene_p >= 0.05:
    # Perform ANOVA (if all groups are normal and variances are equal)
    f_stat, f_p_value = stats.f_oneway(*[longi_todos[longi_todos['Grupo_ordinal'] == group]['Edad_II'] for group in sorted(longi_todos['Grupo_ordinal'].unique())])
    print(f"\nANOVA Test: Statistic={f_stat:.4f}, P-value={f_p_value:.4f}")
elif all(normality_results) and levene_p < 0.05:
    # Perform Welch's ANOVA (if all groups are normal but variances are unequal)
    welch_stat, welch_p_value = stats.ttest_ind(*[longi_todos[longi_todos['Grupo_ordinal'] == group]['Edad_II'] for group in sorted(longi_todos['Grupo_ordinal'].unique())], equal_var=False)
    print(f"\nWelch's ANOVA: Statistic={welch_stat:.4f}, P-value={welch_p_value:.4f}")
else:
    # Perform Kruskal-Wallis Test (if normality is violated in any group)
    kruskal_stat, kruskal_p_value = stats.kruskal(*[longi_todos[longi_todos['Grupo_ordinal'] == group]['Edad_II'] for group in sorted(longi_todos['Grupo_ordinal'].unique())])
    print(f"\nKruskal-Wallis Test: Statistic={kruskal_stat:.4f}, P-value={kruskal_p_value:.4f}")

from scipy.stats import chi2_contingency

data_2x4 = np.array([
    [13, 23],  # Group 1
    [1, 5], # Group 2
    [6, 9],  # Group 3
    [19, 37]  # Group 4
])

# Perform Chi-Square Test
chi2, p, dof, expected = chi2_contingency(data_2x4)

print("Chi2 Statistic:", chi2)
print("P-value:", p)
print("Degrees of Freedom:", dof)
print("Expected Frequencies:\n", expected)

# Reshape data into long format
timepoint_I = longi_todos[['ID', 'Edad_I', 'sexo(M=1;F=0)_I', 'brainPAD_standardized_I', 'grupo_I', 'n_euler', 'grupo_cat']].copy()
timepoint_I['Time'] = 'I'
timepoint_I.rename(columns={
    'Edad_I': 'Edad',
    'sexo(M=1;F=0)_I': 'sexo',
    'brainPAD_standardized_I': 'brainPAD_standardized',
    'grupo_I': 'grupo'
}, inplace=True)

timepoint_II = longi_todos[['ID', 'Edad_II', 'sexo(M=1;F=0)_II', 'brainPAD_standardized_II', 'grupo_II', 'n_euler', 'grupo_cat']].copy()
timepoint_II['Time'] = 'II'
timepoint_II.rename(columns={
    'Edad_II': 'Edad',
    'sexo(M=1;F=0)_II': 'sexo',
    'brainPAD_standardized_II': 'brainPAD_standardized',
    'grupo_II': 'grupo'
}, inplace=True)

long_data = pd.concat([timepoint_I, timepoint_II], ignore_index=True)

# Map 'grupo' to 'Grupo_ordinal' if needed
mapping = {'0': 0, '1': 1, '2': 2, '3': 3}
long_data['Grupo_ordinal'] = long_data['grupo'].map(mapping)

# Define the GEE formula
gee_formula = 'brainPAD_standardized ~ grupo_cat * Time + Edad'

# Fit the GEE model
gee_model = smf.gee(formula=gee_formula,
                    data=long_data,
                    groups=long_data['ID'],
                    family=sm.families.Gaussian(),
                    cov_struct=sm.cov_struct.Exchangeable()).fit()

# Print the summary
print(gee_model.summary())


# Define the GEE formula for trend analysis
gee_formula_trend = 'brainPAD_standardized ~ Grupo_ordinal * Time'

# Fit the GEE model for trend analysis
gee_model_trend = smf.gee(formula=gee_formula_trend,
                          data=long_data,
                          groups=long_data['ID'],
                          family=sm.families.Gaussian(),
                          cov_struct=sm.cov_struct.Exchangeable()).fit()

# Print the summary
print(gee_model_trend.summary())

hue_order = [0, 1, 2, 3]
labels = ['Controls', 'Suspected', 'Definite', 'Clinical Disorder']

# Plot observed means by group at each timepoint
sns.lineplot(data=long_data, x='Time', y='brainPAD_standardized', hue='grupo', estimator='mean', ci='sd', marker='o')

plt.title('Brain Age Predictions Timepoints')
plt.xlabel('Time')
plt.ylabel('Brain Age (Predicted)')

# Get current legend and change labels
handles, _ = plt.gca().get_legend_handles_labels()
plt.legend(handles=handles, labels=labels, title='Group')

plt.show()

# Extract residuals
residuals = gee_model.resid

# Plot Q-Q plot
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot of GEE Residuals')
plt.show()

# Plot residuals vs. fitted values
plt.scatter(gee_model.fittedvalues, residuals)
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('GEE Residuals vs Fitted Values')
plt.show()

# Shapiro-Wilk test for residuals
stat, p_value = stats.shapiro(residuals)
print(f"Shapiro-Wilk Test: W-statistic={stat:.4f}, p-value={p_value:.4f}")
if p_value < 0.05:
    print("Residuals are not normally distributed (reject H0).")
else:
    print("Residuals are normally distributed (fail to reject H0).")

# Levene's test for homogeneity of variances
fitted_value_groups = pd.qcut(gee_model.fittedvalues, q=4)
stat, p_value = stats.levene(*[residuals[fitted_value_groups == group]
                               for group in fitted_value_groups.unique()])
print(f"Levene's Test: W-statistic={stat:.4f}, p-value={p_value:.4f}")
if p_value < 0.05:
    print("Residual variances are significantly different (reject H0).")
else:
    print("Residual variances are equal (fail to reject H0).")

print('===== Permutation test CAT =====')

data_permu = timepoint_I['brainPAD_standardized'].values - timepoint_II['brainPAD_standardized'].values
longi_todos['Delta'] = data_permu
longi_todos['Edad_dif'] = longi_todos['Edad_II'] - longi_todos['Edad_I']
longi_todos['grupo_cat'] = timepoint_I['grupo_cat'].map({'Enfermos': 1, 'Controles': 0})

# Merge the DataFrames on 'ID'
n_permutations = 5000
formula = 'Delta ~ grupo_cat + n_euler + Edad_dif'

# Coefficients of interest
covariates = ['grupo_cat', 'n_euler', 'Edad_dif']

# Step 1: Fit GLM on the original data to get observed coefficients
observed_coefs = fit_glm_and_get_all_coef(longi_todos, formula)

# Step 2: Initialize a dictionary to store permutation coefficients for each covariate
perm_coefs = {covariate: [] for covariate in covariates}

# Create a copy of the DataFrame for permutations
df_perm = longi_todos.copy()

# Step 3: Perform permutations
for _ in range(n_permutations):
    # Permute the dependent variable
    df_perm['Delta'] = np.random.permutation(longi_todos['Delta'])

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
    print(f"Coeficiente observado cat  ({covariate}): {observed_coef}")
    print(f"p-valor de la prueba de permutación cat ({covariate}): {p_value}")


print('===== Permutation test trend =====')

# Merge the DataFrames on 'ID'
n_permutations = 5000
formula = 'DeltaBrainPAD ~ pliks18TH_I + n_euler + Edad_dif'

# Coefficients of interest
covariates = ['pliks18TH_I', 'n_euler', 'Edad_dif']

# Step 1: Fit GLM on the original data to get observed coefficients
observed_coefs = fit_glm_and_get_all_coef(longi_todos, formula)

# Step 2: Initialize a dictionary to store permutation coefficients for each covariate
perm_coefs = {covariate: [] for covariate in covariates}

# Create a copy of the DataFrame for permutations
df_perm = longi_todos.copy()

# Step 3: Perform permutations
for _ in range(n_permutations):
    # Permute the dependent variable
    df_perm['DeltaBrainPAD'] = np.random.permutation(longi_todos['DeltaBrainPAD'])

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


from utils_Train import rain_cloud_plot_VI, rain_cloud_plot_VII, rain_cloud_plot_VIII

rain_cloud_plot_VI(longi_todos)
rain_cloud_plot_VIII(longi_todos)
