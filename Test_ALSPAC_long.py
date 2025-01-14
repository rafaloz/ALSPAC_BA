import pandas as pd
import numpy as np

from statsmodels.formula.api import glm
from scipy.stats import kruskal, chi2_contingency
import statsmodels.api as sm
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

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

ALSPAC_I_merged = pd.read_csv('/home/rafa/PycharmProjects/ALSPAC_BA/ALSPAC_I_merged.csv')
ALSPAC_II_merged = pd.read_csv('/home/rafa/PycharmProjects/ALSPAC_BA/ALSPAC_II_merged_II.csv')

new_values = [value[4:] + '_brain' for value in ALSPAC_II_merged['ID']]
ALSPAC_II_merged['ID'] = new_values

X_test_Cardiff_I = ALSPAC_I_merged[ALSPAC_I_merged['ID'].isin(ALSPAC_II_merged['ID'])]
X_test_Cardiff_II = ALSPAC_II_merged[ALSPAC_II_merged['ID'].isin(ALSPAC_I_merged['ID'])]

# Crear una nueva columna 'grupo' con un valor por defecto
X_test_Cardiff_II.loc[:, 'grupo'] = 'NoDefinido'
X_test_Cardiff_II.loc[:, 'grupo_b'] = 'NoDefinido'

# Asignar el grupo correspondiente según las condiciones
X_test_Cardiff_II.loc[(X_test_Cardiff_II['pliks18TH'] >= 1) & (X_test_Cardiff_II['pliks30TH'] >= 1), 'grupo'] = 'Persistente'
X_test_Cardiff_II.loc[(X_test_Cardiff_II['pliks18TH'] >= 1) & (X_test_Cardiff_II['pliks30TH'] == 0), 'grupo'] = 'NoPersistente'
X_test_Cardiff_II.loc[(X_test_Cardiff_II['pliks18TH'] == 0) & (X_test_Cardiff_II['pliks30TH'] == 0), 'grupo'] = 'Control'
X_test_Cardiff_II.loc[(X_test_Cardiff_II['pliks18TH'] == 0) & (X_test_Cardiff_II['pliks30TH'] >= 1), 'grupo'] = 'Incidente'

# Modificar la clasificación basada en las nuevas condiciones
X_test_Cardiff_II.loc[(X_test_Cardiff_II['pliks18TH'] == 0) & (X_test_Cardiff_II['pliks30TH'] == 0), 'grupo_b'] = 'Control'
X_test_Cardiff_II.loc[(X_test_Cardiff_II['pliks18TH'] == X_test_Cardiff_II['pliks30TH']) & (X_test_Cardiff_II['pliks18TH'] != 0), 'grupo_b'] = 'Persistente'
X_test_Cardiff_II.loc[(X_test_Cardiff_II['pliks18TH'] < X_test_Cardiff_II['pliks30TH']), 'grupo_b'] = 'Aumentado'
X_test_Cardiff_II.loc[(X_test_Cardiff_II['pliks18TH'] > X_test_Cardiff_II['pliks30TH']), 'grupo_b'] = 'Disminuido'

X_test_Cardiff_I['pliks30TH'] = X_test_Cardiff_II['pliks30TH'].values
X_test_Cardiff_I.loc[(X_test_Cardiff_I['pliks18TH'] >= 1) & (X_test_Cardiff_I['pliks30TH'] >= 1), 'grupo'] = 'Persistente'
X_test_Cardiff_I.loc[(X_test_Cardiff_I['pliks18TH'] >= 1) & (X_test_Cardiff_I['pliks30TH'] == 0), 'grupo'] = 'NoPersistente'
X_test_Cardiff_I.loc[(X_test_Cardiff_I['pliks18TH'] == 0) & (X_test_Cardiff_I['pliks30TH'] == 0), 'grupo'] = 'Control'
X_test_Cardiff_I.loc[(X_test_Cardiff_I['pliks18TH'] == 0) & (X_test_Cardiff_I['pliks30TH'] >= 1), 'grupo'] = 'Incidente'

X_test_Cardiff_I.loc[(X_test_Cardiff_I['pliks18TH'] == 0) & (X_test_Cardiff_I['pliks30TH'] == 0), 'grupo_II'] = 'Control'
X_test_Cardiff_I.loc[(X_test_Cardiff_I['pliks18TH'] == X_test_Cardiff_I['pliks30TH']) & (X_test_Cardiff_I['pliks18TH'] != 0), 'grupo_b'] = 'Persistente'
X_test_Cardiff_I.loc[(X_test_Cardiff_I['pliks18TH'] < X_test_Cardiff_I['pliks30TH']), 'grupo_b'] = 'Aumentado'
X_test_Cardiff_I.loc[(X_test_Cardiff_I['pliks18TH'] > X_test_Cardiff_I['pliks30TH']), 'grupo_b'] = 'Disminuido'

Longi_I_cardiff = X_test_Cardiff_I[['ID', 'Edad', 'sexo(M=1;F=0)', 'pliks18TH', 'pliks20TH', 'pliks30TH', 'brainPAD_standardized', 'grupo']]
Longi_I_cardiff.columns = ['ID', 'Edad_I', 'sexo(M=1;F=0)_I', 'pliks18TH_I', 'pliks20TH_I', 'pliks30TH_I', 'brainPAD_standardized_I', 'grupo_I']
Longi_II_cardiff = X_test_Cardiff_II[['ID', 'Edad', 'sexo(M=1;F=0)', 'pliks18TH', 'pliks20TH', 'pliks30TH', 'brainPAD_standardized', 'grupo']]
Longi_II_cardiff.columns = ['ID', 'Edad_II', 'sexo(M=1;F=0)_II', 'pliks18TH_II', 'pliks20TH_II', 'pliks30TH_II', 'brainPAD_standardized_II', 'grupo_II']

longi_todos = pd.merge(Longi_I_cardiff, Longi_II_cardiff, on='ID', how='inner')
longi_todos.reset_index(drop=True, inplace=True)
longi_todos['DeltaBrainPAD'] = Longi_II_cardiff['brainPAD_standardized_II'].values-Longi_I_cardiff['brainPAD_standardized_I'].values

Control_Cardiff_Longi_control = longi_todos[longi_todos['grupo_II']=='Control']
Control_Cardiff_Longi_NoPersi = longi_todos[longi_todos['grupo_II']=='NoPersistente']
Control_Cardiff_Longi_Persi = longi_todos[longi_todos['grupo_II']=='Persistente']
Control_Cardiff_Longi_Inci = longi_todos[longi_todos['grupo_II']=='Incidente']

print('\nGrupo Control')
print('nº cases total: '+str(Control_Cardiff_Longi_control.shape[0]))
print('Age average: '+str(Control_Cardiff_Longi_control['Edad_II'].mean()))
print('Age std: '+str(Control_Cardiff_Longi_control['Edad_II'].std()))
print('Age max: '+str(Control_Cardiff_Longi_control['Edad_II'].max()))
print('Age min: '+str(Control_Cardiff_Longi_control['Edad_II'].min()))
print('Males: '+str(Control_Cardiff_Longi_control['sexo(M=1;F=0)_II'].sum()))
print('Females: '+str(Control_Cardiff_Longi_control.shape[0]-Control_Cardiff_Longi_control['sexo(M=1;F=0)_II'].sum()))

print('\nGrupo NoPersistente')
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

print('\nGrupo Incidente')
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

print("longi noPersi:", Control_Cardiff_Longi_NoPersi['DeltaBrainPAD'].mean())
print("longi Persi:", Control_Cardiff_Longi_Persi['DeltaBrainPAD'].mean())
print("longi control:", Control_Cardiff_Longi_control['DeltaBrainPAD'].mean())
print("longi Inci:", Control_Cardiff_Longi_Inci['DeltaBrainPAD'].mean())

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

# Create a dictionary to map the old values to new values
mapping = {
    'NoPersistente': 0,
    'Control': 1,
    'Persistente': 2,
    'Incidente': 3
}

print("longi Control:", Control_Cardiff_Longi_control['DeltaBrainPAD'].mean())
print("longi NoPersistente:", Control_Cardiff_Longi_NoPersi['DeltaBrainPAD'].mean())
print("longi Persistente:", Control_Cardiff_Longi_Persi['DeltaBrainPAD'].mean())
print("longi Incidente:", Control_Cardiff_Longi_Inci['DeltaBrainPAD'].mean())

# Create a new column based on the old column values
longi_todos['Grupo_ordinal'] = longi_todos['grupo_II'].map(mapping)

longi_todos['n_euler'] = 2

# Reshape data into long format
timepoint_I = longi_todos[['ID', 'Edad_I', 'sexo(M=1;F=0)_I', 'brainPAD_standardized_I', 'grupo_I', 'n_euler']].copy()
timepoint_I['Time'] = 'I'
timepoint_I.rename(columns={
    'Edad_I': 'Edad',
    'sexo(M=1;F=0)_I': 'sexo',
    'brainPAD_standardized_I': 'brainPAD_standardized',
    'grupo_I': 'grupo'
}, inplace=True)

timepoint_II = longi_todos[['ID', 'Edad_II', 'sexo(M=1;F=0)_II', 'brainPAD_standardized_II', 'grupo_II', 'n_euler']].copy()
timepoint_II['Time'] = 'II'
timepoint_II.rename(columns={
    'Edad_II': 'Edad',
    'sexo(M=1;F=0)_II': 'sexo',
    'brainPAD_standardized_II': 'brainPAD_standardized',
    'grupo_II': 'grupo'
}, inplace=True)

long_data = pd.concat([timepoint_I, timepoint_II], ignore_index=True)

hue_order = ['NoPersistente', 'Control', 'Persistente', 'Incidente']
labels = ['Longitudinal Controls', 'Incident', 'Remittent', 'Persistent']

# Plot observed means by group at each timepoint
sns.lineplot(data=long_data, x='Time', y='brainPAD_standardized', hue='grupo', estimator='mean', ci='sd', marker='o')

plt.title('Brain Age Predictions Timepoints')
plt.xlabel('Time')
plt.ylabel('Brain Age (Predicted)')

# Get current legend and change labels
handles, _ = plt.gca().get_legend_handles_labels()
plt.legend(handles=handles, labels=labels, title='Group')

plt.show()

print('-----------------------------')
print('====== TREND ANALYSIS =======')
print('-----------------------------')

print('--- check Age ---')

# Shapiro-Wilk Test for Normality within each group
print("Shapiro-Wilk Test Results for Normality (Age): ")
for group in sorted(longi_todos['grupo_II'].unique()):
    sample = longi_todos[longi_todos['grupo_II'] == group]['Edad_II']
    # Perform the Shapiro-Wilk test on the sample
    stat, p_value = stats.shapiro(sample)
    print(f"Group {group}: W-statistic={stat:.4f}, p-value={p_value:.4f}")

# Levene's Test for Homogeneity of Variances
w, p_value = stats.levene(*[longi_todos[longi_todos['grupo_II'] == group]['Edad_II'] for group in sorted(longi_todos['grupo_II'].unique())])
print(f"\nLevene's Test (Age): W-statistic={w:.4f}, p-value={p_value:.4f}")

# Extract unique groups
groups = longi_todos['grupo_II'].unique()

# Prepare age data for each group
age_data = [longi_todos[longi_todos['grupo_II'] == group]['Edad_II'].values for group in groups]

# Perform Kruskal-Wallis H Test
statistic, p_value = kruskal(*age_data)

print(f"\nKruskal-Wallis H Test Statistic (Age): {statistic:.4f}")
print(f"P-Value: {p_value:.4f}")

# Contingency table (replace with actual counts)
data = np.array([[37, 19], [22, 13], [3, 2], [12, 5]])

# Chi-square test
chi2, p, dof, expected = chi2_contingency(data)

print(f"\nChi-square statistic: {chi2}")
print(f"p-value: {p}")

print('\n--- check std brainPAD ---')

# Shapiro-Wilk Test for Normality within each group
print("Shapiro-Wilk Test Results for Normality (std brainPAD):")
for group in sorted(longi_todos['grupo_II'].unique()):
    sample = longi_todos[longi_todos['grupo_II'] == group]['DeltaBrainPAD']
    # Perform the Shapiro-Wilk test on the sample
    stat, p_value = stats.shapiro(sample)
    print(f"Group {group}: W-statistic={stat:.4f}, p-value={p_value:.4f}")

# Levene's Test for Homogeneity of Variances
w, p_value = stats.levene(*[longi_todos[longi_todos['grupo_II'] == group]['DeltaBrainPAD'] for group in sorted(longi_todos['grupo_II'].unique())])
print(f"\nLevene's Test (std brainPAD): W-statistic={w:.4f}, p-value={p_value:.4f}")

longi_todos['Edad_dif'] = longi_todos['Edad_II'] - longi_todos['Edad_I']

print('===== Permutation test trend =====')

# Merge the DataFrames on 'ID'
n_permutations = 5000
formula = 'DeltaBrainPAD ~ Grupo_ordinal + n_euler + Edad_dif'

# Coefficients of interest
covariates = ['Grupo_ordinal', 'n_euler', 'Edad_dif']

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

from scipy.stats import spearmanr

# Compute Spearman's rank correlation
correlation, p_value = spearmanr(longi_todos['DeltaBrainPAD'], longi_todos['Grupo_ordinal'])

# Display the results
print(f"Spearman's correlation coefficient: {correlation}")
print(f"P-value: {p_value}")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from tqdm import tqdm

longi_todos.loc[:, 'grupo_cat'] = 'NoDefinido'
longi_todos.loc[(longi_todos['Grupo_ordinal'] != 1), 'grupo_cat'] = 1
longi_todos.loc[(longi_todos['Grupo_ordinal'] == 1), 'grupo_cat'] = 0

print('===== Permutation test Cat =====')

# Merge the DataFrames on 'ID'
n_permutations = 5000
formula = 'DeltaBrainPAD ~ grupo_cat + n_euler + Edad_dif'

# Coefficients of interest
covariates = ['grupo_cat[T.1]', 'n_euler', 'Edad_dif']

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


from utils_Train import rain_cloud_plot_VI, rain_cloud_plot_VII

rain_cloud_plot_VI(longi_todos)
rain_cloud_plot_VII(longi_todos)


print('pause')
