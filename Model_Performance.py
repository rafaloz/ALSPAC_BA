from utils import *
import ast

x_test = pd.read_csv('/home/rafa/PycharmProjects/ALSPAC_BA/Model/Datos_Test_sample.csv')
X_test_OutSample = pd.read_csv('/home/rafa/PycharmProjects/ALSPAC_BA/Model/Datos_AgeRisk_To_Test.csv')

features = pd.read_csv('/home/rafa/PycharmProjects/ALSPAC_BA/Model/stored_features.csv')
features = ast.literal_eval(features.iloc[0, 2])

# Evaluate test out-of-sample (Age Risk)
print("# 1 ## Starting evaluation of test dataset...")

all_data_test = x_test.copy()

# Separate features and results
X_test_results = x_test.iloc[:, :7]
X_test = pd.concat([x_test.iloc[:, [2]], x_test.iloc[:, 7:]], axis=1)
features_used = X_test.columns.tolist()

# Evaluate the model
print("Evaluating the model...")
result_x_test = model_evaluation(X_test, X_test_results)
print("Model evaluation completed.")

# Plot the comparison of age vs. predicted age
print("Generating the Age vs Predicted Age figure...")
age_vs_predicted_age_figure(result_x_test['Edad'].values, result_x_test['pred_Edad'].values)
print("Figure generated successfully.")

# Out-of-Sample Test Evaluation (Age Risk)
print("# 2 ## Starting evaluation of the out-of-sample dataset (Age Risk)...")

all_data_test = X_test_OutSample.copy()

# Separate features and results
Age_Risk_results = X_test_OutSample.iloc[:, :7]
X_test = pd.concat([X_test_OutSample.iloc[:, [2]], X_test_OutSample.iloc[:, 7:]], axis=1)

# Evaluate the model
print("Evaluating the model on the out-of-sample data...")
result_AgeRisk = model_evaluation(X_test, Age_Risk_results)
print("Out-of-sample model evaluation completed.")

# Plot the comparison of age vs. predicted age
print("Generating the Age vs Predicted Age figure for the out-of-sample data...")
age_vs_predicted_age_figure(result_AgeRisk['Edad'].values, result_AgeRisk['pred_Edad'].values)
print("Figure generated successfully for the out-of-sample test.")


