from utils import *
import ast

x_test = '/home/rafa/PycharmProjects/ALSPAC_BA/Model/Test_Set.csv'
x_test_OutSample = '/home/rafa/PycharmProjects/ALSPAC_BA/Model/AgeRisk_Test.csv'

features = pd.read_csv('/home/rafa/PycharmProjects/ALSPAC_BA/Model/stored_features.csv')
features = ast.literal_eval(features.iloc[0, 2])

# Evaluate test out-of-sample (Age Risk)
print("# 1 ## Starting evaluation of test dataset...")

# Separate features and results
X_test_results = pd.read_csv(x_test).iloc[:, :7]

# Evaluate the model
print("Evaluating the model...")
result_x_test = model_evaluation(x_test, X_test_results)
print("Model evaluation completed.")

# Plot the comparison of age vs. predicted age
print("Generating the Age vs Predicted Age figure...")
age_vs_predicted_age_figure(result_x_test['Age'].values, result_x_test['pred_Age'].values)
print("Figure generated successfully.")

# Out-of-Sample Test Evaluation (Age Risk)
print("# 2 ## Starting evaluation of the out-of-sample dataset (Age Risk)...")

# Separate features and results
Age_Risk_results = pd.read_csv(x_test_OutSample).iloc[:, :7]

# Evaluate the model
print("Evaluating the model on the out-of-sample data...")
result_AgeRisk = model_evaluation(x_test_OutSample, Age_Risk_results)
print("Out-of-sample model evaluation completed.")

# Plot the comparison of age vs. predicted age
print("Generating the Age vs Predicted Age figure for the out-of-sample data...")
age_vs_predicted_age_figure(result_AgeRisk['Age'].values, result_AgeRisk['pred_Age'].values)
print("Figure generated successfully for the out-of-sample test.")


