import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# Create output directory for plots
os.makedirs('output', exist_ok=True)

print("=== CRISP-DM Phase 1: Business Understanding ===")
print("Goal: Predict the quality of red wine based on its physicochemical tests.")
print("This will help winemakers understand which factors contribute most to wine quality.")

print("\n=== CRISP-DM Phase 2: Data Understanding ===")
# Fetch data directly from UCI Machine Learning Repository
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
df = pd.read_csv(url, sep=';')
print(f"Dataset shape: {df.shape}")
print("Features:", df.columns.tolist())
print(df.head())

# Basic EDA
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig('output/correlation_matrix.png')
plt.close()

print("\n=== CRISP-DM Phase 3: Data Preparation ===")
# Check for missing values
print("Missing values in each column:\n", df.isnull().sum())

# Define X and y
X = df.drop('quality', axis=1)
y = df['quality']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Add constant for statsmodels
X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)

print("\n=== CRISP-DM Phase 4: Modeling (with Feature Selection) ===")
# Initial Full Model
full_model = sm.OLS(y_train, X_train_sm).fit()
print("Initial Model AIC:", full_model.aic)

# Feature Selection (Backward Elimination using p-values)
def backward_elimination(data, target, significance_level=0.05):
    features = data.columns.tolist()
    while len(features) > 1: # Keep at least constant
        model = sm.OLS(target, data[features]).fit()
        p_values = model.pvalues
        max_p_value = p_values.max()
        if max_p_value > significance_level:
            excluded_feature = p_values.idxmax()
            if excluded_feature == 'const':
                # If constant is the highest, check the next highest
                p_values_no_const = p_values.drop('const')
                if not p_values_no_const.empty and p_values_no_const.max() > significance_level:
                    excluded_feature = p_values_no_const.idxmax()
                else:
                    break
            features.remove(excluded_feature)
            print(f"Removed {excluded_feature} with p-value {max_p_value:.4f}")
        else:
            break
    return model, features

final_model, selected_features = backward_elimination(X_train_sm, y_train)
print("\nSelected Features:", selected_features)
print(final_model.summary())

print("\n=== CRISP-DM Phase 5: Evaluation ===")
# Predict on testing set
X_test_selected = X_test_sm[selected_features]
y_pred = final_model.predict(X_test_selected)

# Get prediction intervals
predictions = final_model.get_prediction(X_test_selected)
pred_summary = predictions.summary_frame(alpha=0.05) # 95% confidence/prediction intervals

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test MSE:  {mse:.4f}")
print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE:  {mae:.4f}")
print(f"Test R^2:  {r2:.4f}")

# Enhanced Visualization
sns.set_theme(style="whitegrid", palette="muted")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: Actual vs Predicted with Prediction Intervals
results_df = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred.values,
    'Lower': pred_summary['obs_ci_lower'].values,
    'Upper': pred_summary['obs_ci_upper'].values
})
results_df = results_df.sort_values(by='Actual').reset_index(drop=True)

ax1.plot(results_df.index, results_df['Actual'], 'o', color='darkred', label='Actual Quality', markersize=5, alpha=0.7)
ax1.plot(results_df.index, results_df['Predicted'], '-', color='royalblue', label='Predicted Quality', linewidth=2)
ax1.fill_between(results_df.index, results_df['Lower'], results_df['Upper'], color='royalblue', alpha=0.15, label='95% Prediction Interval')

ax1.set_title('Actual vs Predicted Red Wine Quality', fontsize=14, fontweight='bold')
ax1.set_xlabel('Samples (Sorted by Actual Quality)', fontsize=12)
ax1.set_ylabel('Wine Quality', fontsize=12)
ax1.legend(loc='upper left')

# Subplot 2: Residual Plot
residuals = y_test - y_pred
sns.residplot(x=y_pred, y=residuals, lowess=True, color="darkred", line_kws={"color": "royalblue", "linewidth": 2}, ax=ax2)
ax2.set_title('Residuals vs Predicted Values', fontsize=14, fontweight='bold')
ax2.set_xlabel('Predicted Quality', fontsize=12)
ax2.set_ylabel('Residuals', fontsize=12)
ax2.axhline(0, color='gray', linestyle='--')

plt.tight_layout()
plt.savefig('output/prediction_plot.png', dpi=300)
plt.close()
print("Prediction plot saved to 'output/prediction_plot.png'")

print("\n=== CRISP-DM Phase 6: Deployment ===")
print("Model could be deployed as a REST API or embedded in a web application (e.g., Streamlit).")
print("Saved the trained model parameters if needed.")
# For example, save the model using pickle
import pickle
with open('output/wine_quality_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)
print("Model saved to 'output/wine_quality_model.pkl'")
