#!/usr/bin/env python
# coding: utf-8

# ## SVM

# In[3]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load datasets
data = pd.read_csv('export.csv')
lastMonth = pd.read_csv('lastMonth.csv')

# Remove unnecessary columns
data.drop(columns=['wpgt', 'tsun', 'date'], inplace=True)
lastMonth.drop(columns=['wpgt', 'tsun'], inplace=True)

# Fill missing values with forward fill
data.fillna(method='ffill', inplace=True)
lastMonth.fillna(method='ffill', inplace=True)

# Save cleaned datasets
data.to_csv('cleaned_export.csv', index=False)
lastMonth.to_csv('cleaned_month.csv', index=False)

# Define features (X) and targets (y)
X = data.drop(columns=['tavg', 'prcp'])
y_tavg = data['tavg']  # Target: temperature
y_prcp = data['prcp']  # Target: precipitation


# In[4]:


# Split data for 'tavg' prediction
X_train_tavg, X_test_tavg, y_train_tavg, y_test_tavg = train_test_split(X, y_tavg, test_size=0.2, random_state=42)

# Split data for 'prcp' prediction
X_train_prcp, X_test_prcp, y_train_prcp, y_test_prcp = train_test_split(X, y_prcp, test_size=0.2, random_state=42)


# SVMs assume that the data it works with is in a standard range, like 0 to 1 so we must standardize the data before feeding it to the SVM. 

# In[5]:

# Initialize scalers
scaler_tavg = StandardScaler()
scaler_prcp = StandardScaler()

# Scale training and testing data for 'tavg'
X_train_tavg_scaled = scaler_tavg.fit_transform(X_train_tavg)
X_test_tavg_scaled = scaler_tavg.transform(X_test_tavg)

# Scale training and testing data for 'prcp'
X_train_prcp_scaled = scaler_prcp.fit_transform(X_train_prcp)
X_test_prcp_scaled = scaler_prcp.transform(X_test_prcp)


# In[6]:


# Create SVM models for 'tavg' and 'prcp'
svr_tavg = SVR(kernel='rbf') 
svr_prcp = SVR(kernel='rbf')

# Train the SVM model for 'tavg'
svr_tavg.fit(X_train_tavg_scaled, y_train_tavg)

# Train the SVM model for 'prcp'
svr_prcp.fit(X_train_prcp_scaled, y_train_prcp)


# ## Numerical Metrics

# In[8]:


# Predict 'tavg' using the trained SVR model
y_pred_tavg = svr_tavg.predict(X_test_tavg_scaled)

# Predict 'prcp' using the trained SVR model
y_pred_prcp = svr_prcp.predict(X_test_prcp_scaled)

# Metrics for 'tavg'
tavg_rmse = mean_squared_error(y_test_tavg, y_pred_tavg, squared=False)
tavg_mae = mean_absolute_error(y_test_tavg, y_pred_tavg)
tavg_r2 = r2_score(y_test_tavg, y_pred_tavg)

print("TAVG Metrics:")
print(f"RMSE: {tavg_rmse}")
print(f"MAE: {tavg_mae}")
print(f"R^2: {tavg_r2}")

# Metrics for 'prcp'
prcp_rmse = mean_squared_error(y_test_prcp, y_pred_prcp, squared=False)
prcp_mae = mean_absolute_error(y_test_prcp, y_pred_prcp)
prcp_r2 = r2_score(y_test_prcp, y_pred_prcp)

print("\nPRCP Metrics:")
print(f"RMSE: {prcp_rmse}")
print(f"MAE: {prcp_mae}")
print(f"R^2: {prcp_r2}")


# # In[12]:


# dates_last_month = lastMonth['date']

# X_last_month = lastMonth.drop(columns=['tavg', 'prcp', 'date'])
# y_last_month_tavg = lastMonth['tavg']
# y_last_month_prcp = lastMonth['prcp']

# # Scale the data using the already fitted scalers
# X_last_month_scaled_tavg = scaler_tavg.transform(X_last_month)
# X_last_month_scaled_prcp = scaler_prcp.transform(X_last_month)

# y_pred_last_month_tavg = svr_tavg.predict(X_last_month_scaled_tavg)
# y_pred_last_month_prcp = svr_prcp.predict(X_last_month_scaled_prcp)

# # Create results DataFrame for last month's predictions
# results_last_month = pd.DataFrame({
#     'Date': dates_last_month,
#     'Actual_TAVG': y_last_month_tavg,
#     'Predicted_TAVG': y_pred_last_month_tavg,
#     'Actual_PRCP': y_last_month_prcp,
#     'Predicted_PRCP': y_pred_last_month_prcp
# })

# # Save results for last month
# results_last_month.to_csv('predicted_vs_actual_last_month.csv', index=False)

# print("Results for last month saved to 'predicted_vs_actual_last_month.csv'.")
# print(results_last_month.head())


# # ## Model Performance Visualizations 

# # In[118]:

# # Plot Predicted vs Actual for 'tavg'
# plt.figure(figsize=(10, 5))
# plt.scatter(y_test_tavg, y_pred_tavg, alpha=0.5, color='blue', label='tavg Predictions')
# plt.plot([min(y_test_tavg), max(y_test_tavg)], [min(y_test_tavg), max(y_test_tavg)], color='red', linestyle='--')
# plt.xlabel("Actual TAVG")
# plt.ylabel("Predicted TAVG")
# plt.title("Predicted vs Actual TAVG")
# plt.legend()
# plt.show()

# # Plot Predicted vs Actual for 'prcp'
# plt.figure(figsize=(10, 5))
# plt.scatter(y_test_prcp, y_pred_prcp, alpha=0.5, color='green', label='prcp Predictions')
# plt.plot([min(y_test_prcp), max(y_test_prcp)], [min(y_test_prcp), max(y_test_prcp)], color='red', linestyle='--')
# plt.xlabel("Actual PRCP")
# plt.ylabel("Predicted PRCP")
# plt.title("Predicted vs Actual PRCP")
# plt.legend()
# plt.show()


# # In[119]:


# # Residual plot for 'tavg'
# residuals_tavg = y_test_tavg - y_pred_tavg
# plt.figure(figsize=(10, 5))
# plt.scatter(y_pred_tavg, residuals_tavg, alpha=0.5, color='blue')
# plt.axhline(y=0, color='red', linestyle='--')
# plt.xlabel("Predicted TAVG")
# plt.ylabel("Residuals")
# plt.title("Residuals for TAVG Predictions")
# plt.show()

# # Residual plot for 'prcp'
# residuals_prcp = y_test_prcp - y_pred_prcp
# plt.figure(figsize=(10, 5))
# plt.scatter(y_pred_prcp, residuals_prcp, alpha=0.5, color='green')
# plt.axhline(y=0, color='red', linestyle='--')
# plt.xlabel("Predicted PRCP")
# plt.ylabel("Residuals")
# plt.title("Residuals for PRCP Predictions")
# plt.show()


# # In[3]:





# # In[ ]:




def apply_svm_model(input_data):
    """
    Predicts temperature and precipitation using the pre-trained models.

    Parameters:
    input_data (pd.DataFrame): The input data for prediction.

    Returns:
    pd.DataFrame: A DataFrame containing the predicted temperature and precipitation.
    """
    # Ensure input data has the same features as the training data
    features = ['tmin', 'tmax', 'snow', 'wspd', 'pres', 'wdir']
    input_data = input_data[features]
    
    # Make predictions
    predicted_tavg = svr_tavg.predict(input_data)
    predicted_prcp = svr_prcp.predict(input_data)
    
    # Create a DataFrame with the predictions
    predictions = pd.DataFrame({
        'Predicted Temperature': predicted_tavg,
        'Predicted Precipitation': predicted_prcp
    })
    
    return predictions