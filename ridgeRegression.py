import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np


# Read in the data
data = pd.read_csv('export.csv')
lastMonthData = pd.read_csv('lastMonth.csv')

# Drop columns with NaN values
whatWeAreDroping = ['date','wdir', 'wpgt', 'tsun']
whatWeAreDropingForLastMonth = ['date','wpgt', 'tsun']

# Drop columns with NaN values in the data
data.drop(whatWeAreDroping, inplace=True, axis=1)
lastMonthData.drop(whatWeAreDropingForLastMonth, inplace=True, axis=1)

# Fill NaN values with the mean of the column
data.fillna(method='ffill', inplace=True)
lastMonthData.fillna(method='ffill', inplace=True)

# Save the data to new CSV files
data.to_csv('newExport.csv', index=False)
lastMonthData.to_csv('newLastMonth.csv', index=False)

# Separate features and target variables for tavg (temperature) and prcp (precipitation)
X = data.drop(columns=['tavg', 'prcp'])  # Remove targets from features
y_tavg = data['tavg']  # Target variable: average temperature
y_prcp = data['prcp']  # Target variable: precipitation

X_last_month = lastMonthData.drop(columns=['tavg', 'prcp'])  # Remove targets from features for last month

# Split the data into training and testing sets for tavg
X_train_tavg, X_test_tavg, y_train_tavg, y_test_tavg = train_test_split(X, y_tavg, test_size=0.2, random_state=42)

# Split the data into training and testing sets for prcp
X_train_prcp, X_test_prcp, y_train_prcp, y_test_prcp = train_test_split(X, y_prcp, test_size=0.2, random_state=42)

# Initialize Ridge Regression models for tavg and prcp
ridge_tavg = Ridge(alpha=1.0)  # Model for tavg
ridge_prcp = Ridge(alpha=1.0)  # Model for prcp

# Fit the models to the training data
ridge_tavg.fit(X_train_tavg, y_train_tavg)
ridge_prcp.fit(X_train_prcp, y_train_prcp)

# Predict on the test set for tavg and prcp
y_pred_tavg = ridge_tavg.predict(X_test_tavg)
y_pred_prcp = ridge_prcp.predict(X_test_prcp)

# Evaluate the models using Mean Squared Error
mse_tavg = mean_squared_error(y_test_tavg, y_pred_tavg)
mse_prcp = mean_squared_error(y_test_prcp, y_pred_prcp)
print(f"Mean Squared Error for Average Temperature (tavg): {mse_tavg}")
print(f"Mean Squared Error for Precipitation (prcp): {mse_prcp}")

# # Ensure matching columns between training and last month's data
# X_last_month = X_last_month.reindex(columns=X.columns, fill_value=0)

# # Predict on the last month's data
# y_last_month_pred_tavg = ridge_tavg.predict(X_last_month)
# y_last_month_pred_prcp = ridge_prcp.predict(X_last_month)

# # Set negative values for precipitation predictions to zero
# y_last_month_pred_prcp = np.maximum(y_last_month_pred_prcp, 0)

# # Add predictions as new columns to the last month's data
# lastMonthData['Predicted_Tavg'] = y_last_month_pred_tavg
# lastMonthData['Predicted_Prcp'] = y_last_month_pred_prcp

# # Plotting the results for tavg (average temperature)
# plt.figure(figsize=(12, 6))

# # Plot the actual vs predicted temperature for the test set
# plt.subplot(1, 2, 1)
# plt.scatter(y_test_tavg, y_pred_tavg, color='blue', label='Predicted vs Actual')
# plt.plot([min(y_test_tavg), max(y_test_tavg)], [min(y_test_tavg), max(y_test_tavg)], color='red', linestyle='--')
# plt.title('Tavg (Temperature) - Actual vs Predicted')
# plt.xlabel('Actual Temperature (°C)')
# plt.ylabel('Predicted Temperature (°C)')
# plt.legend()

# # Plot the actual vs predicted precipitation for the test set
# plt.subplot(1, 2, 2)
# plt.scatter(y_test_prcp, y_pred_prcp, color='blue', label='Predicted vs Actual')
# plt.plot([min(y_test_prcp), max(y_test_prcp)], [min(y_test_prcp), max(y_test_prcp)], color='red', linestyle='--')
# plt.title('Prcp (Precipitation) - Actual vs Predicted')
# plt.xlabel('Actual Precipitation (mm)')
# plt.ylabel('Predicted Precipitation (mm)')
# plt.legend()

# plt.tight_layout()
# plt.show()

# # Check if actual values exist in the last month's data
# if 'tavg' in lastMonthData.columns and 'prcp' in lastMonthData.columns:
#     # Print the actual and predicted values
#     print("\nOct. 17 2024 - Nov. 17 2024 Data with Actual and Predicted Values for tavg and prcp:")
#     print(lastMonthData[['tavg', 'prcp', 'Predicted_Tavg', 'Predicted_Prcp']])
# else:
#     print("\nActual values for tavg and prcp are not available in lastMonthData. Showing predictions only:")
#     print(lastMonthData[['Predicted_Tavg', 'Predicted_Prcp']])

def apply_rr_model(input_data):
    """
    Predicts temperature and precipitation using the pre-trained models.

    Parameters:
    input_data (pd.DataFrame): The input data for prediction.

    Returns:
    pd.DataFrame: A DataFrame containing the predicted temperature and precipitation.
    """
    # Ensure input data has the same features as the training data
    features = ['tmin', 'tmax', 'snow', 'wspd', 'pres']
    input_data = input_data[features]
    
    # Make predictions
    predicted_tavg = ridge_tavg.predict(input_data)
    predicted_prcp = ridge_prcp.predict(input_data)
    
    # Create a DataFrame with the predictions
    predictions = pd.DataFrame({
        'Predicted Temperature': predicted_tavg,
        'Predicted Precipitation': predicted_prcp
    })
    
    return predictions