import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Read the historical weather data from 'export.csv'
data = pd.read_csv('export.csv', na_values=['', ' ', 'NA', 'NaN'])

# Convert 'date' column to datetime type
data['date'] = pd.to_datetime(data['date'])

# Create target variables by shifting 'tavg' and 'prcp' to get next day's values
data['tavg_next_day'] = data['tavg'].shift(-1)
data['prcp_next_day'] = data['prcp'].shift(-1)

# Drop the last row as it will have NaN in the target variables due to shifting
data.dropna(subset=['tavg_next_day', 'prcp_next_day'], inplace=True)

# Updated list of features to use in the model (removed 'wpgt', 'tsun', and 'wdir')
features = ['tavg', 'tmin', 'tmax', 'prcp', 'snow', 'wspd', 'pres']

# Convert features to numeric data types and coerce errors to NaN
for feature in features:
    data[feature] = pd.to_numeric(data[feature], errors='coerce')

# Check the number of missing values in each feature
print("Number of missing values in each feature before filling:")
print(data[features].isna().sum())

# Fill missing values in features with the mean of each column
data[features] = data[features].fillna(data[features].mean())

# Verify that there are no NaNs left in X
X = data[features]
print("\nNumber of NaNs in X after filling missing values:", X.isna().sum().sum())

# Target variables
y_tavg = data['tavg_next_day']
y_prcp = data['prcp_next_day']

# Ensure there are no NaN values in y
print("Number of NaNs in y_tavg:", y_tavg.isna().sum())
print("Number of NaNs in y_prcp:", y_prcp.isna().sum())

# Split data into training and testing sets for temperature prediction
X_train_tavg, X_test_tavg, y_train_tavg, y_test_tavg = train_test_split(
    X, y_tavg, test_size=0.2, random_state=42)

# Split data into training and testing sets for precipitation prediction
X_train_prcp, X_test_prcp, y_train_prcp, y_test_prcp = train_test_split(
    X, y_prcp, test_size=0.2, random_state=42)

# Create a Linear Regression model for temperature prediction
lr_model_tavg = LinearRegression()
lr_model_tavg.fit(X_train_tavg, y_train_tavg)  # Train the model
y_pred_tavg = lr_model_tavg.predict(X_test_tavg)  # Predict on test set

# Evaluate the Linear Regression model for temperature prediction
mse_tavg = mean_squared_error(y_test_tavg, y_pred_tavg)
mae_tavg = mean_absolute_error(y_test_tavg, y_pred_tavg)
r2_tavg = r2_score(y_test_tavg, y_pred_tavg)

print('\nTemperature Prediction - Linear Regression')
print('MSE:', mse_tavg)
print('MAE:', mae_tavg)
print('R^2 Score:', r2_tavg)

# Create a Linear Regression model for precipitation prediction
lr_model_prcp = LinearRegression()
lr_model_prcp.fit(X_train_prcp, y_train_prcp)  # Train the model
y_pred_prcp = lr_model_prcp.predict(X_test_prcp)  # Predict on test set

# Evaluate the Linear Regression model for precipitation prediction
mse_prcp = mean_squared_error(y_test_prcp, y_pred_prcp)
mae_prcp = mean_absolute_error(y_test_prcp, y_pred_prcp)
r2_prcp = r2_score(y_test_prcp, y_pred_prcp)

print('\nPrecipitation Prediction - Linear Regression')
print('MSE:', mse_prcp)
print('MAE:', mae_prcp)
print('R^2 Score:', r2_prcp)

# Create a Random Forest Regressor for temperature prediction
rf_model_tavg = RandomForestRegressor(random_state=42)
rf_model_tavg.fit(X_train_tavg, y_train_tavg)  # Train the model
y_pred_rf_tavg = rf_model_tavg.predict(X_test_tavg)  # Predict on test set

# Evaluate the Random Forest model for temperature prediction
mse_rf_tavg = mean_squared_error(y_test_tavg, y_pred_rf_tavg)
mae_rf_tavg = mean_absolute_error(y_test_tavg, y_pred_rf_tavg)
r2_rf_tavg = r2_score(y_test_tavg, y_pred_rf_tavg)

print('\nTemperature Prediction - Random Forest')
print('MSE:', mse_rf_tavg)
print('MAE:', mae_rf_tavg)
print('R^2 Score:', r2_rf_tavg)

# Create a Random Forest Regressor for precipitation prediction
rf_model_prcp = RandomForestRegressor(random_state=42)
rf_model_prcp.fit(X_train_prcp, y_train_prcp)  # Train the model
y_pred_rf_prcp = rf_model_prcp.predict(X_test_prcp)  # Predict on test set

# Evaluate the Random Forest model for precipitation prediction
mse_rf_prcp = mean_squared_error(y_test_prcp, y_pred_rf_prcp)
mae_rf_prcp = mean_absolute_error(y_test_prcp, y_pred_rf_prcp)
r2_rf_prcp = r2_score(y_test_prcp, y_pred_rf_prcp)

print('\nPrecipitation Prediction - Random Forest')
print('MSE:', mse_rf_prcp)
print('MAE:', mae_rf_prcp)
print('R^2 Score:', r2_rf_prcp)

# # Define parameter grid for hyperparameter tuning of Random Forest
# param_grid = {
#     'n_estimators': [50, 100, 200],     # Number of trees
#     'max_depth': [None, 10, 20],        # Maximum depth of trees
#     'min_samples_split': [2, 5],        # Minimum number of samples required to split
# }

# # Read last month's weather data from 'lastMonth.csv' with proper NaN handling
# last_month_data = pd.read_csv('lastMonth.csv', na_values=['', ' ', 'NA', 'NaN'])

# # Convert 'date' column to datetime type
# last_month_data['date'] = pd.to_datetime(last_month_data['date'])

# # Convert features to numeric data types and coerce errors to NaN
# for feature in features:
#     last_month_data[feature] = pd.to_numeric(last_month_data[feature], errors='coerce')

# # Check the number of missing values in each feature
# print("\nNumber of missing values in last_month_data before filling:")
# print(last_month_data[features].isna().sum())

# # Fill missing values in last month's data
# last_month_data[features] = last_month_data[features].fillna(last_month_data[features].mean())

# # Create actual next day's 'tavg' and 'prcp' in last month's data by shifting
# last_month_data['tavg_next_day'] = last_month_data['tavg'].shift(-1)
# last_month_data['prcp_next_day'] = last_month_data['prcp'].shift(-1)

# # Feature matrix for last month's data
# X_last = last_month_data[features]

# # Predict next day's average temperature using the linear regression model
# last_month_data['lr_tavg_pred'] = lr_model_tavg.predict(X_last)

# # Predict next day's precipitation using the linear regression model
# last_month_data['lr_prcp_pred'] = lr_model_prcp.predict(X_last)

# # Predict next day's average temperature using the rain forest model
# last_month_data['rf_tavg_pred'] = rf_model_tavg.predict(X_last)

# # Predict next day's precipitation using the rain forest model
# last_month_data['rf_prcp_pred'] = rf_model_prcp.predict(X_last)

# # Create a new DataFrame with predictions and actual values
# results = last_month_data[['date', 'tavg_next_day', 'lr_tavg_pred', 'rf_tavg_pred', 'prcp_next_day', 'lr_prcp_pred', 'rf_prcp_pred']]

# # Drop rows with NaN values
# results.dropna(subset=['tavg_next_day', 'prcp_next_day'], inplace=True)

# # Write the results to a new CSV file 'predictions.csv'
# results.to_csv('LR_and_RF_predictions.csv', index=False)

# # Plot the actual and predicted average temperature
# plt.figure(figsize=(14, 7))

# plt.subplot(2, 1, 1)
# plt.plot(results['date'], results['tavg_next_day'], label='Actual Temperature', color='black')
# plt.plot(results['date'], results['lr_tavg_pred'], label='Linear Regression Prediction', color='blue')
# plt.xlabel('Date')
# plt.ylabel('Average Temperature')
# plt.title('Actual vs Predicted Avg. Temperature')
# plt.legend()

# plt.subplot(2, 1, 2)
# plt.plot(results['date'], results['prcp_next_day'], label='Actual Precipitation', color='black')
# plt.plot(results['date'], results['lr_prcp_pred'], label='Linear Regression Prediction', color='blue')
# plt.xlabel('Date')
# plt.ylabel('Precipitation')
# plt.title('Actual vs Predicted Precipitation')
# plt.legend()


# # Plot the actual and predicted precipitation
# plt.figure(figsize=(14, 7))
# plt.subplot(2, 1, 1)
# plt.plot(results['date'], results['tavg_next_day'], label='Actual Temperature', color='black')
# plt.plot(results['date'], results['rf_tavg_pred'], label='Random Forest Prediction', color='green')
# plt.xlabel('Date')
# plt.ylabel('Average Temperature')
# plt.title('Actual vs Predicted Avg. Temperature')
# plt.legend()

# # Plot the actual and predicted precipitation
# plt.subplot(2, 1, 2)
# plt.plot(results['date'], results['prcp_next_day'], label='Actual Precipitation', color='black')
# plt.plot(results['date'], results['rf_prcp_pred'], label='Random Forest Prediction', color='green')
# plt.xlabel('Date')
# plt.ylabel('Precipitation')
# plt.title('Actual vs Predicted Precipitation')
# plt.legend()

# plt.tight_layout()
# plt.show()

def apply_linear_regression_model(input_data):
    """
    Apply the linear regression models to the input data and return the predictions.

    Parameters:
    input_data (pd.DataFrame): The input data to predict on.

    Returns:
    pd.DataFrame: The predictions made by the models.
    """
    # Ensure input data has the same features as the training data
    features = ['tavg', 'tmin', 'tmax', 'prcp', 'snow', 'wspd', 'pres']
    input_data = input_data[features]

    # Convert features to numeric data types and coerce errors to NaN
    for feature in features:
        input_data[feature] = pd.to_numeric(input_data[feature], errors='coerce')

    # Fill missing values in features with the mean of each column
    input_data = input_data.fillna(input_data.mean())

    # Predict using the linear regression models
    tavg_predictions = lr_model_tavg.predict(input_data)
    prcp_predictions = lr_model_prcp.predict(input_data)

    # Create a DataFrame with the predictions
    predictions = pd.DataFrame({
        'Predicted Temperature': tavg_predictions,
        'Predicted Precipitation': prcp_predictions
    })

    return predictions

def apply_random_forest_model(input_data):
    """
    Apply the random forest models to the input data and return the predictions.

    Parameters:
    input_data (pd.DataFrame): The input data to predict on.

    Returns:
    pd.DataFrame: The predictions made by the models.
    """
    # Ensure input data has the same features as the training data
    features = ['tavg', 'tmin', 'tmax', 'prcp', 'snow', 'wspd', 'pres']
    input_data = input_data[features]

    # Convert features to numeric data types and coerce errors to NaN
    for feature in features:
        input_data[feature] = pd.to_numeric(input_data[feature], errors='coerce')

    # Fill missing values in features with the mean of each column
    input_data = input_data.fillna(input_data.mean())

    # Predict using the random forest models
    tavg_predictions = rf_model_tavg.predict(input_data)
    prcp_predictions = rf_model_prcp.predict(input_data)

    # Create a DataFrame with the predictions
    predictions = pd.DataFrame({
        'Predicted Temperature': tavg_predictions,
        'Predicted Precipitation': prcp_predictions
    })

    return predictions