import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('export.csv', na_values=['', ' ', 'NA', 'NaN'])

# Convert 'date' column to datetime type
data['date'] = pd.to_datetime(data['date'])

# Create target variables by shifting 'tavg' and 'prcp' to get next day's values
data['tavg_next_day'] = data['tavg'].shift(-1)
data['prcp_next_day'] = data['prcp'].shift(-1)

# Drop the last row as it will have NaN in the target variables due to shifting
data.dropna(subset=['tavg_next_day', 'prcp_next_day'], inplace=True)

# Features to use in the model
features = ['tavg', 'tmin', 'tmax', 'prcp', 'snow', 'wspd', 'pres']

# Feature Engineering: Add lagged features and rolling statistics
data['tavg_lag1'] = data['tavg'].shift(1)
data['prcp_lag1'] = data['prcp'].shift(1)
data['tavg_roll_mean'] = data['tavg'].rolling(window=7).mean()
data['prcp_roll_sum'] = data['prcp'].rolling(window=3).sum()

# Update features list
features += ['tavg_lag1', 'prcp_lag1', 'tavg_roll_mean', 'prcp_roll_sum']

# Handle missing values in features
data[features] = data[features].fillna(data[features].mean())

# Convert features to numeric
for feature in features:
    data[feature] = pd.to_numeric(data[feature], errors='coerce')

# Feature matrix and target variables
X = data[features]
y_tavg = data['tavg_next_day']
y_prcp = data['prcp_next_day']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train_tavg, X_test_tavg, y_train_tavg, y_test_tavg = train_test_split(
    X_scaled, y_tavg, test_size=0.1, random_state=42)

X_train_prcp, X_test_prcp, y_train_prcp, y_test_prcp = train_test_split(
    X_scaled, y_prcp, test_size=0.1, random_state=42)

# Set manual parameters for KNN models
knn_tavg = KNeighborsRegressor(n_neighbors=19, weights='distance', metric='manhattan')
knn_prcp = KNeighborsRegressor(n_neighbors=20, weights='uniform', metric='chebyshev')

# Train the models
knn_tavg.fit(X_train_tavg, y_train_tavg)
knn_prcp.fit(X_train_prcp, y_train_prcp)

# Predict on test set
y_pred_knn_tavg = knn_tavg.predict(X_test_tavg)
y_pred_knn_prcp = knn_prcp.predict(X_test_prcp)

# Evaluate the models on the test set
print('\nTemperature Prediction - KNN')
print('MSE:', mean_squared_error(y_test_tavg, y_pred_knn_tavg))
print('MAE:', mean_absolute_error(y_test_tavg, y_pred_knn_tavg))
print('R^2 Score:', r2_score(y_test_tavg, y_pred_knn_tavg))

print('\nPrecipitation Prediction - KNN')
print('MSE:', mean_squared_error(y_test_prcp, y_pred_knn_prcp))
print('MAE:', mean_absolute_error(y_test_prcp, y_pred_knn_prcp))
print('R^2 Score:', r2_score(y_test_prcp, y_pred_knn_prcp))

# Read and preprocess last month's data
last_month_data = pd.read_csv('lastMonth.csv', na_values=['', ' ', 'NA', 'NaN'])
last_month_data['date'] = pd.to_datetime(last_month_data['date'])

# Recreate lagged features and rolling statistics for last_month_data
last_month_data['tavg_lag1'] = last_month_data['tavg'].shift(1)
last_month_data['prcp_lag1'] = last_month_data['prcp'].shift(1)
last_month_data['tavg_roll_mean'] = last_month_data['tavg'].rolling(window=3).mean()
last_month_data['prcp_roll_sum'] = last_month_data['prcp'].rolling(window=3).sum()

# Handle missing values in new features
for feature in features:
    last_month_data[feature] = pd.to_numeric(last_month_data[feature], errors='coerce')
last_month_data[features] = last_month_data[features].fillna(last_month_data[features].mean())

# Scale the features
X_last = scaler.transform(last_month_data[features])

# Predict next day's values
last_month_data['knn_tavg_pred'] = knn_tavg.predict(X_last)
last_month_data['knn_prcp_pred'] = knn_prcp.predict(X_last)

# Create results DataFrame
results = last_month_data[['date', 'knn_tavg_pred', 'knn_prcp_pred']]

# Write results to CSV
results.to_csv('KNN_predictions.csv', index=False)

# Plot the results with clean, smooth lines
plt.figure(figsize=(14, 8))

# Subplot for Temperature
plt.subplot(2, 1, 1)
plt.plot(results['date'], last_month_data['tavg'], label='Actual Temperature', color='blue', linestyle='-')
plt.plot(results['date'], results['knn_tavg_pred'], label='Predicted Temperature', color='orange', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Average Temperature (°C)')
plt.title('Actual vs Predicted Avg. Temperature (KNN Model)')
plt.grid(visible=True, linestyle=':', linewidth=0.5)
plt.legend()

# Subplot for Precipitation
plt.subplot(2, 1, 2)
plt.plot(results['date'], last_month_data['prcp'], label='Actual Precipitation', color='blue', linestyle='-')
plt.plot(results['date'], results['knn_prcp_pred'], label='Predicted Precipitation', color='orange', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Precipitation (mm)')
plt.title('Actual vs Predicted Precipitation (KNN Model)')
plt.grid(visible=True, linestyle=':', linewidth=0.5)
plt.legend()

# Adjust layout and display the plot
plt.tight_layout()
plt.show()
