import pandas as pd  
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Read in our data.
data = pd.read_csv('export.csv')
# Will be used to test our model.
lastMonth = pd.read_csv('lastMonth.csv')

# Drop columns with null values and dates that cannot be used directly.
columns_to_drop = ['wpgt', 'tsun', 'date']
# Drop the date and null columns in the training set.
data.drop(columns=columns_to_drop, inplace=True)
# Drop the date and null columns in the testing set.
lastMonth.drop(columns=columns_to_drop, inplace=True)

# Fill NaN values.
data.fillna(method='ffill', inplace=True)
lastMonth.fillna(method='ffill', inplace=True)

# Print cleaned data as a check.
print("\nCSV Data after cleaning and converting non-numeric columns:\n")
print(lastMonth.head(5))
print("\n")

# Save this new cleaned data to a file.
data.to_csv('cleaned_export.csv', index=False)
lastMonth.to_csv('cleaned_month.csv', index=False)

# Seperate the target features from the training data.
X = data.drop(columns=['tavg', 'prcp'])
# Average temperature target.
y_tavg = data['tavg']
# Precipitation target.
y_prcp = data['prcp']

# Split training data into training and validation sets (e.g., 80% training, 20% validation).
X_train, X_val, y_train_tavg, y_val_tavg = train_test_split(X, y_tavg, test_size=0.2, random_state=42)
_, _, y_train_prcp, y_val_prcp = train_test_split(X, y_prcp, test_size=0.2, random_state=42)

# Create models for temperature and precipitation.
model_tavg = DecisionTreeRegressor(random_state=42)
model_prcp = DecisionTreeRegressor(random_state=42)

# Train the models on the data.
model_tavg.fit(X_train, y_train_tavg)
model_prcp.fit(X_train, y_train_prcp)

# Evaluate on the validation set for temperature.
y_val_pred_tavg = model_tavg.predict(X_val)
val_mse_tavg = mean_squared_error(y_val_tavg, y_val_pred_tavg)
val_r2_tavg = r2_score(y_val_tavg, y_val_pred_tavg)

# Evaluate on the validation set for precipitation.
y_val_pred_prcp = model_prcp.predict(X_val)
val_mse_prcp = mean_squared_error(y_val_prcp, y_val_pred_prcp)
val_r2_prcp = r2_score(y_val_prcp, y_val_pred_prcp)

# Print out the results to view it.
print(f"Validation Mean Squared Error for tavg: {val_mse_tavg}")
print(f"Validation R^2 Score for tavg: {val_r2_tavg}")
print(f"Validation Mean Squared Error for prcp: {val_mse_prcp}")
print(f"Validation R^2 Score for prcp: {val_r2_prcp}")

# Evaluate on the test set (last month's data).
X_test = lastMonth.drop(columns=['tavg', 'prcp'])
# Actual average temperature values.
y_test_tavg = lastMonth['tavg']
# Actual precipitation values.
y_test_prcp = lastMonth['prcp']

# Make predictions for last month.
y_test_pred_tavg = model_tavg.predict(X_test)
y_test_pred_prcp = model_prcp.predict(X_test)

# Calculate metrics for last month's predictions on temperature.
test_mse_tavg = mean_squared_error(y_test_tavg, y_test_pred_tavg)
test_r2_tavg = r2_score(y_test_tavg, y_test_pred_tavg)

# Calculate metrics for last month's predictions on precipitation.
test_mse_prcp = mean_squared_error(y_test_prcp, y_test_pred_prcp)
test_r2_prcp = r2_score(y_test_prcp, y_test_pred_prcp)

# Print out the results to view it.
print(f"Test Mean Squared Error for tavg: {test_mse_tavg}")
print(f"Test R^2 Score for tavg: {test_r2_tavg}")
print(f"Test Mean Squared Error for prcp: {test_mse_prcp}")
print(f"Test R^2 Score for prcp: {test_r2_prcp}")

# Create a DataFrame to display predictions and actual values for last month
results = pd.DataFrame({
    'Actual Temperature': y_test_tavg,
    'Predicted Temperature': y_test_pred_tavg,
    'Actual Precipitation': y_test_prcp,
    'Predicted Precipitation': y_test_pred_prcp
})

# Print the results
print("\nPredicted vs Actual Values for Last Month:\n")
print(results)

# Plot Predicted vs Actual for Temperature (tavg)
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_test_tavg, y_test_pred_tavg, color='blue', label='Predicted vs Actual')
plt.plot([y_test_tavg.min(), y_test_tavg.max()], [y_test_tavg.min(), y_test_tavg.max()], color='red', linewidth=2)
plt.title("Temperature: Predicted vs Actual")
plt.xlabel("Actual Temperature")
plt.ylabel("Predicted Temperature")
plt.legend()

# Plot Predicted vs Actual for Precipitation (prcp)
plt.subplot(1, 2, 2)
plt.scatter(y_test_prcp, y_test_pred_prcp, color='green', label='Predicted vs Actual')
plt.plot([y_test_prcp.min(), y_test_prcp.max()], [y_test_prcp.min(), y_test_prcp.max()], color='red', linewidth=2)
plt.title("Precipitation: Predicted vs Actual")
plt.xlabel("Actual Precipitation")
plt.ylabel("Predicted Precipitation")
plt.legend()

plt.tight_layout()
plt.show()