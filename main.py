import requests
import pandas as pd
import json
import matplotlib.pyplot as plt
from datetime import datetime
from meteostat import Point, Daily
import linearAndRF
import DecTree
import KNN_model
import ridgeRegression
import weatherSVM
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QLabel, QWidget, QTextEdit
)
import sys

# Example usage
params = {
    "latitude": 41.871968766034165,
    "longitude": -87.6479624107713,
    "hourly": "temperature_2m,precipitation,wind_speed_10m",
    "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
    "temperature_unit": "celsius",
    "wind_speed_unit": "kmh",
    "precipitation_unit": "mm",
    "timezone": "auto",
    "forecast_days": 7
}

def open_meteo_api_data_reading(params):
    """
    Fetches and organizes weather data from the Open-Meteo API.

    Args:
        params (dict): Dictionary of parameters for the API request.

    Returns:
        dict: Structured forecast data including hourly and daily information, or None if the request fails.
    """
    # Base URL for the Open-Meteo API
    url = "https://api.open-meteo.com/v1/forecast"

    # Make the request to the API
    response = requests.get(url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse JSON response
        data = response.json()

        # Organize the data
        forecast_data = {
            "hourly": data.get("hourly", {}),
            "daily": data.get("daily", {}),
            "location": {
                "latitude": params.get("latitude"),
                "longitude": params.get("longitude"),
                "timezone": params.get("timezone", "GMT")  # Default timezone if not specified
            },
            "units": {
                "temperature": params.get("temperature_unit", "celsius"),
                "wind_speed": params.get("wind_speed_unit", "kmh"),
                "precipitation": params.get("precipitation_unit", "mm")
            }
        }
        return forecast_data
    else:
        print(f"Request failed with status code {response.status_code}")
        return None
    
def graph_predictions():
    # Read actual data
    actual_data = pd.read_csv('/Users/changwonchoi/workspace/CS412/412FinalProject/lastMonth.csv')
    actual_tavg = actual_data['tavg']
    actual_prcp = actual_data['prcp']

    # Read predicted data
    lr_rf_predictions = pd.read_csv('/Users/changwonchoi/workspace/CS412/412FinalProject/LR_and_RF_predictions.csv')
    svm_predictions = pd.read_csv('/Users/changwonchoi/workspace/CS412/412FinalProject/SVMpredictions.csv')
    rr_predictions = pd.read_csv('/Users/changwonchoi/workspace/CS412/412FinalProject/Weather_Predictions_RR.csv')
    dt_predictions = pd.read_csv('/Users/changwonchoi/workspace/CS412/412FinalProject/last_month_predictions.csv')
    knn_predictions = pd.read_csv('/Users/changwonchoi/workspace/CS412/412FinalProject/KNN_predictions.csv')

    # Extract predicted values
    lr_tavg_pred = lr_rf_predictions['lr_tavg_pred']
    lr_prcp_pred = lr_rf_predictions['lr_prcp_pred']
    rf_tavg_pred = lr_rf_predictions['rf_tavg_pred']
    rf_prcp_pred = lr_rf_predictions['rf_prcp_pred']
    svm_tavg_pred = svm_predictions['Predicted_TAVG']
    svm_prcp_pred = svm_predictions['Predicted_PRCP']
    rr_tavg_pred = rr_predictions['Predicted_Tavg']
    rr_prcp_pred = rr_predictions['Predicted_Prcp']
    dt_tavg_pred = dt_predictions['Predicted Temperature']
    dt_prcp_pred = dt_predictions['Predicted Precipitation']
    knn_tavg_pred = knn_predictions['knn_tavg_pred']
    knn_prcp_pred = knn_predictions['knn_prcp_pred']

    # Plotting the data
    plt.figure(figsize=(14, 10))

    # Temperature comparison
    plt.subplot(2, 1, 1)
    plt.plot(actual_tavg, label='Actual TAVG', color='black')
    plt.plot(lr_tavg_pred, label='LR TAVG Pred', linestyle='--')
    plt.plot(rf_tavg_pred, label='RF TAVG Pred', linestyle='--')
    plt.plot(svm_tavg_pred, label='SVM TAVG Pred', linestyle='--')
    plt.plot(rr_tavg_pred, label='RR TAVG Pred', linestyle='--')
    plt.plot(dt_tavg_pred, label='DT TAVG Pred', linestyle='--')
    plt.plot(knn_tavg_pred, label='KNN TAVG Pred', linestyle='--')
    plt.title('Temperature Comparison')
    plt.xlabel('Day')
    plt.ylabel('Temperature (°C)')
    plt.legend()

    # Precipitation comparison
    plt.subplot(2, 1, 2)
    plt.plot(actual_prcp, label='Actual PRCP', color='black')
    plt.plot(lr_prcp_pred, label='LR PRCP Pred', linestyle='--')
    plt.plot(rf_prcp_pred, label='RF PRCP Pred', linestyle='--')
    plt.plot(svm_prcp_pred, label='SVM PRCP Pred', linestyle='--')
    plt.plot(rr_prcp_pred, label='RR PRCP Pred', linestyle='--')
    plt.plot(dt_prcp_pred, label='DT PRCP Pred', linestyle='--')
    plt.plot(knn_prcp_pred, label='KNN PRCP Pred', linestyle='--')
    plt.title('Precipitation Comparison')
    plt.xlabel('Day')
    plt.ylabel('Precipitation (mm)')
    plt.legend()

    plt.tight_layout()
    plt.show()

def get_meteostat_data(start, end, location):
    data = Daily(location, start, end)
    data = data.fetch()

    return data

def compare_models(input_data):
    linear_regression_results = linearAndRF.apply_linear_regression_model(input_data.head(1))
    random_forest_results = linearAndRF.apply_random_forest_model(input_data.head(1))
    decision_tree_results = DecTree.apply_decision_tree_model(input_data.head(1))
    knn_results = KNN_model.apply_knn_model(input_data.head(1))
    rr_results = ridgeRegression.apply_rr_model(input_data.head(1))
    svm_results = weatherSVM.apply_svm_model(input_data.head(1))

    results = {
        "Linear Regression": linear_regression_results,
        "Random Forest": random_forest_results,
        "Decision Tree": decision_tree_results,
        "KNN": knn_results,
        "Ridge Regression": rr_results,
        "SVM": svm_results,
        "Actual Data": pd.DataFrame({
            "Actual Temperature": [input_data['tavg'].iloc[1]],
            "Actual Precipitation": [input_data['prcp'].iloc[1]]
        })
    }
    return results

class ModelComparisonApp(QMainWindow):
    def __init__(self, input_data):
        super().__init__()
        self.input_data = input_data
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Weather Prediction Model Comparison")
        
        # Set the initial size of the window
        self.resize(800, 600)  # Width: 800, Height: 600

        # First screen: A button to start the comparison
        self.start_button = QPushButton("Start Comparison")
        self.start_button.setStyleSheet("font-size: 18px; padding: 10px;")
        self.start_button.clicked.connect(self.show_results)

        # Display input data
        input_data_label = QLabel("Input Data (Last 2 Days):")
        input_data_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        input_data_text = QTextEdit()
        input_data_text.setReadOnly(True)
        input_data_text.setText(self.input_data.to_string())
        input_data_text.setStyleSheet("font-size: 14px;")

        self.first_screen_layout = QVBoxLayout()
        self.first_screen_layout.addWidget(QLabel("Weather Prediction Model Comparison", self))
        self.first_screen_layout.addWidget(input_data_label)
        self.first_screen_layout.addWidget(input_data_text)
        self.first_screen_layout.addWidget(self.start_button)

        self.first_screen_widget = QWidget()
        self.first_screen_widget.setLayout(self.first_screen_layout)
        self.setCentralWidget(self.first_screen_widget)

    def show_results(self):
        # Run the comparison
        results = compare_models(self.input_data)

        # Create second screen
        results_widget = QWidget()
        results_layout = QVBoxLayout()

        # Display results with styling
        results_text = QTextEdit()
        results_text.setReadOnly(True)
        
        # Format the results with some styling
        formatted_results = "<h2>Model Comparison Results</h2>"
        for key, value in results.items():
            tavg_pred_text = value.keys()[0]
            tavg_pred_value = f"{value.loc[0][0]:.2f}"
            prcp_pred_text = value.keys()[1]
            prcp_pred_value = f"{value.loc[0][1]:.2f}"
            formatted_results += f"<h3 style='color: blue;'>{key}</h3>"
            formatted_results += f"<p><b>{tavg_pred_text}:</b> <span style='color: green;'>{tavg_pred_value}</span></p>"
            formatted_results += f"<p><b>{prcp_pred_text}:</b> <span style='color: green;'>{prcp_pred_value}</span></p>"
        
        
        results_text.setHtml(formatted_results)

        results_layout.addWidget(results_text)

        # Add an exit button
        exit_button = QPushButton("Exit")
        exit_button.clicked.connect(self.close)
        results_layout.addWidget(exit_button)

        results_widget.setLayout(results_layout)
        self.setCentralWidget(results_widget)

if __name__ == "__main__":
    # # Call the function and store the result
    # forecast_data = open_meteo_api_data_reading(params)

    # # Print the structured data for easy viewing if the request was successful
    # if forecast_data:
    #     print(json.dumps(forecast_data, indent=2))

    # graph_predictions()

    start = datetime.now() - pd.Timedelta(days=2)
    end = datetime.now()
    location = Point(41.85, -87.65, 180)

    input_data = get_meteostat_data(start, end, location)
    
    app = QApplication(sys.argv)
    window = ModelComparisonApp(input_data)
    window.show()
    sys.exit(app.exec_())