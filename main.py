import requests
import json

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

# Call the function and store the result
forecast_data = open_meteo_api_data_reading(params)

# Print the structured data for easy viewing if the request was successful
if forecast_data:
    print(json.dumps(forecast_data, indent=2))