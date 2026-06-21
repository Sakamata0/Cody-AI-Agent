"""
Weather API Tool using Open-Meteo (free, no API key required).

Fetches current weather data for a given city using:
1. Open-Meteo Geocoding API to convert city name → coordinates
2. Open-Meteo Forecast API to get current weather
"""

import requests
from langchain_core.tools import tool


@tool
def weather_tool(city: str) -> str:
    """
    Get the current weather for a city.
    Use this tool when the user asks about weather, temperature, or
    atmospheric conditions for a specific location.
    Input should be a city name (e.g., "Paris", "New York", "Tokyo").
    """
    try:
        # Step 1: Geocode city name to latitude/longitude.
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"
        geo_response = requests.get(geo_url, params={"name": city, "count": 1}, timeout=10)
        geo_response.raise_for_status()
        geo_data = geo_response.json()

        if not geo_data.get("results"):
            return f"City '{city}' not found. Please check the spelling."

        location = geo_data["results"][0]
        lat = location["latitude"]
        lon = location["longitude"]
        resolved_name = location.get("name", city)
        country = location.get("country", "")

        # Step 2: Fetch current weather using coordinates.
        weather_url = "https://api.open-meteo.com/v1/forecast"
        weather_params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
            "timezone": "auto",
        }
        weather_response = requests.get(weather_url, params=weather_params, timeout=10)
        weather_response.raise_for_status()
        weather_data = weather_response.json()

        current = weather_data["current"]
        temp = current["temperature_2m"]
        humidity = current["relative_humidity_2m"]
        wind = current["wind_speed_10m"]
        code = current["weather_code"]

        # Map weather codes to descriptions.
        weather_descriptions = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Foggy", 48: "Depositing rime fog",
            51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
            61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
            71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
            80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
            95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
        }
        description = weather_descriptions.get(code, f"Weather code {code}")

        return (
            f"Current weather in {resolved_name}, {country}:\n"
            f"  Condition: {description}\n"
            f"  Temperature: {temp}°C\n"
            f"  Humidity: {humidity}%\n"
            f"  Wind Speed: {wind} km/h"
        )

    except requests.RequestException as e:
        return f"Error fetching weather data: {str(e)}"
