from app.core.tools import tool
import requests
from requests import HTTPError


@tool
def geocode(city_name: str):
    "Geocodes a city name into latitude and longitude data"
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=10&language=en&format=json"

    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"Other error occurred: {err}")
    else:
        return response.json()


@tool
def weather(latitude: float, longitude: float):
    "Returns the weather conditions for a given latitude and longitude"
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,is_day,precipitation,rain,showers,snowfall&timezone=America%2FChicago"

    try:
        response = requests.get(url)
        response.raise_for_status()
    except HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"Other error occurred: {err}")
    else:
        return response.json()
