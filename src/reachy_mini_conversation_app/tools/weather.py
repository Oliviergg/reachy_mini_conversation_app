import logging
from typing import Any, Dict

import httpx

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


# Antony, France
ANTONY_LATITUDE = 48.7539
ANTONY_LONGITUDE = 2.2989
ANTONY_CITY_NAME = "Antony, France"

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# WMO weather codes (https://open-meteo.com/en/docs)
WEATHER_CODE_DESCRIPTIONS = {
    0: "ciel dégagé",
    1: "principalement clair",
    2: "partiellement nuageux",
    3: "couvert",
    45: "brouillard",
    48: "brouillard givrant",
    51: "bruine légère",
    53: "bruine modérée",
    55: "bruine dense",
    56: "bruine verglaçante légère",
    57: "bruine verglaçante dense",
    61: "pluie légère",
    63: "pluie modérée",
    65: "pluie forte",
    66: "pluie verglaçante légère",
    67: "pluie verglaçante forte",
    71: "neige légère",
    73: "neige modérée",
    75: "neige forte",
    77: "grains de neige",
    80: "averses légères",
    81: "averses modérées",
    82: "averses violentes",
    85: "averses de neige légères",
    86: "averses de neige fortes",
    95: "orage",
    96: "orage avec grêle légère",
    99: "orage avec grêle forte",
}


class Weather(Tool):
    """Get the current weather for Antony, France."""

    name = "weather"
    description = "Get the current weather conditions for Antony, France (temperature, conditions, wind, humidity)."
    parameters_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Fetch current weather from Open-Meteo for Antony, France."""
        logger.info("Tool call: weather city=%s", ANTONY_CITY_NAME)

        params = {
            "latitude": ANTONY_LATITUDE,
            "longitude": ANTONY_LONGITUDE,
            "current": "temperature_2m,weather_code,wind_speed_10m,relative_humidity_2m",
            "timezone": "Europe/Paris",
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(OPEN_METEO_URL, params=params)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as e:
            logger.exception("weather: HTTP error")
            return {"error": f"weather API error: {e}"}

        current = data.get("current") or {}
        weather_code = current.get("weather_code")
        return {
            "city": ANTONY_CITY_NAME,
            "temperature_celsius": current.get("temperature_2m"),
            "conditions": WEATHER_CODE_DESCRIPTIONS.get(weather_code, f"code météo {weather_code}"),
            "wind_speed_kmh": current.get("wind_speed_10m"),
            "humidity_percent": current.get("relative_humidity_2m"),
            "observed_at": current.get("time"),
        }
