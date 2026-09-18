"""
Weather Forecast Tool

Provides get_weather_forecast(city, date_or_month) using Open-Meteo,
a free, keyless weather API. Since Open-Meteo's free forecast endpoint
only covers ~16 days ahead, we use its climate/historical averages
approach for month-level lookups (typical for trip planning, which is
usually weeks or months in advance) via the Open-Meteo Geocoding API
(to resolve city -> lat/lon) plus the Historical Weather API averaged
over past years for that month, as a proxy for "typical" conditions.
"""

import logging
import requests
from datetime import datetime
from typing import Optional

logger = logging.getLogger("tripmate.weather_tool")

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

MONTH_NAME_TO_NUM = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


class WeatherToolError(Exception):
    """Raised when the weather tool cannot produce a result."""


def _geocode_city(city: str) -> Optional[dict]:
    """Resolve a city name to latitude/longitude using Open-Meteo's free geocoder."""
    try:
        resp = requests.get(
            GEOCODING_URL,
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        logger.error("Geocoding request failed for city=%r: %s", city, e)
        raise WeatherToolError(f"Could not reach geocoding service: {e}")

    results = data.get("results")
    if not results:
        logger.warning("No geocoding match found for city=%r", city)
        return None

    top = results[0]
    return {
        "latitude": top["latitude"],
        "longitude": top["longitude"],
        "resolved_name": top.get("name", city),
        "country": top.get("country", ""),
    }


def _parse_month(date_or_month: str) -> int:
    """Extract a month number (1-12) from a free-text date/month string."""
    text = date_or_month.strip().lower()

    for name, num in MONTH_NAME_TO_NUM.items():
        if name in text or name[:3] in text:
            return num

    # Try common numeric date formats, e.g. "2025-12-10" or "12/10/2025"
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y-%m"):
        try:
            return datetime.strptime(text, fmt).month
        except ValueError:
            continue

    raise ValueError(
        f"Could not parse a month from '{date_or_month}'. "
        "Try a month name (e.g. 'December') or a date like '2025-12-10'."
    )


def _fetch_historical_average(lat: float, lon: float, month: int) -> dict:
    """
    Fetch daily min/max temps for that month across the last 3 past years
    and average them, as a proxy for 'typical' conditions in that month.
    """
    current_year = datetime.utcnow().year
    highs, lows, precip_days = [], [], []

    for year_offset in range(1, 4):  # last 3 full years
        year = current_year - year_offset
        start = f"{year}-{month:02d}-01"
        end_day = 28 if month == 2 else 30
        end = f"{year}-{month:02d}-{end_day}"

        try:
            resp = requests.get(
                ARCHIVE_URL,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "start_date": start,
                    "end_date": end,
                    "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                    "timezone": "auto",
                },
                timeout=10,
            )
            resp.raise_for_status()
            daily = resp.json().get("daily", {})
        except requests.exceptions.RequestException as e:
            logger.warning("Historical fetch failed for year=%d: %s", year, e)
            continue

        highs.extend([t for t in daily.get("temperature_2m_max", []) if t is not None])
        lows.extend([t for t in daily.get("temperature_2m_min", []) if t is not None])
        precip_days.extend(daily.get("precipitation_sum", []))

    if not highs or not lows:
        raise WeatherToolError(
            "No historical weather data available for this location/month."
        )

    avg_high = round(sum(highs) / len(highs), 1)
    avg_low = round(sum(lows) / len(lows), 1)
    rainy_days = sum(1 for p in precip_days if p and p > 1.0)
    rain_ratio = rainy_days / len(precip_days) if precip_days else 0

    if rain_ratio > 0.4:
        conditions = "frequent rain likely"
    elif rain_ratio > 0.15:
        conditions = "occasional rain possible"
    else:
        conditions = "mostly dry"

    return {
        "temp_range_c": [avg_low, avg_high],
        "conditions": conditions,
        "basis": f"averaged from {len(highs)} days across the last {year_offset} years",
    }


def get_weather_forecast(city: str, date_or_month: str) -> dict:
    """
    Tool function exposed to the LLM agent.

    Args:
        city: e.g. "Tokyo"
        date_or_month: e.g. "December", "2025-12-10"

    Returns:
        dict with keys: city, month, temp_range_c, conditions, basis
        or an "error" key if the lookup failed.
    """
    logger.info(
        "get_weather_forecast called with city=%r, date_or_month=%r",
        city, date_or_month
    )

    if not city or not city.strip():
        raise ValueError("City must not be empty.")
    if not date_or_month or not date_or_month.strip():
        raise ValueError("date_or_month must not be empty.")

    try:
        location = _geocode_city(city.strip())
        if location is None:
            return {
                "error": f"Could not find a location matching '{city}'. "
                         "Please check the spelling or try a nearby major city."
            }

        month_num = _parse_month(date_or_month)
        result = _fetch_historical_average(
            location["latitude"], location["longitude"], month_num
        )

        response = {
            "city": location["resolved_name"],
            "month": month_num,
            **result,
        }
        logger.info("get_weather_forecast result: %s", response)
        return response

    except ValueError as e:
        logger.warning("get_weather_forecast validation error: %s", e)
        return {"error": str(e)}
    except WeatherToolError as e:
        logger.error("get_weather_forecast tool error: %s", e)
        return {"error": str(e)}
    except Exception as e:
        logger.error("get_weather_forecast unexpected error: %s", e)
        return {"error": f"Unexpected error fetching weather: {e}"}