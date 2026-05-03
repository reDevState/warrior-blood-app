"""
weather.py — OpenWeatherMap + AirVisual client for Warrior Blood.

Uses a 6-hour in-memory cache to avoid exceeding free-tier API limits.
All weather data is optional — the app degrades gracefully when offline.

References:
    Nolan et al. (2008) — temperature and SCD hospitalisation
    Yallop et al. (2007) — AQI and SCD
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional

import requests

CACHE_TTL = 6 * 3600  # 6 hours in seconds

_cache: dict[str, tuple[float, dict]] = {}


@dataclass
class WeatherData:
    """Ambient environmental data for a patient location."""

    ambient_temp_c: Optional[float] = None
    feels_like_c: Optional[float] = None
    humidity_pct: Optional[float] = None
    aqi: Optional[int] = None           # OpenWeatherMap AQI index 1–5
    pm25_ugm3: Optional[float] = None   # PM2.5 µg/m³
    condition: Optional[str] = None     # e.g. "Clear", "Rain"
    cold_stress_alert: bool = False     # temp < 15 °C — vasoconstriction risk
    heat_stress_alert: bool = False     # temp > 35 °C — dehydration risk
    aqi_alert: bool = False             # AQI >= 3 (Yallop et al. 2007)
    source: str = "offline"


def get_weather(lat: float, lon: float) -> WeatherData:
    """
    Fetch current weather and air quality for a location.

    Results are cached per coordinate (rounded to 2 dp) for 6 hours to
    stay within OpenWeatherMap's free-tier limit (1 000 calls/day).

    Args:
        lat: Latitude (-90 to 90).
        lon: Longitude (-180 to 180).

    Returns:
        WeatherData with temperature, humidity, AQI, and clinical risk flags.
        Returns an offline WeatherData if keys are absent or the call fails.
    """
    owm_key = os.getenv("OPENWEATHERMAP_API_KEY", "")
    if not owm_key:
        return WeatherData(source="no_api_key")

    cache_key = f"{round(lat, 2)},{round(lon, 2)}"
    now = time.time()

    if cache_key in _cache:
        cached_at, cached_data = _cache[cache_key]
        if now - cached_at < CACHE_TTL:
            return WeatherData(**cached_data, source="cache")

    try:
        resp = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"lat": lat, "lon": lon, "appid": owm_key, "units": "metric"},
            timeout=5,
        )
        resp.raise_for_status()
        w = resp.json()
        temp = float(w["main"]["temp"])
        feels = float(w["main"]["feels_like"])
        humid = float(w["main"]["humidity"])
        cond = w["weather"][0]["main"] if w.get("weather") else None

        aqi, pm25 = None, None
        air = requests.get(
            "https://api.openweathermap.org/data/2.5/air_pollution",
            params={"lat": lat, "lon": lon, "appid": owm_key},
            timeout=5,
        )
        if air.ok:
            aqi = air.json()["list"][0]["main"]["aqi"]
            pm25 = air.json()["list"][0]["components"].get("pm2_5")

        result: dict = {
            "ambient_temp_c": round(temp, 1),
            "feels_like_c": round(feels, 1),
            "humidity_pct": round(humid, 1),
            "aqi": aqi,
            "pm25_ugm3": round(pm25, 1) if pm25 is not None else None,
            "condition": cond,
            "cold_stress_alert": temp < 15.0,
            "heat_stress_alert": temp > 35.0,
            "aqi_alert": (aqi or 0) >= 3,
        }
        _cache[cache_key] = (now, result)
        return WeatherData(**result, source="live")

    except Exception:
        return WeatherData(source="error")
