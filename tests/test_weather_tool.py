"""
Unit tests for src.tools.weather_tool

Uses real Open-Meteo calls where feasible (it's free and keyless), plus
input-validation tests that don't need network access.
"""

import pytest
from src.tools.weather_tool import get_weather_forecast, _parse_month


def test_valid_city_and_month_returns_expected_schema():
    result = get_weather_forecast("Tokyo", "December")
    assert "error" not in result
    assert "temp_range_c" in result
    assert "conditions" in result
    assert isinstance(result["temp_range_c"], list)
    assert len(result["temp_range_c"]) == 2


def test_empty_city_raises_value_error():
    with pytest.raises(ValueError):
        get_weather_forecast("", "December")


def test_empty_date_raises_value_error():
    with pytest.raises(ValueError):
        get_weather_forecast("Tokyo", "")


def test_unknown_city_returns_error_dict():
    result = get_weather_forecast("Notarealcityxyz123", "December")
    assert "error" in result


def test_parse_month_from_month_name():
    assert _parse_month("December") == 12
    assert _parse_month("december") == 12
    assert _parse_month("Dec") == 12


def test_parse_month_from_iso_date():
    assert _parse_month("2025-12-10") == 12


def test_parse_month_unparseable_raises_value_error():
    with pytest.raises(ValueError):
        _parse_month("not a real date")