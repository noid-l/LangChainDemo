from __future__ import annotations

import unittest

from chainmaster.config import Settings
from chainmaster.weather import (
    AmbiguousLocationError,
    LocationNotFoundError,
    WeatherApiError,
    WeatherConfigError,
    format_weather_report,
    query_weather,
)

from tests.conftest import build_settings, build_transport


class WeatherServiceTests(unittest.TestCase):
    def test_query_weather_by_city(self) -> None:
        result = query_weather(
            build_settings(),
            location="北京",
            transport=build_transport(),
        )
        self.assertEqual(result.resolved_location.location_id, "101010100")
        self.assertEqual(result.current_weather.text, "晴")
        self.assertEqual(len(result.daily_forecast), 3)

    def test_query_weather_by_coordinates(self) -> None:
        result = query_weather(
            build_settings(),
            location="116.41,39.92",
            transport=build_transport(),
        )
        self.assertEqual(result.resolved_location.name, "北京")

    def test_ambiguous_city_raises_error(self) -> None:
        with self.assertRaises(AmbiguousLocationError):
            query_weather(
                build_settings(),
                location="西安",
                transport=build_transport(),
            )

    def test_location_not_found_raises_error(self) -> None:
        with self.assertRaises(LocationNotFoundError):
            query_weather(
                build_settings(),
                location="不存在",
                transport=build_transport(),
            )

    def test_api_error_raises_weather_api_error(self) -> None:
        with self.assertRaises(WeatherApiError):
            query_weather(
                build_settings(),
                location="北京",
                forecast_days=7,
                transport=build_transport(),
            )

    def test_jwt_mode_rejects_legacy_devapi_host(self) -> None:
        settings = build_settings()
        settings = Settings(
            **{
                **settings.__dict__,
                "qweather_api_host": "https://devapi.qweather.com",
            }
        )
        with self.assertRaises(WeatherConfigError):
            query_weather(
                settings,
                location="北京",
                transport=build_transport(),
            )

    def test_missing_private_key_path_raises_error(self) -> None:
        settings = build_settings()
        settings = Settings(
            **{
                **settings.__dict__,
                "qweather_private_key_path": None,
            }
        )
        with self.assertRaises(WeatherConfigError):
            query_weather(
                settings,
                location="北京",
                transport=build_transport(),
            )

    def test_format_weather_report(self) -> None:
        result = query_weather(
            build_settings(),
            location="北京",
            transport=build_transport(),
        )
        report = format_weather_report(result)
        self.assertIn("位置: 北京, 北京市, 中国", report)
        self.assertIn("当前天气:", report)
        self.assertIn("未来 3 天预报:", report)


if __name__ == "__main__":
    unittest.main()
