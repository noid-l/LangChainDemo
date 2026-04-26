"""测试结构化输出（穿衣建议）。"""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from langchaindemo.weather import query_weather
from langchaindemo.weather_structured import (
    ClothingAdvice,
    _format_advice,
    deterministic_advice,
)

from tests.test_weather import build_settings, build_transport


class DeterministicAdviceTests(unittest.TestCase):
    def test_cold_weather(self) -> None:
        """寒冷天气：需要外套、帽子等"""
        from langchaindemo.weather import CurrentWeather

        current = CurrentWeather(
            obs_time="2026-04-26T08:00+08:00",
            temp="-2",
            feels_like="-5",
            text="晴",
            wind_dir="北风",
            wind_scale="4",
            wind_speed="20",
            humidity="30",
            precip="0.0",
            pressure="1020",
            vis="15",
            cloud="0",
            dew="-10",
            icon="150",
        )
        advice = deterministic_advice(current)
        self.assertIn("羽绒", advice.outerwear)
        self.assertEqual(advice.comfort_level, "cold")
        self.assertIn("帽子", advice.accessories)
        self.assertIn("手套", advice.accessories)

    def test_hot_weather(self) -> None:
        """炎热天气：不需要外套"""
        from langchaindemo.weather import CurrentWeather

        current = CurrentWeather(
            obs_time="2026-07-26T14:00+08:00",
            temp="35",
            feels_like="38",
            text="晴",
            wind_dir="南风",
            wind_scale="2",
            wind_speed="8",
            humidity="60",
            precip="0.0",
            pressure="1005",
            vis="25",
            cloud="10",
            dew="20",
            icon="100",
        )
        advice = deterministic_advice(current)
        self.assertIn("不需要外套", advice.outerwear)
        self.assertEqual(advice.comfort_level, "hot")

    def test_rainy_weather(self) -> None:
        """雨天：建议带伞"""
        from langchaindemo.weather import CurrentWeather, DailyForecast

        current = CurrentWeather(
            obs_time="2026-04-26T10:00+08:00",
            temp="18",
            feels_like="17",
            text="小雨",
            wind_dir="东风",
            wind_scale="2",
            wind_speed="8",
            humidity="85",
            precip="5.2",
            pressure="1010",
            vis="8",
            cloud="90",
            dew="15",
            icon="300",
        )
        forecast = DailyForecast(
            fx_date="2026-04-26",
            temp_min="14",
            temp_max="20",
            text_day="小雨",
            text_night="中雨",
            wind_dir_day="东风",
            wind_scale_day="3",
            wind_speed_day="10",
            humidity="90",
            precip="10.5",
            pressure="1009",
            vis="6",
            uv_index="2",
            sunrise="05:40",
            sunset="18:50",
        )
        advice = deterministic_advice(current, [forecast])
        self.assertIn("雨伞", advice.accessories)

    def test_uv_warning(self) -> None:
        """高紫外线指数"""
        from langchaindemo.weather import CurrentWeather, DailyForecast

        current = CurrentWeather(
            obs_time="2026-06-26T12:00+08:00",
            temp="30",
            feels_like="32",
            text="晴",
            wind_dir="南风",
            wind_scale="2",
            wind_speed="8",
            humidity="40",
            precip="0.0",
            pressure="1008",
            vis="30",
            cloud="5",
            dew="10",
            icon="100",
        )
        forecast = DailyForecast(
            fx_date="2026-06-26",
            temp_min="22",
            temp_max="33",
            text_day="晴",
            text_night="晴",
            wind_dir_day="南风",
            wind_scale_day="2",
            wind_speed_day="8",
            humidity="35",
            precip="0.0",
            pressure="1008",
            vis="30",
            uv_index="8",
            sunrise="05:10",
            sunset="19:20",
        )
        advice = deterministic_advice(current, [forecast])
        self.assertTrue(advice.uv_warning)
        self.assertIn("防晒霜", advice.accessories)
        self.assertIn("墨镜", advice.accessories)

    def test_format_advice(self) -> None:
        """格式化输出"""
        advice = ClothingAdvice(
            outerwear="薄外套",
            accessories=["雨伞"],
            uv_warning=False,
            comfort_level="comfortable",
            summary="体感17°C，建议穿薄外套。",
        )
        text = _format_advice(advice)
        self.assertIn("薄外套", text)
        self.assertIn("舒适", text)
        self.assertIn("雨伞", text)

    def test_format_advice_json(self) -> None:
        """JSON 输出"""
        advice = ClothingAdvice(
            outerwear="短袖",
            accessories=[],
            uv_warning=False,
            comfort_level="warm",
            summary="体感25°C，短袖即可。",
        )
        json_str = json.dumps(advice.model_dump(), ensure_ascii=False)
        data = json.loads(json_str)
        self.assertEqual(data["outerwear"], "短袖")
        self.assertEqual(data["comfort_level"], "warm")


if __name__ == "__main__":
    unittest.main()
