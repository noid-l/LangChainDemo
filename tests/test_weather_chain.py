"""测试 LCEL 天气摘要链。"""

from __future__ import annotations

import io
import unittest
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from langchaindemo.weather import query_weather

from tests.conftest import build_settings, build_transport


def _query_beijing() -> object:
    """复用 fixture 查询北京天气。"""
    return query_weather(build_settings(), location="北京", transport=build_transport())


class WeatherChainTests(unittest.TestCase):
    def test_build_chain(self) -> None:
        """验证 LCEL 链可以正常构建"""
        from langchaindemo.weather.chain import build_weather_summary_chain

        settings = build_settings()
        chain = build_weather_summary_chain(settings, style="brief")
        self.assertIsNotNone(chain)

    def test_summarize_weather_deterministic(self) -> None:
        """确定性路径：手动调用模型"""
        from langchaindemo.weather.chain import summarize_weather_deterministic

        settings = build_settings()
        transport = build_transport()
        result = query_weather(settings, location="北京", transport=transport)

        fake_model = MagicMock()
        fake_model.invoke.return_value = AIMessage(content="北京今天晴，适合出行。")

        with patch("langchaindemo.weather.chain.build_chat_model", return_value=fake_model):
            with patch("langchaindemo.weather.chain.ensure_chat_api_key"):
                summary = summarize_weather_deterministic(result, settings, style="brief")

        self.assertEqual(summary, "北京今天晴，适合出行。")
        fake_model.invoke.assert_called_once()

    def test_stream_output(self) -> None:
        """流式输出：逐 token 输出"""
        from langchaindemo.weather.chain import summarize_weather_stream

        settings = build_settings()
        transport = build_transport()

        output = io.StringIO()
        weather_result = query_weather(
            settings, location="北京", transport=transport
        )

        with patch("langchaindemo.weather.chain.query_weather", return_value=weather_result):
            with patch("langchaindemo.weather.chain.build_weather_summary_chain") as mock_build:
                fake_chain = MagicMock()
                fake_chain.stream.return_value = iter(["北京", "今天", "晴。"])
                mock_build.return_value = fake_chain
                with patch("langchaindemo.weather.chain.ensure_chat_api_key"):
                    summarize_weather_stream("北京", settings, file=output)

        self.assertIn("北京今天晴。", output.getvalue())

    def test_batch_output(self) -> None:
        """批量处理：batch 并发"""
        from langchaindemo.weather.chain import summarize_weather_batch

        settings = build_settings()
        transport = build_transport()
        weather_result = query_weather(
            settings, location="北京", transport=transport
        )

        with patch("langchaindemo.weather.chain.query_weather", return_value=weather_result):
            with patch("langchaindemo.weather.chain.build_weather_summary_chain") as mock_build:
                fake_chain = MagicMock()
                fake_chain.batch.return_value = ["摘要1", "摘要2"]
                mock_build.return_value = fake_chain
                with patch("langchaindemo.weather.chain.ensure_chat_api_key"):
                    results = summarize_weather_batch(
                        ["北京", "上海"], settings, style="brief"
                    )

        self.assertEqual(results, ["摘要1", "摘要2"])

    def test_style_instructions(self) -> None:
        """验证三种风格都有对应指令"""
        from langchaindemo.weather.chain import STYLE_INSTRUCTIONS

        self.assertIn("brief", STYLE_INSTRUCTIONS)
        self.assertIn("detailed", STYLE_INSTRUCTIONS)
        self.assertIn("casual", STYLE_INSTRUCTIONS)
        for key, instruction in STYLE_INSTRUCTIONS.items():
            self.assertTrue(len(instruction) > 10, f"{key} 指令过短")

    def test_summarize_weather_with_mode(self) -> None:
        """测试 summarize_weather 切换 deterministic 模式"""
        from langchaindemo.weather.chain import summarize_weather

        settings = build_settings()
        transport = build_transport()
        weather_result = query_weather(
            settings, location="北京", transport=transport
        )

        fake_model = MagicMock()
        fake_model.invoke.return_value = AIMessage(content="确定性摘要")

        with patch("langchaindemo.weather.chain.query_weather", return_value=weather_result):
            with patch("langchaindemo.weather.chain.build_chat_model", return_value=fake_model):
                with patch("langchaindemo.weather.chain.ensure_chat_api_key"):
                    result = summarize_weather(
                        "北京", settings, style="brief", mode="deterministic"
                    )

        self.assertEqual(result, "确定性摘要")


if __name__ == "__main__":
    unittest.main()
