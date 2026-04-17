from __future__ import annotations

import unittest

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from langchaindemo.weather_langchain import (
    answer_weather_question,
    build_weather_tool,
    extract_agent_answer,
)
from test_weather import build_settings, build_transport


class ToolCompatibleFakeChatModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


class WeatherLangChainTests(unittest.TestCase):
    def test_weather_tool_returns_weather_report(self) -> None:
        tool = build_weather_tool(build_settings(), transport=build_transport())
        result = tool.invoke(
            {
                "location": "北京",
                "forecast_days": 3,
                "lang": "zh",
                "unit": "m",
            }
        )
        self.assertIn("当前天气:", result)
        self.assertIn("未来 3 天预报:", result)

    def test_extract_agent_answer_from_ai_message(self) -> None:
        answer = extract_agent_answer(
            {"messages": [AIMessage(content="明天北京晴，最高 26°C。")]}
        )
        self.assertEqual(answer, "明天北京晴，最高 26°C。")

    def test_answer_weather_question_uses_langchain_agent(self) -> None:
        fake_model = ToolCompatibleFakeChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "weather_lookup",
                            "args": {
                                "location": "北京",
                                "forecast_days": 3,
                                "lang": "zh",
                                "unit": "m",
                            },
                            "id": "call-weather-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="根据天气工具结果，北京今天天气晴，未来三天整体晴到多云。"),
            ]
        )

        answer = answer_weather_question(
            "明天北京天气怎么样？",
            build_settings(),
            model=fake_model,
            transport=build_transport(),
        )
        self.assertIn("北京", answer)
        self.assertIn("天气", answer)


if __name__ == "__main__":
    unittest.main()
