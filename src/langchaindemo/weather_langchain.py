from __future__ import annotations

from typing import Literal

import httpx
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from .config import Settings
from .logging_utils import get_logger
from .openai_support import build_chat_model, ensure_chat_api_key
from .weather import (
    WeatherError,
    ensure_qweather_jwt_config,
    format_weather_report,
    query_weather,
)


logger = get_logger(__name__)

WEATHER_AGENT_SYSTEM_PROMPT = "\n".join(
    [
        "你是一个天气查询助手。",
        "你的任务是理解用户的自然语言天气问题，并通过 weather_lookup 工具查询真实天气。",
        "当用户在询问天气、气温、降水、风力、未来几天天气时，必须先调用 weather_lookup 工具。",
        "工具返回的是可信的天气结果文本，你只能基于工具结果回答，不要编造天气信息。",
        "如果用户问题没有明确地点，可结合上下文尽量推断；若仍不明确，请明确提示用户补充地点。",
        "输出使用中文，结构清晰，直接回答用户问题。",
    ]
)


class WeatherToolInput(BaseModel):
    location: str = Field(
        ...,
        description="城市名/地区名，或 `longitude,latitude` 格式的经纬度。",
    )
    adm: str | None = Field(
        default=None,
        description="上级行政区，用于重名城市消歧，例如 `陕西`。",
    )
    lang: str = Field(default="zh", description="返回语言，默认 zh。")
    unit: Literal["m", "i"] = Field(
        default="m",
        description="单位：m 为公制，i 为英制。",
    )
    forecast_days: Literal[3, 7] = Field(
        default=3,
        description="短期预报天数，目前仅支持 3 或 7。",
    )


def build_weather_tool(
    settings: Settings,
    *,
    transport: httpx.BaseTransport | None = None,
) -> StructuredTool:
    def weather_lookup(
        location: str,
        adm: str | None = None,
        lang: str = "zh",
        unit: Literal["m", "i"] = "m",
        forecast_days: Literal[3, 7] = 3,
    ) -> str:
        """
        查询指定地点的当前天气和短期天气预报。

        适用于用户询问当前天气、未来几天天气、温度、降水、风力等信息。
        """
        logger.info(
            "LangChain weather tool 被调用: location=%s, adm=%s, lang=%s, unit=%s, forecast_days=%s",
            location,
            adm,
            lang,
            unit,
            forecast_days,
        )
        result = query_weather(
            settings,
            location=location,
            adm=adm,
            lang=lang,
            unit=unit,
            forecast_days=forecast_days,
            transport=transport,
        )
        return format_weather_report(result)

    return StructuredTool.from_function(
        func=weather_lookup,
        name="weather_lookup",
        description=(
            "查询真实天气信息。输入地点后，返回当前天气和未来几天天气预报。"
            "当用户问天气时必须调用这个工具，而不是直接凭空回答。"
        ),
        args_schema=WeatherToolInput,
    )


def extract_agent_answer(agent_result: dict[str, object]) -> str:
    messages = agent_result.get("messages")
    if not isinstance(messages, list):
        raise WeatherError("Agent 未返回有效消息列表。")

    for message in reversed(messages):
        if isinstance(message, AIMessage):
            content = message.content
            if isinstance(content, str) and content.strip():
                return content.strip()
            if isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict) and item.get("type") == "text":
                        text = item.get("text")
                        if isinstance(text, str):
                            parts.append(text)
                answer = "\n".join(part.strip() for part in parts if part.strip())
                if answer:
                    return answer

    raise WeatherError("Agent 未生成最终回答。")


def answer_weather_question(
    question: str,
    settings: Settings,
    *,
    model=None,
    transport: httpx.BaseTransport | None = None,
) -> str:
    ensure_chat_api_key(settings)
    ensure_qweather_jwt_config(settings)

    logger.info("开始执行 LangChain Agent 天气问答: question=%s", question)
    weather_tool = build_weather_tool(settings, transport=transport)
    agent = create_agent(
        model=model or build_chat_model(settings),
        tools=[weather_tool],
        system_prompt=WEATHER_AGENT_SYSTEM_PROMPT,
    )
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    answer = extract_agent_answer(result)
    logger.info("LangChain Agent 天气问答完成。")
    return answer
