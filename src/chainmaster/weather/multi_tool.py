"""多工具 Agent 示例。

演示 LangChain 核心概念：
- 多个 StructuredTool 注册到同一 Agent
- LLM 的工具选择推理——根据问题自动选择合适的工具
- 工具间可以组合（如 clothing_advisor 内部调用天气数据）

对照学习：
- 单工具 Agent：只有 weather_lookup，LLM 无法展示工具选择能力
- 多工具 Agent：三个工具，LLM 必须判断用哪个

工具清单：
1. weather_lookup（现有）— 查询天气数据
2. weather_compare — 对比两个城市的天气
3. clothing_advisor — 基于天气数据给出穿衣建议
"""

from __future__ import annotations

from typing import Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ..config import Settings
from ..logging_utils import get_logger
from ..openai_support import build_chat_model, ensure_chat_api_key
from .agent import build_weather_tool
from .service import (
    WeatherError,
    ensure_qweather_jwt_config,
    format_location_summary,
    format_weather_report,
    query_weather,
)
from .structured import ClothingAdvice, _format_advice, deterministic_advice

logger = get_logger(__name__)


class WeatherCompareInput(BaseModel):
    location1: str = Field(description="第一个城市名")
    location2: str = Field(description="第二个城市名")
    adm1: str | None = Field(default=None, description="第一个城市的上级行政区")
    adm2: str | None = Field(default=None, description="第二个城市的上级行政区")


class ClothingAdvisorInput(BaseModel):
    location: str = Field(description="城市名")
    adm: str | None = Field(default=None, description="上级行政区")


def _build_compare_tool(settings: Settings) -> StructuredTool:
    """构建天气对比工具。"""

    def compare_weather(
        location1: str,
        location2: str,
        adm1: str | None = None,
        adm2: str | None = None,
    ) -> str:
        logger.info(
            "weather_compare 工具被调用: %s vs %s", location1, location2
        )
        try:
            r1 = query_weather(settings, location=location1, adm=adm1)
            r2 = query_weather(settings, location=location2, adm=adm2)
        except WeatherError as exc:
            return f"查询失败: {exc}"

        temp_unit = "°C" if r1.unit == "m" else "°F"
        lines = [
            f"天气对比：{format_location_summary(r1.resolved_location)} vs {format_location_summary(r2.resolved_location)}",
            "",
            f"{'项目':<8} {r1.resolved_location.name:<10} {r2.resolved_location.name:<10}",
            f"{'温度':<8} {r1.current_weather.temp}{temp_unit:<10} {r2.current_weather.temp}{temp_unit}",
            f"{'体感':<8} {r1.current_weather.feels_like}{temp_unit:<10} {r2.current_weather.feels_like}{temp_unit}",
            f"{'天气':<8} {r1.current_weather.text:<10} {r2.current_weather.text}",
            f"{'湿度':<8} {r1.current_weather.humidity}%{'':<9} {r2.current_weather.humidity}%",
            f"{'降水':<8} {r1.current_weather.precip}mm{'':<8} {r2.current_weather.precip}mm",
        ]
        return "\n".join(lines)

    return StructuredTool.from_function(
        func=compare_weather,
        name="weather_compare",
        description=(
            "对比两个城市的天气数据。当用户问'哪个更热'、'对比天气'时使用。"
        ),
        args_schema=WeatherCompareInput,
    )


def _build_clothing_advisor_tool(settings: Settings) -> StructuredTool:
    """构建穿衣建议工具（内部复用 deterministic_advice 确定性逻辑）。"""

    def clothing_advisor(
        location: str,
        adm: str | None = None,
    ) -> str:
        logger.info("clothing_advisor 工具被调用: location=%s", location)
        try:
            result = query_weather(settings, location=location, adm=adm)
        except WeatherError as exc:
            return f"查询失败: {exc}"

        advice = deterministic_advice(
            result.current_weather, result.daily_forecast
        )
        location_summary = format_location_summary(result.resolved_location)
        return f"{location_summary} 穿衣建议：\n{_format_advice(advice)}"

    return StructuredTool.from_function(
        func=clothing_advisor,
        name="clothing_advisor",
        description=(
            "根据天气数据给出穿衣建议。"
            "当用户问'穿什么'、'需要带伞吗'、'带什么衣服'时使用。"
        ),
        args_schema=ClothingAdvisorInput,
    )


def build_multi_tool_agent(settings: Settings, *, model=None):
    """构建多工具天气 Agent。"""
    from langchain.agents import create_agent

    ensure_chat_api_key(settings)
    ensure_qweather_jwt_config(settings)

    tools = [
        build_weather_tool(settings),
        _build_compare_tool(settings),
        _build_clothing_advisor_tool(settings),
    ]

    agent = create_agent(
        model=model or build_chat_model(settings),
        tools=tools,
        system_prompt=(
            "你是一个天气查询助手，拥有以下工具：\n"
            "1. weather_lookup — 查询单个城市天气\n"
            "2. weather_compare — 对比两个城市天气\n"
            "3. clothing_advisor — 根据天气给出穿衣建议\n\n"
            "请根据用户的问题选择合适的工具。"
            "如果问题涉及对比，用 weather_compare。"
            "如果问题涉及穿衣建议，用 clothing_advisor。"
            "其他天气查询用 weather_lookup。"
            "输出使用中文。"
        ),
    )
    logger.info("多工具天气 Agent 构建完成: 工具数=%s", len(tools))
    return agent


def answer_with_multi_tool(
    question: str,
    settings: Settings,
    *,
    model=None,
) -> str:
    """使用多工具 Agent 回答天气问题。"""
    agent = build_multi_tool_agent(settings, model=model)
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})

    messages = result.get("messages", [])
    for message in reversed(messages):
        from langchain_core.messages import AIMessage

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
    return "（未获得有效回答）"
