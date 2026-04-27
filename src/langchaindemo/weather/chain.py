"""LCEL（LangChain Expression Language）链示例。

演示 LangChain 核心概念：
- LCEL 管道操作符 `|`，将组件串联为 Runnable 链
- Runnable 接口的三种调用方式：invoke / stream / batch
- StrOutputParser 从 AIMessage 中提取纯文本

对照学习：
- 确定性路径：手动调用 model.invoke() + 从 AIMessage 中取 .content
- LCEL 路径：prompt | model | StrOutputParser() 一行搞定
"""

from __future__ import annotations

import sys
from typing import Literal

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from ..config import Settings
from ..logging_utils import get_logger
from ..openai_support import build_chat_model, ensure_chat_api_key
from .service import WeatherQueryResult, format_weather_report, query_weather

logger = get_logger(__name__)

# 风格名称 → prompt 指令映射
STYLE_INSTRUCTIONS: dict[str, str] = {
    "brief": "用 2-3 句话简要概括天气要点，适合快速浏览。",
    "detailed": "用分段结构详细描述天气状况，包含穿衣建议和出行提醒。",
    "casual": "用轻松口语化的风格描述，就像朋友间聊天气一样自然。",
}


def build_weather_summary_prompt(
    style: Literal["brief", "detailed", "casual"] = "brief",
) -> ChatPromptTemplate:
    """构建天气摘要 prompt 模板。"""
    instruction = STYLE_INSTRUCTIONS.get(style, STYLE_INSTRUCTIONS["brief"])
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一名天气播报助手。根据提供的天气数据，用中文生成天气摘要。\n{instruction}",
            ),
            ("human", "以下是天气数据，请生成摘要：\n\n{{weather_report}}"),
        ]
    ).partial(instruction=instruction)


def build_weather_summary_chain(
    settings: Settings,
    style: Literal["brief", "detailed", "casual"] = "brief",
) -> Runnable:
    """构建 LCEL 天气摘要链：prompt | model | StrOutputParser。

    LCEL 的 `|` 操作符将每个组件串联为 Runnable，
    数据自动从上一个组件的输出流向下一个组件的输入：
      ChatPromptValue → AIMessage → str
    """
    ensure_chat_api_key(settings)
    prompt = build_weather_summary_prompt(style)
    model = build_chat_model(settings)
    chain = prompt | model | StrOutputParser()
    logger.info("LCEL 天气摘要链构建完成: style=%s", style)
    return chain


def summarize_weather_deterministic(
    result: WeatherQueryResult,
    settings: Settings,
    style: Literal["brief", "detailed", "casual"] = "brief",
) -> str:
    """确定性路径：手动调用模型并提取结果。

    对比 LCEL 链的封装效果——这里的每一步 LCEL 都自动处理了：
    1. 手动构建 prompt 消息列表
    2. 手动调用 model.invoke()
    3. 手动从 AIMessage.content 中提取文本
    """
    ensure_chat_api_key(settings)
    model = build_chat_model(settings)
    instruction = STYLE_INSTRUCTIONS.get(style, STYLE_INSTRUCTIONS["brief"])
    report = format_weather_report(result)

    messages = [
        ("system", f"你是一名天气播报助手。根据提供的天气数据，用中文生成天气摘要。\n{instruction}"),
        ("human", f"以下是天气数据，请生成摘要：\n\n{report}"),
    ]
    response = model.invoke(messages)
    return response.content if isinstance(response.content, str) else str(response.content)


def summarize_weather(
    location: str,
    settings: Settings,
    *,
    style: Literal["brief", "detailed", "casual"] = "brief",
    adm: str | None = None,
    mode: Literal["lcel", "deterministic"] = "lcel",
) -> str:
    """查天气并生成摘要（单次调用）。"""
    logger.info("开始天气摘要: location=%s, style=%s, mode=%s", location, style, mode)
    result = query_weather(settings, location=location, adm=adm)

    if mode == "deterministic":
        return summarize_weather_deterministic(result, settings, style)

    chain = build_weather_summary_chain(settings, style)
    report = format_weather_report(result)
    return chain.invoke({"weather_report": report})


def summarize_weather_batch(
    locations: list[str],
    settings: Settings,
    *,
    style: Literal["brief", "detailed", "casual"] = "brief",
) -> list[str]:
    """批量天气摘要——演示 Runnable.batch() 并发处理多个输入。

    batch() 会并行处理所有输入，比逐个 invoke 更高效。
    """
    logger.info("开始批量天气摘要: locations=%s, style=%s", locations, style)
    chain = build_weather_summary_chain(settings, style)

    inputs: list[dict[str, str]] = []
    for location in locations:
        result = query_weather(settings, location=location)
        report = format_weather_report(result)
        inputs.append({"weather_report": report})

    results = chain.batch(inputs)
    logger.info("批量天气摘要完成: count=%s", len(results))
    return results


def summarize_weather_stream(
    location: str,
    settings: Settings,
    *,
    style: Literal["brief", "detailed", "casual"] = "brief",
    adm: str | None = None,
    file: object = None,
) -> None:
    """流式天气摘要——演示 Runnable.stream() 逐 token 输出。

    stream() 返回迭代器，每个 chunk 是一个 token 片段，
    可以实现类似 ChatGPT 的"打字机"效果。
    """
    logger.info("开始流式天气摘要: location=%s, style=%s", location, style)
    result = query_weather(settings, location=location, adm=adm)
    chain = build_weather_summary_chain(settings, style)
    report = format_weather_report(result)

    output = file or sys.stdout
    for chunk in chain.stream({"weather_report": report}):
        output.write(chunk)
        output.flush()
    output.write("\n")
