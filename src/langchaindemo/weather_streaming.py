"""流式输出示例。

演示 LangChain 核心概念：
- Agent / Chain 的 stream() 方法
- stream_mode="messages" 获取 token 级增量输出
- AIMessageChunk 与 AIMessage 的区别
- 对比 invoke() 阻塞等待 vs stream() 逐 token 输出

对照学习：
- 确定性路径：将格式化报告逐行打印（模拟"流式"效果）
- LangChain 路径：agent.stream() 逐 token 生成，反映真实 LLM 推理过程
"""

from __future__ import annotations

import sys
from typing import Any

from langchain_core.messages import AIMessageChunk

from .config import Settings
from .logging_utils import get_logger
from .openai_support import build_chat_model, ensure_chat_api_key
from .weather import WeatherQueryResult, format_weather_report, query_weather
from .weather_langchain import build_weather_tool

logger = get_logger(__name__)


def stream_weather_report_lines(
    result: WeatherQueryResult,
    *,
    file: Any | None = None,
    delay: float = 0.02,
) -> None:
    """确定性路径：逐行打印格式化天气报告。

    这不是真正的"流式"——只是把完整结果分段输出。
    对比 LLM 的 stream()，后者是模型真正在逐 token 生成。
    """
    import time

    output = file or sys.stdout
    report = format_weather_report(result)
    for line in report.splitlines():
        output.write(line + "\n")
        output.flush()
        time.sleep(delay)


def stream_weather_agent_answer(
    question: str,
    settings: Settings,
    *,
    model=None,
    file: Any | None = None,
) -> None:
    """LangChain 路径：流式输出天气 Agent 回答。

    使用 agent.stream(stream_mode="messages") 逐 token 输出，
    这是真正的流式——每个 token 都是 LLM 实时生成的。
    """
    from langchain.agents import create_agent

    ensure_chat_api_key(settings)
    from .weather import ensure_qweather_jwt_config

    ensure_qweather_jwt_config(settings)

    weather_tool = build_weather_tool(settings)
    agent = create_agent(
        model=model or build_chat_model(settings),
        tools=[weather_tool],
        system_prompt=(
            "你是一个天气查询助手。通过 weather_lookup 工具查询真实天气数据来回答用户问题。"
            "输出使用中文，结构清晰。"
        ),
    )

    output = file or sys.stdout
    logger.info("开始流式天气 Agent 问答: question=%s", question)

    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="messages",
        version="v2",
    ):
        if chunk["type"] != "messages":
            continue
        token, _metadata = chunk["data"]
        if isinstance(token, AIMessageChunk):
            # 提取文本内容
            if isinstance(token.content, str) and token.content:
                output.write(token.content)
                output.flush()
            elif isinstance(token.content, list):
                for block in token.content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        output.write(block.get("text", ""))
                        output.flush()

    output.write("\n")
    logger.info("流式天气 Agent 问答完成。")
