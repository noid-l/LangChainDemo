"""LangGraph 工作流示例。

演示 LangChain 核心概念：
- StateGraph：显式状态图替代隐式 Agent 循环
- add_node / add_edge / add_conditional_edges：定义图结构
- InMemorySaver：检查点持久化，支持多轮对话
- 条件路由：根据意图分类路由到不同处理节点

对照学习：
- create_agent：隐式循环，路由逻辑封装在 Agent 内部
- StateGraph：显式图，每一步、每条边都清晰可见

图结构：
  START → classify_intent → weather_query / weather_compare / clothing_advise
                              ↓               ↓                 ↓
                           format_response ← ← ← ← ← ← ← ← ←
                              ↓
                             END
"""

from __future__ import annotations

from typing import Annotated, Literal

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from ..config import Settings
from ..logging_utils import get_logger
from ..openai_support import build_chat_model, ensure_chat_api_key
from .service import (
    WeatherError,
    ensure_qweather_jwt_config,
    format_location_summary,
    query_weather,
)
from .structured import _format_advice, deterministic_advice

logger = get_logger(__name__)

# 全局 checkpointer 实例（进程内）
_checkpointer = InMemorySaver()


class WeatherGraphState(TypedDict):
    """天气工作流状态。"""
    messages: Annotated[list, add_messages]
    intent: str | None
    tool_result: str | None


def classify_intent(state: WeatherGraphState) -> dict:
    """意图分类节点：分析用户问题，判断应使用哪个工具。"""
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = state["messages"]
    user_message = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_message = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    logger.info("意图分类: question=%s", user_message[:100])

    # 简单的规则分类（实际项目中可用 LLM 分类）
    lower = user_message.lower()
    if any(kw in lower for kw in ["对比", "比较", "vs", "哪个热", "哪个冷"]):
        intent = "compare"
    elif any(kw in lower for kw in ["穿", "带", "伞", "衣服", "外套", "建议"]):
        intent = "advise"
    else:
        intent = "query"

    logger.info("意图分类结果: intent=%s", intent)
    return {"intent": intent}


def _extract_location(text: str) -> str:
    """从文本中简单提取城市名（教学简化版）。"""
    import re
    # 尝试匹配常见城市名模式
    for pattern in [
        r"([一-鿿]{2,4}(?:市|省|县)?)的?天气",
        r"天气.*?([一-鿿]{2,4})",
        r"([一-鿿]{2,4})",
    ]:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return "北京"  # 默认


def weather_query_node(state: WeatherGraphState) -> dict:
    """天气查询节点。"""
    from langchain_core.messages import HumanMessage

    messages = state["messages"]
    user_message = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_message = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    location = _extract_location(user_message)
    logger.info("天气查询节点: location=%s", location)

    # 需要 settings，这里通过 import 延迟获取
    from ..config import load_settings
    settings = load_settings()

    try:
        result = query_weather(settings, location=location)
        from .service import format_weather_report
        tool_result = format_weather_report(result)
    except WeatherError as exc:
        tool_result = f"查询失败: {exc}"

    return {"tool_result": tool_result}


def weather_compare_node(state: WeatherGraphState) -> dict:
    """天气对比节点。"""
    from langchain_core.messages import HumanMessage
    import re

    messages = state["messages"]
    user_message = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_message = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    # 简单提取两个城市名
    cities = re.findall(r"[一-鿿]{2,4}", user_message)
    if len(cities) >= 2:
        city1, city2 = cities[0], cities[1]
    else:
        city1, city2 = "北京", "上海"

    logger.info("天气对比节点: %s vs %s", city1, city2)

    from ..config import load_settings
    settings = load_settings()

    try:
        r1 = query_weather(settings, location=city1)
        r2 = query_weather(settings, location=city2)
        temp_unit = "°C" if r1.unit == "m" else "°F"
        tool_result = (
            f"天气对比：{format_location_summary(r1.resolved_location)} vs "
            f"{format_location_summary(r2.resolved_location)}\n"
            f"- 温度：{r1.current_weather.temp}{temp_unit} vs {r2.current_weather.temp}{temp_unit}\n"
            f"- 天气：{r1.current_weather.text} vs {r2.current_weather.text}\n"
            f"- 湿度：{r1.current_weather.humidity}% vs {r2.current_weather.humidity}%"
        )
    except WeatherError as exc:
        tool_result = f"查询失败: {exc}"

    return {"tool_result": tool_result}


def clothing_advise_node(state: WeatherGraphState) -> dict:
    """穿衣建议节点。"""
    from langchain_core.messages import HumanMessage

    messages = state["messages"]
    user_message = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_message = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    location = _extract_location(user_message)
    logger.info("穿衣建议节点: location=%s", location)

    from ..config import load_settings
    settings = load_settings()

    try:
        result = query_weather(settings, location=location)
        advice = deterministic_advice(result.current_weather, result.daily_forecast)
        tool_result = f"{format_location_summary(result.resolved_location)} 穿衣建议：\n{_format_advice(advice)}"
    except WeatherError as exc:
        tool_result = f"查询失败: {exc}"

    return {"tool_result": tool_result}


def format_response_node(state: WeatherGraphState) -> dict:
    """格式化响应节点：将工具结果转为自然语言回答。"""
    from langchain_core.messages import AIMessage

    tool_result = state.get("tool_result", "未能获取天气数据。")
    logger.info("格式化响应节点")

    # 直接使用工具结果作为回答（教学简化，实际可用 LLM 润色）
    return {"messages": [AIMessage(content=tool_result)]}


def route_by_intent(state: WeatherGraphState) -> str:
    """条件路由：根据意图分类结果路由到不同节点。"""
    intent = state.get("intent", "query")
    routing = {
        "query": "weather_query",
        "compare": "weather_compare",
        "advise": "clothing_advise",
    }
    target = routing.get(intent, "weather_query")
    logger.info("条件路由: intent=%s -> node=%s", intent, target)
    return target


def build_weather_graph() -> StateGraph:
    """构建天气工作流状态图。"""
    builder = StateGraph(WeatherGraphState)

    # 添加节点
    builder.add_node("classify_intent", classify_intent)
    builder.add_node("weather_query", weather_query_node)
    builder.add_node("weather_compare", weather_compare_node)
    builder.add_node("clothing_advise", clothing_advise_node)
    builder.add_node("format_response", format_response_node)

    # 添加边
    builder.add_edge(START, "classify_intent")
    builder.add_conditional_edges("classify_intent", route_by_intent)
    builder.add_edge("weather_query", "format_response")
    builder.add_edge("weather_compare", "format_response")
    builder.add_edge("clothing_advise", "format_response")
    builder.add_edge("format_response", END)

    logger.info("天气工作流状态图构建完成")
    return builder


def answer_weather_graph(
    question: str,
    settings: Settings,
    *,
    thread_id: str = "default",
) -> str:
    """使用 LangGraph 工作流回答天气问题。"""
    ensure_chat_api_key(settings)
    ensure_qweather_jwt_config(settings)

    graph = build_weather_graph().compile(checkpointer=_checkpointer)
    config = {"configurable": {"thread_id": thread_id}}

    logger.info("LangGraph 天气工作流启动: question=%s, thread=%s", question, thread_id)
    result = graph.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config=config,
    )

    # 提取最终回答
    from langchain_core.messages import AIMessage
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, str) and content.strip():
                return content.strip()

    return "（未获得有效回答）"
