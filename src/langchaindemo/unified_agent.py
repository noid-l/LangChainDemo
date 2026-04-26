"""统一 Agent 入口——超级 Agent REPL。

将所有功能整合为一个 Agent，LLM 自动判断使用哪个工具：
- weather_lookup: 天气查询
- weather_compare: 天气对比
- clothing_advisor: 穿衣建议
- knowledge_search: RAG 知识库检索

支持多轮对话（InMemoryChatMessageHistory）、流式输出、回调追踪。
"""

from __future__ import annotations

import sys
from typing import Any

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from .config import Settings
from .logging_utils import get_logger
from .openai_support import build_chat_model, ensure_chat_api_key

logger = get_logger(__name__)

_sessions: dict[str, InMemoryChatMessageHistory] = {}


def _get_session(session_id: str = "default") -> InMemoryChatMessageHistory:
    if session_id not in _sessions:
        _sessions[session_id] = InMemoryChatMessageHistory()
    return _sessions[session_id]


class KnowledgeSearchInput(BaseModel):
    question: str = Field(description="要在知识库中搜索的问题")


def _build_knowledge_search_tool(settings: Settings) -> StructuredTool:
    """构建知识库检索工具。"""
    from .rag import answer_question

    def knowledge_search(question: str) -> str:
        logger.info("knowledge_search 工具被调用: question=%s", question[:100])
        try:
            result = answer_question(question=question, settings=settings)
            return result.answer
        except Exception as exc:
            logger.error("知识库检索失败: %s", exc)
            return f"知识库检索失败: {exc}"

    return StructuredTool.from_function(
        func=knowledge_search,
        name="knowledge_search",
        description=(
            "从本地知识库中检索信息。当用户问关于 LangChain、RAG、OpenAI 代理等"
            "技术概念的问题时使用。不适用于天气查询。"
        ),
        args_schema=KnowledgeSearchInput,
    )


def build_all_tools(settings: Settings) -> list[StructuredTool]:
    """收集所有工具。天气工具需要 JWT 配置，如果缺失则跳过。"""
    tools: list[StructuredTool] = []

    # 知识库检索
    tools.append(_build_knowledge_search_tool(settings))

    # 天气相关工具（需要 JWT 配置）
    try:
        from .weather import ensure_qweather_jwt_config
        ensure_qweather_jwt_config(settings)

        from .weather_langchain import build_weather_tool
        from .weather_multi_tool import _build_compare_tool, _build_clothing_advisor_tool

        tools.append(build_weather_tool(settings))
        tools.append(_build_compare_tool(settings))
        tools.append(_build_clothing_advisor_tool(settings))
        logger.info("天气工具已加载（共 3 个）。")
    except Exception:
        logger.warning("天气工具加载跳过（JWT 配置缺失）。")

    # 网页搜索工具（需要 TAVILY_API_KEY）
    try:
        from .web_search import build_web_search_tool
        tools.append(build_web_search_tool(settings))
        logger.info("搜索工具已加载。")
    except Exception:
        logger.warning("搜索工具加载跳过（TAVILY_API_KEY 缺失）。")

    # 翻译工具
    try:
        from .translate import build_translate_tool
        tools.append(build_translate_tool(settings))
        logger.info("翻译工具已加载。")
    except Exception:
        logger.warning("翻译工具加载跳过。")

    # 文档问答工具
    try:
        from .document_qa import build_document_qa_tool
        tools.append(build_document_qa_tool(settings))
        logger.info("文档问答工具已加载。")
    except Exception:
        logger.warning("文档问答工具加载跳过。")

    # 数据分析工具
    try:
        from .data_analysis import build_data_analysis_tool
        tools.append(build_data_analysis_tool(settings))
        logger.info("数据分析工具已加载。")
    except Exception:
        logger.warning("数据分析工具加载跳过。")

    logger.info("统一 Agent 工具列表: %s", [t.name for t in tools])
    return tools


UNIFIED_SYSTEM_PROMPT = "\n".join([
    "你是一个多功能助手，拥有以下工具：",
    "",
    "1. weather_lookup — 查询指定城市的天气数据",
    "2. weather_compare — 对比两个城市的天气",
    "3. clothing_advisor — 根据天气给出穿衣建议",
    "4. knowledge_search — 从知识库中检索技术文档",
    "5. web_search — 搜索互联网获取实时信息",
    "6. translate — 翻译文本到指定语言",
    "7. document_qa — 读取文档文件（PDF/Word/TXT）并回答问题",
    "8. data_analysis — 分析 CSV 数据文件",
    "",
    "请根据用户的问题选择合适的工具：",
    "- 天气相关 → weather_lookup / weather_compare / clothing_advisor",
    "- 技术概念、知识库 → knowledge_search",
    "- 实时信息、最新新闻 → web_search",
    "- 翻译 → translate",
    "- 文档问题（提到文件路径）→ document_qa",
    "- 数据分析（提到 CSV/数据）→ data_analysis",
    "- 闲聊或通用问题 → 直接回答",
    "",
    "如果用户没有明确地点但问天气，可追问。多轮对话中结合上下文推断。",
    "输出使用中文，结构清晰。",
])


def build_unified_agent(settings: Settings, *, model=None):
    """构建统一 Agent。"""
    from langchain.agents import create_agent

    ensure_chat_api_key(settings)
    tools = build_all_tools(settings)

    agent = create_agent(
        model=model or build_chat_model(settings),
        tools=tools,
        system_prompt=UNIFIED_SYSTEM_PROMPT,
    )
    logger.info("统一 Agent 构建完成: 工具数=%s", len(tools))
    return agent


def chat_unified(
    question: str,
    settings: Settings,
    *,
    session_id: str = "default",
    model=None,
    config: dict[str, Any] | None = None,
) -> str:
    """单次问答。"""
    agent = build_unified_agent(settings, model=model)
    history = _get_session(session_id)

    messages: list[dict[str, str]] = []
    for msg in history.messages:
        if isinstance(msg, HumanMessage):
            messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            messages.append({"role": "assistant", "content": msg.content})
    messages.append({"role": "user", "content": question})

    logger.info("统一 Agent 问答: question=%s, history=%s轮", question[:80], len(history.messages) // 2)

    result = agent.invoke({"messages": messages}, config=config or {})
    answer = _extract_answer(result)

    history.add_user_message(question)
    history.add_ai_message(answer)
    return answer


def chat_unified_stream(
    question: str,
    settings: Settings,
    *,
    session_id: str = "default",
    model=None,
    file: Any | None = None,
) -> None:
    """流式问答。"""
    agent = build_unified_agent(settings, model=model)
    history = _get_session(session_id)

    messages: list[dict[str, str]] = []
    for msg in history.messages:
        if isinstance(msg, HumanMessage):
            messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            messages.append({"role": "assistant", "content": msg.content})
    messages.append({"role": "user", "content": question})

    output = file or sys.stdout
    collected: list[str] = []

    for chunk in agent.stream(
        {"messages": messages},
        stream_mode="messages",
        version="v2",
    ):
        if chunk["type"] != "messages":
            continue
        token, _ = chunk["data"]
        if isinstance(token, AIMessageChunk):
            text = ""
            if isinstance(token.content, str):
                text = token.content
            elif isinstance(token.content, list):
                for block in token.content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text += block.get("text", "")
            if text:
                output.write(text)
                output.flush()
                collected.append(text)

    output.write("\n")
    full_answer = "".join(collected)
    history.add_user_message(question)
    history.add_ai_message(full_answer)


def _extract_answer(agent_result: dict) -> str:
    messages = agent_result.get("messages", [])
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
                answer = "\n".join(p.strip() for p in parts if p.strip())
                if answer:
                    return answer
    return "（未获得有效回答）"
