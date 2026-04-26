"""对话记忆示例。

演示 LangChain 核心概念：
- InMemoryChatMessageHistory 管理多轮对话上下文
- Agent 本身不管理记忆——记忆由调用方维护
- 多会话隔离（不同 session_id 独立历史）

对照学习：
- 确定性路径：用 list 手动拼接消息
- LangChain 路径：ChatMessageHistory 提供标准化的消息管理

教学要点：agent.invoke() 每次都是无状态的，
多轮对话的关键在于调用方把历史消息传入 messages 列表。
"""

from __future__ import annotations

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from .config import Settings
from .logging_utils import get_logger
from .openai_support import build_chat_model, ensure_chat_api_key
from .weather import ensure_qweather_jwt_config
from .weather_langchain import build_weather_tool

logger = get_logger(__name__)

# 全局会话存储（进程内）
_sessions: dict[str, InMemoryChatMessageHistory] = {}


def get_session(session_id: str = "default") -> InMemoryChatMessageHistory:
    """获取或创建指定 ID 的会话。"""
    if session_id not in _sessions:
        _sessions[session_id] = InMemoryChatMessageHistory()
        logger.info("创建新会话: session_id=%s", session_id)
    return _sessions[session_id]


def list_sessions() -> list[str]:
    """列出所有活跃会话。"""
    return list(_sessions.keys())


def clear_session(session_id: str) -> bool:
    """清除指定会话。"""
    if session_id in _sessions:
        del _sessions[session_id]
        logger.info("已清除会话: session_id=%s", session_id)
        return True
    return False


def format_history(history: InMemoryChatMessageHistory) -> str:
    """格式化会话历史为可读文本。"""
    messages = history.messages
    if not messages:
        return "（空会话）"
    lines = []
    for msg in messages:
        role = "用户" if isinstance(msg, HumanMessage) else "助手"
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        preview = content[:100] + "..." if len(content) > 100 else content
        lines.append(f"[{role}] {preview}")
    return "\n".join(lines)


def chat_turn(
    question: str,
    settings: Settings,
    *,
    session_id: str = "default",
    model=None,
) -> str:
    """执行一轮对话——将历史消息传入 agent，然后追加本轮问答。

    这是多轮对话的核心模式：
    1. 从 ChatMessageHistory 取出历史消息
    2. 拼接新的用户消息
    3. 调用 agent（agent 看到完整上下文）
    4. 将本轮问答追加回历史
    """
    from langchain.agents import create_agent

    ensure_chat_api_key(settings)
    ensure_qweather_jwt_config(settings)

    history = get_session(session_id)
    weather_tool = build_weather_tool(settings)
    agent = create_agent(
        model=model or build_chat_model(settings),
        tools=[weather_tool],
        system_prompt=(
            "你是一个天气查询助手，正在与用户进行多轮对话。"
            "通过 weather_lookup 工具查询真实天气数据来回答用户问题。"
            "如果用户的问题需要参考之前的对话内容（如'明天呢？'、'那里的呢？'），"
            "请根据上下文推断具体地点和时间。"
            "输出使用中文，结构清晰。"
        ),
    )

    # 构建消息列表：历史 + 本轮
    messages: list[dict[str, str]] = []
    for msg in history.messages:
        if isinstance(msg, HumanMessage):
            messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            messages.append({"role": "assistant", "content": msg.content})

    messages.append({"role": "user", "content": question})

    logger.info(
        "多轮对话: session=%s, 历史轮数=%s, 新问题=%s",
        session_id,
        len(history.messages) // 2,
        question,
    )

    result = agent.invoke({"messages": messages})

    # 提取回答
    answer = _extract_answer(result)

    # 追加到历史
    history.add_user_message(question)
    history.add_ai_message(answer)

    logger.info("多轮对话完成: session=%s", session_id)
    return answer


def _extract_answer(agent_result: dict) -> str:
    """从 agent 结果中提取最终回答。"""
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
                answer = "\n".join(part.strip() for part in parts if part.strip())
                if answer:
                    return answer
    return "（未获得有效回答）"
