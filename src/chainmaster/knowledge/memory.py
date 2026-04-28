"""知识库多轮对话。

演示 LangChain 核心概念：
- InMemoryChatMessageHistory 管理多轮对话上下文
- RAG 检索与对话记忆的结合——每轮基于当前问题检索，历史消息作为上下文传入
- 多会话隔离（不同 session_id 独立历史）

对照学习：
- 单轮模式：knowledge/rag.py 的 answer_question（无历史）
- 多轮模式：本模块的 chat_turn（带历史 + 检索）

教学要点：多轮 RAG 的关键在于"检索用当前问题，回答用完整上下文"，
这样追问（如"详细说说第二个"）能被 LLM 正确理解。
"""

from __future__ import annotations

from time import perf_counter

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.vectorstores import InMemoryVectorStore

from ..config import Settings
from ..logging_utils import get_logger
from ..openai_support import (
    build_chat_model,
    build_embeddings,
    ensure_chat_api_key,
    ensure_embedding_api_key,
)
from ..prompting import format_documents
from .rag import build_index

logger = get_logger(__name__)

# 全局会话存储（进程内）
_sessions: dict[str, InMemoryChatMessageHistory] = {}


def get_session(session_id: str = "default") -> InMemoryChatMessageHistory:
    """获取或创建指定 ID 的会话。"""
    if session_id not in _sessions:
        _sessions[session_id] = InMemoryChatMessageHistory()
        logger.info("创建新会话: session_id=%s", session_id)
    return _sessions[session_id]


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
    top_k: int | None = None,
) -> str:
    """执行一轮知识库多轮对话。

    核心流程：
    1. 获取会话历史
    2. 基于当前问题执行 RAG 检索
    3. 拼接消息列表：历史 + 当前轮（含检索上下文）
    4. 调用 LLM，提取回答
    5. 追加本轮 Q&A 到历史
    """
    ensure_chat_api_key(settings)
    ensure_embedding_api_key(settings)

    # 确保索引存在
    if not settings.vector_store_path.exists():
        logger.info("未检测到可用索引，开始构建索引。")
        build_index(settings)

    history = get_session(session_id)

    # 加载向量索引并检索
    embeddings = build_embeddings(settings)
    vector_store = InMemoryVectorStore.load(
        str(settings.vector_store_path),
        embedding=embeddings,
    )
    used_top_k = top_k or settings.rag_top_k
    retrieval_started_at = perf_counter()
    documents = vector_store.similarity_search(question, k=used_top_k)
    logger.info(
        "多轮 RAG 检索完成，耗时 %.0f ms，top_k=%s，命中=%s",
        (perf_counter() - retrieval_started_at) * 1000,
        used_top_k,
        len(documents),
    )

    context = format_documents(documents)

    # 构建消息列表：system（含检索上下文）+ 历史 + 当前问题
    system_content = (
        "你是一名严谨的知识库问答助手，正在与用户进行多轮对话。"
        "请基于提供的上下文和之前的对话内容回答，不要编造不存在的事实。"
        "如果用户的问题需要参考之前的对话内容（如'详细说说第二个'、'还有呢？'），"
        "请根据上下文推断具体含义。若上下文不足，请明确说明。"
        "回答末尾请附上引用来源。\n\n"
        f"当前检索到的上下文：\n{context}"
    )

    messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]

    # 追加历史消息
    for msg in history.messages:
        if isinstance(msg, HumanMessage):
            messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            messages.append({"role": "assistant", "content": msg.content})

    # 追加当前问题
    messages.append({"role": "user", "content": question})

    logger.info(
        "多轮 RAG 对话: session=%s, 历史轮数=%s, 新问题=%s",
        session_id,
        len(history.messages) // 2,
        question,
    )

    # 直接调用模型（多轮对话不需要 prompt template，消息列表已包含完整上下文）
    model = build_chat_model(settings)
    answer_started_at = perf_counter()
    response = model.invoke(messages)
    answer = response.content if isinstance(response.content, str) else str(response.content)
    logger.info(
        "多轮 RAG 回答生成完成，耗时 %.0f ms",
        (perf_counter() - answer_started_at) * 1000,
    )

    # 追加到历史
    history.add_user_message(question)
    history.add_ai_message(answer)

    return answer
