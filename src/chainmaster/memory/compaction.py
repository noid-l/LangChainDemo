"""会话短期压缩——自动摘要。

当会话历史达到 Token/字符阈值时，自动调用 LLM 生成摘要，
替换旧消息以保持上下文"新鲜"且不超限。

展示了 LangChain 与 LLM 的 Memory 集成模式。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ..logging_utils import get_logger

if TYPE_CHECKING:
    from .store import ChatHistoryStore

logger = get_logger(__name__)

DEFAULT_MAX_CHARS = 12000
DEFAULT_HARD_MAX_CHARS = 36000
DEFAULT_KEEP_LAST = 4
_SUMMARY_PROMPT = (
    "请将以下对话历史压缩为一段简洁的摘要，保留关键事实、用户偏好和重要决策。"
    "用中文输出，不超过 300 字。\n\n"
    "对话历史：\n{history}"
)


def estimate_tokens(text: str) -> int:
    """粗略估算 Token 数（中文约 1.5 字符/token，英文约 4 字符/token）。"""
    cn_chars = sum(1 for c in text if "一" <= c <= "鿿")
    other = len(text) - cn_chars
    return int(cn_chars / 1.5 + other / 4)


def compact_if_needed(
    store: ChatHistoryStore,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    hard_max_chars: int = DEFAULT_HARD_MAX_CHARS,
    keep_last: int = DEFAULT_KEEP_LAST,
    model=None,
) -> bool:
    """检查会话长度并在超限时执行压缩。

    返回 True 表示执行了压缩，False 表示无需压缩。
    超过 hard_max_chars 时强制截断，不依赖 LLM 摘要。
    """
    total = store.total_chars()
    if total < max_chars:
        return False

    logger.info(
        "会话 [%s] 达到压缩阈值: %d / %d 字符，开始摘要...",
        store.session_id, total, max_chars,
    )

    summary = _generate_summary(store, model=model)
    if not summary:
        if total >= hard_max_chars:
            logger.warning(
                "会话 [%s] 超过硬限制 (%d / %d)，强制截断。",
                store.session_id, total, hard_max_chars,
            )
            store.remove_older_than(keep_last)
            return True
        logger.warning("摘要生成失败，跳过压缩。")
        return False

    deleted = store.remove_older_than(keep_last)
    store.prepend_message(SystemMessage(content=f"[会话摘要] {summary}"))

    logger.info(
        "会话 [%s] 压缩完成: 删除 %d 条消息，摘要已注入。",
        store.session_id, deleted,
    )
    return True


def _generate_summary(store: ChatHistoryStore, *, model=None) -> str | None:
    """调用 LLM 为历史消息生成摘要。"""
    messages = store.messages
    if not messages:
        return None

    history_text = _format_history(messages)
    prompt = _SUMMARY_PROMPT.format(history=history_text)

    try:
        if model is None:
            from ..openai_support import build_chat_model
            from ..config import load_settings
            settings = load_settings()
            model = build_chat_model(settings)

        response = model.invoke([HumanMessage(content=prompt)])
        content = response.content
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = [
                block.get("text", "") for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            return "".join(parts).strip()
        return None
    except Exception:
        logger.error("摘要生成失败", exc_info=True)
        return None


def _format_history(messages: list) -> str:
    """将消息列表格式化为可读文本。"""
    lines: list[str] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            prefix = "用户"
        elif isinstance(msg, AIMessage):
            prefix = "助手"
        elif isinstance(msg, SystemMessage):
            prefix = "系统"
        else:
            prefix = "未知"
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        lines.append(f"{prefix}: {content}")
    return "\n".join(lines)
