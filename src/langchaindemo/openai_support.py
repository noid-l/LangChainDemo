from __future__ import annotations

from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .config import Settings
from .logging_utils import get_logger


logger = get_logger(__name__)


class ChatDeepSeekAdapter(ChatDeepSeek):
    """DeepSeek 适配器，确保 reasoning_content 在多轮对话中正确回传。

    ChatDeepSeek._create_chat_result 会将 reasoning_content 存入
    AIMessage.additional_kwargs，但基类 _get_request_payload 在序列化
    消息时只提取 content / tool_calls / function_call，丢弃了 reasoning_content。
    DeepSeek API 要求后续请求必须原样回传 reasoning_content，否则返回 400。

    修复方式：重写 _get_request_payload，在序列化后的 dict 中把
    reasoning_content 注入回对应的 assistant 消息。
    """

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)

        # 从原始消息收集各 assistant 位置的 reasoning_content
        messages = self._convert_input(input_).to_messages()
        reasoning_by_index: dict[int, str] = {}
        ai_idx = 0
        for msg in messages:
            if isinstance(msg, AIMessage):
                rc = msg.additional_kwargs.get("reasoning_content")
                if rc:
                    reasoning_by_index[ai_idx] = rc
                ai_idx += 1

        if not reasoning_by_index:
            return payload

        # 按 assistant 消息的位置注入到 payload
        ai_idx = 0
        for msg_dict in payload.get("messages", []):
            if msg_dict.get("role") == "assistant":
                if ai_idx in reasoning_by_index:
                    msg_dict["reasoning_content"] = reasoning_by_index[ai_idx]
                ai_idx += 1

        return payload


def ensure_chat_api_key(settings: Settings) -> None:
    if settings.chat_api_key:
        return

    logger.error("聊天模型 API Key 缺失。")
    raise SystemExit(
        "未检测到聊天模型 API Key。请先复制 .env.example 为 .env，"
        "并填写 OPENAI_API_KEY 或 CHAT_API_KEY。"
    )


def ensure_embedding_api_key(settings: Settings) -> None:
    if settings.embedding_api_key:
        return

    logger.error("Embedding API Key 缺失。")
    raise SystemExit(
        "未检测到 Embedding API Key。请先在 .env 中填写 "
        "OPENAI_EMBEDDING_API_KEY、EMBEDDING_API_KEY，"
        "或沿用 OPENAI_API_KEY。"
    )


def ensure_vision_api_key(settings: Settings) -> None:
    if settings.vision_api_key:
        return

    logger.error("Vision API Key 缺失。")
    raise SystemExit(
        "未检测到 Vision API Key。请先在 .env 中填写 "
        "VISION_API_KEY，或沿用 OPENAI_API_KEY。"
    )


def build_chat_model(settings: Settings) -> ChatOpenAI | ChatDeepSeekAdapter:
    ensure_chat_api_key(settings)
    logger.info(
        "构建聊天模型客户端: provider=%s, model=%s, base_url=%s",
        settings.chat_provider,
        settings.chat_model,
        settings.chat_base_url,
    )

    if settings.chat_provider == "deepseek":
        return ChatDeepSeekAdapter(
            api_key=settings.chat_api_key,
            api_base=settings.chat_base_url,
            model=settings.chat_model,
            temperature=0.2,
        )

    return ChatOpenAI(
        api_key=settings.chat_api_key,
        base_url=settings.chat_base_url,
        model=settings.chat_model,
        temperature=0.2,
    )


def build_embeddings(settings: Settings) -> OpenAIEmbeddings:
    ensure_embedding_api_key(settings)
    logger.info(
        "构建 Embedding 客户端: model=%s, base_url=%s",
        settings.embedding_model,
        settings.embedding_base_url,
    )
    return OpenAIEmbeddings(
        api_key=settings.embedding_api_key,
        base_url=settings.embedding_base_url,
        model=settings.embedding_model,
    )


def build_vision_model(settings: Settings) -> ChatOpenAI:
    ensure_vision_api_key(settings)
    logger.info(
        "构建 Vision 客户端: model=%s, base_url=%s",
        settings.vision_model,
        settings.vision_base_url,
    )
    return ChatOpenAI(
        api_key=settings.vision_api_key,
        base_url=settings.vision_base_url,
        model=settings.vision_model,
        temperature=0,
    )
