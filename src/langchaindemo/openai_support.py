from __future__ import annotations

import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .config import Settings
from .logging_utils import get_logger


logger = get_logger(__name__)


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


def build_chat_model(settings: Settings) -> ChatOpenAI:
    ensure_chat_api_key(settings)
    logger.info(
        "构建聊天模型客户端: model=%s, base_url=%s, proxy=%s",
        settings.chat_model,
        settings.chat_base_url,
        settings.proxy_url,
    )
    return ChatOpenAI(
        api_key=settings.chat_api_key,
        base_url=settings.chat_base_url,
        model=settings.chat_model,
        openai_proxy=settings.proxy_url,
        temperature=0.2,
    )


def build_embeddings(settings: Settings) -> OpenAIEmbeddings:
    ensure_embedding_api_key(settings)
    logger.info(
        "构建 Embedding 客户端: model=%s, base_url=%s, proxy=%s",
        settings.embedding_model,
        settings.embedding_base_url,
        settings.proxy_url,
    )
    return OpenAIEmbeddings(
        api_key=settings.embedding_api_key,
        base_url=settings.embedding_base_url,
        model=settings.embedding_model,
        openai_proxy=settings.proxy_url,
    )
