from __future__ import annotations

import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .config import Settings


def ensure_chat_api_key(settings: Settings) -> None:
    if settings.chat_api_key:
        return

    raise SystemExit(
        "未检测到聊天模型 API Key。请先复制 .env.example 为 .env，"
        "并填写 OPENAI_API_KEY 或 CHAT_API_KEY。"
    )


def ensure_embedding_api_key(settings: Settings) -> None:
    if settings.embedding_api_key:
        return

    raise SystemExit(
        "未检测到 Embedding API Key。请先在 .env 中填写 "
        "OPENAI_EMBEDDING_API_KEY、EMBEDDING_API_KEY，"
        "或沿用 OPENAI_API_KEY。"
    )


def build_chat_model(settings: Settings) -> ChatOpenAI:
    ensure_chat_api_key(settings)
    return ChatOpenAI(
        api_key=settings.chat_api_key,
        base_url=settings.chat_base_url,
        model=settings.chat_model,
        openai_proxy=settings.proxy_url,
        temperature=0.2,
    )


def build_embeddings(settings: Settings) -> OpenAIEmbeddings:
    ensure_embedding_api_key(settings)
    return OpenAIEmbeddings(
        api_key=settings.embedding_api_key,
        base_url=settings.embedding_base_url,
        model=settings.embedding_model,
        openai_proxy=settings.proxy_url,
    )
