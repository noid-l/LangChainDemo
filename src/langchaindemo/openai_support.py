"""兼容旧 import 路径 — 所有模型构建逻辑已迁移到 providers.py。"""
from .providers import (  # noqa: F401
    ChatDeepSeekAdapter,
    build_chat_model,
    build_embeddings,
    build_vision_model,
    ensure_chat_api_key,
    ensure_embedding_api_key,
    ensure_vision_api_key,
)
