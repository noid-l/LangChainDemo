from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .logging_utils import get_logger

DEFAULT_CHAT_MODEL = "gpt-4.1-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_PROXY_URL = "http://127.0.0.1:7890"
DEFAULT_NO_PROXY = "localhost,127.0.0.1"
DEFAULT_KNOWLEDGE_DIR = "data/knowledge"
DEFAULT_VECTOR_STORE_PATH = ".cache/vector_store.json"
DEFAULT_RAG_TOP_K = 4
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120

logger = get_logger(__name__)


def _read_optional_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return None


def _read_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def _is_explicitly_blank(*names: str) -> bool:
    return any(name in os.environ and not os.environ[name].strip() for name in names)


def get_project_root() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    logger.debug("解析项目根目录: %s", project_root)
    return project_root


def resolve_proxy_url() -> str:
    proxy = _read_optional_env(
        "OPENAI_PROXY",
        "HTTPS_PROXY",
        "https_proxy",
        "HTTP_PROXY",
        "http_proxy",
    )
    resolved = proxy or DEFAULT_PROXY_URL
    logger.debug("解析代理地址: %s", resolved)
    return resolved


def apply_proxy_defaults(proxy_url: str, no_proxy: str) -> None:
    os.environ.setdefault("HTTP_PROXY", proxy_url)
    os.environ.setdefault("HTTPS_PROXY", proxy_url)
    os.environ.setdefault("NO_PROXY", no_proxy)
    os.environ.setdefault("OPENAI_PROXY", proxy_url)
    logger.debug("已应用代理默认值: proxy=%s, no_proxy=%s", proxy_url, no_proxy)


@dataclass(frozen=True)
class Settings:
    project_root: Path
    knowledge_dir: Path
    vector_store_path: Path
    chat_api_key: str | None
    chat_base_url: str | None
    chat_model: str
    embedding_api_key: str | None
    embedding_base_url: str | None
    embedding_model: str
    proxy_url: str
    no_proxy: str
    rag_top_k: int
    chunk_size: int
    chunk_overlap: int


def load_settings() -> Settings:
    project_root = get_project_root()
    knowledge_dir = project_root / os.getenv("KNOWLEDGE_DIR", DEFAULT_KNOWLEDGE_DIR)
    vector_store_path = project_root / os.getenv(
        "VECTOR_STORE_PATH", DEFAULT_VECTOR_STORE_PATH
    )
    proxy_url = resolve_proxy_url()
    no_proxy = _read_optional_env("NO_PROXY", "no_proxy") or DEFAULT_NO_PROXY

    apply_proxy_defaults(proxy_url=proxy_url, no_proxy=no_proxy)

    embedding_api_key = _read_optional_env(
        "OPENAI_EMBEDDING_API_KEY",
        "EMBEDDING_API_KEY",
    )
    if embedding_api_key is None and not _is_explicitly_blank(
        "OPENAI_EMBEDDING_API_KEY",
        "EMBEDDING_API_KEY",
    ):
        embedding_api_key = _read_optional_env(
            "OPENAI_API_KEY",
            "DEEPSEEK_API_KEY",
        )

    embedding_base_url = _read_optional_env(
        "OPENAI_EMBEDDING_API_BASE",
        "OPENAI_EMBEDDING_BASE_URL",
        "EMBEDDING_BASE_URL",
    )
    if embedding_base_url is None and not _is_explicitly_blank(
        "OPENAI_EMBEDDING_API_BASE",
        "OPENAI_EMBEDDING_BASE_URL",
        "EMBEDDING_BASE_URL",
    ):
        embedding_base_url = _read_optional_env(
            "OPENAI_API_BASE",
            "OPENAI_BASE_URL",
            "DEEPSEEK_BASE_URL",
        )
    embedding_base_url = embedding_base_url or DEFAULT_OPENAI_BASE_URL

    embedding_model = _read_optional_env("OPENAI_EMBEDDING_MODEL")
    if embedding_model is None and not _is_explicitly_blank("OPENAI_EMBEDDING_MODEL"):
        embedding_model = DEFAULT_EMBEDDING_MODEL

    settings = Settings(
        project_root=project_root,
        knowledge_dir=knowledge_dir,
        vector_store_path=vector_store_path,
        chat_api_key=_read_optional_env(
            "OPENAI_API_KEY",
            "CHAT_API_KEY",
            "DEEPSEEK_API_KEY",
        ),
        chat_base_url=_read_optional_env(
            "OPENAI_API_BASE",
            "OPENAI_BASE_URL",
            "CHAT_BASE_URL",
            "DEEPSEEK_BASE_URL",
        )
        or DEFAULT_OPENAI_BASE_URL,
        chat_model=_read_optional_env("OPENAI_MODEL", "OPENAI_CHAT_MODEL")
        or DEFAULT_CHAT_MODEL,
        embedding_api_key=embedding_api_key,
        embedding_base_url=embedding_base_url,
        embedding_model=embedding_model,
        proxy_url=proxy_url,
        no_proxy=no_proxy,
        rag_top_k=_read_int_env("RAG_TOP_K", DEFAULT_RAG_TOP_K),
        chunk_size=_read_int_env("RAG_CHUNK_SIZE", DEFAULT_CHUNK_SIZE),
        chunk_overlap=_read_int_env("RAG_CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP),
    )
    logger.info(
        "配置加载完成: knowledge_dir=%s, vector_store_path=%s, chat_model=%s, "
        "embedding_model=%s, rag_top_k=%s, chunk_size=%s, chunk_overlap=%s, "
        "chat_api_key=%s, embedding_api_key=%s",
        settings.knowledge_dir,
        settings.vector_store_path,
        settings.chat_model,
        settings.embedding_model,
        settings.rag_top_k,
        settings.chunk_size,
        settings.chunk_overlap,
        "set" if settings.chat_api_key else "missing",
        "set" if settings.embedding_api_key else "missing",
    )
    return settings
