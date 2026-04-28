from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .logging_utils import get_logger

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_VISION_MODEL = "zai-org/GLM-4.6V"

PROVIDER_DEFAULTS: dict[str, dict[str, str | None]] = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4.1-mini",
        "key_env": "OPENAI_API_KEY",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "key_env": "DEEPSEEK_API_KEY",
    },
    "qwen": {
        "base_url": None,
        "model": "qwen-max",
        "key_env": "DASHSCOPE_API_KEY",
    },
}
DEFAULT_KNOWLEDGE_DIR = "data/knowledge"
DEFAULT_VECTOR_STORE_PATH = ".cache/vector_store.json"
DEFAULT_RAG_TOP_K = 4
DEFAULT_CHUNK_SIZE = 400
DEFAULT_CHUNK_OVERLAP = 80
DEFAULT_QWEATHER_JWT_TTL_SECONDS = 900
DEFAULT_WEATHER_LANG = "zh"
DEFAULT_WEATHER_UNIT = "m"
DEFAULT_WEATHER_FORECAST_DAYS = 3
DEFAULT_WEATHER_TIMEOUT_SECONDS = 10.0

logger = get_logger(__name__)


def _read_optional_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return None


def _read_required_env(*names: str, label: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    raise SystemExit(
        f"缺少必填配置 {label}。请在 .env 中设置 {names[0]}。"
    )


def _read_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def _read_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def _normalize_url(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    if "://" not in normalized:
        normalized = f"https://{normalized}"
    return normalized.rstrip("/")


def get_project_root() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    logger.debug("解析项目根目录: %s", project_root)
    return project_root


@dataclass(frozen=True)
class Settings:
    project_root: Path
    knowledge_dir: Path
    vector_store_path: Path
    chat_provider: str  # openai, deepseek, qwen
    chat_api_key: str | None
    chat_base_url: str | None
    chat_model: str
    embedding_api_key: str | None
    embedding_base_url: str | None
    embedding_model: str
    vision_api_key: str | None
    vision_base_url: str | None
    vision_model: str
    rag_top_k: int
    chunk_size: int
    chunk_overlap: int
    qweather_project_id: str | None
    qweather_key_id: str | None
    qweather_private_key_path: str | None
    qweather_api_host: str | None
    qweather_jwt_ttl_seconds: int
    weather_lang: str
    weather_unit: str
    weather_forecast_days: int
    weather_timeout_seconds: float


def load_settings() -> Settings:
    project_root = get_project_root()
    knowledge_dir = project_root / os.getenv("KNOWLEDGE_DIR", DEFAULT_KNOWLEDGE_DIR)
    vector_store_path = project_root / os.getenv(
        "VECTOR_STORE_PATH", DEFAULT_VECTOR_STORE_PATH
    )

    # --- 聊天模型 ---
    chat_provider = _read_required_env("CHAT_PROVIDER", label="CHAT_PROVIDER")
    provider_defaults = PROVIDER_DEFAULTS.get(chat_provider)
    if provider_defaults is None:
        valid = ", ".join(PROVIDER_DEFAULTS)
        raise SystemExit(
            f"未知的 CHAT_PROVIDER={chat_provider!r}，可选: {valid}"
        )

    key_env = provider_defaults["key_env"]
    chat_api_key = _read_optional_env("CHAT_API_KEY", key_env)
    chat_base_url = _read_optional_env(
        "CHAT_BASE_URL",
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
    ) or provider_defaults["base_url"]
    chat_model = _read_optional_env(
        "CHAT_MODEL",
        "OPENAI_MODEL",
        "OPENAI_CHAT_MODEL",
    ) or provider_defaults["model"]  # type: ignore[assignment]

    # --- 向量模型 ---
    embedding_api_key = _read_optional_env("EMBEDDING_API_KEY") or chat_api_key
    embedding_base_url = _read_optional_env("EMBEDDING_BASE_URL") or chat_base_url
    embedding_model = _read_optional_env("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL

    # --- 视觉模型 ---
    vision_api_key = _read_optional_env("VISION_API_KEY") or chat_api_key
    vision_base_url = _read_optional_env("VISION_BASE_URL") or chat_base_url
    vision_model = _read_optional_env("VISION_MODEL") or DEFAULT_VISION_MODEL

    settings = Settings(
        project_root=project_root,
        knowledge_dir=knowledge_dir,
        vector_store_path=vector_store_path,
        chat_provider=chat_provider,
        chat_api_key=chat_api_key,
        chat_base_url=chat_base_url,
        chat_model=chat_model,
        embedding_api_key=embedding_api_key,
        embedding_base_url=embedding_base_url,
        embedding_model=embedding_model,
        vision_api_key=vision_api_key,
        vision_base_url=vision_base_url,
        vision_model=vision_model,
        rag_top_k=_read_int_env("RAG_TOP_K", DEFAULT_RAG_TOP_K),
        chunk_size=_read_int_env("RAG_CHUNK_SIZE", DEFAULT_CHUNK_SIZE),
        chunk_overlap=_read_int_env("RAG_CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP),
        qweather_project_id=_read_optional_env("QWEATHER_PROJECT_ID"),
        qweather_key_id=_read_optional_env("QWEATHER_KEY_ID"),
        qweather_private_key_path=_read_optional_env("QWEATHER_PRIVATE_KEY_PATH"),
        qweather_api_host=_normalize_url(
            _read_optional_env("QWEATHER_API_HOST", "QWEATHER_BASE_URL")
        ),
        qweather_jwt_ttl_seconds=_read_int_env(
            "QWEATHER_JWT_TTL_SECONDS", DEFAULT_QWEATHER_JWT_TTL_SECONDS
        ),
        weather_lang=_read_optional_env("WEATHER_LANG") or DEFAULT_WEATHER_LANG,
        weather_unit=_read_optional_env("WEATHER_UNIT") or DEFAULT_WEATHER_UNIT,
        weather_forecast_days=_read_int_env(
            "WEATHER_FORECAST_DAYS", DEFAULT_WEATHER_FORECAST_DAYS
        ),
        weather_timeout_seconds=_read_float_env(
            "WEATHER_TIMEOUT_SECONDS", DEFAULT_WEATHER_TIMEOUT_SECONDS
        ),
    )
    logger.info(
        "配置加载完成: chat_provider=%s, chat_model=%s, "
        "embedding_model=%s, vision_model=%s, "
        "chat_api_key=%s, embedding_api_key=%s, vision_api_key=%s",
        settings.chat_provider,
        settings.chat_model,
        settings.embedding_model,
        settings.vision_model,
        "set" if settings.chat_api_key else "missing",
        "set" if settings.embedding_api_key else "missing",
        "set" if settings.vision_api_key else "missing",
    )
    return settings
