from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .logging_utils import get_logger

DEFAULT_CHAT_MODEL = "gpt-4.1-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_VISION_MODEL = "zai-org/GLM-4.6V"
DEFAULT_VISION_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
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


def _is_explicitly_blank(*names: str) -> bool:
    return any(name in os.environ and not os.environ[name].strip() for name in names)


def get_project_root() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    logger.debug("解析项目根目录: %s", project_root)
    return project_root


@dataclass(frozen=True)
class Settings:
    project_root: Path
    knowledge_dir: Path
    vector_store_path: Path
    chat_provider: str  # openai, deepseek 等
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

    # 视觉模型配置：独立 key → 回退到 chat key
    vision_api_key = _read_optional_env(
        "VISION_API_KEY",
        "OPENAI_VISION_API_KEY",
    )
    if vision_api_key is None and not _is_explicitly_blank(
        "VISION_API_KEY",
        "OPENAI_VISION_API_KEY",
    ):
        vision_api_key = _read_optional_env(
            "OPENAI_API_KEY",
            "CHAT_API_KEY",
            "DEEPSEEK_API_KEY",
        )

    vision_base_url = _read_optional_env(
        "VISION_BASE_URL",
        "OPENAI_VISION_API_BASE",
        "OPENAI_VISION_BASE_URL",
    )
    if vision_base_url is None and not _is_explicitly_blank(
        "VISION_BASE_URL",
        "OPENAI_VISION_API_BASE",
        "OPENAI_VISION_BASE_URL",
    ):
        vision_base_url = None  # 使用下面的默认值逻辑

    if vision_base_url is None:
        # 如果有独立的 vision key，说明用户想用专门的视觉平台（如硅基流动）
        # 此时使用默认视觉平台 URL；否则回退到 chat base_url
        if _read_optional_env("VISION_API_KEY", "OPENAI_VISION_API_KEY"):
            vision_base_url = DEFAULT_VISION_BASE_URL
        else:
            vision_base_url = _read_optional_env(
                "OPENAI_API_BASE",
                "OPENAI_BASE_URL",
                "CHAT_BASE_URL",
                "DEEPSEEK_BASE_URL",
            ) or DEFAULT_OPENAI_BASE_URL

    vision_model = _read_optional_env("VISION_MODEL")
    if vision_model is None and not _is_explicitly_blank("VISION_MODEL"):
        vision_model = DEFAULT_VISION_MODEL

    chat_model = _read_optional_env("OPENAI_MODEL", "OPENAI_CHAT_MODEL") or DEFAULT_CHAT_MODEL
    chat_provider = _read_optional_env("CHAT_PROVIDER")
    if not chat_provider:
        if "deepseek" in chat_model.lower():
            chat_provider = "deepseek"
        else:
            chat_provider = "openai"

    settings = Settings(
        project_root=project_root,
        knowledge_dir=knowledge_dir,
        vector_store_path=vector_store_path,
        chat_provider=chat_provider,
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
        "配置加载完成: knowledge_dir=%s, vector_store_path=%s, chat_model=%s, "
        "embedding_model=%s, vision_model=%s, rag_top_k=%s, chunk_size=%s, chunk_overlap=%s, "
        "chat_api_key=%s, embedding_api_key=%s, vision_api_key=%s, qweather_project_id=%s, "
        "qweather_key_id=%s, qweather_private_key_path=%s, "
        "qweather_api_host=%s, qweather_jwt_ttl_seconds=%s, "
        "weather_lang=%s, weather_unit=%s, weather_forecast_days=%s",
        settings.knowledge_dir,
        settings.vector_store_path,
        settings.chat_model,
        settings.embedding_model,
        settings.vision_model,
        settings.rag_top_k,
        settings.chunk_size,
        settings.chunk_overlap,
        "set" if settings.chat_api_key else "missing",
        "set" if settings.embedding_api_key else "missing",
        "set" if settings.vision_api_key else "missing",
        "set" if settings.qweather_project_id else "missing",
        "set" if settings.qweather_key_id else "missing",
        "set" if settings.qweather_private_key_path else "missing",
        settings.qweather_api_host or "missing",
        settings.qweather_jwt_ttl_seconds,
        settings.weather_lang,
        settings.weather_unit,
        settings.weather_forecast_days,
    )
    return settings
