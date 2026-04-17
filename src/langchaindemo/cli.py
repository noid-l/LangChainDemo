from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from .config import load_settings
from .logging_utils import configure_logging, get_logger
from .openai_support import build_chat_model, ensure_chat_api_key
from .prompting import build_summary_prompt
from .rag import answer_question, build_index, preview_question
from .weather import (
    AmbiguousLocationError,
    WeatherError,
    format_location_summary,
    format_weather_report,
    query_weather,
)
from .weather_langchain import answer_weather_question


KNOWN_COMMANDS = {"prompt", "rag", "config", "weather"}
logger = get_logger(__name__)


def summarize_text(text: str, limit: int = 120) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3]}..."


def log_documents(documents: list[Document]) -> None:
    if not documents:
        logger.info("未检索到相关文档。")
        return

    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "unknown")
        chunk = document.metadata.get("chunk", "?")
        preview = summarize_text(document.page_content)
        logger.info(
            f"检索结果 {index}: source={source}, chunk={chunk}, preview={preview}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LangChain + OpenAI + PromptTemplate + RAG demo project."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prompt_parser = subparsers.add_parser("prompt", help="运行 PromptTemplate 示例")
    prompt_parser.add_argument("--topic", default="什么是 LangChain？")
    prompt_parser.add_argument("--tone", default="清晰易懂")
    prompt_parser.add_argument("--dry-run", action="store_true")
    prompt_parser.set_defaults(handler=handle_prompt)

    rag_parser = subparsers.add_parser("rag", help="运行 RAG 示例")
    rag_subparsers = rag_parser.add_subparsers(dest="rag_command", required=True)

    rag_build_parser = rag_subparsers.add_parser(
        "build", help="为知识库构建向量索引"
    )
    rag_build_parser.set_defaults(handler=handle_rag_build)

    rag_ask_parser = rag_subparsers.add_parser(
        "ask", help="基于知识库执行检索增强问答"
    )
    rag_ask_parser.add_argument("question", help="要提问的问题")
    rag_ask_parser.add_argument("--top-k", type=int, default=None)
    rag_ask_parser.add_argument("--rebuild-index", action="store_true")
    rag_ask_parser.add_argument("--dry-run", action="store_true")
    rag_ask_parser.set_defaults(handler=handle_rag_ask)

    weather_parser = subparsers.add_parser("weather", help="天气查询示例")
    weather_subparsers = weather_parser.add_subparsers(
        dest="weather_command", required=True
    )

    weather_query_parser = weather_subparsers.add_parser(
        "query", help="直接查询天气"
    )
    weather_query_parser.add_argument(
        "location", help="城市名/地区名，或 `longitude,latitude`"
    )
    weather_query_parser.add_argument(
        "--adm",
        default=None,
        help="上级行政区，用于消歧，例如 `黑龙江` 或 `beijing`",
    )
    weather_query_parser.add_argument(
        "--lang",
        default=None,
        help="语言设置，默认使用 WEATHER_LANG 或 zh",
    )
    weather_query_parser.add_argument(
        "--unit",
        choices=("m", "i"),
        default=None,
        help="单位设置：m 为公制，i 为英制",
    )
    weather_query_parser.add_argument(
        "--days",
        type=int,
        choices=(3, 7),
        default=None,
        help="预报天数，目前支持 3 或 7",
    )
    weather_query_parser.set_defaults(handler=handle_weather_query)

    weather_ask_parser = weather_subparsers.add_parser(
        "ask", help="使用 LangChain Agent 理解自然语言天气问题"
    )
    weather_ask_parser.add_argument(
        "question",
        help="自然语言天气问题，例如 `明天北京天气怎么样？`",
    )
    weather_ask_parser.set_defaults(handler=handle_weather_ask)

    config_parser = subparsers.add_parser("config", help="查看当前有效配置")
    config_parser.set_defaults(handler=handle_config)
    return parser


def normalize_argv(argv: list[str]) -> list[str]:
    if not argv:
        return ["prompt"]
    if argv[0] == "weather" and len(argv) >= 2:
        if argv[1] not in {"query", "ask"} and not argv[1].startswith("-"):
            return ["weather", "query", *argv[1:]]
    if argv[0] in KNOWN_COMMANDS:
        return argv
    if argv[0].startswith("-"):
        return ["prompt", *argv]
    return ["prompt", *argv]


def handle_prompt(args: argparse.Namespace) -> None:
    logger.info("开始执行 prompt 命令。")
    settings = load_settings()
    prompt = build_summary_prompt()

    if args.dry_run:
        logger.info("prompt 命令处于 dry-run 模式。")
        print(prompt.invoke({"topic": args.topic, "tone": args.tone}).to_string())
        return

    ensure_chat_api_key(settings)
    chain = prompt | build_chat_model(settings) | StrOutputParser()
    logger.info("Prompt 调用链准备完成，开始请求模型。")
    print(chain.invoke({"topic": args.topic, "tone": args.tone}))


def handle_rag_build(_args: argparse.Namespace) -> None:
    logger.info("开始执行 rag build 命令。")
    settings = load_settings()
    result = build_index(settings)
    logger.info("rag build 命令执行完成。")
    print(f"已构建索引: {result.vector_store_path}")
    print(f"源文档数: {result.source_count}")
    print(f"切分块数: {result.chunk_count}")


def handle_rag_ask(args: argparse.Namespace) -> None:
    settings = load_settings()

    if args.dry_run:
        print(preview_question(args.question, settings, top_k=args.top_k))
        return

    logger.info(f"开始执行 RAG 问答，question={args.question}")
    logger.info(f"knowledge_dir={settings.knowledge_dir}")
    logger.info(f"vector_store_path={settings.vector_store_path}")
    logger.info(f"chat_model={settings.chat_model}")
    logger.info(f"embedding_model={settings.embedding_model}")
    logger.info(f"top_k={args.top_k or settings.rag_top_k}")

    result = answer_question(
        question=args.question,
        settings=settings,
        rebuild_index=args.rebuild_index,
        top_k=args.top_k,
    )

    logger.info(f"检索完成，命中文档数={len(result.documents)}")
    log_documents(result.documents)
    logger.info(f"上下文长度={len(result.context)} 字符")
    logger.info("答案生成完成，开始输出最终结果。")
    print(result.answer)


def handle_config(_args: argparse.Namespace) -> None:
    logger.info("开始执行 config 命令。")
    settings = load_settings()
    if settings.qweather_api_host and "devapi.qweather.com" in settings.qweather_api_host:
        logger.warning(
            "检测到旧的 QWEATHER API Host=%s。JWT 模式下应改为控制台分配的专属 Host。",
            settings.qweather_api_host,
        )
    print(f"project_root={settings.project_root}")
    print(f"knowledge_dir={settings.knowledge_dir}")
    print(f"vector_store_path={settings.vector_store_path}")
    print(f"chat_api_key={'set' if settings.chat_api_key else 'missing'}")
    print(f"chat_base_url={settings.chat_base_url}")
    print(f"chat_model={settings.chat_model}")
    print(f"embedding_api_key={'set' if settings.embedding_api_key else 'missing'}")
    print(f"embedding_base_url={settings.embedding_base_url}")
    print(f"embedding_model={settings.embedding_model}")
    print(f"proxy_url={settings.proxy_url}")
    print(f"no_proxy={settings.no_proxy}")
    print(f"rag_top_k={settings.rag_top_k}")
    print(f"chunk_size={settings.chunk_size}")
    print(f"chunk_overlap={settings.chunk_overlap}")
    print(
        f"qweather_project_id={'set' if settings.qweather_project_id else 'missing'}"
    )
    print(f"qweather_key_id={'set' if settings.qweather_key_id else 'missing'}")
    print(
        f"qweather_private_key={'set' if settings.qweather_private_key else 'missing'}"
    )
    print(
        "qweather_private_key_path="
        f"{settings.qweather_private_key_path or 'missing'}"
    )
    print(f"qweather_api_host={settings.qweather_api_host or 'missing'}")
    print(f"qweather_jwt_ttl_seconds={settings.qweather_jwt_ttl_seconds}")
    print(f"weather_lang={settings.weather_lang}")
    print(f"weather_unit={settings.weather_unit}")
    print(f"weather_forecast_days={settings.weather_forecast_days}")
    print(f"weather_timeout_seconds={settings.weather_timeout_seconds}")
    logger.info("config 命令输出完成。")


def handle_weather_query(args: argparse.Namespace) -> None:
    logger.info("开始执行 weather query 命令。")
    settings = load_settings()
    try:
        result = query_weather(
            settings,
            location=args.location,
            adm=args.adm,
            lang=args.lang,
            unit=args.unit,
            forecast_days=args.days,
        )
    except AmbiguousLocationError as exc:
        logger.error("地点匹配到多个候选，请使用 --adm 或经纬度重新查询。")
        for index, candidate in enumerate(exc.candidates, start=1):
            logger.error("候选 %s: %s", index, format_location_summary(candidate))
        raise SystemExit(1) from exc
    except WeatherError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc

    logger.info("weather query 命令执行完成，开始输出结果。")
    print(format_weather_report(result))


def handle_weather_ask(args: argparse.Namespace) -> None:
    logger.info("开始执行 weather ask 命令。")
    settings = load_settings()
    try:
        answer = answer_weather_question(args.question, settings)
    except WeatherError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc

    logger.info("weather ask 命令执行完成，开始输出结果。")
    print(answer)


def main() -> None:
    load_dotenv()
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(normalize_argv(sys.argv[1:]))
    logger.info("命令解析完成: command=%s", getattr(args, "command", "unknown"))
    try:
        args.handler(args)
    except Exception:
        logger.exception("命令执行失败。")
        raise SystemExit(1)
