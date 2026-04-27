from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

from .config import load_settings
from .logging_utils import configure_logging, get_logger
from .openai_support import build_chat_model, ensure_chat_api_key
from .prompting import build_summary_prompt

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LangChain + OpenAI + PromptTemplate + RAG demo project."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # === 顶层命令 ===

    chat_parser = subparsers.add_parser(
        "chat",
        help="统一 Agent 入口（天气 + 知识库 + 闲聊，自动路由）",
    )
    chat_parser.add_argument(
        "question", nargs="?", default=None,
        help="问题（省略则进入交互模式）",
    )
    chat_parser.add_argument("--stream", action="store_true", help="流式输出")
    chat_parser.add_argument("--trace", action="store_true", help="启用回调追踪")
    chat_parser.add_argument("--trace-output", default=None, help="追踪日志输出文件路径")
    chat_parser.add_argument("--session", default="default", help="会话 ID（默认 default）")
    chat_parser.set_defaults(handler=handle_chat)

    prompt_parser = subparsers.add_parser("prompt", help="运行 PromptTemplate 示例")
    prompt_parser.add_argument("--topic", default="什么是 LangChain？")
    prompt_parser.add_argument("--tone", default="清晰易懂")
    prompt_parser.add_argument("--dry-run", action="store_true")
    prompt_parser.set_defaults(handler=handle_prompt)

    config_parser = subparsers.add_parser("config", help="查看当前有效配置")
    config_parser.set_defaults(handler=handle_config)

    # === 领域子命令（由各包自注册）===

    from .weather import register_handlers as register_weather
    from .knowledge import register_handlers as register_knowledge
    from .tools import register_handlers as register_tools

    register_weather(subparsers)
    register_knowledge(subparsers)
    register_tools(subparsers)

    return parser


def normalize_argv(argv: list[str]) -> list[str]:
    KNOWN_COMMANDS = {
        "chat", "prompt", "rag", "config", "weather",
        "search", "doc", "analyze", "translate", "convert",
    }
    if not argv:
        return ["chat"]
    if argv[0] == "weather" and len(argv) >= 2:
        weather_subcmds = {"query", "ask", "summarize", "summarize-batch", "advise", "chat", "graph"}
        if argv[1] not in weather_subcmds and not argv[1].startswith("-"):
            return ["weather", "query", *argv[1:]]
    if argv[0] in KNOWN_COMMANDS:
        return argv
    return ["chat", *argv]


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
    print(f"chat_provider={settings.chat_provider}")
    print(f"chat_api_key={'set' if settings.chat_api_key else 'missing'}")
    print(f"chat_base_url={settings.chat_base_url}")
    print(f"chat_model={settings.chat_model}")
    print(f"embedding_api_key={'set' if settings.embedding_api_key else 'missing'}")
    print(f"embedding_base_url={settings.embedding_base_url}")
    print(f"embedding_model={settings.embedding_model}")
    print(f"vision_api_key={'set' if settings.vision_api_key else 'missing'}")
    print(f"vision_base_url={settings.vision_base_url}")
    print(f"vision_model={settings.vision_model}")
    print(f"rag_top_k={settings.rag_top_k}")
    print(f"chunk_size={settings.chunk_size}")
    print(f"chunk_overlap={settings.chunk_overlap}")
    print(
        f"qweather_project_id={'set' if settings.qweather_project_id else 'missing'}"
    )
    print(f"qweather_key_id={'set' if settings.qweather_key_id else 'missing'}")
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


def handle_chat(args: argparse.Namespace) -> None:
    from .agent import chat_unified, chat_unified_stream
    from .weather.tracing import WeatherTraceHandler

    logger.info("开始执行统一 chat 命令。")
    settings = load_settings()

    trace_handler = None
    if args.trace:
        trace_handler = WeatherTraceHandler(output_file=args.trace_output)
    config = {"callbacks": [trace_handler]} if trace_handler else {}

    # 单次问答模式
    if args.question:
        try:
            if args.stream:
                chat_unified_stream(
                    args.question, settings, session_id=args.session,
                )
            else:
                answer = chat_unified(
                    args.question, settings,
                    session_id=args.session, config=config or None,
                )
                print(answer)
        except Exception as exc:
            logger.error("统一 Agent 出错: %s", exc)
            raise SystemExit(1) from exc

        if trace_handler:
            print(f"\n--- 追踪摘要: {trace_handler.summary} ---")
            if args.trace_output:
                trace_handler.save_trace(args.trace_output)
                print(f"追踪日志已保存到: {args.trace_output}")
        logger.info("统一 chat 单次问答完成。")
        return

    # 交互 REPL 模式
    print("统一 Agent 已启动。支持天气查询、知识库检索、闲聊。")
    print("输入问题开始对话，输入 'quit' 或 'exit' 退出。")
    print()

    try:
        while True:
            try:
                question = input("你: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not question:
                continue
            if question.lower() in ("quit", "exit", "q"):
                break
            try:
                if args.stream:
                    sys.stdout.write("助手: ")
                    chat_unified_stream(
                        question, settings,
                        session_id=args.session, file=sys.stdout,
                    )
                else:
                    answer = chat_unified(
                        question, settings,
                        session_id=args.session,
                        config=config or None,
                    )
                    print(f"\n助手: {answer}\n")
            except Exception as exc:
                logger.error("对话出错: %s", exc)
                print(f"出错: {exc}\n")
    finally:
        logger.info("统一 chat REPL 结束。")


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
