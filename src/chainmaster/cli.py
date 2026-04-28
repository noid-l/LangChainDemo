from __future__ import annotations

import argparse
import shlex
import sys

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import InMemoryHistory

from .config import load_settings
from .logging_utils import configure_logging, get_logger
from .openai_support import build_chat_model, ensure_chat_api_key
from .prompting import build_summary_prompt

logger = get_logger(__name__)


class NoExitArgumentParser(argparse.ArgumentParser):
    """自定义 ArgumentParser，在解析错误时不退出程序。"""

    def exit(self, status=0, message=None):
        if message:
            self._print_message(message, sys.stderr)
        raise SystemExit(status)

    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: error: {message}\n")


def build_parser() -> NoExitArgumentParser:
    parser = NoExitArgumentParser(
        prog="",
        description="LangChain Demo 交互式命令行。",
        add_help=True,
    )
    subparsers = parser.add_subparsers(dest="command")

    # === 顶层命令 ===

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


def handle_prompt(args: argparse.Namespace) -> None:
    logger.info("开始执行 prompt 命令。")
    settings = load_settings()
    prompt = build_summary_prompt()

    if args.dry_run:
        logger.info("prompt 命令处于 dry-run 模式。")
        print(prompt.invoke({"topic": args.topic, "tone": args.topic}).to_string())
        return

    ensure_chat_api_key(settings)
    chain = prompt | build_chat_model(settings) | StrOutputParser()
    logger.info("Prompt 调用链准备完成，开始请求模型。")
    print(chain.invoke({"topic": args.topic, "tone": args.tone}))


def handle_config(_args: argparse.Namespace) -> None:
    logger.info("开始执行 config 命令。")
    settings = load_settings()
    if (
        settings.qweather_api_host
        and "devapi.qweather.com" in settings.qweather_api_host
    ):
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
    print(f"qweather_project_id={'set' if settings.qweather_project_id else 'missing'}")
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


def main() -> None:
    load_dotenv()
    configure_logging()
    settings = load_settings()
    parser = build_parser()

    # 获取所有子命令用于补全
    commands = []
    if parser._subparsers:
        for action in parser._subparsers._actions:
            if isinstance(action, argparse._SubParsersAction):
                commands.extend(action.choices.keys())
    
    completer = WordCompleter(
        [f"/{cmd}" for cmd in commands] + ["/exit", "/quit", "/help", "/history"],
        ignore_case=True,
    )
    session = PromptSession(history=InMemoryHistory(), completer=completer)

    print("ChainMaster 交互式终端已启动。")
    print("输入问题开始对话，输入 /help 查看命令，输入 /history 查看历史，输入 /exit 退出。")
    print()

    from .agent import chat_unified_stream, get_session_history

    while True:
        try:
            text = session.prompt(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not text:
            continue

        if text.lower() in ("/exit", "/quit"):
            break

        if text.startswith("/"):
            cmd_line = text[1:]
            if cmd_line.lower() == "help":
                parser.print_help()
                continue

            if cmd_line.lower() == "history":
                print("\n--- 会话历史 ---")
                print(get_session_history())
                print("----------------\n")
                continue
            
            try:
                args = parser.parse_args(shlex.split(cmd_line))
                if hasattr(args, "handler"):
                    args.handler(args)
                else:
                    parser.print_help()
            except SystemExit:
                # argparse 报错或帮助信息已打印，捕获退出信号以继续循环
                continue
            except Exception as exc:
                logger.error("命令执行出错: %s", exc)
                print(f"出错: {exc}")
        else:
            # 普通聊天
            try:
                sys.stdout.write("助手: ")
                chat_unified_stream(text, settings)
                print("\n")
            except Exception as exc:
                logger.error("对话出错: %s", exc)
                print(f"出错: {exc}\n")

    print("再见！")


if __name__ == "__main__":
    main()
