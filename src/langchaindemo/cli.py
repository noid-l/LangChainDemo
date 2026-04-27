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
from .weather_chain import summarize_weather, summarize_weather_batch, summarize_weather_stream
from .weather_langchain import answer_weather_question
from .weather_streaming import stream_weather_agent_answer, stream_weather_report_lines
from .weather_structured import advise_weather
from .weather_memory import chat_turn, clear_session, format_history, get_session, list_sessions
from .weather_multi_tool import answer_with_multi_tool
from .weather_tracing import WeatherTraceHandler
from .weather_graph import answer_weather_graph
from .unified_agent import chat_unified, chat_unified_stream
from .web_search import search_and_answer
from .document_qa import answer_document_question
from .data_analysis import analyze_csv
from .translate import translate_text, translate_batch
from .markitdown_tool import convert_to_markdown, answer_with_markitdown


KNOWN_COMMANDS = {"chat", "prompt", "rag", "config", "weather", "search", "doc", "analyze", "translate", "convert"}
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

    chat_parser = subparsers.add_parser(
        "chat",
        help="统一 Agent 入口（天气 + 知识库 + 闲聊，自动路由）",
    )
    chat_parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help="问题（省略则进入交互模式）",
    )
    chat_parser.add_argument(
        "--stream",
        action="store_true",
        help="流式输出",
    )
    chat_parser.add_argument(
        "--trace",
        action="store_true",
        help="启用回调追踪",
    )
    chat_parser.add_argument(
        "--trace-output",
        default=None,
        help="追踪日志输出文件路径",
    )
    chat_parser.add_argument(
        "--session",
        default="default",
        help="会话 ID（默认 default）",
    )
    chat_parser.set_defaults(handler=handle_chat)

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
    weather_ask_parser.add_argument(
        "--stream",
        action="store_true",
        help="流式输出（逐 token 打印，演示 Runnable.stream）",
    )
    weather_ask_parser.add_argument(
        "--multi-tool",
        action="store_true",
        help="启用多工具 Agent（天气查询+对比+穿衣建议）",
    )
    weather_ask_parser.add_argument(
        "--trace",
        action="store_true",
        help="启用回调追踪（演示 BaseCallbackHandler）",
    )
    weather_ask_parser.add_argument(
        "--trace-output",
        default=None,
        help="追踪日志输出文件路径",
    )
    weather_ask_parser.set_defaults(handler=handle_weather_ask)

    weather_summarize_parser = weather_subparsers.add_parser(
        "summarize",
        help="使用 LCEL 链生成天气摘要（演示 Runnable invoke/stream/batch）",
    )
    weather_summarize_parser.add_argument("location", help="城市名或地区名")
    weather_summarize_parser.add_argument(
        "--style",
        choices=("brief", "detailed", "casual"),
        default="brief",
        help="摘要风格（默认 brief）",
    )
    weather_summarize_parser.add_argument(
        "--adm",
        default=None,
        help="上级行政区，用于消歧",
    )
    weather_summarize_parser.add_argument(
        "--mode",
        choices=("lcel", "deterministic"),
        default="lcel",
        help="实现模式：lcel 使用 LCEL 链，deterministic 使用手动调用（默认 lcel）",
    )
    weather_summarize_parser.add_argument(
        "--stream",
        action="store_true",
        help="使用流式输出（逐 token 打印）",
    )
    weather_summarize_parser.set_defaults(handler=handle_weather_summarize)

    weather_batch_parser = weather_subparsers.add_parser(
        "summarize-batch",
        help="批量天气摘要（演示 Runnable.batch 并发处理）",
    )
    weather_batch_parser.add_argument(
        "locations", nargs="+", help="多个城市名，空格分隔"
    )
    weather_batch_parser.add_argument(
        "--style",
        choices=("brief", "detailed", "casual"),
        default="brief",
        help="摘要风格（默认 brief）",
    )
    weather_batch_parser.set_defaults(handler=handle_weather_summarize_batch)

    weather_advise_parser = weather_subparsers.add_parser(
        "advise",
        help="结构化穿衣建议（演示 with_structured_output / PydanticOutputParser）",
    )
    weather_advise_parser.add_argument("location", help="城市名或地区名")
    weather_advise_parser.add_argument(
        "--adm", default=None, help="上级行政区，用于消歧"
    )
    weather_advise_parser.add_argument(
        "--mode",
        choices=("deterministic", "langchain"),
        default="deterministic",
        help="实现模式（默认 deterministic）",
    )
    weather_advise_parser.add_argument(
        "--json",
        dest="output_json",
        action="store_true",
        help="输出原始 JSON",
    )
    weather_advise_parser.set_defaults(handler=handle_weather_advise)

    weather_chat_parser = weather_subparsers.add_parser(
        "chat",
        help="多轮天气对话（演示 ChatMessageHistory 对话记忆）",
    )
    weather_chat_parser.add_argument(
        "--session",
        default="default",
        help="会话名称（默认 default）",
    )
    weather_chat_parser.add_argument(
        "--history",
        action="store_true",
        help="打印会话历史后退出",
    )
    weather_chat_parser.set_defaults(handler=handle_weather_chat)

    weather_graph_parser = weather_subparsers.add_parser(
        "graph",
        help="LangGraph 工作流天气问答（演示 StateGraph 条件路由）",
    )
    weather_graph_parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help="天气问题",
    )
    weather_graph_parser.add_argument(
        "--thread",
        default="default",
        help="会话线程 ID（默认 default）",
    )
    weather_graph_parser.add_argument(
        "--repl",
        action="store_true",
        help="交互式多轮对话模式",
    )
    weather_graph_parser.set_defaults(handler=handle_weather_graph)

    # === 新领域子命令 ===

    search_parser = subparsers.add_parser(
        "search", help="网页搜索（演示 Web Search 工具）"
    )
    search_parser.add_argument("question", help="搜索问题")
    search_parser.set_defaults(handler=handle_search)

    doc_parser = subparsers.add_parser(
        "doc", help="文档问答（演示 Document Loaders）"
    )
    doc_parser.add_argument("file_path", help="文档路径（PDF/Word/TXT）")
    doc_parser.add_argument("question", help="关于文档的问题")
    doc_parser.set_defaults(handler=handle_doc)

    analyze_parser = subparsers.add_parser(
        "analyze", help="数据分析（演示代码生成 + 执行）"
    )
    analyze_parser.add_argument("file_path", help="CSV 文件路径")
    analyze_parser.add_argument("question", help="关于数据的问题")
    analyze_parser.set_defaults(handler=handle_analyze)

    translate_parser = subparsers.add_parser(
        "translate", help="翻译（演示 FewShotPromptTemplate）"
    )
    translate_parser.add_argument("text", help="要翻译的文本")
    translate_parser.add_argument(
        "--target-lang", default="中文", help="目标语言（默认中文）"
    )
    translate_parser.add_argument(
        "--source-lang", default="English", help="源语言（默认 English）"
    )
    translate_parser.set_defaults(handler=handle_translate)

    config_parser = subparsers.add_parser("config", help="查看当前有效配置")
    config_parser.set_defaults(handler=handle_config)

    convert_parser = subparsers.add_parser(
        "convert", help="文件格式转换（演示 MarkItDown 工具集成）"
    )
    convert_parser.add_argument("file_path", help="要转换的文件路径")
    convert_parser.add_argument(
        "--question", default=None,
        help="转换后对文档提问（不提供则仅输出 Markdown）",
    )
    convert_parser.set_defaults(handler=handle_convert)
    return parser


def normalize_argv(argv: list[str]) -> list[str]:
    if not argv:
        return ["chat"]
    if argv[0] == "weather" and len(argv) >= 2:
        weather_subcmds = {"query", "ask", "summarize", "summarize-batch", "advise", "chat", "graph"}
        if argv[1] not in weather_subcmds and not argv[1].startswith("-"):
            return ["weather", "query", *argv[1:]]
    if argv[0] in KNOWN_COMMANDS:
        return argv
    # 裸参数或未知命令路由到 chat
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
    print(f"chat_provider={settings.chat_provider}")
    print(f"chat_api_key={'set' if settings.chat_api_key else 'missing'}")
    print(f"chat_base_url={settings.chat_base_url}")
    print(f"chat_model={settings.chat_model}")
    print(f"embedding_api_key={'set' if settings.embedding_api_key else 'missing'}")
    print(f"embedding_base_url={settings.embedding_base_url}")
    print(f"embedding_model={settings.embedding_model}")
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

    # 构建回调追踪配置
    trace_handler = None
    if args.trace:
        trace_handler = WeatherTraceHandler(output_file=args.trace_output)

    try:
        if args.stream:
            stream_weather_agent_answer(args.question, settings)
        elif args.multi_tool:
            from langchain.agents import create_agent
            from .weather_multi_tool import build_multi_tool_agent

            agent = build_multi_tool_agent(settings)
            config = {"callbacks": [trace_handler]} if trace_handler else {}
            result = agent.invoke(
                {"messages": [{"role": "user", "content": args.question}]},
                config=config,
            )
            # 提取回答
            from .weather_langchain import extract_agent_answer
            answer = extract_agent_answer(result)
            print(answer)
        else:
            if trace_handler:
                from langchain.agents import create_agent
                from .weather_langchain import build_weather_tool

                weather_tool = build_weather_tool(settings)
                from .openai_support import build_chat_model
                agent = create_agent(
                    model=build_chat_model(settings),
                    tools=[weather_tool],
                    system_prompt="你是一个天气查询助手。通过 weather_lookup 工具查询真实天气数据来回答用户问题。输出使用中文。",
                )
                result = agent.invoke(
                    {"messages": [{"role": "user", "content": args.question}]},
                    config={"callbacks": [trace_handler]},
                )
                from .weather_langchain import extract_agent_answer
                answer = extract_agent_answer(result)
                print(answer)
            else:
                answer = answer_weather_question(args.question, settings)
                print(answer)
    except WeatherError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc

    if trace_handler:
        print(f"\n--- 追踪摘要: {trace_handler.summary} ---")
        if args.trace_output:
            trace_handler.save_trace(args.trace_output)
            print(f"追踪日志已保存到: {args.trace_output}")

    logger.info("weather ask 命令执行完成。")


def handle_weather_summarize(args: argparse.Namespace) -> None:
    logger.info("开始执行 weather summarize 命令。")
    settings = load_settings()
    try:
        if args.stream:
            summarize_weather_stream(
                args.location,
                settings,
                style=args.style,
                adm=args.adm,
            )
        else:
            result = summarize_weather(
                args.location,
                settings,
                style=args.style,
                adm=args.adm,
                mode=args.mode,
            )
            print(result)
    except WeatherError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc
    logger.info("weather summarize 命令执行完成。")


def handle_weather_summarize_batch(args: argparse.Namespace) -> None:
    logger.info("开始执行 weather summarize-batch 命令。")
    settings = load_settings()
    try:
        results = summarize_weather_batch(
            args.locations,
            settings,
            style=args.style,
        )
    except WeatherError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc

    for location, summary in zip(args.locations, results):
        print(f"--- {location} ---")
        print(summary)
    logger.info("weather summarize-batch 命令执行完成。")


def handle_weather_advise(args: argparse.Namespace) -> None:
    logger.info("开始执行 weather advise 命令。")
    settings = load_settings()
    try:
        result = advise_weather(
            args.location,
            settings,
            adm=args.adm,
            mode=args.mode,
            output_json=args.output_json,
        )
        print(result)
    except WeatherError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc
    logger.info("weather advise 命令执行完成。")


def handle_weather_chat(args: argparse.Namespace) -> None:
    logger.info("开始执行 weather chat 命令。")
    settings = load_settings()

    if args.history:
        session = get_session(args.session)
        print(f"会话 '{args.session}' 历史记录：")
        print(format_history(session))
        return

    print(f"天气对话已启动（会话: {args.session}）")
    print("输入问题进行天气查询，输入 'quit' 或 'exit' 退出，输入 'history' 查看历史。")
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
            if question.lower() == "history":
                session = get_session(args.session)
                print(format_history(session))
                print()
                continue

            try:
                answer = chat_turn(question, settings, session_id=args.session)
                print(f"\n助手: {answer}\n")
            except Exception as exc:
                logger.error("对话出错: %s", exc)
                print(f"出错: {exc}\n")
    finally:
        logger.info("weather chat 命令结束。")


def handle_weather_graph(args: argparse.Namespace) -> None:
    logger.info("开始执行 weather graph 命令。")
    settings = load_settings()

    if args.repl:
        print(f"LangGraph 天气工作流已启动（线程: {args.thread}）")
        print("输入问题进行天气查询，输入 'quit' 或 'exit' 退出。")
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
                    answer = answer_weather_graph(
                        question, settings, thread_id=args.thread
                    )
                    print(f"\n助手: {answer}\n")
                except Exception as exc:
                    logger.error("工作流出错: %s", exc)
                    print(f"出错: {exc}\n")
        finally:
            logger.info("weather graph REPL 结束。")
    else:
        if not args.question:
            print("请提供天气问题，例如: weather graph 北京天气怎么样")
            raise SystemExit(1)
        try:
            answer = answer_weather_graph(
                args.question, settings, thread_id=args.thread
            )
            print(answer)
        except Exception as exc:
            logger.error("工作流出错: %s", exc)
            raise SystemExit(1) from exc

    logger.info("weather graph 命令执行完成。")


def handle_search(args: argparse.Namespace) -> None:
    logger.info("开始执行 search 命令。")
    settings = load_settings()
    try:
        answer = search_and_answer(args.question, settings)
        print(answer)
    except Exception as exc:
        logger.error("搜索失败: %s", exc)
        raise SystemExit(1) from exc
    logger.info("search 命令执行完成。")


def handle_doc(args: argparse.Namespace) -> None:
    logger.info("开始执行 doc 命令。")
    settings = load_settings()
    try:
        answer = answer_document_question(args.file_path, args.question, settings)
        print(answer)
    except Exception as exc:
        logger.error("文档问答失败: %s", exc)
        raise SystemExit(1) from exc
    logger.info("doc 命令执行完成。")


def handle_analyze(args: argparse.Namespace) -> None:
    logger.info("开始执行 analyze 命令。")
    settings = load_settings()
    try:
        answer = analyze_csv(args.file_path, args.question, settings)
        print(answer)
    except Exception as exc:
        logger.error("数据分析失败: %s", exc)
        raise SystemExit(1) from exc
    logger.info("analyze 命令执行完成。")


def handle_translate(args: argparse.Namespace) -> None:
    logger.info("开始执行 translate 命令。")
    settings = load_settings()
    try:
        result = translate_text(
            args.text, settings,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
        )
        print(result)
    except Exception as exc:
        logger.error("翻译失败: %s", exc)
        raise SystemExit(1) from exc
    logger.info("translate 命令执行完成。")


def handle_convert(args: argparse.Namespace) -> None:
    logger.info("开始执行 convert 命令。")
    settings = load_settings()
    try:
        if args.question:
            answer = answer_with_markitdown(args.file_path, args.question, settings)
            print(answer)
        else:
            markdown = convert_to_markdown(args.file_path, settings=settings)
            print(markdown)
    except Exception as exc:
        logger.error("文档转换失败: %s", exc)
        raise SystemExit(1) from exc
    logger.info("convert 命令执行完成。")


def handle_chat(args: argparse.Namespace) -> None:
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
