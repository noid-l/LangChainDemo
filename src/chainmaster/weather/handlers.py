"""天气相关 CLI 子命令的 handler 函数与 argparse 注册。"""

from __future__ import annotations

import argparse

from ..config import load_settings
from ..logging_utils import get_logger
from .service import (
    AmbiguousLocationError,
    WeatherError,
    format_location_summary,
    format_weather_report,
    query_weather,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Handler 函数
# ---------------------------------------------------------------------------


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
    from .tracing import WeatherTraceHandler

    logger.info("开始执行 weather ask 命令。")
    settings = load_settings()

    # 构建回调追踪配置
    trace_handler = None
    if args.trace:
        trace_handler = WeatherTraceHandler(output_file=args.trace_output)

    try:
        if args.stream:
            from .streaming import stream_weather_agent_answer

            stream_weather_agent_answer(args.question, settings)
        elif args.multi_tool:
            from langchain.agents import create_agent
            from .multi_tool import build_multi_tool_agent

            agent = build_multi_tool_agent(settings)
            config = {"callbacks": [trace_handler]} if trace_handler else {}
            result = agent.invoke(
                {"messages": [{"role": "user", "content": args.question}]},
                config=config,
            )
            # 提取回答
            from .agent import extract_agent_answer
            answer = extract_agent_answer(result)
            print(answer)
        else:
            if trace_handler:
                from langchain.agents import create_agent
                from .agent import build_weather_tool
                from ..openai_support import build_chat_model

                weather_tool = build_weather_tool(settings)
                agent = create_agent(
                    model=build_chat_model(settings),
                    tools=[weather_tool],
                    system_prompt="你是一个天气查询助手。通过 weather_lookup 工具查询真实天气数据来回答用户问题。输出使用中文。",
                )
                result = agent.invoke(
                    {"messages": [{"role": "user", "content": args.question}]},
                    config={"callbacks": [trace_handler]},
                )
                from .agent import extract_agent_answer
                answer = extract_agent_answer(result)
                print(answer)
            else:
                from .agent import answer_weather_question

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
    from .chain import summarize_weather, summarize_weather_stream

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
    from .chain import summarize_weather_batch

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
    from .structured import advise_weather

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


def handle_weather_graph(args: argparse.Namespace) -> None:
    from .graph import answer_weather_graph

    logger.info("开始执行 weather graph 命令。")
    settings = load_settings()

    if not args.question:
        print("请提供天气问题，例如: /weather graph 北京天气怎么样")
        return

    try:
        answer = answer_weather_graph(
            args.question, settings, thread_id=args.thread
        )
        print(answer)
    except Exception as exc:
        logger.error("工作流出错: %s", exc)
        print(f"出错: {exc}")

    logger.info("weather graph 命令执行完成。")


# ---------------------------------------------------------------------------
# argparse 注册
# ---------------------------------------------------------------------------


def register_handlers(subparsers: argparse._SubParsersAction) -> None:
    """注册天气相关 CLI 子命令。"""

    weather_parser = subparsers.add_parser("weather", help="天气查询示例")
    weather_subparsers = weather_parser.add_subparsers(
        dest="weather_command", required=True
    )

    # query
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

    # ask
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

    # summarize
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

    # summarize-batch
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

    # advise
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

    # graph
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
    weather_graph_parser.set_defaults(handler=handle_weather_graph)
