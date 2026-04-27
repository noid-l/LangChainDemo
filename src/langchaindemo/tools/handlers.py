from __future__ import annotations

import argparse

from ..config import load_settings
from ..logging_utils import get_logger
from .data_analysis import analyze_csv
from .document_qa import answer_document_question
from .markitdown import answer_with_markitdown, convert_to_markdown
from .translate import translate_text
from .web_search import search_and_answer

logger = get_logger(__name__)


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


def register_handlers(subparsers) -> None:
    """注册工具相关的 CLI 子命令。"""

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

    convert_parser = subparsers.add_parser(
        "convert", help="文件格式转换（演示 MarkItDown 工具集成）"
    )
    convert_parser.add_argument("file_path", help="要转换的文件路径")
    convert_parser.add_argument(
        "--question", default=None,
        help="转换后对文档提问（不提供则仅输出 Markdown）",
    )
    convert_parser.set_defaults(handler=handle_convert)
