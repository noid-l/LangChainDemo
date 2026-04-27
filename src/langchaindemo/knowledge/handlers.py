from __future__ import annotations

import argparse

from langchain_core.documents import Document

from ..config import load_settings
from ..logging_utils import get_logger
from .rag import answer_question, build_index, preview_question

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


def register_handlers(subparsers) -> None:
    """注册知识库相关的 CLI 子命令。"""

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
