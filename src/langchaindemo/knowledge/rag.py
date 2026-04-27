from __future__ import annotations

import re
from time import perf_counter
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import InMemoryVectorStore

from ..config import Settings
from .loader import ChunkingOptions, load_knowledge_documents, split_documents
from ..logging_utils import get_logger
from ..openai_support import (
    build_chat_model,
    build_embeddings,
    ensure_chat_api_key,
    ensure_embedding_api_key,
)
from ..prompting import build_rag_prompt, format_documents

logger = get_logger(__name__)


@dataclass(frozen=True)
class IndexBuildResult:
    source_count: int
    chunk_count: int
    vector_store_path: Path


@dataclass(frozen=True)
class RagAnswer:
    answer: str
    documents: list[Document]
    context: str
    top_k: int
    vector_store_path: Path
    rebuilt_index: bool
    index_build_result: IndexBuildResult | None


def build_index(settings: Settings) -> IndexBuildResult:
    logger.info("开始构建向量索引。")
    ensure_embedding_api_key(settings)
    source_documents = load_knowledge_documents(
        knowledge_dir=settings.knowledge_dir,
        project_root=settings.project_root,
    )
    chunks = split_documents(
        source_documents,
        options=ChunkingOptions(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        ),
    )
    if not chunks:
        raise SystemExit("知识库中没有可索引的文档内容。")

    embeddings = build_embeddings(settings)
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(chunks)
    settings.vector_store_path.parent.mkdir(parents=True, exist_ok=True)
    vector_store.dump(str(settings.vector_store_path))

    result = IndexBuildResult(
        source_count=len(source_documents),
        chunk_count=len(chunks),
        vector_store_path=settings.vector_store_path,
    )
    logger.info(
        "向量索引构建结束: source_count=%s, chunk_count=%s, path=%s",
        result.source_count,
        result.chunk_count,
        result.vector_store_path,
    )
    return result


def answer_question(
    question: str,
    settings: Settings,
    *,
    rebuild_index: bool = False,
    top_k: int | None = None,
) -> RagAnswer:
    total_started_at = perf_counter()
    index_build_result: IndexBuildResult | None = None
    needs_rebuild = rebuild_index or not settings.vector_store_path.exists()
    if needs_rebuild:
        rebuild_started_at = perf_counter()
        logger.info("未检测到可用索引或显式要求重建，开始构建索引。")
        index_build_result = build_index(settings)
        logger.info(
            "索引构建完成，耗时 %.0f ms: source_count=%s, chunk_count=%s, path=%s",
            (perf_counter() - rebuild_started_at) * 1000,
            index_build_result.source_count,
            index_build_result.chunk_count,
            index_build_result.vector_store_path,
        )
    else:
        logger.info("复用现有索引: %s", settings.vector_store_path)

    ensure_chat_api_key(settings)
    ensure_embedding_api_key(settings)

    embedding_client_started_at = perf_counter()
    embeddings = build_embeddings(settings)
    logger.info(
        "Embedding 客户端初始化完成，耗时 %.0f ms",
        (perf_counter() - embedding_client_started_at) * 1000,
    )

    vector_store_load_started_at = perf_counter()
    vector_store = InMemoryVectorStore.load(
        str(settings.vector_store_path),
        embedding=embeddings,
    )
    logger.info(
        "向量索引加载完成，耗时 %.0f ms",
        (perf_counter() - vector_store_load_started_at) * 1000,
    )

    used_top_k = top_k or settings.rag_top_k

    retrieval_started_at = perf_counter()
    documents = vector_store.similarity_search(question, k=used_top_k)
    logger.info(
        "相似度检索完成，耗时 %.0f ms，top_k=%s，命中=%s",
        (perf_counter() - retrieval_started_at) * 1000,
        used_top_k,
        len(documents),
    )

    context_started_at = perf_counter()
    context = format_documents(documents)
    logger.info(
        "上下文组装完成，耗时 %.0f ms，context_length=%s",
        (perf_counter() - context_started_at) * 1000,
        len(context),
    )

    chat_model_started_at = perf_counter()
    chain = build_rag_prompt() | build_chat_model(settings) | StrOutputParser()
    logger.info(
        "聊天模型与调用链初始化完成，耗时 %.0f ms",
        (perf_counter() - chat_model_started_at) * 1000,
    )

    answer_started_at = perf_counter()
    answer = chain.invoke({"question": question, "context": context})
    logger.info(
        "答案生成完成，耗时 %.0f ms",
        (perf_counter() - answer_started_at) * 1000,
    )
    logger.info("RAG 总耗时 %.0f ms", (perf_counter() - total_started_at) * 1000)

    return RagAnswer(
        answer=answer,
        documents=documents,
        context=context,
        top_k=used_top_k,
        vector_store_path=settings.vector_store_path,
        rebuilt_index=needs_rebuild,
        index_build_result=index_build_result,
    )


def preview_question(
    question: str,
    settings: Settings,
    *,
    top_k: int | None = None,
) -> str:
    logger.info("开始生成 RAG dry-run 预览: question=%s", question)
    source_documents = load_knowledge_documents(
        knowledge_dir=settings.knowledge_dir,
        project_root=settings.project_root,
    )
    chunks = split_documents(
        source_documents,
        options=ChunkingOptions(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        ),
    )
    documents = lexical_search(question=question, documents=chunks, k=top_k or 3)
    logger.info("Dry-run 词法检索完成: matched=%s", len(documents))
    prompt_value = build_rag_prompt().invoke(
        {"question": question, "context": format_documents(documents)}
    )
    logger.info("Dry-run 提示词生成完成。")
    return prompt_value.to_string()


def lexical_search(question: str, documents: list[Document], k: int) -> list[Document]:
    logger.debug("开始执行词法检索: question=%s, document_count=%s, top_k=%s", question, len(documents), k)
    query_terms = {term for term in tokenize(question) if len(term) >= 2}
    ranked = sorted(
        documents,
        key=lambda document: lexical_score(document.page_content, query_terms),
        reverse=True,
    )
    results: list[Document] = []
    for document in ranked:
        if len(results) >= k:
            break
        if lexical_score(document.page_content, query_terms) > 0:
            results.append(document)
    logger.debug("词法检索完成: query_terms=%s, matched=%s", sorted(query_terms), len(results))
    return results


def lexical_score(text: str, query_terms: set[str]) -> int:
    if not query_terms:
        return 0
    lower_text = text.lower()
    return sum(lower_text.count(term) for term in query_terms)


def tokenize(text: str) -> list[str]:
    cleaned = text.lower()
    return [
        part
        for part in re.split(r"[^0-9a-zA-Z一-鿿]+", cleaned)
        if part
    ]
