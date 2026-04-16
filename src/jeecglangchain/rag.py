from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import InMemoryVectorStore

from .config import Settings
from .knowledge import ChunkingOptions, load_knowledge_documents, split_documents
from .openai_support import (
    build_chat_model,
    build_embeddings,
    ensure_chat_api_key,
    ensure_embedding_api_key,
)
from .prompting import build_rag_prompt, format_documents


@dataclass(frozen=True)
class IndexBuildResult:
    source_count: int
    chunk_count: int
    vector_store_path: Path


@dataclass(frozen=True)
class RagAnswer:
    answer: str
    documents: list[Document]


def build_index(settings: Settings) -> IndexBuildResult:
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

    return IndexBuildResult(
        source_count=len(source_documents),
        chunk_count=len(chunks),
        vector_store_path=settings.vector_store_path,
    )


def answer_question(
    question: str,
    settings: Settings,
    *,
    rebuild_index: bool = False,
    top_k: int | None = None,
) -> RagAnswer:
    if rebuild_index or not settings.vector_store_path.exists():
        build_index(settings)

    ensure_chat_api_key(settings)
    ensure_embedding_api_key(settings)
    embeddings = build_embeddings(settings)
    vector_store = InMemoryVectorStore.load(
        str(settings.vector_store_path),
        embedding=embeddings,
    )
    documents = vector_store.similarity_search(question, k=top_k or settings.rag_top_k)
    context = format_documents(documents)
    chain = build_rag_prompt() | build_chat_model(settings) | StrOutputParser()
    answer = chain.invoke({"question": question, "context": context})
    return RagAnswer(answer=answer, documents=documents)


def preview_question(
    question: str,
    settings: Settings,
    *,
    top_k: int | None = None,
) -> str:
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
    prompt_value = build_rag_prompt().invoke(
        {"question": question, "context": format_documents(documents)}
    )
    return prompt_value.to_string()


def lexical_search(question: str, documents: list[Document], k: int) -> list[Document]:
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
        for part in re.split(r"[^0-9a-zA-Z\u4e00-\u9fff]+", cleaned)
        if part
    ]
