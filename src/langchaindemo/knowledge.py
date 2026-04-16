from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document

from .logging_utils import get_logger

SUPPORTED_EXTENSIONS = {".md", ".txt"}

logger = get_logger(__name__)


@dataclass(frozen=True)
class ChunkingOptions:
    chunk_size: int
    chunk_overlap: int


def load_knowledge_documents(knowledge_dir: Path, project_root: Path) -> list[Document]:
    if not knowledge_dir.exists():
        logger.error("知识库目录不存在: %s", knowledge_dir)
        raise SystemExit(f"知识库目录不存在: {knowledge_dir}")

    logger.info("开始扫描知识库目录: %s", knowledge_dir)
    documents: list[Document] = []
    skipped_empty = 0
    for path in sorted(knowledge_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        text = path.read_text(encoding="utf-8").strip()
        if not text:
            skipped_empty += 1
            continue

        relative_path = path.relative_to(project_root).as_posix()
        documents.append(
            Document(
                page_content=_normalize_text(text),
                metadata={"source": relative_path},
            )
        )
        logger.debug("已加载知识文档: source=%s, length=%s", relative_path, len(text))
    logger.info(
        "知识库扫描完成: loaded=%s, skipped_empty=%s, supported_extensions=%s",
        len(documents),
        skipped_empty,
        sorted(SUPPORTED_EXTENSIONS),
    )
    return documents


def split_documents(
    documents: list[Document], options: ChunkingOptions
) -> list[Document]:
    logger.info(
        "开始切分文档: document_count=%s, chunk_size=%s, chunk_overlap=%s",
        len(documents),
        options.chunk_size,
        options.chunk_overlap,
    )
    chunks: list[Document] = []
    for document in documents:
        source = document.metadata["source"]
        chunk_texts = split_text(document.page_content, options=options)
        logger.debug("文档切分完成: source=%s, chunk_count=%s", source, len(chunk_texts))
        for index, chunk_text in enumerate(
            chunk_texts,
            start=1,
        ):
            chunks.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        "source": source,
                        "chunk": index,
                    },
                )
            )
    logger.info("文档切分完成: total_chunks=%s", len(chunks))
    return chunks


def split_text(text: str, options: ChunkingOptions) -> list[str]:
    normalized = _normalize_text(text)
    if len(normalized) <= options.chunk_size:
        logger.debug("文本长度未超过 chunk_size，直接返回单块: length=%s", len(normalized))
        return [normalized]

    chunks: list[str] = []
    start = 0
    text_length = len(normalized)

    while start < text_length:
        end = min(start + options.chunk_size, text_length)
        end = _find_chunk_boundary(normalized, start=start, proposed_end=end)
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_length:
            break
        start = max(end - options.chunk_overlap, start + 1)

    logger.debug("文本切分完成: source_length=%s, chunks=%s", text_length, len(chunks))
    return chunks


def _find_chunk_boundary(text: str, start: int, proposed_end: int) -> int:
    if proposed_end >= len(text):
        return len(text)

    search_start = max(start, proposed_end - 80)
    for marker in ("\n\n", "\n", "。", "！", "？", ".", " "):
        position = text.rfind(marker, search_start, proposed_end)
        if position > start:
            return position + len(marker)
    return proposed_end


def _normalize_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n")
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()
