"""知识库与 RAG 子包。

模块与 LangChain 概念的对应关系：
- **loader** — 文档加载与切分：从文件系统加载知识文档，使用 RecursiveCharacterTextSplitter 风格的文本切分
- **rag** — RAG（Retrieval-Augmented Generation）：InMemoryVectorStore 索引构建、相似度检索、LCEL 问答链
"""

from .loader import (
    ChunkingOptions,
    SUPPORTED_EXTENSIONS,
    load_knowledge_documents,
    split_documents,
    split_text,
)

from .rag import (
    IndexBuildResult,
    RagAnswer,
    answer_question,
    build_index,
    lexical_search,
    lexical_score,
    preview_question,
    tokenize,
)

__all__ = [
    # loader
    "ChunkingOptions",
    "SUPPORTED_EXTENSIONS",
    "load_knowledge_documents",
    "split_documents",
    "split_text",
    # rag
    "IndexBuildResult",
    "RagAnswer",
    "answer_question",
    "build_index",
    "lexical_search",
    "lexical_score",
    "preview_question",
    "tokenize",
]


def register_handlers(subparsers):
    """注册知识库相关的 CLI 子命令。"""
    from .handlers import register_handlers as _register
    _register(subparsers)