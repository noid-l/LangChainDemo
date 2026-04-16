from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from .logging_utils import get_logger


logger = get_logger(__name__)


def build_summary_prompt() -> PromptTemplate:
    logger.debug("构建摘要 PromptTemplate。")
    return PromptTemplate.from_template(
        "\n".join(
            [
                "你是一名专业的 LangChain 助手。",
                "请用{tone}的语气，围绕“{topic}”写一段入门说明。",
                "回答要求：",
                "1. 使用中文。",
                "2. 先解释概念，再给一个最小示例。",
                "3. 保持清晰、精炼、可落地。",
            ]
        )
    )


def build_rag_prompt() -> ChatPromptTemplate:
    logger.debug("构建 RAG ChatPromptTemplate。")
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一名严谨的知识库问答助手。请只基于提供的上下文回答，"
                "不要编造不存在的事实。若上下文不足，请明确说明。"
                "回答末尾请附上引用来源。",
            ),
            (
                "human",
                "\n".join(
                    [
                        "问题：{question}",
                        "",
                        "上下文：",
                        "{context}",
                        "",
                        "请给出结构清晰的中文回答，并在末尾列出引用来源。",
                    ]
                ),
            ),
        ]
    )


def format_documents(documents: list[Document]) -> str:
    if not documents:
        logger.info("没有检索到文档，将返回空上下文提示。")
        return "没有检索到相关文档。"

    logger.debug("开始格式化检索文档: count=%s", len(documents))
    blocks: list[str] = []
    for document in documents:
        source = document.metadata.get("source", "unknown")
        chunk = document.metadata.get("chunk", "?")
        blocks.append(f"[{source}#chunk-{chunk}]\n{document.page_content}")
    logger.debug("检索文档格式化完成: count=%s", len(blocks))
    return "\n\n".join(blocks)
