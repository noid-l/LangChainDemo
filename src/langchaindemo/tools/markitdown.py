"""MarkItDown 文档转换工具。

演示 LangChain 概念：
- 外部工具集成：将第三方库封装为 LangChain StructuredTool
- 文档预处理管道：MarkItDown → Markdown → 向量化 → 检索
- Tool 组合：与 RAG pipeline 组合实现多格式文档问答
- Vision 集成：通过配置的视觉模型（如 GLM-4.6V）进行 OCR

对照学习：
- 现有 document_qa.py：手动处理 PDF（PyPDF）/ Word（python-docx），每种格式独立编码
- MarkItDown：统一接口支持 PDF/Word/Excel/PPT/图片/HTML/音频等 20+ 格式
- MarkItDown 生态定位：专为 LLM/RAG 管线设计，输出 Markdown 适合 LLM 消费
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool
from langchain_core.vectorstores import InMemoryVectorStore
from pydantic import BaseModel, Field

from ..config import Settings
from ..logging_utils import get_logger
from ..openai_support import build_chat_model, build_embeddings, build_vision_model, ensure_chat_api_key, ensure_embedding_api_key, ensure_vision_api_key

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".pptx", ".ppt",
    ".xlsx", ".xls", ".csv",
    ".html", ".htm", ".xml",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff",
    ".mp3", ".wav",
    ".txt", ".md", ".json", ".zip",
}


class ConvertInput(BaseModel):
    file_path: str = Field(description="要转换的文件路径")


class ConvertQAInput(BaseModel):
    file_path: str = Field(description="文档文件路径")
    question: str = Field(description="关于文档的问题")


def _build_markitdown_client(settings: Settings):
    """构建 MarkItDown 实例，如果配置了视觉模型则启用 OCR。"""
    from markitdown import MarkItDown

    if settings.vision_api_key:
        try:
            from openai import OpenAI

            vision_client = OpenAI(
                api_key=settings.vision_api_key,
                base_url=settings.vision_base_url,
            )
            md = MarkItDown(
                llm_client=vision_client,
                llm_model=settings.vision_model,
            )
            logger.info(
                "MarkItDown OCR 模式: model=%s, base_url=%s",
                settings.vision_model,
                settings.vision_base_url,
            )
            return md
        except Exception as exc:
            logger.warning("视觉模型初始化失败，回退到基础模式: %s", exc)

    md = MarkItDown()
    logger.info("MarkItDown 基础模式（无 OCR）。")
    return md


def convert_to_markdown(file_path: str, settings: Settings | None = None) -> str:
    """将文件转换为 Markdown。

    使用 Microsoft MarkItDown 统一处理 20+ 种文件格式。
    如果配置了视觉模型（Vision API Key），则自动启用 LLM OCR。
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    md = _build_markitdown_client(settings) if settings else MarkItDown()
    result = md.convert(str(path))
    logger.info("MarkItDown 转换完成: file=%s, chars=%s", file_path, len(result.markdown))
    return result.markdown


def answer_with_markitdown(
    file_path: str,
    question: str,
    settings: Settings,
    *,
    top_k: int = 4,
) -> str:
    """使用 MarkItDown 转换文档后进行 RAG 问答。

    管道：文件 → MarkItDown（+OCR）→ Markdown → 切分 → 向量化 → 检索 → LLM 问答
    """
    ensure_chat_api_key(settings)
    ensure_embedding_api_key(settings)

    markdown_text = convert_to_markdown(file_path, settings=settings)
    if not markdown_text or not markdown_text.strip():
        return "文档转换后内容为空，无法回答问题。"

    chunks = _split_markdown(
        markdown_text,
        file_path,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    logger.info("Markdown 切分完成: chunks=%s", len(chunks))

    embeddings = build_embeddings(settings)
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(chunks)

    retrieved = vector_store.similarity_search(question, k=min(top_k, len(chunks)))
    logger.info("检索完成: retrieved=%s", len(retrieved))

    context = "\n\n".join(doc.page_content for doc in retrieved)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个文档问答助手。只根据提供的文档内容回答，不要编造。若文档不足，请说明。用中文回答。"),
        ("human", "文档内容：\n{context}\n\n问题：{question}"),
    ])

    chain = prompt | build_chat_model(settings) | StrOutputParser()
    return chain.invoke({"context": context, "question": question})


def _split_markdown(
    text: str,
    source: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[Document]:
    """将 Markdown 文本切分为固定大小的块。"""
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]
        chunks.append(Document(
            page_content=chunk_text,
            metadata={"source": source, "chunk": idx},
        ))
        idx += 1
        start += chunk_size - chunk_overlap
    return chunks


def build_convert_tool(settings: Settings) -> StructuredTool:
    """构建文档转换工具（供 Agent 使用）。"""

    def convert(file_path: str) -> str:
        logger.info("markitdown_convert 工具被调用: file=%s", file_path)
        try:
            return convert_to_markdown(file_path, settings=settings)
        except Exception as exc:
            return f"文档转换失败: {exc}"

    return StructuredTool.from_function(
        func=convert,
        name="markitdown_convert",
        description=(
            "将文件转换为 Markdown 文本。支持 PDF/Word/Excel/PPT/图片/HTML 等格式。"
            "当用户需要查看文件内容或文件格式不在 document_qa 支持范围内时使用。"
            "如果配置了视觉模型，PDF/图片中的内容会通过 OCR 提取。"
        ),
        args_schema=ConvertInput,
    )


def build_markitdown_qa_tool(settings: Settings) -> StructuredTool:
    """构建 MarkItDown 问答工具（供 Agent 使用）。"""

    def doc_qa(file_path: str, question: str) -> str:
        logger.info("markitdown_qa 工具被调用: file=%s, question=%s", file_path, question[:80])
        try:
            return answer_with_markitdown(file_path, question, settings)
        except Exception as exc:
            return f"文档问答失败: {exc}"

    return StructuredTool.from_function(
        func=doc_qa,
        name="markitdown_qa",
        description=(
            "读取文件并回答问题。支持 PDF/Word/Excel/PPT/图片/HTML 等格式。"
            "使用 MarkItDown 统一转换后进行 RAG 检索问答。"
            "如果配置了视觉模型，PDF/图片中的内容会通过 OCR 提取。"
        ),
        args_schema=ConvertQAInput,
    )
