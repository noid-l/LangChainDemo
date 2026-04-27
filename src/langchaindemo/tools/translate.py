"""翻译助手。

演示 LangChain 概念：
- FewShotPromptTemplate：通过示例引导翻译风格
- 翻译链（LCEL）：source_lang → target_lang
- 批量翻译：Runnable.batch() 处理多条文本

对照学习：
- 普通翻译：直接让 LLM 翻译
- FewShot 翻译：提供术语表和翻译示例，确保术语一致性和风格统一
"""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ..config import Settings
from ..logging_utils import get_logger
from ..openai_support import build_chat_model, ensure_chat_api_key

logger = get_logger(__name__)

# FewShot 示例：展示翻译风格和术语一致性
TRANSLATION_EXAMPLES = [
    {
        "input": "LangChain is a framework for building LLM-powered applications.",
        "output": "LangChain 是一个用于构建 LLM 驱动应用的框架。",
    },
    {
        "input": "Retrieval-Augmented Generation (RAG) combines search with generation.",
        "output": "检索增强生成（RAG）将搜索与生成相结合。",
    },
    {
        "input": "The agent uses tool calling to interact with external APIs.",
        "output": "智能体通过工具调用与外部 API 交互。",
    },
]

# 技术术语表
GLOSSARY = {
    "agent": "智能体",
    "tool calling": "工具调用",
    "chain": "链",
    "prompt": "提示词",
    "embedding": "嵌入",
    "vector store": "向量存储",
    "retriever": "检索器",
    "document loader": "文档加载器",
    "output parser": "输出解析器",
    "streaming": "流式输出",
    "structured output": "结构化输出",
}


def build_translate_prompt(
    source_lang: str = "English",
    target_lang: str = "中文",
) -> ChatPromptTemplate:
    """构建 FewShot 翻译 prompt。"""
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}"),
    ])

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=TRANSLATION_EXAMPLES,
    )

    glossary_text = "\n".join(f"- {en} → {zh}" for en, zh in GLOSSARY.items())

    return ChatPromptTemplate.from_messages([
        ("system",
         f"你是一个专业翻译。将{source_lang}翻译为{target_lang}。\n"
         f"请参考以下术语表确保术语一致性：\n{glossary_text}\n\n"
         f"保持技术术语的准确性和一致性。"),
        few_shot_prompt,
        ("human", "{text}"),
    ])


def translate_text(
    text: str,
    settings: Settings,
    *,
    source_lang: str = "English",
    target_lang: str = "中文",
) -> str:
    """翻译文本。"""
    ensure_chat_api_key(settings)
    prompt = build_translate_prompt(source_lang, target_lang)
    model = build_chat_model(settings)
    chain = prompt | model | StrOutputParser()
    logger.info("翻译: %s → %s, text=%s...", source_lang, target_lang, text[:50])
    return chain.invoke({"text": text})


def translate_batch(
    texts: list[str],
    settings: Settings,
    *,
    source_lang: str = "English",
    target_lang: str = "中文",
) -> list[str]:
    """批量翻译。"""
    ensure_chat_api_key(settings)
    prompt = build_translate_prompt(source_lang, target_lang)
    model = build_chat_model(settings)
    chain = prompt | model | StrOutputParser()
    logger.info("批量翻译: %s 条, %s → %s", len(texts), source_lang, target_lang)
    inputs = [{"text": t} for t in texts]
    return chain.batch(inputs)


def build_translate_tool(settings: Settings) -> StructuredTool:
    """构建翻译工具（供 Agent 使用）。"""

    class TranslateInput(BaseModel):
        text: str = Field(description="要翻译的文本")
        target_lang: str = Field(default="中文", description="目标语言")

    def translate(text: str, target_lang: str = "中文") -> str:
        logger.info("translate 工具被调用: target_lang=%s, text=%s", target_lang, text[:50])
        try:
            return translate_text(text, settings, target_lang=target_lang)
        except Exception as exc:
            return f"翻译失败: {exc}"

    return StructuredTool.from_function(
        func=translate,
        name="translate",
        description="翻译文本到指定语言。当用户需要翻译时使用。",
        args_schema=TranslateInput,
    )
