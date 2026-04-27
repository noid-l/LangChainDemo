"""网页搜索助手。

演示 LangChain 概念：
- Web 搜索工具集成（langchain-tavily）
- 实时信息获取 vs 知识库检索的区别
- 搜索结果摘要链（LCEL）

对照学习：
- 知识库检索：从本地静态文档中搜索，数据可能过时
- 网页搜索：获取实时信息，但需要 API Key 且有调用成本
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ..config import Settings
from ..logging_utils import get_logger
from ..openai_support import build_chat_model, ensure_chat_api_key

logger = get_logger(__name__)


class WebSearchInput(BaseModel):
    query: str = Field(description="搜索关键词")


def build_web_search_tool(settings: Settings) -> StructuredTool:
    """构建网页搜索工具。需要 TAVILY_API_KEY 环境变量。"""
    import os

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("未配置 TAVILY_API_KEY。请在 .env 中添加。")

    def web_search(query: str) -> str:
        logger.info("web_search 工具被调用: query=%s", query)
        from langchain_tavily import TavilySearch

        search = TavilySearch(max_results=5, topic="general")
        results = search.invoke(query)

        if isinstance(results, list):
            snippets = []
            for item in results[:5]:
                title = item.get("title", "")
                content = item.get("content", "")
                url = item.get("url", "")
                snippets.append(f"- {title}\n  {content}\n  来源: {url}")
            return "\n\n".join(snippets)
        return str(results)

    return StructuredTool.from_function(
        func=web_search,
        name="web_search",
        description="搜索互联网获取实时信息。当用户问最新新闻、近期事件、实时数据时使用。",
        args_schema=WebSearchInput,
    )


def build_search_summary_chain(settings: Settings):
    """构建搜索结果摘要链。"""
    ensure_chat_api_key(settings)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个信息摘要助手。根据搜索结果，用中文简洁回答用户问题。标注信息来源。"),
        ("human", "问题：{question}\n\n搜索结果：\n{search_results}"),
    ])
    model = build_chat_model(settings)
    return prompt | model | StrOutputParser()


def search_and_answer(
    question: str,
    settings: Settings,
) -> str:
    """搜索并生成摘要回答。"""
    import os

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "未配置 TAVILY_API_KEY，无法搜索。请在 .env 中添加。"

    from langchain_tavily import TavilySearch

    logger.info("开始搜索: question=%s", question)
    search = TavilySearch(max_results=5, topic="general")
    results = search.invoke(question)

    if isinstance(results, list):
        snippets = []
        for item in results[:5]:
            title = item.get("title", "")
            content = item.get("content", "")
            url = item.get("url", "")
            snippets.append(f"- {title}\n  {content}\n  来源: {url}")
        search_text = "\n\n".join(snippets)
    else:
        search_text = str(results)

    logger.info("搜索完成，开始生成摘要。")
    chain = build_search_summary_chain(settings)
    return chain.invoke({"question": question, "search_results": search_text})
