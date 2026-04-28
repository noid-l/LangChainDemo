"""工具子包 -- 整合所有 LangChain 工具模块。

模块与 LangChain 概念的对应关系：
- web_search: Web 搜索工具集成（langchain-tavily）、搜索结果摘要链（LCEL）
- document_qa: Document Loaders（PDF/Word/TXT）、文档切分 + 向量检索 + 问答
- data_analysis: Python 代码生成 + 受限执行、自然语言 → 代码 → 结果转换链
- translate: FewShotPromptTemplate、翻译链（LCEL）、批量翻译（Runnable.batch）
- markitdown: 外部工具集成（StructuredTool）、文档预处理管道、Vision 集成（OCR）
"""

# web_search -- 网页搜索
from chainmaster.tools.web_search import (
    WebSearchInput,
    build_web_search_tool,
    build_search_summary_chain,
    search_and_answer,
)

# document_qa -- 文档问答
from chainmaster.tools.document_qa import (
    DocumentQuestionInput,
    SUPPORTED_EXTENSIONS as DOCUMENT_QA_SUPPORTED_EXTENSIONS,
    load_document,
    split_into_chunks,
    answer_document_question,
    build_document_qa_tool,
)

# data_analysis -- 数据分析
from chainmaster.tools.data_analysis import (
    DataAnalysisInput,
    _run_pandas_code,
    analyze_csv,
    build_data_analysis_tool,
)

# translate -- 翻译
from chainmaster.tools.translate import (
    TRANSLATION_EXAMPLES,
    GLOSSARY,
    build_translate_prompt,
    translate_text,
    translate_batch,
    build_translate_tool,
)

# markitdown -- MarkItDown 文档转换
from chainmaster.tools.markitdown import (
    ConvertInput,
    ConvertQAInput,
    SUPPORTED_EXTENSIONS as MARKITDOWN_SUPPORTED_EXTENSIONS,
    convert_to_markdown,
    answer_with_markitdown,
    build_convert_tool,
    build_markitdown_qa_tool,
)


def register_handlers(subparsers):
    """注册工具相关的 CLI 子命令。"""
    from .handlers import register_handlers as _register
    _register(subparsers)


__all__ = [
    # web_search
    "WebSearchInput",
    "build_web_search_tool",
    "build_search_summary_chain",
    "search_and_answer",
    # document_qa
    "DocumentQuestionInput",
    "DOCUMENT_QA_SUPPORTED_EXTENSIONS",
    "load_document",
    "split_into_chunks",
    "answer_document_question",
    "build_document_qa_tool",
    # data_analysis
    "DataAnalysisInput",
    "_run_pandas_code",
    "analyze_csv",
    "build_data_analysis_tool",
    # translate
    "TRANSLATION_EXAMPLES",
    "GLOSSARY",
    "build_translate_prompt",
    "translate_text",
    "translate_batch",
    "build_translate_tool",
    # markitdown
    "ConvertInput",
    "ConvertQAInput",
    "MARKITDOWN_SUPPORTED_EXTENSIONS",
    "convert_to_markdown",
    "answer_with_markitdown",
    "build_convert_tool",
    "build_markitdown_qa_tool",
    # cli
    "register_handlers",
]
