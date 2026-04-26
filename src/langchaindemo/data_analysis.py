"""数据分析助手。

演示 LangChain 概念：
- Python 代码生成 + 受限执行
- 自然语言 → 代码 → 结果的转换链
- 结构化数据处理的 Agent 工具化

对照学习：
- 传统方式：手动写 pandas 代码分析数据
- LangChain 方式：用自然语言提问，Agent 生成并执行代码

安全性说明：LLM 生成的代码在受限环境中执行，
仅允许 pandas 操作，__builtins__ 为空。
"""

from __future__ import annotations

import io
import traceback
from pathlib import Path

import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from .config import Settings
from .logging_utils import get_logger
from .openai_support import build_chat_model, ensure_chat_api_key

logger = get_logger(__name__)


class DataAnalysisInput(BaseModel):
    file_path: str = Field(description="CSV 文件路径")
    question: str = Field(description="关于数据的问题")


def _run_pandas_code(code: str, df: pd.DataFrame) -> str:
    """在受限环境中执行 pandas 代码。

    安全措施：__builtins__ 为空，只有 df 和 pd 可用。
    """
    safe_globals = {"__builtins__": {}}
    local_vars: dict = {"df": df, "pd": pd}

    try:
        # noqa: S102 - 受限环境，仅允许 pandas 操作
        exec(code, safe_globals, local_vars)  # noqa: S102
    except Exception:
        return f"执行出错:\n{traceback.format_exc()}"

    results = []
    for key, val in local_vars.items():
        if key in ("df", "pd"):
            continue
        if isinstance(val, pd.DataFrame):
            results.append(f"结果 ({key}):\n{val.to_string()}")
        elif isinstance(val, (str, int, float)):
            results.append(f"{key}: {val}")
        else:
            results.append(f"{key}: {val}")

    if not results:
        return "代码执行成功。（将结果赋值给变量以查看）"
    return "\n\n".join(results)


def analyze_csv(
    file_path: str,
    question: str,
    settings: Settings,
) -> str:
    """加载 CSV 并用 LLM 生成的代码分析数据。"""
    ensure_chat_api_key(settings)

    path = Path(file_path)
    if not path.exists():
        return f"文件不存在: {file_path}"
    if path.suffix.lower() != ".csv":
        return f"仅支持 CSV 文件，当前文件: {path.suffix}"

    df = pd.read_csv(str(path))
    logger.info("CSV 加载完成: rows=%s, cols=%s", len(df), len(df.columns))

    buf = io.StringIO()
    df.info(buf=buf)
    info_text = buf.getvalue()
    head = df.head(3).to_string()
    describe = df.describe(include="all").to_string()

    schema_desc = (
        f"CSV 文件: {len(df)} 行, {len(df.columns)} 列\n"
        f"列名: {', '.join(df.columns.tolist())}\n"
        f"数据类型:\n{info_text}\n"
        f"前 3 行:\n{head}\n"
        f"统计摘要:\n{describe}"
    )

    code_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是数据分析专家。根据 CSV 数据结构，生成 pandas 代码回答用户问题。\n"
         "变量 `df` 已加载为 DataFrame，`pd` 可用。只输出 Python 代码，不要解释。"
         "将最终结果赋值给 `result`。"),
        ("human", "数据结构：\n{schema}\n\n问题：{question}"),
    ])

    chain = code_prompt | build_chat_model(settings) | StrOutputParser()
    code = chain.invoke({"schema": schema_desc, "question": question})

    # 清理 markdown 代码块标记
    code = code.strip()
    if code.startswith("```"):
        code = "\n".join(code.split("\n")[1:])
    if code.endswith("```"):
        code = "\n".join(code.split("\n")[:-1])
    code = code.strip()

    logger.info("生成的分析代码:\n%s", code)
    return _run_pandas_code(code, df)


def build_data_analysis_tool(settings: Settings) -> StructuredTool:
    """构建数据分析工具（供 Agent 使用）。"""

    def analyze(file_path: str, question: str) -> str:
        logger.info("data_analysis 工具被调用: file=%s, question=%s", file_path, question[:80])
        try:
            return analyze_csv(file_path, question, settings)
        except Exception as exc:
            return f"分析失败: {exc}"

    return StructuredTool.from_function(
        func=analyze,
        name="data_analysis",
        description="分析 CSV 数据文件。输入文件路径和问题，自动生成代码执行分析。",
        args_schema=DataAnalysisInput,
    )
