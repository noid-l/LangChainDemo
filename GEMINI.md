# GEMINI.md

This file provides guidance to Gemini CLI when working with code in this repository.

## 项目定位

教学型 LangChain Demo。所有功能优先服务于学习 LangChain 能力边界，而非产品完整性。新功能应说明它展示了哪种 LangChain 概念。同一需求可同时保留"确定性实现"和"LangChain 实现"便于对照学习。

## 常用命令

```bash
# 安装依赖
uv sync

# 运行 CLI（所有命令通过此入口）
uv run langchaindemo <command>

# 运行全部测试
uv run python -m pytest tests/

# 运行单个测试文件
uv run python -m pytest tests/test_weather.py

# 运行单个测试方法
uv run python -m pytest tests/test_weather.py::WeatherServiceTests::test_query_weather_by_city

# 也可以用 unittest 运行
uv run python -m unittest tests/test_weather.py
```

## 架构概览

CLI 入口在 `src/langchaindemo/cli.py`，使用 argparse 子命令组织，通过 `normalize_argv` 兼容旧式裸参数调用。所有子命令的 handler 函数在同一个文件中。

配置系统在 `config.py`，用 frozen dataclass `Settings` 统一管理所有环境变量，支持多级回退（如 embedding key 回退到 chat key）。`project_root` 通过 `Path(__file__).parents[2]` 解析。

模块与 LangChain 概念的对应关系：

- **PromptTemplate** — `prompting.py`：`PromptTemplate` 文本生成 + `ChatPromptTemplate` RAG 提示词
- **ChatOpenAI** — `openai_support.py`：聊天模型与 embeddings 工厂，统一通过 `langchain-openai` 接入
- **RAG** — `knowledge.py`（文档加载/切分）+ `rag.py`（索引构建/检索/问答）：使用 `InMemoryVectorStore`，支持持久化到 JSON
- **Tool Calling / Agent** — `weather_langchain.py`：将天气查询封装为 `StructuredTool`，通过 `create_agent` 构建 Agent，用 `FakeMessagesListChatModel` 做测试
- **LCEL / Runnable** — `weather_chain.py`：`prompt | model | StrOutputParser()` 管道链，演示 invoke/stream/batch 三种调用方式
- **Streaming** — `weather_streaming.py`：Agent 流式输出，`stream_mode="messages"` 逐 token 生成
- **Structured Output** — `weather_structured.py`：`with_structured_output(PydanticModel)` 让 LLM 返回结构化数据，对比确定性阈值逻辑
- **Memory** — `weather_memory.py`：`InMemoryChatMessageHistory` 多轮对话上下文管理，演示 Agent 无状态 vs 调用方管理记忆
- **Multi-Tool Agent** — `weather_multi_tool.py`：多工具选择推理，天气查询/对比/穿衣建议三个工具
- **Callbacks** — `weather_tracing.py`：`BaseCallbackHandler` 框架级可观测性，不修改链代码即可追踪每一步
- **LangGraph** — `weather_graph.py`：`StateGraph` 显式状态图，条件路由 + `InMemorySaver` 检查点持久化
- **天气服务** — `weather.py`：纯确定性实现，包含 JWT 鉴权、地点解析、API 调用，不依赖 LangChain
- **FewShot Prompt** — `translate.py`：`FewShotChatMessagePromptTemplate` 翻译示例 + 术语表，`Runnable.batch()` 批量翻译
- **Document Loaders** — `document_qa.py`：PDF（PyPDF）/ Word（python-docx）/ TXT 多格式加载 → 切分 → 向量检索 → 问答
- **Web Search** — `web_search.py`：Tavily 搜索 API 集成，搜索结果摘要链（LCEL）
- **Code Generation** — `data_analysis.py`：LLM 生成 pandas 代码 → 受限执行（`__builtins__` 为空），自然语言 → 代码 → 结果
- **统一 Agent** — `unified_agent.py`：超级 Agent REPL，整合所有工具（天气 ×3、知识库、搜索、翻译、文档问答、数据分析），`InMemoryChatMessageHistory` 多轮会话
- **MarkItDown** — `markitdown_tool.py`：Microsoft MarkItDown 统一文件转换（PDF/Word/Excel/PPT/图片/HTML 等 20+ 格式 → Markdown），封装为 StructuredTool，支持 RAG 问答

数据流：CLI → handler → service 层（weather / rag / web_search / document_qa / data_analysis / translate / markitdown_tool）→ 外部 API 或 LangChain chain。统一入口 `chat` 命令通过 Agent 自动路由到对应工具。

## 测试约定

测试位于 `tests/`，使用 `unittest`。天气相关测试通过 `httpx.MockTransport` mock HTTP 层，LangChain Agent 测试通过 `FakeMessagesListChatModel` mock 模型层。`test_weather_langchain.py` 从 `test_weather.py` 导入 `build_settings` 和 `build_transport` 复用 fixture。

## 环境变量

所有配置通过 `.env` 加载（`python-dotenv`）。模板见 `.env.example`。包括 OpenAI 兼容接口配置（支持 DeepSeek 等）、embedding 独立配置、和风天气 JWT 凭据、RAG 参数、Tavily 搜索 API Key（`TAVILY_API_KEY`）。

## 语言

代码注释、日志、CLI 输出均为中文。使用简体中文回答问题。
