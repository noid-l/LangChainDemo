# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

# 也可以用 unittest 运行
uv run python -m unittest discover -s tests -v
```

## 架构概览

```
src/langchaindemo/
├── cli.py              # CLI 解析 + 调度（~170 行），领域子命令由各包自注册
├── config.py           # frozen dataclass Settings，多级回退配置
├── providers.py        # 模型提供者注册表（openai/deepseek/qwen），可扩展
├── openai_support.py   # 兼容 shim，re-export providers.py
├── prompting.py        # PromptTemplate 文本生成
├── logging_utils.py    # 日志工具
├── agent.py            # 统一 Agent 编排器，整合所有工具
│
├── weather/            # 天气领域包
│   ├── service.py      # 核心 API（JWT 鉴权、地点解析、API 调用）
│   ├── agent.py        # Tool Calling / Agent
│   ├── chain.py        # LCEL / Runnable
│   ├── structured.py   # Structured Output
│   ├── streaming.py    # Streaming
│   ├── memory.py       # Memory
│   ├── multi_tool.py   # Multi-Tool Agent
│   ├── graph.py        # LangGraph StateGraph
│   ├── tracing.py      # Callbacks
│   └── handlers.py     # CLI handler + register_handlers()
│
├── knowledge/          # RAG 领域包
│   ├── loader.py       # 文档加载/切分
│   ├── rag.py          # 索引构建/检索/问答
│   └── handlers.py     # CLI handler + register_handlers()
│
└── tools/              # 独立工具包
    ├── web_search.py   # Tavily 搜索
    ├── document_qa.py  # PDF/Word/TXT 问答
    ├── data_analysis.py # CSV 数据分析
    ├── translate.py    # FewShot 翻译
    ├── markitdown.py   # MarkItDown 文件转换 + OCR
    └── handlers.py     # CLI handler + register_handlers()
```

### 扩展新领域

每个领域包通过 `register_handlers(subparsers)` 自注册 CLI 子命令。新增领域只需：
1. 创建新包（含 `__init__.py`、模块、`handlers.py`）
2. 在 `cli.py` 的 `build_parser()` 中加一行 `from .new_domain import register_handlers; register_handlers(subparsers)`
3. 在 `agent.py` 的 `build_all_tools()` 中加一个 try/except 加载工具

### 模块与 LangChain 概念对应

**基础设施**：
- **PromptTemplate** — `prompting.py`
- **模型提供者注册表** — `providers.py`：`register_provider()` 可扩展注册，内置 OpenAI / DeepSeek / Qwen 三个提供者
- **ChatOpenAI / Embeddings / Vision** — `openai_support.py`（re-export shim）

**天气领域**（`weather/`）：
- **Tool Calling / Agent** — `agent.py`
- **LCEL / Runnable** — `chain.py`
- **Streaming** — `streaming.py`
- **Structured Output** — `structured.py`
- **Memory** — `memory.py`
- **Multi-Tool Agent** — `multi_tool.py`
- **Callbacks** — `tracing.py`
- **LangGraph** — `graph.py`
- **确定性实现** — `service.py`（不依赖 LangChain）

**RAG 领域**（`knowledge/`）：
- **Document Loaders** — `loader.py`
- **RAG** — `rag.py`：InMemoryVectorStore，持久化到 JSON

**工具领域**（`tools/`）：
- **FewShot Prompt** — `translate.py`
- **Document Loaders** — `document_qa.py`
- **Web Search** — `web_search.py`
- **Code Generation** — `data_analysis.py`
- **MarkItDown + Vision OCR** — `markitdown.py`

**统一 Agent** — `agent.py`：整合所有工具，InMemoryChatMessageHistory 多轮会话

### 数据流

CLI → handler（各包 handlers.py）→ service 层 → 外部 API 或 LangChain chain。`chat` 命令通过 Agent 自动路由到对应工具。

## 测试约定

测试位于 `tests/`，使用 `unittest`。共享 fixture 在 `tests/conftest.py`（`build_settings`、`build_transport`）。天气测试通过 `httpx.MockTransport` mock HTTP 层，Agent 测试通过 `FakeMessagesListChatModel` mock 模型层。

## 环境变量

所有配置通过 `.env` 加载（`python-dotenv`）。模板见 `.env.example`。

**`CHAT_PROVIDER` 为必填项**（openai / deepseek / qwen），决定使用哪个模型 SDK。环境变量统一命名：
- 聊天：`CHAT_API_KEY`、`CHAT_MODEL`、`CHAT_BASE_URL`
- 向量：`EMBEDDING_API_KEY`、`EMBEDDING_MODEL`、`EMBEDDING_BASE_URL`（不填则沿用聊天配置）
- 视觉：`VISION_API_KEY`、`VISION_MODEL`、`VISION_BASE_URL`（不填则沿用聊天配置）

每个提供者有专属 key 回退：`CHAT_API_KEY` 未设时，按 provider 查找 `OPENAI_API_KEY` / `DEEPSEEK_API_KEY` / `DASHSCOPE_API_KEY`。

## 语言

代码注释、日志、CLI 输出均为中文。
