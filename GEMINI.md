# GEMINI.md

This file provides guidance to Gemini CLI when working with code in this repository.

## 项目定位

教学型 LangChain Demo。所有功能优先服务于学习 LangChain 能力边界，而非产品完整性。新功能应说明它展示了哪种 LangChain 概念。同一需求可同时保留"确定性实现"和"LangChain 实现"便于对照学习。

## 常用命令

```bash
# 安装依赖
uv sync

# 启动交互式终端（推荐）
uv run chainmaster

# 运行全部测试
uv run python -m pytest tests/

# 运行单个测试文件
uv run python -m pytest tests/test_weather.py

# 也可以用 unittest 运行
uv run python -m unittest discover -s tests -v
```

## 架构概览

```
src/chainmaster/
├── cli.py              # 交互式 REPL 入口，支持斜杠命令（如 /config, /mcp, /skills）
├── config.py           # frozen dataclass Settings，多级回退配置
├── providers.py        # 模型提供者注册表（openai/deepseek/qwen），可扩展
├── openai_support.py   # 兼容 shim，re-export providers.py
├── prompting.py        # PromptTemplate 文本生成
├── logging_utils.py    # 日志工具
├── agent.py            # 统一 Agent 编排器，整合所有工具与记忆系统
│
├── memory/             # 核心记忆系统
│   ├── store.py        # SQLite 持久化历史，支持 FTS5 全文检索
│   └── compaction.py   # 会话自动压缩（摘要生成）
│
├── mcp/                # Model Context Protocol 集成
│   ├── client.py       # MCP 客户端，连接外部工具服务
│   ├── adapter.py      # MCP 工具适配层，映射为 LangChain StructuredTool
│   └── server/         # 内置 MCP Server（如知识图谱记忆）
│
├── skills/             # 技能系统（Progressive Disclosure）
│   ├── registry.py     # 扫描 data/skills/*.md 构建轻量索引
│   └── loader.py       # 按需加载完整指令内容
│
├── weather/            # 天气领域包（教学示例集）
│   ├── service.py      # 确定性 API 实现（JWT、地点解析）
│   ├── agent.py        # Tool Calling / Agent
│   ├── chain.py        # LCEL / Runnable
│   ├── structured.py   # Structured Output
│   ├── streaming.py    # Streaming
│   ├── memory.py       # Memory
│   ├── multi_tool.py   # Multi-Tool Agent
│   ├── graph.py        # LangGraph StateGraph 实现
│   ├── tracing.py      # Callbacks
│   └── handlers.py     # CLI handler
│
├── knowledge/          # RAG 领域包
│   ├── loader.py       # 文档加载/切分
│   ├── rag.py          # 索引构建/检索/问答（InMemoryVectorStore）
│   └── handlers.py     # CLI handler
│
└── tools/              # 独立工具包
    ├── web_search.py   # Tavily 搜索
    ├── document_qa.py  # PDF/Word/TXT 问答
    ├── data_analysis.py # CSV 数据分析
    ├── translate.py    # FewShot 翻译
    ├── markitdown.py   # MarkItDown 文件转换 + OCR
    └── handlers.py     # CLI handler
```

### 扩展新领域

每个领域包通过 `register_handlers(subparsers)` 自注册 CLI 子命令。新增领域只需：
1. 创建新包（含 `__init__.py`、模块、`handlers.py`）
2. 在 `cli.py` 的 `build_parser()` 中加一行 `from .new_domain import register_handlers; register_handlers(subparsers)`
3. 在 `agent.py` 的 `build_all_tools()` 中加一个 try/except 加载工具

### 模块与 LangChain 概念对应

**基础设施**：
- **PromptTemplate** — `prompting.py`
- **模型提供者注册表** — `providers.py`：`register_provider()`
- **ChatMessageHistory** — `memory/store.py` (SQLite 持久化)
- **Automatic Summary** — `memory/compaction.py` (会话压缩)
- **Model Context Protocol** — `mcp/` (工具生态扩展)

**天气领域**（`weather/`）：
- **Tool Calling / Agent** — `agent.py`
- **LCEL / Runnable** — `chain.py`
- **Streaming** — `streaming.py`
- **Structured Output** — `structured.py`
- **Memory** — `memory.py`
- **Multi-Tool Agent** — `multi_tool.py`
- **Callbacks** — `tracing.py`
- **LangGraph** — `graph.py` (StateGraph 工作流)

**RAG 领域**（`knowledge/`）：
- **Document Loaders** — `loader.py`
- **RAG** — `rag.py`：InMemoryVectorStore，持久化到 JSON

**技能系统**（`skills/`）：
- **Progressive Disclosure** — `registry.py` (只加载元数据，按需加载指令)
- **System Prompt Optimization** — `loader.py` (动态注入指令)

**统一 Agent** — `agent.py`：整合所有工具，ChatHistoryStore 持久化会话，支持自动压缩。

### 数据流

交互式终端 → `cli.py`。
- 如果输入以 `/` 开头：路由到对应的子命令 handler。
- 如果是普通文本：路由到 `agent.py` 的统一 Agent 进行自动处理。

## 测试约定

测试位于 `tests/`，使用 `unittest`。共享 fixture 在 `tests/conftest.py`（`build_settings`、`build_transport`）。天气测试通过 `httpx.MockTransport` mock HTTP 层，Agent 测试通过 `FakeMessagesListChatModel` mock 模型层。

## 环境变量

所有配置通过 `.env` 加载（`python-dotenv`）。模板见 `.env.example`。

**`CHAT_PROVIDER` 为必填项**（openai / deepseek / qwen）。环境变量统一命名：
- 聊天：`CHAT_API_KEY`、`CHAT_MODEL`、`CHAT_BASE_URL`
- 向量：`EMBEDDING_API_KEY`、`EMBEDDING_MODEL`、`EMBEDDING_BASE_URL`
- 视觉：`VISION_API_KEY`、`VISION_MODEL`、`VISION_BASE_URL`

## 语言

代码注释、日志、CLI 输出均为中文。使用简体中文回答问题。
