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
- **天气服务** — `weather.py`：纯确定性实现，包含 JWT 鉴权、地点解析、API 调用，不依赖 LangChain

数据流：CLI → handler → service 层（weather / rag）→ 外部 API 或 LangChain chain。

## 测试约定

测试位于 `tests/`，使用 `unittest`。天气相关测试通过 `httpx.MockTransport` mock HTTP 层，LangChain Agent 测试通过 `FakeMessagesListChatModel` mock 模型层。`test_weather_langchain.py` 从 `test_weather.py` 导入 `build_settings` 和 `build_transport` 复用 fixture。

## 环境变量

所有配置通过 `.env` 加载（`python-dotenv`）。模板见 `.env.example`。包括 OpenAI 兼容接口配置（支持 DeepSeek 等）、embedding 独立配置、和风天气 JWT 凭据、RAG 参数。

## 语言

代码注释、日志、CLI 输出均为中文。
