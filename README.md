# LangChainDemo

使用 `uv` 初始化的 LangChain 项目骨架，包含以下能力：

- `PromptTemplate` 文本生成示例
- `ChatOpenAI` 聊天模型接入
- 本地知识库 RAG
- OpenAI 兼容接口与代理配置
- 向量索引持久化

## 环境要求

- Python 3.13
- `uv`
- 聊天模型 API Key
- 可访问模型服务的代理

## 快速开始

```bash
/opt/homebrew/bin/uv sync
cp .env.example .env
```

默认代理配置参考 `/Users/lishuo/Workspace.localized/env/bin/codex` 中的设置：

```bash
HTTP_PROXY=http://127.0.0.1:7890
HTTPS_PROXY=http://127.0.0.1:7890
NO_PROXY=localhost,127.0.0.1
```

项目还会自动把 `OPENAI_PROXY` 对齐到同一地址，确保 `langchain-openai` 的聊天模型和 embeddings 都通过代理访问 OpenAI。

聊天模型的 `api_key` 和 `base_url` 都从 `.env` 读取。默认使用：

```bash
OPENAI_API_KEY=your_chat_api_key
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_MODEL=gpt-4.1-mini
```

如果你要改成 DeepSeek 的 OpenAI 兼容接口，可以这样写：

```bash
OPENAI_API_KEY=your_deepseek_api_key
OPENAI_API_BASE=https://api.deepseek.com
OPENAI_MODEL=deepseek-chat
```

RAG 的 embeddings 也支持单独配置，避免聊天模型和 embedding 服务绑死在同一个提供方：

```bash
OPENAI_EMBEDDING_API_KEY=
OPENAI_EMBEDDING_API_BASE=https://api.openai.com/v1
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

如果 `OPENAI_EMBEDDING_API_KEY` 为空，代码会回退使用 `OPENAI_API_KEY`。

## 命令示例

离线查看 PromptTemplate 渲染结果：

```bash
/opt/homebrew/bin/uv run langchaindemo --topic "介绍 LangChain" --tone "专业" --dry-run
```

调用 OpenAI 执行普通生成：

```bash
/opt/homebrew/bin/uv run langchaindemo prompt --topic "介绍 LangChain" --tone "专业"
```

构建本地知识库索引：

```bash
/opt/homebrew/bin/uv run langchaindemo rag build
```

离线预览 RAG 提示词：

```bash
/opt/homebrew/bin/uv run langchaindemo rag ask "RAG 的典型流程是什么？" --dry-run
```

执行真实 RAG 问答：

```bash
/opt/homebrew/bin/uv run langchaindemo rag ask "RAG 的典型流程是什么？"
```

查看当前生效配置：

```bash
/opt/homebrew/bin/uv run langchaindemo config
```

## 项目结构

```text
.
├── data/knowledge/
├── pyproject.toml
├── uv.lock
├── .env.example
├── .cache/
├── skills/
└── src/langchaindemo/
```

## 模块说明

- `src/langchaindemo/config.py`：环境变量、路径与代理配置
- `src/langchaindemo/openai_support.py`：OpenAI 聊天模型与 embeddings 工厂
- `src/langchaindemo/prompting.py`：PromptTemplate 与 RAG 提示词模板
- `src/langchaindemo/knowledge.py`：知识库文档加载与文本切分
- `src/langchaindemo/rag.py`：索引构建、检索、RAG 问答与离线预览
- `src/langchaindemo/cli.py`：命令行入口
