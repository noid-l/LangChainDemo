# LangChainDemo

这是一个以学习 LangChain 为核心目标的教学型 demo 项目。

项目中的每一项功能都不是为了做一个完备的业务系统，而是为了借助具体需求来练习 LangChain 的能力边界、工程组织方式和常见集成模式。新增功能时，优先考虑“这个功能能帮助学习哪种 LangChain 能力”，其次才是业务完整性。

当前示例能力包括：

- `PromptTemplate` 文本生成示例
- `ChatOpenAI` 聊天模型接入
- 本地知识库 RAG
- 和风天气查询
- OpenAI 兼容接口与代理配置
- 向量索引持久化

## 项目定位

- 这是一个教学项目，目标是通过真实但可控的小需求学习 LangChain
- 功能会围绕 LangChain 能力逐步添加，例如 Prompt、RAG、Tool Calling、Agent
- 同一个需求在必要时会同时保留“确定性实现”和“LangChain 实现”，便于对照学习
- 设计取向优先可理解、可验证、可演示，而不是追求产品级复杂度

## 环境要求

- Python 3.13
- `uv`
- 聊天模型 API Key
- 和风天气 JWT 凭据
- 可访问模型服务的代理

## 快速开始

```bash
uv sync
cp .env.example .env
```

如果你的网络环境需要代理，可以按实际情况配置：

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

天气查询使用和风天气 API，当前项目已经完全切换到 JWT 鉴权。按照和风官方文档，你需要准备 Ed25519 密钥对、在控制台上传公钥、拿到 `Project ID`、`Key ID` 和专属 `API Host`。建议优先使用私钥文件路径配置：

```bash
QWEATHER_PROJECT_ID=your_qweather_project_id
QWEATHER_KEY_ID=your_qweather_key_id
QWEATHER_PRIVATE_KEY_PATH=/absolute/path/to/ed25519-private.pem
QWEATHER_API_HOST=https://your-api-host.qweatherapi.com
QWEATHER_JWT_TTL_SECONDS=900
WEATHER_LANG=zh
WEATHER_UNIT=m
WEATHER_FORECAST_DAYS=3
WEATHER_TIMEOUT_SECONDS=10
```

和风官方文档：

- 身份认证：https://dev.qweather.com/docs/configuration/authentication/
- API 配置与 Host：https://dev.qweather.com/docs/configuration/api-config/

## 命令示例

离线查看 PromptTemplate 渲染结果：

```bash
uv run langchaindemo --topic "介绍 LangChain" --tone "专业" --dry-run
```

调用 OpenAI 执行普通生成：

```bash
uv run langchaindemo prompt --topic "介绍 LangChain" --tone "专业"
```

构建本地知识库索引：

```bash
uv run langchaindemo rag build
```

离线预览 RAG 提示词：

```bash
uv run langchaindemo rag ask "RAG 的典型流程是什么？" --dry-run
```

执行真实 RAG 问答：

```bash
uv run langchaindemo rag ask "RAG 的典型流程是什么？"
```

查看当前生效配置：

```bash
uv run langchaindemo config
```

查询城市天气：

```bash
uv run langchaindemo weather "北京"
```

使用 LangChain Agent 处理自然语言天气问题：

```bash
uv run langchaindemo weather ask "明天北京天气怎么样？"
```

通过上级行政区消歧：

```bash
uv run langchaindemo weather "西安" --adm "陕西"
```

使用经纬度查询天气：

```bash
uv run langchaindemo weather "116.41,39.92"
```

查询 7 天天气预报：

```bash
uv run langchaindemo weather "上海" --days 7
```

## LangChain 学习点

天气功能现在分成两层，便于学习：

- `weather query`：底层确定性天气服务，负责真实天气 API 调用、地点解析和结果格式化
- `weather ask`：基于 LangChain Agent 的自然语言天气问答，展示 Tool Calling 的典型能力

其中 `weather ask` 的学习重点包括：

- 使用 LangChain Tool 暴露真实天气查询能力
- 让聊天模型通过 Tool Calling 决定何时、如何调用天气工具
- 基于工具返回的真实结果生成最终自然语言回答

整体上，这个项目中的所有能力都遵循同一个原则：功能本身只是载体，真正要学习的是 LangChain 在不同场景下的设计方式和组合方法。

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
- `src/langchaindemo/weather.py`：天气地点解析、JWT 生成、天气 API 查询与结果格式化
- `src/langchaindemo/weather_langchain.py`：LangChain 天气 Tool 与 Agent 问答示例
- `src/langchaindemo/cli.py`：命令行入口
