# OpenAI 代理访问说明

如果本地网络环境不能直接访问 OpenAI，可以通过 HTTP 代理转发请求。当前项目默认采用与 codex 启动脚本一致的代理配置：

- `HTTP_PROXY=http://127.0.0.1:7890`
- `HTTPS_PROXY=http://127.0.0.1:7890`
- `NO_PROXY=localhost,127.0.0.1`

为了兼容 `langchain-openai`，项目还会自动设置 `OPENAI_PROXY`。这样聊天模型和 Embeddings 都会走同一条代理链路。
