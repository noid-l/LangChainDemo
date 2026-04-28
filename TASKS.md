# ChainMaster MCP 支持与多层级记忆系统实现计划

## 0. 已完成里程碑 (Completed Milestones)
- [x] **项目品牌重塑**：从 `LangChainDemo` 全面更名为 `ChainMaster`，完成目录、导入、配置及文档的同步更新。
- [x] **入口点收紧**：移除分散的命令行工具，实现基于 `prompt_toolkit` 的统一交互式 REPL。
- [x] **斜杠命令支持**：在交互终端中支持 `/config`, `/rag`, `/help`, `/history` 等标准化指令。
- [x] **先进 Agent 调研**：深入分析了 `Goose` 和 `Hermes Agent` 的架构设计、工具调用及记忆管理机制。
- [x] **记忆系统规划**：制定了基于 MCP 的多层级（工作记忆、情节记忆、语义记忆）设计方案。

## 1. 背景与目标
受 Goose 和 Hermes Agent 项目的启发，我们计划在 ChainMaster 中实现完整的 **MCP (Model Context Protocol)** 支持。核心目标是构建一个基于 **知识图谱 (Knowledge Graph)** 的 Python 版 Memory Server，作为 MCP 的首个标杆应用，为 Agent 提供结构化、可推理的长期记忆。

## 2. 记忆系统深度设计 (Memory Layers)

参考先进 Agent 的实践，我们将记忆划分为三个层级：

1.  **工作记忆 (Working Memory - 借鉴 Goose)**：
    *   **自动压缩 (Auto-Compaction)**：当会话 Token 消耗达到阈值时，触发 LLM 生成摘要。
    *   **上下文清理**：通过派生子代理执行任务，保持主会话清洁。
2.  **情节记忆 (Episodic Memory - 借鉴 Hermes)**：
    *   **持久化会话**：使用 SQLite 存储历史对话。
    *   **全文检索 (FTS5)**：通过关键词快速回溯特定历史事实。
3.  **语义与事实记忆 (Semantic/Fact Memory - 核心实现)**：
    *   **基于知识图谱的 Python Memory Server**：参考 MCP 官方 TypeScript 版实现，使用实体（Entity）和关系（Relation）构建用户的“知识大脑”。

## 3. Python 版 Memory Server (Graph-based) 实现细节

我们将实现一个完全兼容 MCP 标准的 Python 内存服务器，其核心逻辑如下：

### 3.1 数据模型
内存以 **属性图 (Property Graph)** 结构存储在本地 `memory_graph.json` 中：
*   **Entity (实体)**：如“人”、“项目”、“技术栈”。包含名称、类型和 **Observations (观察记录)**。
*   **Relation (关系)**：连接两个实体，包含关系类型（如“使用”、“属于”、“位于”）。

### 3.2 核心工具集 (Tools)
Agent 将通过以下工具与记忆交互：
1.  `create_entities`: 创建一组新实体（名称、类型、描述）。
2.  `create_relations`: 在现有实体间建立逻辑连接。
3.  `add_observations`: 为特定实体添加新的观察事实（如：“用户提到他喜欢用 Dark Mode”）。
4.  `search_nodes`: 基于关键词搜索相关的实体和观察记录。
5.  `open_nodes`: 获取特定实体的完整子图（包括其所有关联关系和观察），用于深入理解某个主题。
6.  `delete_entities/relations`: 允许 Agent 清理过时或错误的记忆。

### 3.3 存储策略
*   **本地化**：默认存储在 `~/.chainmaster/memory_graph.json`。
*   **原子性**：每次修改后立即持久化，防止程序崩溃导致丢数据。

## 4. 实施步骤 (Phases)

### 第一阶段：基础设施与依赖
- [x] 更新 `pyproject.toml`：`uv add mcp pydantic`。
- [x] 创建 `mcp_servers.json` 配置文件。
- [x] 实现 `MCPManager` 类，管理 stdio 进程和生命周期。

### 第二阶段：开发 Python Memory Server
- [x] 在 `src/chainmaster/mcp/server/memory.py` 实现图数据结构。
- [x] 使用 `mcp.server` SDK 封装上述 6 个核心工具。
- [x] 实现 JSON 序列化持久化逻辑。
- [x] **独立验证**：Server 已实现原子性写入，可独立运行。

### 第三阶段：ChainMaster 客户端对接
- [x] 实现 `src/chainmaster/mcp/adapter.py`，将异步 MCP 工具映射为同步 `StructuredTool`。
- [x] 在 `agent.py` 中自动启动并挂载该本地 Memory Server。
- [x] **自动注入**：已实现 `_inject_memory_context` 在会话开始时注入图谱背景。

### 第四阶段：Agent 行为优化
- [x] 更新系统提示词，教会 Agent 如何利用“实体”和“关系”来构建用户的知识画像。
- [x] **多层级记忆闭环**：结合了自动压缩（Compaction）和 SQLite 会话持久化。

## 5. 技术难点与处理状态
*   **图检索策略**：[待优化] 目前使用 `read_graph` 全量注入，图谱较大时会超限。
*   **异步桥接**：[已解决] 使用后台线程和 `asyncio.run_coroutine_threadsafe` 实现了完美的同步/异步桥接。
*   **进程管理**：[已解决] `MCPManager` 已包含生命周期管理，但需确保全局触发。

## 6. 预期成果
ChainMaster 现已进化为一个具备**结构化长期记忆**和**标准化扩展协议**的先进 Agent 平台。

---

## 7. Code Review & 修改意见 (Revision Suggestions)

经代码审查，当前项目实现度极高（>90%），基础架构非常扎实。以下为进一步完善的意见：

1.  **长期记忆检索优化 (Read Graph Efficiency)**：
    *   [x] 已优化：`_inject_memory_context` 现在根据图谱大小自动选择策略——小图谱（<=50）全量注入，大图谱基于用户提问进行 `search_nodes` Top-K 检索。

2.  **优雅退出与资源清理 (Graceful Shutdown)**：
    *   [x] 已修复：`cli.py` 中通过 `atexit` 注册清理函数 + `try...finally` 双重保障，确保 MCP 子进程在任何退出路径下都能被清理。

3.  **代码清理 (Code Cleanup)**：
    *   [x] 已清理：移除了冗余的 `fact_store.py`，知识图谱方案为唯一的语义记忆实现。

4.  **子代理机制落地 (Subagents)**：
    *   [x] 已实现：新增 `delegate_task` 工具，Agent 可派生轻量级子代理处理高密度任务（日志分析、文档摘要、数据整理），只回传最终结论，避免主会话上下文污染。

