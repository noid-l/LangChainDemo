# ChainMaster 发展规划

> 定位：教学型 LangChain Demo，所有功能优先展示 LangChain / AI Agent 核心概念。

## 当前状态

已完成多层级记忆系统（SQLite 持久化、知识图谱、自动压缩）、MCP Server（知识图谱 Memory Server）、12+ 内置工具、统一 Agent REPL。

---

## 第一阶段：Agent 智能化与行为控制 (1-2 周)

> 核心目标：解构 Agent 推理过程，实现从“自动驾驶”到“辅助驾驶”的精细化控制。

### 1.1 推理策略多样化 (Advanced Reasoning)
- [ ] **多模式切换架构**：在 `agent.py` 中抽象 `BaseReasoning` 接口，支持运行时动态切换。
- [ ] **ReAct 深度集成**：
    - [ ] 实现标准 ReAct 循环，并在 CLI 中以颜色区分“思考(Thought)”和“行动(Action)”。
    - [ ] 优化 Observation 解析，支持处理工具返回的超长结果（自动摘要）。
- [ ] **Plan-and-Execute 模式**：
    - [ ] 实现 `Planner`：将复杂任务拆解为 Pydantic 定义的 `Plan` 对象。
    - [ ] 实现 `StepExecutor`：支持带状态的步骤执行，步骤间可传递中间结果。
    - [ ] 实现 `Re-planner`：当步骤执行结果与预期偏离时，自动更新剩余计划。

### 1.2 自我演进与反思 (Self-Reflection & Correction)
- [ ] **后置质量审计**：实现 `ReflectionChain`，作为 Agent 的“良知”，对输出进行逻辑一致性和安全性检查。
- [ ] **自动纠错回路**：针对解析错误（Parsing Error）或逻辑错误，实现自动重试机制（Max 3 times），并将错误经验注入上下文。
- [ ] **反思经验持久化**：将 Agent 的自我修正过程记录到 SQLite `reflection_logs` 表，便于后续分析。

### 1.3 交互式受控执行 (Human-in-the-Loop)
- [ ] **工具调用断点 (Breakpoints)**：
    - [ ] 基于 LangGraph 的 `interrupt` 机制，在敏感工具执行前实现强制挂起。
    - [ ] CLI 实现交互式审批界面：`[A]pprove / [R]eject / [E]dit arguments`。
- [ ] **中间状态干预**：
    - [ ] 实现 `/state` 命令：查看 Agent 当前的 Memory 和已搜集的外部信息。
    - [ ] 支持在断点处动态修改状态变量，纠正 Agent 的偏差方向。

**展示概念**：LCEL 动态路由、LangGraph 循环与中断、Agent 容错设计、人机协作最佳实践。

---

## 第二阶段：多 Agent 协作与图编排架构 (2-3 周)

> 核心目标：利用 LangGraph 构建复杂的 Agent 拓扑结构，实现从单兵作战到“数字工厂”的演进。

### 2.1 编排策略与模式设计 (Collaboration Patterns)
- [ ] **中心化管理 (Supervisor Pattern)**：
    - [ ] 实现 `Master Supervisor` 节点：使用 LLM 决定任务路由。
    - [ ] 实现专业 Agent 节点簇：`SearchExpert`, `DataAnalyst`, `KnowledgeConsultant`。
- [ ] **工作流拓扑实验**：
    - [ ] **线性流 (Chain)**：固定顺序的流水线任务。
    - [ ] **循环流 (Cycle)**：包含 Review-Fix 回路的循环任务，直到满足特定条件。
    - [ ] **动态流 (Router)**：基于逻辑判断（而非 LLM 意图）的硬编码路由。
- [ ] **子图隔离 (Sub-graphs)**：展示如何将复杂的子任务封装为独立的图结构，并在主图中作为一个 Node 调用。

### 2.2 状态工程与长时会话 (State & Persistence)
- [ ] **强类型状态机设计**：
    - [ ] 使用 `TypedDict` 和 `Pydantic` 定义多维度度的 `AgentState`。
    - [ ] 实现状态过滤：不同 Agent 只能看到或修改与其职责相关的 State 部分（State Masking）。
- [ ] **持久化与断点恢复 (Checkpoints)**：
    - [ ] 集成 `SqliteSaver`：实现会话级别的状态持久化。
    - [ ] 实现“时间旅行”调试：支持回溯到历史某个状态节点重新执行。
- [ ] **跨 Agent 上下文压缩**：在 Agent 切换时，自动提炼当前进展，防止上下文窗口因多 Agent 冗余对话而爆炸。

### 2.3 高级编排特性 (Advanced Orchestration)
- [ ] **并行任务分发 (Parallel Nodes)**：
    - [ ] 实现 `Fan-out` 逻辑：同时触发多个非依赖工具调用。
    - [ ] 实现 `Sync/Merge` 逻辑：等待所有并行分支返回后进行结果聚合。
- [ ] **Agent 通信协议化**：
    - [ ] 定义结构化指令包：包含 `TaskID`, `Payload`, `Priority`, `Sender`。
    - [ ] 实现 Agent 间的“求助”和“委派”机制。

**展示概念**：LangGraph StateGraph、Nodes/Edges、Shared State、Checkpointer、Multi-Agent Coordination。

---

## 第三阶段：高级 RAG 与知识演进架构 (2-3 周)

> 核心目标：解决检索幻觉与召回瓶颈，实现从“匹配”到“推理”的知识发现。

### 3.1 检索策略演进 (Retrieval Engineering)
- [ ] **多查询转换 (Multi-Query/HyDE)**：
    - [ ] 实现 `MultiQueryRetriever`：自动从不同视角拆解用户问题。
    - [ ] 实现 `HyDE (Hypothetical Document Embeddings)`：通过假设性回答提升语义召回精度。
- [ ] **混合搜索后端 (Hybrid Search)**：
    - [ ] 引入 `BM25` 关键词检索，与向量相似度进行 `RRF (Reciprocal Rank Fusion)` 融合。
    - [ ] 抽象 `VectorStoreBackend`：支持在配置中一键切换 ChromaDB, FAISS 或 DuckDB。
- [ ] **增量索引管理**：实现基于文件指纹（Hash）的更新机制，支持只重新处理变更文档。

### 3.2 召回精度与上下文优化 (Precision & Context)
- [ ] **两阶段重排序 (Reranking)**：
    - [ ] 引入 `RerankExecutor`：对初步召回结果进行二次精排，显著降低干扰信息。
- [ ] **父子文档检索 (Parent-Document Retrieval)**：
    - [ ] 实现检索细粒度 Chunk、召回完整大 Context 的策略，平衡检索灵活性与理解深度。
- [ ] **上下文动态压缩 (Compression)**：
    - [ ] 实现 `ContextualCompressor`：在喂给 LLM 前，根据问题自动提取段落核心摘要。
    - [ ] 支持长文本处理策略（Map-Reduce / Refine 链模式）。

### 3.3 知识图谱推理 (GraphRAG & Reasoning)
- [ ] **自动化实体提取流**：在 Loader 阶段自动识别实体及关系，并写入 MCP 图谱 Server。
- [ ] **多跳推理工具 (Multi-hop Query)**：
    - [ ] 实现 `GraphSearch` 工具：支持 Agent 沿着“实体-关系-实体”链条进行深度追踪。
- [ ] **认知图谱可视化**：
    - [ ] 实现导出 Mermaid 关系图的功能，直观展示 Agent 的知识储备。
    - [ ] 支持通过 `/memory graph` 命令交互式探索知识网络。

**展示概念**：Multi-Query Retrieval、Hybrid Search、Cross-Encoder Reranking、Parent-Document Strategy、GraphRAG。

---

## 第四阶段：MCP 生态与标准化工具体系 (2-3 周)

> 核心目标：构建基于 Model Context Protocol 的可插拔工具生态，实现工具与核心逻辑的深度解耦。

### 4.1 MCP 基础设施与生命周期管理 (Infrastructure)
- [ ] **动态服务发现与接入**：
    - [ ] 实现从配置文件自动加载外部 MCP Server（基于 stdio/sse）。
    - [ ] 实现 `/mcp list` 命令：实时列出已连接的 Server 及其暴露的工具 Schema。
- [ ] **服务运行监控 (Health Checks)**：
    - [ ] 实现 MCP Server 的保活与自动重启机制。
    - [ ] 实现资源限制与超时控制，防止异常 Server 拖垮 Agent。
- [ ] **协议交互可视化**：在调试模式下展示完整的 MCP JSON-RPC 报文交互过程。

### 4.2 专业 MCP Server 矩阵开发 (Practical Servers)
- [ ] **FileSystem 安全沙箱 Server**：
    - [ ] 提供受限的文件读写能力，支持设置允许操作的根目录白名单。
- [ ] **高级 Git 助手 Server**：
    - [ ] 封装 Diff 分析、Commit 自动总结、分支切换等高阶研发协作功能。
- [ ] **结构化数据查询 Server**：
    - [ ] 支持 SQLite/PostgreSQL 数据库的元数据自省与结构化 SQL 执行。
- [ ] **安全代码执行 Server**：
    - [ ] 集成沙箱环境（如 Docker），支持 Agent 安全地执行生成的 Python/JS 代码。

### 4.3 工具安全与权限控制 (Security)
- [ ] **工具审批策略 (Policy Engine)**：
    - [ ] 实现基于 Schema 的自动审批与人工审批切换机制。
    - [ ] 针对高危操作（如写文件、删数据）强制执行断点确认。
- [ ] **MCP 工具组合 (Tool Chaining)**：
    - [ ] 展示 Agent 如何协同调用来自不同 MCP Server 的工具完成跨系统任务。

**展示概念**：Model Context Protocol (MCP)、JSON-RPC Over stdio、Service Discovery、Tool Sandboxing、Security Policy。

---

## 第五阶段：Skills 技能系统与指令工程 (1-2 周)

> 核心目标：实现领域知识的模块化封装，通过渐进式加载（Progressive Disclosure）优化上下文窗口。

### 5.1 SKILL.md 2.0 规范与注册表 (Standardization)
- [ ] **增强型技能描述规范**：
    - [ ] 定义技能元数据：`category`, `triggers`, `constraints`, `few_shot_examples`。
    - [ ] 支持技能引用外部资源：代码模板、参考文档、特定提示词片段。
- [ ] **高性能技能索引 (Progressive Registry)**：
    - [ ] 启动时仅加载技能摘要，构建语义索引。
    - [ ] 实现意图驱动的技能激活：根据用户输入自动检索并准备相关的技能包。

### 5.2 渐进式上下文管理 (Context Optimization)
- [ ] **动态系统提示词注入**：
    - [ ] 实现“任务态”上下文切换：当 Agent 进入特定技能领域时，动态替换/扩充 System Message。
    - [ ] 任务结束后自动清理技能上下文，保持上下文窗口的整洁。
- [ ] **智能 Few-shot 选择器**：
    - [ ] 基于向量相似度，从技能包的示例库中动态挑选与当前任务最接近的示范用例。

### 5.3 技能开发与调试工具 (Developer Experience)
- [ ] **技能测试命令**：
    - [ ] 实现 `/skills run <name> <input>`：绕过复杂 Agent 推理，直接在特定技能环境下运行任务。
- [ ] **技能模板脚手架**：
    - [ ] 提供标准化的技能创建模板，支持一键生成技能目录结构及 SKILL.md 模板。
- [ ] **技能链路分析**：
    - [ ] 记录技能调用日志，分析技能激活频率与执行成功率。

**展示概念**：Progressive Disclosure、Context Window Optimization、Modular Prompt Engineering、Few-shot Learning。

---

## 第六阶段：工程化观测与评估体系 (1-2 周)

> 核心目标：解决 Agent 调试难、成本不可控、质量难评估的痛点，构建工业级可观测性。

### 6.1 全链路追踪与成本审计 (Observability)
- [ ] **LangSmith/LangFuse 深度集成**：
    - [ ] 实现针对 LangGraph 每一个 Node/Edge 的颗粒度追踪。
    - [ ] 支持在 Web UI 中可视化查看 Agent 的推理路径与工具调用报文。
- [ ] **实时消耗看板**：
    - [ ] CLI 实时显示当前对话的 Token 数、API 耗时、费用预估（基于各模型单价映射表）。
    - [ ] 汇总 `/stats` 命令：统计今日总消耗、各模型调用频次与平均延迟。
- [ ] **本地执行流预览**：实现 `/trace` 命令，在终端渲染最后一次执行的简版 Mermaid 流程图。

### 6.2 自动化评估框架 (Evaluation Framework)
- [ ] **LLM-as-a-Judge (自动评价器)**：
    - [ ] 构建评测节点：自动对 Agent 的回复进行“事实一致性”、“相关性”、“有用性”多维度评分。
    - [ ] 实现针对 RAG 的“幻觉检测”机制。
- [ ] **单元评估数据集 (Benchmarking)**：
    - [ ] 定义 `evals/` 标准测试集，支持一键运行全量任务并生成质量报表。
    - [ ] 支持 A/B 测试模式：对比不同提示词或不同推理引擎的性能表现（成功率/成本/耗时对比）。

### 6.3 Prompt 演进与工程化调试 (PromptOps)
- [ ] **提示词版本控制**：
    - [ ] 实现本地提示词变更追踪，支持快速回滚到历史表现优异的版本。
- [ ] **热重载调试器**：
    - [ ] 实现 `/prompt dev` 命令：支持在会话过程中动态修改 System Prompt 并在下一轮对话立即生效。
- [ ] **错误堆栈分析器**：
    - [ ] 针对工具调用失败或解析错误，自动提炼错误上下文并生成改进建议。

**展示概念**：LangSmith Tracing、Cost Tracking、LLM-as-a-Judge、Evaluation Metrics、Prompt Management.

---

## 第七阶段：部署、产品化与 Web 交互 (2-3 周)

> 核心目标：将 Agent 转化为稳定可靠的服务，实现多端一致的交互体验。

### 7.1 高级 Web 交互界面 (Rich Web UI)
- [ ] **Gradio/Streamlit 智能对话界面**：
    - [ ] 实现流式响应展示，支持 Markdown 渲染、图片预览与代码高亮。
    - [ ] 集成**思维链路折叠展示**（Thinking Process Blocks）：允许用户点击展开 Agent 的原始思考过程。
- [ ] **知识可视化仪表盘**：
    - [ ] 在 Web 端集成 Mermaid.js，实时渲染知识图谱的网络关系结构。
    - [ ] 实现会话历史搜索与知识片段预览。
- [ ] **动态文件交互**：支持通过 Web 界面拖拽上传 PDF/DOCX，自动触发后台 RAG 索引构建。

### 7.2 FastAPI 异步服务与接口规范 (RESTful API)
- [ ] **高性能异步后端**：
    - [ ] 基于 FastAPI 实现 `chat`, `mcp`, `skills` 等全量功能的 REST 端点。
    - [ ] 实现多用户 Session 隔离与 SQLite 记忆库的自动关联。
- [ ] **标准 SSE (Server-Sent Events) 流**：
    - [ ] 提供统一的流式输出接口，适配主流前端 Agent 客户端协议。
- [ ] **API 密钥管理与鉴权**：
    - [ ] 实现基础的 API Key 认证机制。
    - [ ] 导出 OpenAPI (Swagger) 文档，方便开发者接入。

### 7.3 容器化与生产级部署 (DevOps)
- [ ] **Docker 一键编排 (Docker Compose)**：
    - [ ] 编写 Compose 脚本，同时编排 ChainMaster 主应用与多个 MCP 独立服务。
    - [ ] 支持通过 `.env` 配置文件动态注入多厂商模型参数。
- [ ] **配置热更新界面**：
    - [ ] 实现 Web 端的 `/config` 交互界面，支持在线修改模型配置并即时生效。
- [ ] **日志聚合与错误告警**：
    - [ ] 实现结构化 JSON 日志记录，方便进行 ELK/Graylog 等系统接入。

**展示概念**：Gradio Integration、FastAPI Async Operations、SSE Streaming、Docker Orchestration、Webhooks.

---

## 第八阶段：多模态感知与自适应进化 (长期)

> 核心目标：探索 Agent 的感官扩展与自我优化能力，实现从“被动响应”到“主动自适应”的跨越。

### 8.1 多模态链路闭环 (Multimodal Perception)
- [ ] **视觉驱动的决策与行动 (Vision-to-Action)**：
    - [ ] 实现针对网页或 GUI 截图的深度分析。
    - [ ] 教学展示：Agent 如何解析 UI 元素坐标并执行自动化导航操作。
- [ ] **全流程语音交互 (Speech-to-Speech)**：
    - [ ] 集成 Whisper 等模型实现高性能语音输入。
    - [ ] 集成实时 TTS 引擎，支持带情感色的 Agent 语音反馈。
- [ ] **多模态 RAG 增强**：
    - [ ] 支持对图片、图表、视频关键帧的嵌入（Embedding）与跨模态检索。

### 8.2 个性化画像与动态自适应 (Adaptive Personalization)
- [ ] **用户画像动态建模 (User Profiling)**：
    - [ ] 实现从历史会话中自动提取用户的技术栈、偏好偏向、工作习惯。
    - [ ] 将 Profile 结构化存储于 SQLite，并在每次任务启动前自动注入个性化上下文。
- [ ] **自适应提示词策略 (Adaptive Prompting)**：
    - [ ] 根据用户当前的疲劳度（对话长度）、任务紧迫度自动调整输出的简洁程度或详细程度。

### 8.3 事件驱动与主动 Agent (Autonomous Automation)
- [ ] **外部事件触发流 (Event-driven Agent)**：
    - [ ] 实现文件监控、邮件到达、Webhooks 等外部事件自动唤醒 Agent 执行预设任务。
- [ ] **自修复与自优化 (Self-Correction 2.0)**：
    - [ ] Agent 根据历史工具调用成功率，自动优化工具的选择权重。
    - [ ] 能够通过 `/re-evaluate` 命令自我审视历史回答并生成修正建议。
- [ ] **多 Agent 协作工作流模板**：针对“周报自动生成”、“代码安全扫描”等场景提供开箱即用的自动化模板。

**展示概念**：Vision-to-Action、Speech Processing、User Modeling、Event-driven Architecture、Self-Optimization.

---

## 技术演进路线

```
当前 ─── 第一阶段 ─── 第二阶段 ─── 第三阶段 ─── 第四阶段 ─── 第五阶段 ─── 第六阶段 ─── 第七阶段 ─── 第八阶段
 │          │           │           │           │           │           │           │           │
 │     Agent 智能化   多 Agent    知识升级    MCP 生态    Skills     开发体验    部署产品化   高级特性
 │          │           │           │           │           │           │           │           │
 ├─ ReAct   ├─ Router   ├─ ChromaDB ├─ FS Server ├─ SKILL.md  ├─ LangSmith ├─ Gradio   ├─ 多模态
 ├─ Plan    ├─ Research ├─ FAISS    ├─ Git Server├─ Registry  ├─ 评估框架  ├─ FastAPI  ├─ 个性化
 ├─ Reflect ├─ Writing  ├─ 图谱推理 ├─ DB Server ├─ load_tool ├─ Prompt   ├─ Docker   ├─ 自动化
 └─ HITL    └─ Review   └─ 知识构建 └─ Code Run  └─ 社区技能  └─ Trace    └─ API     └─ 工作流
```

## 每阶段的 LangChain 概念对照

| 阶段 | 核心概念 |
|------|---------|
| 一 | ReAct、Plan-and-Execute、Human-in-the-Loop |
| 二 | LangGraph StateGraph、Multi-Agent、Supervisor Pattern |
| 三 | VectorStore 抽象、Embedding 持久化、Knowledge Graph |
| 四 | MCP Server 开发、工具沙箱、服务发现 |
| 五 | SKILL.md 规范、Progressive Disclosure、Skill Registry |
| 六 | LangSmith Tracing、Agent Evaluation、Prompt Engineering |
| 七 | Gradio、FastAPI、Docker、SSE Streaming |
| 八 | Vision Chain、Whisper、TTS、Workflow Automation |

## 实施原则

1. **教学优先**：每个新功能必须说明展示了哪个 LangChain / AI Agent 概念
2. **对照学习**：同一需求可同时保留"确定性实现"和"AI 实现"
3. **渐进增强**：新功能不应破坏现有功能，通过配置开关控制
4. **独立可运行**：每个模块可独立演示，不依赖其他新模块
5. **中文优先**：代码注释、日志、CLI 输出均为中文
