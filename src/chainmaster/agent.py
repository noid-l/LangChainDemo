"""统一 Agent 入口——超级 Agent REPL。

将所有功能整合为一个 Agent，LLM 自动判断使用哪个工具：
- weather_lookup: 天气查询
- weather_compare: 天气对比
- clothing_advisor: 穿衣建议
- knowledge_search: RAG 知识库检索
- search_history: 历史对话全文检索

支持多轮对话（SQLite 持久化 + 自动压缩）、流式输出、回调追踪。
"""

from __future__ import annotations

import sys
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from .config import Settings
from .logging_utils import get_logger
from .memory.store import ChatHistoryStore, StoreManager
from .memory.compaction import compact_if_needed
from .mcp.client import MCPManager
from .openai_support import build_chat_model, ensure_chat_api_key

logger = get_logger(__name__)

_store_manager: StoreManager | None = None
_mcp_manager: MCPManager | None = None


def _inject_memory_context(question: str = "") -> str:
    """从 MCP Memory Server 检索记忆背景，注入到会话消息中。

    小图谱（<=50 实体）全量注入；大图谱基于用户提问进行 Top-K 检索。
    """
    global _mcp_manager
    if not _mcp_manager or not _mcp_manager.server_names:
        return ""
    try:
        import json

        graph_json = _mcp_manager.call_tool("memory", "read_graph", {})
        graph = json.loads(graph_json)
        entities = graph.get("entities", [])
        relations = graph.get("relations", [])

        if not entities:
            return ""

        if len(entities) <= 50:
            return _format_memory_context(entities, relations)

        if question:
            search_json = _mcp_manager.call_tool(
                "memory", "search_nodes", {"query": question}
            )
            search_result = json.loads(search_json)
            matched_entities = search_result.get("entities", [])
            matched_relations = search_result.get("relations", [])
            if matched_entities:
                return _format_memory_context(matched_entities, matched_relations)

        return _format_memory_context(entities[:10], [])

    except Exception:
        return ""


def _format_memory_context(
    entities: list[dict], relations: list[dict]
) -> str:
    parts = ["[你的长期记忆] 你已记住以下信息："]
    for e in entities[:15]:
        obs = "；".join(e.get("observations", []))
        parts.append(
            f"- {e['name']}（{e['entityType']}）：{obs}"
            if obs
            else f"- {e['name']}（{e['entityType']}）"
        )
    for r in relations[:10]:
        parts.append(f"- {r['from']} → {r['relationType']} → {r['to']}")
    return "\n".join(parts)


def _get_store_manager(settings: Settings) -> StoreManager:
    global _store_manager
    if _store_manager is None:
        _store_manager = StoreManager(project_root=settings.project_root)
        logger.info("会话持久化存储已初始化: %s", settings.project_root / ".cache" / "chat_history.db")
    return _store_manager


def _get_session(session_id: str, settings: Settings) -> ChatHistoryStore:
    return _get_store_manager(settings).get(session_id)


def get_session_history(session_id: str = "default") -> str:
    """获取并格式化会话历史（兼容旧接口，使用默认配置）。"""
    from .config import load_settings
    settings = load_settings()
    history = _get_session(session_id, settings)
    msgs = history.messages
    if not msgs:
        return "（无历史记录）"

    lines = []
    for msg in msgs:
        role = "你" if isinstance(msg, HumanMessage) else "助手"
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


class KnowledgeSearchInput(BaseModel):
    question: str = Field(description="要在知识库中搜索的问题")


class SearchHistoryInput(BaseModel):
    query: str = Field(description="要在历史对话中搜索的关键词")


class DelegateTaskInput(BaseModel):
    task: str = Field(description="要委派给子代理执行的任务描述")
    context: str = Field(default="", description="任务相关的上下文信息（可选）")


def _build_knowledge_search_tool(settings: Settings) -> StructuredTool:
    """构建知识库检索工具。"""
    from .knowledge.rag import answer_question

    def knowledge_search(question: str) -> str:
        logger.info("knowledge_search 工具被调用: question=%s", question[:100])
        try:
            result = answer_question(question=question, settings=settings)
            return result.answer
        except Exception as exc:
            logger.error("知识库检索失败: %s", exc)
            return f"知识库检索失败: {exc}"

    return StructuredTool.from_function(
        func=knowledge_search,
        name="knowledge_search",
        description=(
            "从本地知识库中检索信息。当用户问关于 LangChain、RAG、OpenAI 代理等"
            "技术概念的问题时使用。不适用于天气查询。"
        ),
        args_schema=KnowledgeSearchInput,
    )


def _build_search_history_tool(settings: Settings) -> StructuredTool:
    """构建历史对话全文检索工具。"""

    def search_history(query: str) -> str:
        logger.info("search_history 工具被调用: query=%s", query[:100])
        manager = _get_store_manager(settings)
        results = manager.search_all(query, limit=10)
        if not results:
            return "未找到匹配的历史对话。"
        lines = []
        for r in results:
            role = "用户" if r["role"] == "human" else "助手"
            content = r["content"][:200] + "..." if len(r["content"]) > 200 else r["content"]
            lines.append(f"[{r['created_at'][:16]}] [{r['session_id']}] {role}: {content}")
        return "\n".join(lines)

    return StructuredTool.from_function(
        func=search_history,
        name="search_history",
        description=(
            "从历史对话记录中全文检索。当用户提到「之前说过」「之前讨论过」"
            "「我记得」等回忆场景时使用。"
        ),
        args_schema=SearchHistoryInput,
    )


def _build_delegate_task_tool(settings: Settings) -> StructuredTool:
    """构建子代理委派工具。

    派生一个轻量级 Agent（不带长期记忆背景）处理高密度任务，
    只将最终结论回传给主会话，避免主上下文被执行日志淹没。
    """

    def delegate_task(task: str, context: str = "") -> str:
        logger.info("delegate_task 子代理启动: task=%s", task[:80])
        try:
            ensure_chat_api_key(settings)
            model = build_chat_model(settings)

            prompt_parts = [
                "你是一个专注的任务执行者。请高效完成以下任务，只返回最终结论，不要输出中间步骤。",
                "",
            ]
            if context:
                prompt_parts.append(f"上下文信息：{context}")
                prompt_parts.append("")
            prompt_parts.append(f"任务：{task}")
            prompt = "\n".join(prompt_parts)

            from langchain_core.messages import HumanMessage
            response = model.invoke([HumanMessage(content=prompt)])
            content = response.content
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                parts = [
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                return "".join(parts).strip()
            return str(content)
        except Exception as exc:
            logger.error("delegate_task 执行失败: %s", exc)
            return f"子代理执行失败: {exc}"

    return StructuredTool.from_function(
        func=delegate_task,
        name="delegate_task",
        description=(
            "委派一个独立的子代理去执行任务。适合处理会产生大量中间输出的工作"
            "（如日志分析、文档摘要、数据整理），只将最终结论带回主对话。"
        ),
        args_schema=DelegateTaskInput,
    )


def build_all_tools(settings: Settings) -> list[StructuredTool]:
    """收集所有工具。天气工具需要 JWT 配置，如果缺失则跳过。"""
    tools: list[StructuredTool] = []

    # 知识库检索
    tools.append(_build_knowledge_search_tool(settings))

    # 历史对话全文检索
    try:
        tools.append(_build_search_history_tool(settings))
        logger.info("历史检索工具已加载。")
    except Exception:
        logger.warning("历史检索工具加载跳过。")

    # 子代理委派工具
    try:
        tools.append(_build_delegate_task_tool(settings))
        logger.info("子代理委派工具已加载。")
    except Exception:
        logger.warning("子代理委派工具加载跳过。")

    # 天气相关工具（需要 JWT 配置）
    try:
        from .weather import ensure_qweather_jwt_config

        ensure_qweather_jwt_config(settings)

        from .weather.agent import build_weather_tool
        from .weather.multi_tool import _build_compare_tool, _build_clothing_advisor_tool

        tools.append(build_weather_tool(settings))
        tools.append(_build_compare_tool(settings))
        tools.append(_build_clothing_advisor_tool(settings))
        logger.info("天气工具已加载（共 3 个）。")
    except Exception:
        logger.warning("天气工具加载跳过（JWT 配置缺失）。")

    # 网页搜索工具（需要 TAVILY_API_KEY）
    try:
        from .tools.web_search import build_web_search_tool
        tools.append(build_web_search_tool(settings))
        logger.info("搜索工具已加载。")
    except Exception:
        logger.warning("搜索工具加载跳过（TAVILY_API_KEY 缺失）。")

    # 翻译工具
    try:
        from .tools.translate import build_translate_tool
        tools.append(build_translate_tool(settings))
        logger.info("翻译工具已加载。")
    except Exception:
        logger.warning("翻译工具加载跳过。")

    # 文档问答工具
    try:
        from .tools.document_qa import build_document_qa_tool
        tools.append(build_document_qa_tool(settings))
        logger.info("文档问答工具已加载。")
    except Exception:
        logger.warning("文档问答工具加载跳过。")

    # 数据分析工具
    try:
        from .tools.data_analysis import build_data_analysis_tool
        tools.append(build_data_analysis_tool(settings))
        logger.info("数据分析工具已加载。")
    except Exception:
        logger.warning("数据分析工具加载跳过。")

    # MarkItDown 文档转换/问答工具
    try:
        from .tools.markitdown import build_convert_tool, build_markitdown_qa_tool
        tools.append(build_convert_tool(settings))
        tools.append(build_markitdown_qa_tool(settings))
        logger.info("MarkItDown 工具已加载。")
    except Exception:
        logger.warning("MarkItDown 工具加载跳过。")

    # MCP 工具（自动从 mcp_servers.json 加载）
    try:
        from .mcp.adapter import build_langchain_tools
        global _mcp_manager
        _mcp_manager = MCPManager()
        _mcp_manager.startup()
        mcp_tools = build_langchain_tools(_mcp_manager)
        tools.extend(mcp_tools)
        logger.info("MCP 工具已加载（共 %d 个）。", len(mcp_tools))
    except Exception:
        logger.warning("MCP 工具加载跳过。")

    # Skills 技能工具（load_skill / list_skills）
    try:
        from .skills.loader import build_load_skill_tool, build_list_skills_tool
        tools.append(build_load_skill_tool())
        tools.append(build_list_skills_tool())
        logger.info("Skills 技能工具已加载。")
    except Exception:
        logger.warning("Skills 技能工具加载跳过。")

    logger.info("统一 Agent 工具列表: %s", [t.name for t in tools])
    return tools


UNIFIED_SYSTEM_PROMPT = "\n".join([
    "你是一个多功能助手，拥有以下工具：",
    "",
    "1. weather_lookup — 查询指定城市的天气数据",
    "2. weather_compare — 对比两个城市的天气",
    "3. clothing_advisor — 根据天气给出穿衣建议",
    "4. knowledge_search — 从知识库中检索技术文档",
    "5. web_search — 搜索互联网获取实时信息",
    "6. translate — 翻译文本到指定语言",
    "7. document_qa — 读取文档文件（PDF/Word/TXT）并回答问题",
    "8. data_analysis — 分析 CSV 数据文件",
    "9. markitdown_convert — 将文件转换为 Markdown（支持 20+ 格式）",
    "10. markitdown_qa — 读取多种格式文件并回答问题",
    "11. search_history — 从历史对话中检索之前讨论过的内容",
    "12. delegate_task — 委派子代理处理高密度任务（日志分析、文档摘要等），只回传结论",
    "13. list_skills — 列出所有可用的 Agent 技能",
    "14. load_skill — 加载指定技能的完整指令",
    "",
    "**记忆工具（MCP）**：你拥有长期记忆能力，可以记住用户信息和项目知识：",
    "- mcp_create_entities — 创建实体（人、项目、技术等），自动记录观察",
    "- mcp_create_relations — 在实体间建立关系",
    "- mcp_add_observations — 为已有实体添加新观察",
    "- mcp_search_nodes — 搜索记忆中的实体和知识",
    "- mcp_open_nodes — 获取实体的完整关联信息",
    "- mcp_read_graph — 读取完整知识图谱",
    "",
    "请根据用户的问题选择合适的工具：",
    "- 天气相关 → weather_lookup / weather_compare / clothing_advisor",
    "- 技术概念、知识库 → knowledge_search",
    "- 实时信息、最新新闻 → web_search",
    "- 翻译 → translate",
    "- 文档问题（提到文件路径）→ document_qa 或 markitdown_qa",
    "- 数据分析（提到 CSV/数据）→ data_analysis",
    "- 文件格式转换/查看文件内容 → markitdown_convert",
    "- PPT/Excel/图片等 document_qa 不支持的格式 → markitdown_qa",
    "- 回忆之前讨论的内容（「之前说过」「我记得」）→ search_history",
    "- 大量文本处理、日志分析、文档摘要 → delegate_task（委派子代理，避免上下文污染）",
    "- 记住用户信息（「记住我喜欢...」「我在做...」）→ mcp_create_entities + mcp_add_observations",
    "- 查询已记住的用户信息 → mcp_search_nodes / mcp_open_nodes",
    "- 代码审查、文档摘要、翻译等专业任务 → 先 list_skills 查看可用技能，再 load_skill 加载指令执行",
    "- 闲聊或通用问题 → 直接回答",
    "",
    "**主动记忆策略**：当用户透露个人偏好、项目信息或重要决策时，",
    "主动使用 mcp_create_entities 和 mcp_create_relations 记录下来。",
    "例如用户说「我在做 ChainMaster 项目」时，应创建 Entity 并建立关系。",
    "",
    "如果用户没有明确地点但问天气，可追问。多轮对话中结合上下文推断。",
    "输出使用中文，结构清晰。",
])


def build_unified_agent(settings: Settings, *, model=None):
    """构建统一 Agent。"""
    from langchain.agents import create_agent

    ensure_chat_api_key(settings)
    tools = build_all_tools(settings)

    agent = create_agent(
        model=model or build_chat_model(settings),
        tools=tools,
        system_prompt=UNIFIED_SYSTEM_PROMPT,
    )
    logger.info("统一 Agent 构建完成: 工具数=%s", len(tools))
    return agent


def chat_unified(
    question: str,
    settings: Settings,
    *,
    session_id: str = "default",
    model=None,
    config: dict[str, Any] | None = None,
) -> str:
    """单次问答。"""
    agent = build_unified_agent(settings, model=model)
    history = _get_session(session_id, settings)

    memory_ctx = _inject_memory_context(question)
    messages: list[dict[str, str]] = []
    if memory_ctx:
        messages.append({"role": "system", "content": memory_ctx})
    for msg in history.messages:
        if isinstance(msg, HumanMessage):
            messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            messages.append({"role": "assistant", "content": msg.content})
    messages.append({"role": "user", "content": question})

    logger.info("统一 Agent 问答: question=%s, history=%s轮", question[:80], history.message_count() // 2)

    result = agent.invoke({"messages": messages}, config=config or {})
    answer = _extract_answer(result)

    history.add_user_message(question)
    history.add_ai_message(answer)

    compact_if_needed(history, model=model)
    return answer


def chat_unified_stream(
    question: str,
    settings: Settings,
    *,
    session_id: str = "default",
    model=None,
    file: Any | None = None,
) -> None:
    """流式问答。"""
    agent = build_unified_agent(settings, model=model)
    history = _get_session(session_id, settings)

    memory_ctx = _inject_memory_context(question)
    messages: list[dict[str, str]] = []
    if memory_ctx:
        messages.append({"role": "system", "content": memory_ctx})
    for msg in history.messages:
        if isinstance(msg, HumanMessage):
            messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            messages.append({"role": "assistant", "content": msg.content})
    messages.append({"role": "user", "content": question})

    output = file or sys.stdout
    collected: list[str] = []

    for chunk in agent.stream(
        {"messages": messages},
        stream_mode="messages",
        version="v2",
    ):
        if chunk["type"] != "messages":
            continue
        token, _ = chunk["data"]
        if isinstance(token, AIMessageChunk):
            text = ""
            if isinstance(token.content, str):
                text = token.content
            elif isinstance(token.content, list):
                for block in token.content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text += block.get("text", "")
            if text:
                output.write(text)
                output.flush()
                collected.append(text)

    output.write("\n")
    full_answer = "".join(collected)
    history.add_user_message(question)
    history.add_ai_message(full_answer)

    compact_if_needed(history, model=model)


def _extract_answer(agent_result: dict) -> str:
    messages = agent_result.get("messages", [])
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            content = message.content
            if isinstance(content, str) and content.strip():
                return content.strip()
            if isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict) and item.get("type") == "text":
                        text = item.get("text")
                        if isinstance(text, str):
                            parts.append(text)
                answer = "\n".join(p.strip() for p in parts if p.strip())
                if answer:
                    return answer
    return "（未获得有效回答）"
