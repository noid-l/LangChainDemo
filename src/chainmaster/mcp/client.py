"""MCP 客户端管理器——管理 MCP Server 进程生命周期与工具发现。

负责：
- 读取 mcp_servers.json 配置
- 通过 stdio 传输启动/停止 MCP Server 子进程
- 初始化会话并发现工具
- 提供同步调用接口（内部桥接到 asyncio）

展示了 LangChain 与 MCP 协议的集成模式。
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from pathlib import Path
from typing import Any

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import TextContent

from ..logging_utils import get_logger

logger = get_logger(__name__)


def _find_config_path() -> Path:
    """查找 mcp_servers.json 配置文件路径。"""
    candidates = [
        Path("mcp_servers.json"),
        Path(__file__).resolve().parents[3] / "mcp_servers.json",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return candidates[-1]


def load_mcp_config() -> dict[str, Any]:
    """加载 mcp_servers.json 配置。"""
    config_path = _find_config_path()
    if not config_path.is_file():
        logger.info("MCP 配置文件不存在: %s", config_path)
        return {"mcpServers": {}}

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    servers = config.get("mcpServers", {})
    logger.info("MCP 配置已加载: %s 个服务", len(servers))
    return config


class _ServerConnection:
    """单个 MCP Server 的连接上下文。"""

    def __init__(self, name: str, params: StdioServerParameters) -> None:
        self.name = name
        self.params = params
        self._session: ClientSession | None = None
        self._cm_stack: Any = None
        self._tools: list[dict[str, Any]] = []

    @property
    def tools(self) -> list[dict[str, Any]]:
        return list(self._tools)

    async def start(self) -> None:
        """启动 Server 进程并初始化会话。"""
        logger.info("正在启动 MCP Server [%s]: %s %s", self.name, self.params.command, " ".join(self.params.args or []))
        try:
            self._cm_stack = stdio_client(self.params)
            read, write = await self._cm_stack.__aenter__()
            self._session = ClientSession(read, write)
            await self._session.__aenter__()
            await self._session.initialize()
            await self._refresh_tools()
            logger.info("MCP Server [%s] 已启动，发现 %d 个工具", self.name, len(self._tools))
        except Exception:
            logger.error("MCP Server [%s] 启动失败", self.name, exc_info=True)
            await self.stop()
            raise

    async def _refresh_tools(self) -> None:
        if not self._session:
            return
        result = await self._session.list_tools()
        self._tools = [
            {
                "name": t.name,
                "description": t.description or "",
                "input_schema": t.inputSchema,
            }
            for t in result.tools
        ]

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """调用工具并返回文本结果。"""
        if not self._session:
            raise RuntimeError(f"MCP Server [{self.name}] 未连接")

        logger.info("MCP 工具调用: [%s] %s(%s)", self.name, tool_name, _truncate_args(arguments))
        result = await self._session.call_tool(tool_name, arguments)

        parts: list[str] = []
        for block in result.content:
            if isinstance(block, TextContent):
                parts.append(block.text)
        return "\n".join(parts) if parts else "（工具未返回文本内容）"

    async def stop(self) -> None:
        """关闭会话并终止 Server 进程。"""
        if self._session:
            try:
                await self._session.__aexit__(None, None, None)
            except Exception:
                pass
            self._session = None

        if self._cm_stack:
            try:
                await self._cm_stack.__aexit__(None, None, None)
            except Exception:
                pass
            self._cm_stack = None

        logger.info("MCP Server [%s] 已停止", self.name)


class MCPManager:
    """MCP 客户端管理器。

    管理所有 MCP Server 的连接生命周期，提供工具发现与调用的同步接口。

    用法::

        manager = MCPManager()
        manager.startup()             # 启动所有已配置的 Server
        tools = manager.all_tools()   # 获取所有可用工具
        result = manager.call_tool("server_name", "tool_name", {"arg": "val"})
        manager.shutdown()            # 关闭所有 Server
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or load_mcp_config()
        self._connections: dict[str, _ServerConnection] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        """确保后台事件循环运行。"""
        if self._loop is not None and self._loop.is_running():
            return self._loop

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever,
            daemon=True,
            name="mcp-event-loop",
        )
        self._thread.start()
        return self._loop

    def _run_async(self, coro):
        """在后台事件循环中提交协程并等待结果。"""
        loop = self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout=60)

    def startup(self) -> None:
        """启动配置中所有 MCP Server。"""
        servers = self._config.get("mcpServers", {})
        if not servers:
            logger.info("无 MCP Server 配置，跳过启动。")
            return

        for name, server_conf in servers.items():
            params = StdioServerParameters(
                command=server_conf["command"],
                args=server_conf.get("args", []),
                env={**os.environ, **server_conf.get("env", {})},
            )
            conn = _ServerConnection(name, params)
            try:
                self._run_async(conn.start())
                self._connections[name] = conn
            except Exception:
                logger.warning("MCP Server [%s] 启动失败，已跳过。", name)

        logger.info("MCP Manager 启动完成: %d/%d 个 Server 可用", len(self._connections), len(servers))

    def shutdown(self) -> None:
        """关闭所有 MCP Server 并停止事件循环。"""
        for conn in self._connections.values():
            try:
                self._run_async(conn.stop())
            except Exception:
                pass
        self._connections.clear()

        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._loop = None
        self._thread = None

        logger.info("MCP Manager 已关闭。")

    def all_tools(self) -> list[dict[str, Any]]:
        """返回所有 Server 的工具列表。"""
        tools: list[dict[str, Any]] = []
        for name, conn in self._connections.items():
            for t in conn.tools:
                tools.append({**t, "server": name})
        return tools

    def call_tool(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> str:
        """调用指定 Server 的工具（同步接口）。"""
        conn = self._connections.get(server_name)
        if not conn:
            raise ValueError(f"未找到 MCP Server: {server_name}")
        return self._run_async(conn.call_tool(tool_name, arguments))

    @property
    def server_names(self) -> list[str]:
        return list(self._connections.keys())


def _truncate_args(args: dict, max_len: int = 100) -> str:
    s = str(args)
    return s[:max_len] + "..." if len(s) > max_len else s
