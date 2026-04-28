"""MCP 工具适配层——将 MCP 工具映射为 LangChain StructuredTool。

动态读取 MCP Server 暴露的工具列表，将每个工具包装为
LangChain 可识别的 StructuredTool，供 Agent 统一调度。

展示了 LangChain 与 MCP 协议的工具桥接模式。
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

from ..logging_utils import get_logger
from .client import MCPManager

logger = get_logger(__name__)


def _schema_to_pydantic(
    tool_name: str,
    input_schema: dict[str, Any],
) -> type[BaseModel]:
    """将 JSON Schema 转换为 Pydantic 模型。"""
    properties = input_schema.get("properties", {})
    required = set(input_schema.get("required", []))

    field_definitions: dict[str, Any] = {}
    for prop_name, prop_schema in properties.items():
        prop_type = prop_schema.get("type", "string")
        description = prop_schema.get("description", "")

        if prop_type == "array":
            items = prop_schema.get("items", {})
            item_type = items.get("type", "string")
            if item_type == "object":
                python_type = list[dict[str, Any]]
            elif item_type == "string":
                python_type = list[str]
            else:
                python_type = list[Any]
        elif prop_type == "object":
            python_type = dict[str, Any]
        elif prop_type == "integer":
            python_type = int
        elif prop_type == "number":
            python_type = float
        elif prop_type == "boolean":
            python_type = bool
        else:
            python_type = str

        if prop_name in required:
            field_definitions[prop_name] = (python_type, Field(description=description))
        else:
            default = prop_schema.get("default")
            if default is not None:
                field_definitions[prop_name] = (python_type, Field(default=default, description=description))
            else:
                field_definitions[prop_name] = (python_type | None, Field(default=None, description=description))

    model_name = f"{tool_name}_Input"
    return create_model(model_name, **field_definitions)


def build_langchain_tools(manager: MCPManager) -> list[StructuredTool]:
    """将所有 MCP Server 的工具转换为 LangChain StructuredTool。"""
    tools: list[StructuredTool] = []

    for tool_info in manager.all_tools():
        server_name = tool_info["server"]
        tool_name = tool_info["name"]
        description = tool_info.get("description", "")
        input_schema = tool_info.get("input_schema", {})

        if not input_schema or not input_schema.get("properties"):
            input_schema = {"type": "object", "properties": {}}

        full_name = f"mcp_{tool_name}" if not tool_name.startswith("mcp_") else tool_name
        args_schema = _schema_to_pydantic(full_name, input_schema)

        manager_ref = manager
        srv = server_name
        tn = tool_name

        def _make_call(m: MCPManager, s: str, t: str):
            def _call(**kwargs) -> str:
                try:
                    return m.call_tool(s, t, kwargs)
                except Exception as e:
                    return f"工具调用失败: {e}"
            return _call

        tool = StructuredTool.from_function(
            func=_make_call(manager_ref, srv, tn),
            name=full_name,
            description=f"[MCP:{server_name}] {description}",
            args_schema=args_schema,
        )
        tools.append(tool)

    logger.info("MCP 适配层已加载 %d 个工具: %s", len(tools), [t.name for t in tools])
    return tools
