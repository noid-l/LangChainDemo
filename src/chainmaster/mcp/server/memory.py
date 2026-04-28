"""Memory MCP Server——基于知识图谱的长期记忆服务。

通过 MCP 协议暴露知识图谱操作工具，兼容 Claude Desktop 等标准 MCP 客户端。

核心工具：
- create_entities: 创建实体
- create_relations: 创建关系
- add_observations: 添加观察记录
- search_nodes: 搜索节点
- open_nodes: 打开子图
- delete_entities: 删除实体
- delete_relations: 删除关系
- delete_observations: 删除观察记录
- read_graph: 读取完整图谱

启动方式（stdio）：
    python -m chainmaster.mcp.server.memory
"""

from __future__ import annotations

import json
import os
from typing import Any

from mcp.server.fastmcp import FastMCP

from .graph import (
    Entity,
    GraphStore,
    ObservationInput,
    Relation,
)

MEMORY_PATH_ENV = "MEMORY_PATH"
DEFAULT_MEMORY_PATH = "~/.chainmaster/memory_graph.json"

mcp = FastMCP(
    "ChainMaster Memory",
    instructions=(
        "你是一个知识图谱记忆管理器。通过实体（Entity）和关系（Relation）"
        "来组织用户的个人信息、偏好和项目知识。"
    ),
)

_store: GraphStore | None = None


def _get_store() -> GraphStore:
    global _store
    if _store is None:
        path = os.environ.get(MEMORY_PATH_ENV, DEFAULT_MEMORY_PATH)
        _store = GraphStore(path)
    return _store


def _entity_to_dict(entity: Entity) -> dict[str, Any]:
    return {
        "name": entity.name,
        "entityType": entity.entityType,
        "observations": entity.observations,
    }


def _relation_to_dict(relation: Relation) -> dict[str, Any]:
    return {
        "from": relation.from_,
        "to": relation.to,
        "relationType": relation.relationType,
    }


# --- 工具注册 ---


@mcp.tool()
def create_entities(entities: list[dict[str, Any]]) -> str:
    """创建一组新实体。每个实体包含 name（名称）、entityType（类型）和可选的 observations（观察列表）。

    如果实体已存在，会合并 observations 而不会覆盖。
    """
    store = _get_store()
    entity_objs = [Entity(**e) for e in entities]
    created = store.create_entities(entity_objs)
    return json.dumps(
        {"entities": [_entity_to_dict(e) for e in created]},
        ensure_ascii=False,
    )


@mcp.tool()
def create_relations(relations: list[dict[str, Any]]) -> str:
    """在两个实体之间创建有向关系。每个关系包含 from（源实体名）、to（目标实体名）和 relationType（关系类型）。

    重复的关系会被自动去重。
    """
    store = _get_store()
    relation_objs = [Relation(**r) for r in relations]
    added = store.create_relations(relation_objs)
    return json.dumps(
        {"relations": [_relation_to_dict(r) for r in added]},
        ensure_ascii=False,
    )


@mcp.tool()
def add_observations(observations: list[dict[str, Any]]) -> str:
    """为已有实体添加观察记录。每项包含 entityName（实体名）和 contents（观察内容列表）。

    如果实体不存在，该项会返回错误。重复的观察内容会被自动跳过。
    """
    store = _get_store()
    obs_objs = [ObservationInput(**o) for o in observations]
    results = store.add_observations(obs_objs)
    return json.dumps({"results": results}, ensure_ascii=False)


@mcp.tool()
def search_nodes(query: str) -> str:
    """在知识图谱中搜索实体。匹配实体的名称、类型和观察记录中的关键词。

    返回匹配的实体及其关联关系。
    """
    store = _get_store()
    graph = store.search_nodes(query)
    return json.dumps(
        {
            "entities": [_entity_to_dict(e) for e in graph.entities],
            "relations": [_relation_to_dict(r) for r in graph.relations],
        },
        ensure_ascii=False,
    )


@mcp.tool()
def open_nodes(names: list[str]) -> str:
    """获取指定实体的完整子图，包括其所有关联关系和直接邻居节点。

    适合深入了解某个主题的全部背景信息。
    """
    store = _get_store()
    graph = store.open_nodes(names)
    return json.dumps(
        {
            "entities": [_entity_to_dict(e) for e in graph.entities],
            "relations": [_relation_to_dict(r) for r in graph.relations],
        },
        ensure_ascii=False,
    )


@mcp.tool()
def delete_entities(entityNames: list[str]) -> str:
    """删除指定实体及其所有关联关系。不可恢复。"""
    store = _get_store()
    result = store.delete_entities(entityNames)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def delete_relations(relations: list[dict[str, Any]]) -> str:
    """删除指定的关系。每项需包含 from、to 和 relationType 三个字段。"""
    store = _get_store()
    relation_objs = [Relation(**r) for r in relations]
    result = store.delete_relations(relation_objs)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def delete_observations(deletions: list[dict[str, Any]]) -> str:
    """删除实体的指定观察记录。每项包含 entityName 和 contents（要删除的观察内容列表）。"""
    store = _get_store()
    obs_objs = [ObservationInput(**d) for d in deletions]
    result = store.delete_observations(obs_objs)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def read_graph() -> str:
    """读取完整的知识图谱。包含所有实体和关系。

    注意：图谱较大时返回内容可能很长，建议优先使用 search_nodes。
    """
    store = _get_store()
    graph = store.read_graph()
    return json.dumps(
        {
            "entities": [_entity_to_dict(e) for e in graph.entities],
            "relations": [_relation_to_dict(r) for r in graph.relations],
        },
        ensure_ascii=False,
    )


def main() -> None:
    """启动 Memory MCP Server（stdio 传输）。"""
    mcp.run()


if __name__ == "__main__":
    main()
