"""知识图谱数据模型与 JSON 持久化。

参考 MCP 官方 TypeScript 版 Memory Server 实现，
使用 Entity（实体）和 Relation（关系）构建属性图。

存储格式为 JSON Lines（每行一个 entity 或 relation），
支持原子写入防止数据丢失。
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

DEFAULT_MEMORY_PATH = "~/.chainmaster/memory_graph.json"


class Entity(BaseModel):
    """实体——如人、项目、技术栈。"""

    name: str = Field(description="实体名称，唯一标识")
    entityType: str = Field(description="实体类型，如 person、project、technology")
    observations: list[str] = Field(default_factory=list, description="观察记录列表")


class Relation(BaseModel):
    """关系——连接两个实体的有向边。"""

    from_: str = Field(alias="from", description="源实体名称")
    to: str = Field(description="目标实体名称")
    relationType: str = Field(description="关系类型，如 works_on、uses、located_in")

    model_config = {"populate_by_name": True}


class KnowledgeGraph(BaseModel):
    """知识图谱——包含所有实体和关系。"""

    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)


class ObservationInput(BaseModel):
    """为实体添加观察的输入。"""

    entityName: str = Field(description="目标实体名称")
    contents: list[str] = Field(description="要添加的观察内容")


class GraphStore:
    """知识图谱存储——JSON Lines 持久化。

    每次修改后立即写入磁盘，确保原子性：
    1. 先写入临时文件
    2. 再 rename 覆盖原文件

    用法::

        store = GraphStore(Path("~/.chainmaster/memory_graph.json"))
        store.create_entities([Entity(name="Alice", entityType="person")])
        store.create_relations([Relation(from_="Alice", to="Bob", relationType="knows")])
        results = store.search_nodes("Alice")
    """

    def __init__(self, path: str | Path | None = None) -> None:
        raw = path or DEFAULT_MEMORY_PATH
        self._path = Path(os.path.expanduser(raw))
        self._entities: dict[str, Entity] = {}
        self._relations: list[Relation] = []
        self._load()

    @property
    def path(self) -> Path:
        return self._path

    def _load(self) -> None:
        if not self._path.is_file():
            return
        try:
            with open(self._path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get("type") == "entity":
                        entity = Entity(
                            name=data["name"],
                            entityType=data["entityType"],
                            observations=data.get("observations", []),
                        )
                        self._entities[entity.name] = entity
                    elif data.get("type") == "relation":
                        relation = Relation(
                            from_=data["from"],
                            to=data["to"],
                            relationType=data["relationType"],
                        )
                        self._relations.append(relation)
        except (json.JSONDecodeError, KeyError) as e:
            # 不中断启动，记录错误继续
            import logging
            logging.getLogger(__name__).warning("记忆文件加载出错（部分数据可能丢失）: %s", e)

    def _save(self) -> None:
        """原子写入：先写临时文件，再 rename 覆盖。"""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []

        for entity in self._entities.values():
            lines.append(json.dumps({
                "type": "entity",
                "name": entity.name,
                "entityType": entity.entityType,
                "observations": entity.observations,
            }, ensure_ascii=False))

        for relation in self._relations:
            lines.append(json.dumps({
                "type": "relation",
                "from": relation.from_,
                "to": relation.to,
                "relationType": relation.relationType,
            }, ensure_ascii=False))

        content = "\n".join(lines) + "\n" if lines else ""

        fd, tmp_path = tempfile.mkstemp(
            dir=str(self._path.parent),
            prefix=".memory_",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            os.replace(tmp_path, str(self._path))
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    # --- 核心操作 ---

    def create_entities(self, entities: list[Entity]) -> list[Entity]:
        """创建实体。已存在的实体会合并 observations。"""
        created: list[Entity] = []
        for entity in entities:
            existing = self._entities.get(entity.name)
            if existing:
                for obs in entity.observations:
                    if obs not in existing.observations:
                        existing.observations.append(obs)
                created.append(existing)
            else:
                self._entities[entity.name] = entity.model_copy()
                created.append(entity)
        self._save()
        return created

    def create_relations(self, relations: list[Relation]) -> list[Relation]:
        """创建关系。重复关系会被去重。"""
        existing_set = {
            (r.from_, r.to, r.relationType) for r in self._relations
        }
        added: list[Relation] = []
        for relation in relations:
            key = (relation.from_, relation.to, relation.relationType)
            if key not in existing_set:
                self._relations.append(relation.model_copy())
                added.append(relation)
                existing_set.add(key)
        self._save()
        return added

    def add_observations(self, observations: list[ObservationInput]) -> list[dict[str, Any]]:
        """为实体添加观察记录。返回每个实体实际添加的内容。"""
        results: list[dict[str, Any]] = []
        for obs in observations:
            entity = self._entities.get(obs.entityName)
            if entity is None:
                results.append({
                    "entityName": obs.entityName,
                    "error": f"实体 '{obs.entityName}' 不存在",
                    "addedObservations": [],
                })
                continue

            added: list[str] = []
            for content in obs.contents:
                if content not in entity.observations:
                    entity.observations.append(content)
                    added.append(content)

            results.append({
                "entityName": obs.entityName,
                "addedObservations": added,
            })
        self._save()
        return results

    def search_nodes(self, query: str) -> KnowledgeGraph:
        """按关键词搜索实体及其关联关系。"""
        q = query.lower()
        matched: list[Entity] = []
        matched_names: set[str] = set()

        for entity in self._entities.values():
            if (
                q in entity.name.lower()
                or q in entity.entityType.lower()
                or any(q in obs.lower() for obs in entity.observations)
            ):
                matched.append(entity)
                matched_names.add(entity.name)

        matched_relations = [
            r for r in self._relations
            if r.from_ in matched_names or r.to in matched_names
        ]

        return KnowledgeGraph(entities=matched, relations=matched_relations)

    def open_nodes(self, names: list[str]) -> KnowledgeGraph:
        """获取指定实体的完整子图（包括关联关系和邻居节点）。"""
        name_set = set(names)
        entities: list[Entity] = []
        all_names: set[str] = set()

        for name in names:
            entity = self._entities.get(name)
            if entity:
                entities.append(entity)
                all_names.add(name)

        relations = [
            r for r in self._relations
            if r.from_ in name_set or r.to in name_set
        ]

        for r in relations:
            neighbor = r.to if r.from_ in name_set else r.from_
            if neighbor not in all_names and neighbor in self._entities:
                entities.append(self._entities[neighbor])
                all_names.add(neighbor)

        return KnowledgeGraph(entities=entities, relations=relations)

    def delete_entities(self, entity_names: list[str]) -> dict[str, Any]:
        """删除实体及其关联关系。"""
        name_set = set(entity_names)
        deleted_count = 0

        for name in entity_names:
            if name in self._entities:
                del self._entities[name]
                deleted_count += 1

        before = len(self._relations)
        self._relations = [
            r for r in self._relations
            if r.from_ not in name_set and r.to not in name_set
        ]
        removed_relations = before - len(self._relations)

        self._save()
        return {
            "deletedEntities": deleted_count,
            "deletedRelations": removed_relations,
        }

    def delete_relations(self, relations: list[Relation]) -> dict[str, Any]:
        """删除指定关系。"""
        to_remove = {
            (r.from_, r.to, r.relationType) for r in relations
        }
        before = len(self._relations)
        self._relations = [
            r for r in self._relations
            if (r.from_, r.to, r.relationType) not in to_remove
        ]
        removed = before - len(self._relations)
        self._save()
        return {"deletedRelations": removed}

    def delete_observations(self, deletions: list[ObservationInput]) -> dict[str, Any]:
        """删除实体的指定观察记录。"""
        total_removed = 0
        for deletion in deletions:
            entity = self._entities.get(deletion.entityName)
            if entity:
                before = len(entity.observations)
                to_remove = set(deletion.contents)
                entity.observations = [o for o in entity.observations if o not in to_remove]
                total_removed += before - len(entity.observations)
        self._save()
        return {"deletedObservations": total_removed}

    def read_graph(self) -> KnowledgeGraph:
        """读取完整知识图谱。"""
        return KnowledgeGraph(
            entities=list(self._entities.values()),
            relations=list(self._relations),
        )
