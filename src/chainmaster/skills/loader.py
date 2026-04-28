"""load_skill Agent 工具——将技能系统集成到统一 Agent 中。

当 Agent 判断需要使用某个技能时，调用 load_skill 加载完整指令，
然后按照指令执行任务。

展示概念：Dynamic Tool Loading、Progressive Disclosure
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ..logging_utils import get_logger
from .registry import SkillRegistry

logger = get_logger(__name__)

_global_registry: SkillRegistry | None = None


def get_registry(skills_dir: Path | str | None = None) -> SkillRegistry:
    """获取全局技能注册表（懒初始化）。"""
    global _global_registry
    if _global_registry is None:
        _global_registry = SkillRegistry(skills_dir)
        _global_registry.scan()
    return _global_registry


class LoadSkillInput(BaseModel):
    skill_name: str = Field(description="要加载的技能名称")


def build_load_skill_tool(skills_dir: Path | str | None = None) -> StructuredTool:
    """构建 load_skill 工具，注入到统一 Agent 中。"""

    registry = get_registry(skills_dir)

    def load_skill(skill_name: str) -> str:
        logger.info("load_skill 工具被调用: skill_name=%s", skill_name)

        skill = registry.load_skill(skill_name)
        if skill is None:
            available = [m.name for m in registry.list_skills()]
            return (
                f"技能 '{skill_name}' 不存在。"
                f"可用技能：{', '.join(available)}"
            )

        return (
            f"[已加载技能: {skill.name}]\n"
            f"描述：{skill.description}\n\n"
            f"请按以下指令执行：\n\n{skill.instructions}"
        )

    return StructuredTool.from_function(
        func=load_skill,
        name="load_skill",
        description=(
            "加载指定技能的完整指令。当你需要执行特定任务（如代码审查、"
            "文档摘要、翻译等）时，先调用此工具加载技能，再按照指令执行。"
            "可用技能可通过 list_skills 查看。"
        ),
        args_schema=LoadSkillInput,
    )


class ListSkillsInput(BaseModel):
    query: str = Field(default="", description="可选的关键词过滤")


def build_list_skills_tool(skills_dir: Path | str | None = None) -> StructuredTool:
    """构建 list_skills 工具，让 Agent 能发现可用技能。"""

    registry = get_registry(skills_dir)

    def list_skills(query: str = "") -> str:
        logger.info("list_skills 工具被调用: query=%s", query[:50])

        if query:
            matched = registry.match_trigger(query)
            if not matched:
                all_skills = registry.list_skills()
                q_lower = query.lower()
                matched = [
                    s for s in all_skills
                    if q_lower in s.name.lower() or q_lower in s.description.lower()
                ]
        else:
            matched = registry.list_skills()

        if not matched:
            all_skills = registry.list_skills()
            if not all_skills:
                return "当前没有可用的技能。"
            names = [s.name for s in all_skills]
            return f"未找到匹配的技能。所有可用技能：{', '.join(names)}"

        lines = [f"共找到 {len(matched)} 个技能："]
        for meta in matched:
            trigger_info = ""
            if meta.triggers:
                trigger_info = f"（触发词：{'、'.join(meta.triggers[:3])}）"
            lines.append(f"- **{meta.name}**: {meta.description}{trigger_info}")
        lines.append("\n使用 load_skill 加载具体技能的完整指令。")
        return "\n".join(lines)

    return StructuredTool.from_function(
        func=list_skills,
        name="list_skills",
        description=(
            "列出所有可用的 Agent 技能。当用户需要执行特定任务时，"
            "先查看有哪些技能可用，再加载合适的技能。"
        ),
        args_schema=ListSkillsInput,
    )
