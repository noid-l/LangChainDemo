"""Skills CLI 子命令处理器。

提供 /skills 相关的 REPL 命令：
- /skills list — 列出所有可用技能
- /skills show <name> — 显示技能详情
- /skills scan — 重新扫描技能目录
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..logging_utils import get_logger
from .registry import SkillRegistry

logger = get_logger(__name__)

_registry: SkillRegistry | None = None


def _get_registry(project_root: Path | None = None) -> SkillRegistry:
    global _registry
    if _registry is None:
        if project_root:
            skills_dir = project_root / "data" / "skills"
        else:
            skills_dir = None
        _registry = SkillRegistry(skills_dir)
        _registry.scan()
    return _registry


def register_handlers(subparsers: argparse._SubParsersAction) -> None:
    """注册 /skills CLI 子命令。"""
    skills_parser = subparsers.add_parser(
        "skills",
        help="管理 Agent 技能",
    )
    skills_sub = skills_parser.add_subparsers(dest="skills_command")

    skills_sub.add_parser("list", help="列出所有可用技能")
    skills_sub.add_parser("scan", help="重新扫描技能目录")

    show_parser = skills_sub.add_parser("show", help="显示技能详情")
    show_parser.add_argument("name", help="技能名称")

    skills_parser.set_defaults(func=_handle_skills)


def _handle_skills(args: argparse.Namespace) -> None:
    from ..config import load_settings
    settings = load_settings()
    registry = _get_registry(settings.project_root)

    cmd = getattr(args, "skills_command", None)
    if cmd == "list" or cmd is None:
        _cmd_list(registry)
    elif cmd == "scan":
        _cmd_scan(registry)
    elif cmd == "show":
        _cmd_show(registry, args.name)
    else:
        print("未知命令。使用 /skills list、/skills show <name> 或 /skills scan")


def _cmd_list(registry: SkillRegistry) -> None:
    skills = registry.list_skills()
    if not skills:
        print("当前没有可用的技能。")
        print(f"技能目录：{registry.skills_dir}")
        return

    print(f"可用技能（共 {len(skills)} 个）：")
    print()
    for meta in skills:
        trigger_info = ""
        if meta.triggers:
            trigger_info = f"  触发词：{'、'.join(meta.triggers[:5])}"
        print(f"  {meta.name} — {meta.description}")
        if trigger_info:
            print(trigger_info)
    print()
    print("使用 /skills show <name> 查看技能详情。")


def _cmd_scan(registry: SkillRegistry) -> None:
    count = registry.scan()
    print(f"扫描完成，发现 {count} 个技能。")


def _cmd_show(registry: SkillRegistry, name: str) -> None:
    skill = registry.load_skill(name)
    if skill is None:
        available = [m.name for m in registry.list_skills()]
        print(f"技能 '{name}' 不存在。")
        if available:
            print(f"可用技能：{', '.join(available)}")
        return

    print(f"技能：{skill.name}")
    print(f"描述：{skill.description}")
    if skill.triggers:
        print(f"触发词：{'、'.join(skill.triggers)}")
    print(f"文件：{skill.path}")
    print()
    print("── 指令内容 ──")
    print(skill.instructions)
