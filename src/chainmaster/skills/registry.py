"""技能注册表——扫描、索引和加载 SKILL.md 技能文件。

实现了 Progressive Disclosure 模式：
- 启动时只加载 name + description 构建轻量索引
- Agent 需要时才加载完整指令内容

用法::

    registry = SkillRegistry(Path("data/skills"))
    registry.scan()
    print(registry.list_skills())          # [{"name": "code_review", ...}]
    skill = registry.load_skill("code_review")  # 加载完整内容
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from ..logging_utils import get_logger

logger = get_logger(__name__)

_DEFAULT_SKILLS_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / "data" / "skills"

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass
class SkillMeta:
    """技能元数据（轻量索引）。"""

    name: str
    description: str
    triggers: list[str] = field(default_factory=list)
    path: Path = field(default_factory=lambda: Path("."))


@dataclass
class Skill:
    """完整技能（含指令内容）。"""

    name: str
    description: str
    triggers: list[str]
    instructions: str
    path: Path


class SkillRegistry:
    """技能注册表。

    启动时扫描 skills 目录构建索引，按需加载完整内容。

    用法::

        registry = SkillRegistry()
        registry.scan()
        for meta in registry.list_skills():
            print(meta.name, meta.description)
        skill = registry.load_skill("code_review")
        print(skill.instructions)
    """

    def __init__(self, skills_dir: Path | str | None = None) -> None:
        raw = Path(skills_dir) if skills_dir else _DEFAULT_SKILLS_DIR
        self._dir = raw.resolve()
        self._index: dict[str, SkillMeta] = {}

    @property
    def skills_dir(self) -> Path:
        return self._dir

    def scan(self) -> int:
        """扫描技能目录，构建索引。返回发现的技能数量。"""
        self._index.clear()
        if not self._dir.is_dir():
            logger.warning("技能目录不存在: %s", self._dir)
            return 0

        for skill_dir in sorted(self._dir.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_file = skill_dir / "SKILL.md"
            if not skill_file.is_file():
                continue

            meta = self._parse_meta(skill_file)
            if meta:
                self._index[meta.name] = meta
                logger.debug("发现技能: %s — %s", meta.name, meta.description)

        logger.info("技能扫描完成: 共 %d 个技能", len(self._index))
        return len(self._index)

    def list_skills(self) -> list[SkillMeta]:
        """列出所有已索引技能的元数据。"""
        return list(self._index.values())

    def has_skill(self, name: str) -> bool:
        return name in self._index

    def get_meta(self, name: str) -> SkillMeta | None:
        return self._index.get(name)

    def load_skill(self, name: str) -> Skill | None:
        """按需加载技能完整内容（Progressive Disclosure）。"""
        meta = self._index.get(name)
        if meta is None:
            return None

        content = meta.path.read_text(encoding="utf-8")
        instructions = self._extract_instructions(content)
        if instructions is None:
            instructions = content

        logger.info("加载技能: %s（指令长度 %d 字符）", name, len(instructions))
        return Skill(
            name=meta.name,
            description=meta.description,
            triggers=meta.triggers,
            instructions=instructions,
            path=meta.path,
        )

    def match_trigger(self, text: str) -> list[SkillMeta]:
        """根据用户输入文本匹配可能触发的技能。"""
        text_lower = text.lower()
        matched: list[SkillMeta] = []
        for meta in self._index.values():
            for trigger in meta.triggers:
                if trigger.lower() in text_lower:
                    matched.append(meta)
                    break
        return matched

    def _parse_meta(self, path: Path) -> SkillMeta | None:
        """解析 SKILL.md 的 YAML frontmatter。"""
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as e:
            logger.warning("读取技能文件失败 %s: %s", path, e)
            return None

        m = _FRONTMATTER_RE.match(content)
        if not m:
            logger.warning("技能文件缺少 frontmatter: %s", path)
            return None

        fm = m.group(1)
        name = self._parse_field(fm, "name")
        description = self._parse_field(fm, "description")
        if not name:
            logger.warning("技能文件缺少 name 字段: %s", path)
            return None

        triggers = self._parse_list_field(fm, "triggers")

        return SkillMeta(
            name=name,
            description=description or "",
            triggers=triggers,
            path=path,
        )

    @staticmethod
    def _parse_field(frontmatter: str, field_name: str) -> str | None:
        pattern = re.compile(rf"^{field_name}:\s*(.+)$", re.MULTILINE)
        m = pattern.search(frontmatter)
        return m.group(1).strip().strip('"').strip("'") if m else None

    @staticmethod
    def _parse_list_field(frontmatter: str, field_name: str) -> list[str]:
        pattern = re.compile(rf"^{field_name}:\s*\n((?:\s+- .+\n?)+)", re.MULTILINE)
        m = pattern.search(frontmatter)
        if not m:
            return []
        items = re.findall(r"-\s+(.+)", m.group(1))
        return [item.strip().strip('"').strip("'") for item in items]

    @staticmethod
    def _extract_instructions(content: str) -> str | None:
        """提取 frontmatter 之后的指令内容。"""
        m = _FRONTMATTER_RE.match(content)
        if m:
            return content[m.end():].strip()
        return content.strip()
