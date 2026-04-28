"""Skills 技能系统——可插拔的 Agent 能力扩展。

参考 Anthropic Agent Skills 标准和 LangChain Skills 框架，
使用 SKILL.md 格式描述技能，支持渐进式加载（Progressive Disclosure）。

展示概念：Agent Skills、SKILL.md 规范、Progressive Disclosure、Dynamic Tool Loading
"""

from .handlers import register_handlers

__all__ = ["register_handlers"]
