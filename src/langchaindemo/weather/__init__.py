"""weather 子包 — 天气相关功能的模块集合。

模块与 LangChain 概念对应关系：
- service      — 确定性天气服务（JWT 鉴权、API 调用），不依赖 LangChain
- agent        — Tool Calling / Agent：StructuredTool + create_agent
- chain        — LCEL / Runnable：prompt | model | StrOutputParser() 管道链
- structured   — Structured Output：with_structured_output(PydanticModel)
- streaming    — Streaming：agent.stream(stream_mode="messages") 逐 token 输出
- memory       — Memory：InMemoryChatMessageHistory 多轮对话上下文管理
- multi_tool   — Multi-Tool Agent：多工具选择推理
- graph        — LangGraph：StateGraph 显式状态图 + InMemorySaver 检查点
- tracing      — Callbacks：BaseCallbackHandler 框架级可观测性
"""

from .service import *  # noqa: F401,F403
from .agent import *  # noqa: F401,F403
from .chain import *  # noqa: F401,F403
from .structured import *  # noqa: F401,F403
from .streaming import *  # noqa: F401,F403
from .memory import *  # noqa: F401,F403
from .multi_tool import *  # noqa: F401,F403
from .graph import *  # noqa: F401,F403
from .tracing import *  # noqa: F401,F403


def register_handlers(subparsers):
    """注册天气相关的 CLI 子命令。"""
    from .handlers import register_handlers as _register
    _register(subparsers)
