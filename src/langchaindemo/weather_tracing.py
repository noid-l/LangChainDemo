"""回调追踪示例。

演示 LangChain 核心概念：
- BaseCallbackHandler：框架级可观测性钩子
- 通过 config={"callbacks": [...]} 注入到任何 Runnable/Agent
- 不修改链/Agent 代码即可观察每一步执行

对照学习：
- 确定性路径：用 @trace 装饰器手动埋点
- LangChain 路径：BaseCallbackHandler 框架自动回调

回调钩子：
- on_llm_start / on_llm_end：LLM 调用前后
- on_tool_start / on_tool_end：工具调用前后
- on_chain_start / on_chain_end：链/Agent 调用前后
"""

from __future__ import annotations

import json
import time
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

from .logging_utils import get_logger

logger = get_logger(__name__)


class WeatherTraceHandler(BaseCallbackHandler):
    """自定义回调处理器，记录 Agent 执行的每一步。

    使用方式：
        agent.invoke(input, config={"callbacks": [WeatherTraceHandler()]})
    """

    def __init__(self, *, verbose: bool = True, output_file: str | None = None) -> None:
        self._verbose = verbose
        self._output_file = output_file
        self._trace_log: list[dict[str, Any]] = []
        self._start_times: dict[str, float] = {}

    def _log(self, event: str, data: dict[str, Any]) -> None:
        record = {
            "event": event,
            "timestamp": time.time(),
            **data,
        }
        self._trace_log.append(record)
        if self._verbose:
            logger.info("[Trace] %s: %s", event, json.dumps(data, ensure_ascii=False, default=str))

    def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs) -> None:
        run_id = str(kwargs.get("run_id", ""))
        self._start_times[run_id] = time.time()
        model_name = serialized.get("name", "unknown") if isinstance(serialized, dict) else "unknown"
        self._log("llm_start", {"model": model_name, "run_id": run_id})

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        run_id = str(kwargs.get("run_id", ""))
        elapsed = time.time() - self._start_times.pop(run_id, time.time())
        token_usage = {}
        if response.llm_output and isinstance(response.llm_output, dict):
            token_usage = response.llm_output.get("token_usage", {})
        self._log("llm_end", {
            "run_id": run_id,
            "elapsed_ms": round(elapsed * 1000),
            "token_usage": token_usage,
        })

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs) -> None:
        run_id = str(kwargs.get("run_id", ""))
        self._start_times[run_id] = time.time()
        tool_name = serialized.get("name", "unknown") if isinstance(serialized, dict) else "unknown"
        self._log("tool_start", {"tool": tool_name, "input": input_str[:200], "run_id": run_id})

    def on_tool_end(self, output: str, **kwargs) -> None:
        run_id = str(kwargs.get("run_id", ""))
        elapsed = time.time() - self._start_times.pop(run_id, time.time())
        self._log("tool_end", {
            "run_id": run_id,
            "elapsed_ms": round(elapsed * 1000),
            "output_preview": str(output)[:200],
        })

    def on_chain_start(self, serialized: dict, inputs: dict, **kwargs) -> None:
        run_id = str(kwargs.get("run_id", ""))
        self._start_times[run_id] = time.time()
        chain_name = serialized.get("name", "unknown") if isinstance(serialized, dict) else "unknown"
        self._log("chain_start", {"chain": chain_name, "run_id": run_id})

    def on_chain_end(self, outputs: dict, **kwargs) -> None:
        run_id = str(kwargs.get("run_id", ""))
        elapsed = time.time() - self._start_times.pop(run_id, time.time())
        self._log("chain_end", {
            "run_id": run_id,
            "elapsed_ms": round(elapsed * 1000),
        })

    def on_llm_error(self, error, **kwargs) -> None:
        self._log("llm_error", {"error": str(error)})

    def on_tool_error(self, error, **kwargs) -> None:
        self._log("tool_error", {"error": str(error)})

    def get_trace_log(self) -> list[dict[str, Any]]:
        """获取完整追踪日志。"""
        return list(self._trace_log)

    def save_trace(self, path: str) -> None:
        """保存追踪日志到文件。"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._trace_log, f, ensure_ascii=False, indent=2, default=str)
        logger.info("追踪日志已保存: path=%s, events=%s", path, len(self._trace_log))

    @property
    def summary(self) -> str:
        """生成追踪摘要。"""
        if not self._trace_log:
            return "（无追踪记录）"
        llm_calls = sum(1 for r in self._trace_log if r["event"] == "llm_start")
        tool_calls = sum(1 for r in self._trace_log if r["event"] == "tool_start")
        total_events = len(self._trace_log)
        return f"共 {total_events} 个事件：{llm_calls} 次 LLM 调用，{tool_calls} 次工具调用"
