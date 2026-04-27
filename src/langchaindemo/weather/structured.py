"""结构化输出示例。

演示 LangChain 核心概念：
- model.with_structured_output(PydanticModel) 让 LLM 返回结构化数据
- PydanticOutputParser 手动解析（在 prompt 中注入格式说明）
- 确定性阈值逻辑 vs LLM 语义判断的对比

对照学习：
- 确定性路径：基于温度/降水阈值的 if/else 逻辑
- LangChain 路径：LLM 根据天气数据语义判断，自动填充 Pydantic 字段
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

from ..config import Settings
from ..logging_utils import get_logger
from ..openai_support import build_chat_model, ensure_chat_api_key
from .service import CurrentWeather, DailyForecast, query_weather

__all__ = [
    "ClothingAdvice",
    "CurrentWeather",
    "DailyForecast",
    "_format_advice",
    "advise_weather",
    "deterministic_advice",
    "langchain_structured_advice",
]

logger = get_logger(__name__)


class ClothingAdvice(BaseModel):
    """穿衣建议结构化输出。"""

    outerwear: str = Field(description="外套建议，如'薄外套'、'羽绒服'、'不需要外套'")
    accessories: list[str] = Field(
        description="建议携带的配件，如['雨伞', '墨镜', '围巾']"
    )
    uv_warning: bool = Field(description="是否需要紫外线防护")
    comfort_level: Literal["comfortable", "warm", "cold", "hot"] = Field(
        description="体感舒适度"
    )
    summary: str = Field(description="一句话穿衣建议总结")


def deterministic_advice(
    current: CurrentWeather,
    forecasts: list[DailyForecast] | None = None,
) -> ClothingAdvice:
    """确定性路径：基于阈值的穿衣建议。

    优点：完全可预测，不会产生幻觉。
    缺点：规则固定，无法处理特殊情况（如"虽然 25 度但有大风"）。
    """
    temp = float(current.temp) if current.temp else 20.0
    feels_like = float(current.feels_like) if current.feels_like else temp
    humidity = float(current.humidity) if current.humidity else 50.0
    precip = float(current.precip) if current.precip else 0.0
    uv_index = float(forecasts[0].uv_index) if forecasts and forecasts[0].uv_index else 3.0

    # 外套建议
    if feels_like < 5:
        outerwear = "羽绒服或厚棉服"
    elif feels_like < 12:
        outerwear = "厚外套或大衣"
    elif feels_like < 18:
        outerwear = "薄外套或夹克"
    elif feels_like < 25:
        outerwear = "长袖即可"
    else:
        outerwear = "不需要外套，短袖即可"

    # 配件
    accessories: list[str] = []
    if precip > 0 or humidity > 80:
        accessories.append("雨伞")
    if uv_index >= 6:
        accessories.append("防晒霜")
        accessories.append("墨镜")
    if feels_like < 0:
        accessories.append("围巾")
        accessories.append("手套")
    if feels_like < 10:
        accessories.append("帽子")

    # 紫外线警告
    uv_warning = uv_index >= 6

    # 舒适度
    if feels_like < 10:
        comfort_level = "cold"
    elif feels_like < 18:
        comfort_level = "comfortable"
    elif feels_like < 28:
        comfort_level = "warm"
    else:
        comfort_level = "hot"

    # 总结
    summary_parts = [f"体感{feels_like:.0f}°C"]
    if precip > 0:
        summary_parts.append("有降水")
    summary_parts.append(f"建议穿{outerwear}")
    summary = "，".join(summary_parts) + "。"

    return ClothingAdvice(
        outerwear=outerwear,
        accessories=accessories,
        uv_warning=uv_warning,
        comfort_level=comfort_level,
        summary=summary,
    )


def langchain_structured_advice(
    current: CurrentWeather,
    forecasts: list[DailyForecast] | None = None,
    settings: Settings | None = None,
) -> ClothingAdvice:
    """LangChain 路径：使用 with_structured_output 让 LLM 填充字段。

    优点：能理解语义（如"虽然 25 度但大风天需要防风"）。
    缺点：可能产生幻觉字段值，依赖模型能力。
    """
    if settings is None:
        from ..config import load_settings
        settings = load_settings()

    ensure_chat_api_key(settings)
    model = build_chat_model(settings)
    structured_model = model.with_structured_output(ClothingAdvice)

    weather_description = (
        f"温度 {current.temp}°C，体感 {current.feels_like}°C，"
        f"天气 {current.text}，湿度 {current.humidity}%，"
        f"降水 {current.precip}mm，风速 {current.wind_speed}km/h"
    )
    if forecasts:
        f = forecasts[0]
        weather_description += (
            f"，今日温度范围 {f.temp_min}~{f.temp_max}°C，"
            f"紫外线指数 {f.uv_index}"
        )

    result = structured_model.invoke(
        f"根据以下天气数据，给出穿衣建议：{weather_description}"
    )
    logger.info("LangChain 结构化穿衣建议生成完成。")
    return result


def advise_weather(
    location: str,
    settings: Settings,
    *,
    adm: str | None = None,
    mode: Literal["deterministic", "langchain"] = "deterministic",
    output_json: bool = False,
) -> str:
    """查天气并生成穿衣建议。"""
    logger.info("开始穿衣建议: location=%s, mode=%s", location, mode)
    result = query_weather(settings, location=location, adm=adm)

    if mode == "langchain":
        advice = langchain_structured_advice(
            result.current_weather, result.daily_forecast, settings
        )
    else:
        advice = deterministic_advice(result.current_weather, result.daily_forecast)

    if output_json:
        return json.dumps(advice.model_dump(), ensure_ascii=False, indent=2)

    return _format_advice(advice)


def _format_advice(advice: ClothingAdvice) -> str:
    """将穿衣建议格式化为中文文本。"""
    comfort_labels = {
        "comfortable": "舒适",
        "warm": "偏热",
        "cold": "偏冷",
        "hot": "炎热",
    }
    lines = [
        f"穿衣建议：{advice.outerwear}",
        f"体感：{comfort_labels.get(advice.comfort_level, advice.comfort_level)}",
        f"紫外线警告：{'是' if advice.uv_warning else '否'}",
    ]
    if advice.accessories:
        lines.append(f"建议携带：{', '.join(advice.accessories)}")
    lines.append(f"总结：{advice.summary}")
    return "\n".join(lines)
