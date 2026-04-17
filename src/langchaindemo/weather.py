from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import httpx
import jwt

from .config import Settings
from .logging_utils import get_logger


logger = get_logger(__name__)
SUPPORTED_FORECAST_DAYS = (3, 7)


class WeatherError(Exception):
    """Base exception for weather querying."""


class WeatherConfigError(WeatherError):
    """Raised when weather configuration is invalid or missing."""


class WeatherApiError(WeatherError):
    """Raised when QWeather API returns an unexpected response."""


class LocationNotFoundError(WeatherError):
    """Raised when a location cannot be resolved."""


class AmbiguousLocationError(WeatherError):
    """Raised when a location query matches multiple candidates."""

    def __init__(self, query: str, candidates: list["ResolvedLocation"]) -> None:
        self.query = query
        self.candidates = candidates
        super().__init__(query)


@dataclass(frozen=True)
class ResolvedLocation:
    location_id: str
    name: str
    lat: str
    lon: str
    adm1: str
    adm2: str
    country: str
    tz: str | None
    utc_offset: str | None
    fx_link: str | None


@dataclass(frozen=True)
class CurrentWeather:
    obs_time: str
    temp: str
    feels_like: str
    text: str
    wind_dir: str
    wind_scale: str
    wind_speed: str
    humidity: str
    precip: str
    pressure: str
    vis: str
    cloud: str
    dew: str
    icon: str


@dataclass(frozen=True)
class DailyForecast:
    fx_date: str
    temp_min: str
    temp_max: str
    text_day: str
    text_night: str
    wind_dir_day: str
    wind_scale_day: str
    wind_speed_day: str
    humidity: str
    precip: str
    pressure: str
    vis: str
    uv_index: str
    sunrise: str
    sunset: str


@dataclass(frozen=True)
class WeatherQueryResult:
    requested_location: str
    resolved_location: ResolvedLocation
    current_weather: CurrentWeather
    daily_forecast: list[DailyForecast]
    lang: str
    unit: str
    forecast_days: int


@dataclass(frozen=True)
class QWeatherJwtConfig:
    project_id: str
    key_id: str
    private_key: str
    api_host: str
    ttl_seconds: int


def ensure_qweather_jwt_config(settings: Settings) -> None:
    if (
        settings.qweather_project_id
        and settings.qweather_key_id
        and settings.qweather_api_host
        and (settings.qweather_private_key or settings.qweather_private_key_path)
    ):
        return

    missing_items: list[str] = []
    if not settings.qweather_project_id:
        missing_items.append("QWEATHER_PROJECT_ID")
    if not settings.qweather_key_id:
        missing_items.append("QWEATHER_KEY_ID")
    if not settings.qweather_api_host:
        missing_items.append("QWEATHER_API_HOST")
    if not (settings.qweather_private_key or settings.qweather_private_key_path):
        missing_items.append("QWEATHER_PRIVATE_KEY 或 QWEATHER_PRIVATE_KEY_PATH")

    logger.error("QWeather JWT 配置缺失: missing=%s", ", ".join(missing_items))
    raise WeatherConfigError(
        "未检测到完整的 QWeather JWT 配置。请在 .env 中配置 "
        "QWEATHER_PROJECT_ID、QWEATHER_KEY_ID、QWEATHER_API_HOST，"
        "以及 QWEATHER_PRIVATE_KEY 或 QWEATHER_PRIVATE_KEY_PATH。"
    )


def resolve_qweather_private_key(settings: Settings) -> str:
    if settings.qweather_private_key:
        logger.info("使用环境变量中的 QWeather 私钥内容。")
        return settings.qweather_private_key

    private_key_path = settings.qweather_private_key_path
    if not private_key_path:
        raise WeatherConfigError("未配置 QWEATHER_PRIVATE_KEY 或 QWEATHER_PRIVATE_KEY_PATH。")

    resolved_path = Path(private_key_path).expanduser()
    if not resolved_path.is_absolute():
        resolved_path = settings.project_root / resolved_path

    logger.info("从文件加载 QWeather 私钥: path=%s", resolved_path)
    try:
        return resolved_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise WeatherConfigError(f"QWeather 私钥文件不存在: {resolved_path}") from exc
    except OSError as exc:
        raise WeatherConfigError(f"无法读取 QWeather 私钥文件: {resolved_path}") from exc


def resolve_qweather_jwt_config(settings: Settings) -> QWeatherJwtConfig:
    ensure_qweather_jwt_config(settings)
    ttl_seconds = settings.qweather_jwt_ttl_seconds
    if ttl_seconds <= 0:
        raise WeatherConfigError("QWEATHER_JWT_TTL_SECONDS 必须大于 0。")
    if ttl_seconds > 86400:
        raise WeatherConfigError("QWEATHER_JWT_TTL_SECONDS 不能超过 86400 秒。")
    api_host = settings.qweather_api_host or ""
    if "devapi.qweather.com" in api_host.lower():
        raise WeatherConfigError(
            "JWT 模式下不能继续使用 https://devapi.qweather.com。"
            "请改为和风控制台中分配给项目的专属 QWEATHER_API_HOST。"
        )

    return QWeatherJwtConfig(
        project_id=settings.qweather_project_id or "",
        key_id=settings.qweather_key_id or "",
        private_key=resolve_qweather_private_key(settings),
        api_host=api_host,
        ttl_seconds=ttl_seconds,
    )


def build_qweather_jwt_token(
    jwt_config: QWeatherJwtConfig,
    *,
    now: int | None = None,
) -> str:
    issued_at_base = int(time.time()) if now is None else now
    issued_at = issued_at_base - 30
    expires_at = issued_at + jwt_config.ttl_seconds
    logger.info(
        "开始生成 QWeather JWT: key_id=%s, project_id=%s, ttl_seconds=%s",
        jwt_config.key_id,
        jwt_config.project_id,
        jwt_config.ttl_seconds,
    )
    started_at = perf_counter()
    try:
        token = jwt.encode(
            {
                "sub": jwt_config.project_id,
                "iat": issued_at,
                "exp": expires_at,
            },
            jwt_config.private_key,
            algorithm="EdDSA",
            headers={"kid": jwt_config.key_id},
        )
    except Exception as exc:  # pragma: no cover - defensive wrapper for key parsing/signing
        raise WeatherConfigError(
            "生成 QWeather JWT 失败，请检查 Ed25519 私钥格式、QWEATHER_KEY_ID 和 "
            "QWEATHER_PROJECT_ID 是否正确。"
        ) from exc
    logger.info(
        "QWeather JWT 生成完成，耗时 %.0f ms，iat=%s，exp=%s",
        (perf_counter() - started_at) * 1000,
        issued_at,
        expires_at,
    )
    return token


def normalize_weather_days(days: int) -> int:
    if days not in SUPPORTED_FORECAST_DAYS:
        raise WeatherConfigError(
            f"暂只支持 {', '.join(str(item) for item in SUPPORTED_FORECAST_DAYS)} 天预报。"
        )
    return days


def is_coordinate_query(location: str) -> bool:
    return re.fullmatch(
        r"\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*",
        location,
    ) is not None


def normalize_coordinate_query(location: str) -> str:
    if not is_coordinate_query(location):
        raise WeatherConfigError("经纬度格式无效，请使用 `longitude,latitude`。")

    longitude_text, latitude_text = [part.strip() for part in location.split(",", 1)]
    longitude = float(longitude_text)
    latitude = float(latitude_text)
    if not -180 <= longitude <= 180:
        raise WeatherConfigError("经度超出范围，必须在 -180 到 180 之间。")
    if not -90 <= latitude <= 90:
        raise WeatherConfigError("纬度超出范围，必须在 -90 到 90 之间。")
    return f"{longitude:.2f},{latitude:.2f}"


def format_location_summary(location: ResolvedLocation) -> str:
    parts = [location.name]
    if location.adm2 and location.adm2 != location.name:
        parts.append(location.adm2)
    if location.adm1 and location.adm1 != location.adm2:
        parts.append(location.adm1)
    if location.country:
        parts.append(location.country)
    return ", ".join(part for part in parts if part)


def format_weather_report(result: WeatherQueryResult) -> str:
    temperature_unit = "°C" if result.unit == "m" else "°F"
    precip_unit = "mm" if result.unit == "m" else "in"
    speed_unit = "km/h" if result.unit == "m" else "mph"
    distance_unit = "km" if result.unit == "m" else "mi"

    lines = [
        f"位置: {format_location_summary(result.resolved_location)} "
        f"(LocationID={result.resolved_location.location_id})",
        f"坐标: {result.resolved_location.lon},{result.resolved_location.lat}",
        "当前天气:",
        (
            f"- {result.current_weather.text}，温度 {result.current_weather.temp}{temperature_unit}，"
            f"体感 {result.current_weather.feels_like}{temperature_unit}，"
            f"湿度 {result.current_weather.humidity}%，风向 {result.current_weather.wind_dir} "
            f"{result.current_weather.wind_scale}级，风速 {result.current_weather.wind_speed} {speed_unit}，"
            f"降水 {result.current_weather.precip} {precip_unit}，"
            f"能见度 {result.current_weather.vis} {distance_unit}，"
            f"观测时间 {result.current_weather.obs_time}"
        ),
        f"未来 {result.forecast_days} 天预报:",
    ]

    for item in result.daily_forecast:
        lines.append(
            (
                f"- {item.fx_date}: 白天{item.text_day}/夜间{item.text_night}，"
                f"{item.temp_min}~{item.temp_max}{temperature_unit}，"
                f"风向 {item.wind_dir_day} {item.wind_scale_day}级，"
                f"风速 {item.wind_speed_day} {speed_unit}，"
                f"湿度 {item.humidity}%，降水 {item.precip} {precip_unit}，"
                f"能见度 {item.vis} {distance_unit}，"
                f"日出 {item.sunrise}，日落 {item.sunset}"
            )
        )
    return "\n".join(lines)


class WeatherClient:
    def __init__(
        self,
        *,
        bearer_token: str,
        base_url: str,
        timeout_seconds: float,
        proxy_url: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        logger.info(
            "初始化 WeatherClient: base_url=%s, timeout_seconds=%s",
            base_url,
            timeout_seconds,
        )
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {bearer_token}"},
            timeout=timeout_seconds,
            proxy=proxy_url,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "WeatherClient":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def city_lookup(
        self,
        location: str,
        *,
        adm: str | None,
        lang: str,
        number: int = 10,
    ) -> list[ResolvedLocation]:
        payload = self._request_json(
            "/geo/v2/city/lookup",
            params={
                "location": location,
                "adm": adm,
                "number": number,
                "lang": lang,
            },
        )
        locations = payload.get("location", [])
        logger.info("城市搜索完成: query=%s, candidate_count=%s", location, len(locations))
        return [self._parse_location(item) for item in locations]

    def weather_now(
        self,
        location_id: str,
        *,
        lang: str,
        unit: str,
    ) -> CurrentWeather:
        payload = self._request_json(
            "/v7/weather/now",
            params={"location": location_id, "lang": lang, "unit": unit},
        )
        now = payload["now"]
        return CurrentWeather(
            obs_time=now.get("obsTime", ""),
            temp=now.get("temp", ""),
            feels_like=now.get("feelsLike", ""),
            text=now.get("text", ""),
            wind_dir=now.get("windDir", ""),
            wind_scale=now.get("windScale", ""),
            wind_speed=now.get("windSpeed", ""),
            humidity=now.get("humidity", ""),
            precip=now.get("precip", ""),
            pressure=now.get("pressure", ""),
            vis=now.get("vis", ""),
            cloud=now.get("cloud", ""),
            dew=now.get("dew", ""),
            icon=now.get("icon", ""),
        )

    def weather_daily(
        self,
        location_id: str,
        *,
        days: int,
        lang: str,
        unit: str,
    ) -> list[DailyForecast]:
        payload = self._request_json(
            f"/v7/weather/{days}d",
            params={"location": location_id, "lang": lang, "unit": unit},
        )
        return [
            DailyForecast(
                fx_date=item.get("fxDate", ""),
                temp_min=item.get("tempMin", ""),
                temp_max=item.get("tempMax", ""),
                text_day=item.get("textDay", ""),
                text_night=item.get("textNight", ""),
                wind_dir_day=item.get("windDirDay", ""),
                wind_scale_day=item.get("windScaleDay", ""),
                wind_speed_day=item.get("windSpeedDay", ""),
                humidity=item.get("humidity", ""),
                precip=item.get("precip", ""),
                pressure=item.get("pressure", ""),
                vis=item.get("vis", ""),
                uv_index=item.get("uvIndex", ""),
                sunrise=item.get("sunrise", ""),
                sunset=item.get("sunset", ""),
            )
            for item in payload.get("daily", [])
        ]

    def _request_json(
        self,
        path: str,
        *,
        params: dict[str, object | None],
    ) -> dict[str, object]:
        filtered_params = {key: value for key, value in params.items() if value is not None}
        logger.debug("QWeather 请求开始: path=%s, params=%s", path, filtered_params)
        request_started_at = perf_counter()
        try:
            response = self._client.get(path, params=filtered_params)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise WeatherApiError(f"请求天气服务超时: {path}") from exc
        except httpx.HTTPStatusError as exc:
            raise WeatherApiError(
                f"天气服务返回 HTTP {exc.response.status_code}: {path}"
            ) from exc
        except httpx.RequestError as exc:
            raise WeatherApiError(f"请求天气服务失败: {path}") from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise WeatherApiError(f"天气服务返回了无效 JSON: {path}") from exc

        code = payload.get("code")
        if code != "200":
            raise WeatherApiError(f"天气服务返回业务错误 code={code}: {path}")
        logger.info(
            "QWeather 请求完成: path=%s, status=%s, cost=%.0f ms",
            path,
            response.status_code,
            (perf_counter() - request_started_at) * 1000,
        )
        return payload

    @staticmethod
    def _parse_location(item: dict[str, str]) -> ResolvedLocation:
        return ResolvedLocation(
            location_id=item.get("id", ""),
            name=item.get("name", ""),
            lat=item.get("lat", ""),
            lon=item.get("lon", ""),
            adm1=item.get("adm1", ""),
            adm2=item.get("adm2", ""),
            country=item.get("country", ""),
            tz=item.get("tz"),
            utc_offset=item.get("utcOffset"),
            fx_link=item.get("fxLink"),
        )


class LocationResolver:
    def __init__(self, client: WeatherClient) -> None:
        self._client = client

    def resolve(self, location: str, *, adm: str | None, lang: str) -> ResolvedLocation:
        logger.info("开始解析地点: location=%s, adm=%s", location, adm)
        query = (
            normalize_coordinate_query(location)
            if is_coordinate_query(location)
            else location
        )
        candidates = self._client.city_lookup(query, adm=adm, lang=lang)
        if not candidates:
            raise LocationNotFoundError(f"未找到地点: {location}")
        if is_coordinate_query(location):
            resolved = candidates[0]
            logger.info("经纬度反查完成: %s", format_location_summary(resolved))
            return resolved
        if len(candidates) > 1:
            raise AmbiguousLocationError(location, candidates)
        resolved = candidates[0]
        logger.info("地点解析完成: %s", format_location_summary(resolved))
        return resolved


class WeatherService:
    def __init__(self, client: WeatherClient, resolver: LocationResolver) -> None:
        self._client = client
        self._resolver = resolver

    def query_weather(
        self,
        *,
        location: str,
        adm: str | None,
        lang: str,
        unit: str,
        forecast_days: int,
    ) -> WeatherQueryResult:
        started_at = perf_counter()
        resolved_location = self._resolver.resolve(location, adm=adm, lang=lang)

        current_started_at = perf_counter()
        current_weather = self._client.weather_now(
            resolved_location.location_id,
            lang=lang,
            unit=unit,
        )
        logger.info(
            "实时天气查询完成，耗时 %.0f ms", (perf_counter() - current_started_at) * 1000
        )

        forecast_started_at = perf_counter()
        daily_forecast = self._client.weather_daily(
            resolved_location.location_id,
            days=forecast_days,
            lang=lang,
            unit=unit,
        )
        logger.info(
            "天气预报查询完成，耗时 %.0f ms，days=%s",
            (perf_counter() - forecast_started_at) * 1000,
            forecast_days,
        )
        logger.info("天气查询总耗时 %.0f ms", (perf_counter() - started_at) * 1000)
        return WeatherQueryResult(
            requested_location=location,
            resolved_location=resolved_location,
            current_weather=current_weather,
            daily_forecast=daily_forecast,
            lang=lang,
            unit=unit,
            forecast_days=forecast_days,
        )


def query_weather(
    settings: Settings,
    *,
    location: str,
    adm: str | None = None,
    lang: str | None = None,
    unit: str | None = None,
    forecast_days: int | None = None,
    transport: httpx.BaseTransport | None = None,
) -> WeatherQueryResult:
    jwt_config = resolve_qweather_jwt_config(settings)
    effective_lang = lang or settings.weather_lang
    effective_unit = unit or settings.weather_unit
    effective_days = normalize_weather_days(
        forecast_days if forecast_days is not None else settings.weather_forecast_days
    )
    logger.info(
        "开始查询天气: location=%s, adm=%s, lang=%s, unit=%s, forecast_days=%s",
        location,
        adm,
        effective_lang,
        effective_unit,
        effective_days,
    )
    token = build_qweather_jwt_token(jwt_config)
    with WeatherClient(
        bearer_token=token,
        base_url=jwt_config.api_host,
        timeout_seconds=settings.weather_timeout_seconds,
        proxy_url=None if transport is not None else settings.proxy_url,
        transport=transport,
    ) as client:
        service = WeatherService(client, LocationResolver(client))
        return service.query_weather(
            location=location,
            adm=adm,
            lang=effective_lang,
            unit=effective_unit,
            forecast_days=effective_days,
        )
