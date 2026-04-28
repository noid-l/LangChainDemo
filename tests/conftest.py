"""共享测试 fixture。"""

from __future__ import annotations

from pathlib import Path

import httpx
import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from chainmaster.config import Settings


def build_private_key_pem() -> str:
    private_key = ed25519.Ed25519PrivateKey.generate()
    return private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")


def build_settings() -> Settings:
    project_root = Path("/tmp/chainmaster")
    project_root.mkdir(parents=True, exist_ok=True)
    private_key_path = project_root / "qweather-test-private.pem"
    private_key_path.write_text(build_private_key_pem(), encoding="utf-8")
    return Settings(
        project_root=project_root,
        knowledge_dir=project_root / "data/knowledge",
        vector_store_path=project_root / ".cache/vector_store.json",
        chat_api_key="chat-key",
        chat_base_url="https://api.openai.com/v1",
        chat_model="gpt-4.1-mini",
        embedding_api_key="embedding-key",
        embedding_base_url="https://api.openai.com/v1",
        embedding_model="text-embedding-3-small",
        vision_api_key=None,
        vision_base_url=None,
        vision_model="zai-org/GLM-4.6V",
        chat_provider="openai",
        rag_top_k=4,
        chunk_size=800,
        chunk_overlap=120,
        qweather_project_id="project-123",
        qweather_key_id="kid-123",
        qweather_private_key_path=str(private_key_path),
        qweather_api_host="https://example.com",
        qweather_jwt_ttl_seconds=900,
        weather_lang="zh",
        weather_unit="m",
        weather_forecast_days=3,
        weather_timeout_seconds=10.0,
        langchain_tracing_v2=False,
        langchain_endpoint=None,
        langchain_api_key=None,
        langchain_project="ChainMasterTest",
    )


def build_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        authorization = request.headers.get("Authorization")
        assert authorization is not None
        assert authorization.startswith("Bearer ")
        token = authorization.removeprefix("Bearer ").strip()
        header = jwt.get_unverified_header(token)
        payload = jwt.decode(
            token,
            options={"verify_signature": False},
            algorithms=["EdDSA"],
        )
        assert header["kid"] == "kid-123"
        assert payload["sub"] == "project-123"
        assert payload["exp"] > payload["iat"]

        if request.url.path == "/geo/v2/city/lookup":
            location = request.url.params["location"]
            adm = request.url.params.get("adm")
            if location == "北京" or location == "116.41,39.92":
                return httpx.Response(
                    200,
                    json={
                        "code": "200",
                        "location": [
                            {
                                "name": "北京",
                                "id": "101010100",
                                "lat": "39.90499",
                                "lon": "116.40529",
                                "adm2": "北京",
                                "adm1": "北京市",
                                "country": "中国",
                                "tz": "Asia/Shanghai",
                                "utcOffset": "+08:00",
                                "fxLink": "https://www.qweather.com/weather/beijing-101010100.html",
                            }
                        ],
                    },
                )
            if location == "西安" and adm is None:
                return httpx.Response(
                    200,
                    json={
                        "code": "200",
                        "location": [
                            {
                                "name": "西安",
                                "id": "101110101",
                                "lat": "34.34157",
                                "lon": "108.93984",
                                "adm2": "西安",
                                "adm1": "陕西省",
                                "country": "中国",
                                "tz": "Asia/Shanghai",
                                "utcOffset": "+08:00",
                                "fxLink": "https://www.qweather.com/weather/xian-101110101.html",
                            },
                            {
                                "name": "西安区",
                                "id": "101050210",
                                "lat": "44.57745",
                                "lon": "129.61616",
                                "adm2": "牡丹江",
                                "adm1": "黑龙江省",
                                "country": "中国",
                                "tz": "Asia/Shanghai",
                                "utcOffset": "+08:00",
                                "fxLink": "https://www.qweather.com/weather/xi-an-district-101050210.html",
                            },
                        ],
                    },
                )
            if location == "不存在":
                return httpx.Response(200, json={"code": "200", "location": []})

        if request.url.path == "/v7/weather/now":
            return httpx.Response(
                200,
                json={
                    "code": "200",
                    "now": {
                        "obsTime": "2026-04-16T17:00+08:00",
                        "temp": "22",
                        "feelsLike": "21",
                        "text": "晴",
                        "windDir": "东南风",
                        "windScale": "2",
                        "windSpeed": "10",
                        "humidity": "40",
                        "precip": "0.0",
                        "pressure": "1008",
                        "vis": "30",
                        "cloud": "5",
                        "dew": "8",
                        "icon": "100",
                    },
                },
            )

        if request.url.path == "/v7/weather/3d":
            return httpx.Response(
                200,
                json={
                    "code": "200",
                    "daily": [
                        {
                            "fxDate": "2026-04-16",
                            "tempMin": "15",
                            "tempMax": "25",
                            "textDay": "晴",
                            "textNight": "多云",
                            "windDirDay": "东风",
                            "windScaleDay": "3",
                            "windSpeedDay": "15",
                            "humidity": "41",
                            "precip": "0.0",
                            "pressure": "1009",
                            "vis": "25",
                            "uvIndex": "6",
                            "sunrise": "05:36",
                            "sunset": "18:47",
                        },
                        {
                            "fxDate": "2026-04-17",
                            "tempMin": "16",
                            "tempMax": "26",
                            "textDay": "多云",
                            "textNight": "晴",
                            "windDirDay": "东北风",
                            "windScaleDay": "3",
                            "windSpeedDay": "12",
                            "humidity": "45",
                            "precip": "0.1",
                            "pressure": "1010",
                            "vis": "20",
                            "uvIndex": "5",
                            "sunrise": "05:35",
                            "sunset": "18:48",
                        },
                        {
                            "fxDate": "2026-04-18",
                            "tempMin": "17",
                            "tempMax": "27",
                            "textDay": "阴",
                            "textNight": "小雨",
                            "windDirDay": "北风",
                            "windScaleDay": "2",
                            "windSpeedDay": "11",
                            "humidity": "50",
                            "precip": "1.2",
                            "pressure": "1011",
                            "vis": "18",
                            "uvIndex": "4",
                            "sunrise": "05:34",
                            "sunset": "18:49",
                        },
                    ],
                },
            )

        if request.url.path == "/v7/weather/7d":
            return httpx.Response(200, json={"code": "401"})

        return httpx.Response(404, json={"code": "404"})

    return httpx.MockTransport(handler)
