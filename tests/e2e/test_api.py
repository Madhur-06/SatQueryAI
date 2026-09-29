"""End-to-end API tests: POST /analyze with the synthetic fixtures, one per input configuration."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from satquery.api.app import create_app
from satquery.config import Settings


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> TestClient:
    data_dir = tmp_path_factory.mktemp("data")
    settings = Settings(data_dir=data_dir, ui_dir=data_dir / "no-ui", max_upload_mb=1)
    return TestClient(create_app(settings))


def post(client: TestClient, paths: list[Path], query: str, **form):
    files = [("files", (p.name, p.read_bytes(), "image/tiff")) for p in paths]
    return client.post("/analyze", files=files, data={"query": query, **form})


def assert_artifacts_served(client: TestClient, body: dict) -> None:
    kinds = {a["kind"] for a in body["artifacts"]}
    assert {"quicklook", "overlay", "mask_png", "geotiff", "geojson"} <= kinds
    for artifact in body["artifacts"]:
        r = client.get(artifact["url"])
        assert r.status_code == 200, artifact
    geojson = next(a for a in body["artifacts"] if a["kind"] == "geojson")
    fc = client.get(geojson["url"]).json()
    assert fc["type"] == "FeatureCollection"
    trace = client.get(body["trace_url"]).json()
    assert trace["request_id"] == body["request_id"]


def test_health_and_tools(client: TestClient) -> None:
    health = client.get("/health").json()
    assert health["status"] == "ok"
    assert health["tools"] == 14
    assert health["vlm"]["available"] is False
    tools = client.get("/tools").json()
    assert {t["name"] for t in tools} >= {"ndwi_water_mask", "change_cva", "vlm_vqa"}
    assert all("params_schema" in t for t in tools)


def test_single_optical(client: TestClient, optical_4band) -> None:
    r = post(client, [optical_4band.path], "How much area is covered by water?")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok"
    assert body["trace"]["inputs"]["config"] == "single_opt"
    assert body["answer"]["value"] == pytest.approx(optical_4band.truth_hectares("water"), rel=0.02)
    assert 0 < body["confidence"] <= 0.95
    assert_artifacts_served(client, body)


def test_single_sar(client: TestClient, sar_db) -> None:
    r = post(client, [sar_db.path], "Is there any water?")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["trace"]["inputs"]["config"] == "single_sar"
    assert body["answer"]["value"] is True
    assert_artifacts_served(client, body)


def test_pair_optical_sar(client: TestClient, optical_4band, sar_db) -> None:
    r = post(client, [optical_4band.path, sar_db.path], "Map the water with optical and SAR")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["trace"]["inputs"]["config"] == "pair_opt_sar"
    assert body["trace"]["task"]["task"] == "fusion_extract"
    assert body["answer"]["facts"]["agreement_iou"] > 0.85
    assert {a["layer"] for a in body["artifacts"] if a["kind"] == "overlay"} == {"optical", "sar"}
    assert_artifacts_served(client, body)


def test_pair_temporal(client: TestClient, temporal_pair) -> None:
    t1, t2 = temporal_pair
    r = post(client, [t1.path, t2.path], "Has built-up area increased?")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["trace"]["inputs"]["config"] == "pair_temporal"
    assert body["answer"]["value"] is True
    assert body["answer"]["facts"]["gain"] == pytest.approx(
        t2.truth_hectares("builtup_gain"), rel=0.03
    )
    assert {a["layer"] for a in body["artifacts"] if a["kind"] == "quicklook"} == {"t1", "t2"}
    assert_artifacts_served(client, body)


def test_mock_caption(client: TestClient, optical_4band) -> None:
    body = post(client, [optical_4band.path], "Describe this image").json()
    assert body["answer"]["is_mock"] is True
    assert body["answer"]["text"].startswith("[MOCK]")


def test_change_query_with_one_image_is_422_with_trace(client: TestClient, optical_4band) -> None:
    r = post(client, [optical_4band.path], "Has the lake shrunk since 2020?")
    assert r.status_code == 422
    body = r.json()
    assert body["status"] == "rejected"
    assert body["error"]["code"] == "task_input_mismatch"
    assert body["trace"]["task"]["task"] == "change_vqa"


def test_png_requires_benchmark_flag(client: TestClient, png_rgb) -> None:
    r = post(client, [png_rgb.path], "Describe this image")
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "unsupported_format"
    ok = post(client, [png_rgb.path], "Describe this image", benchmark="true")
    assert ok.status_code == 200


def test_too_many_files(client: TestClient, optical_4band) -> None:
    r = post(client, [optical_4band.path] * 3, "Any water?")
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "too_many_inputs"


def test_overrides(client: TestClient, tmp_path: Path) -> None:
    from satquery.testing.synthetic import write_optical

    anon = write_optical(tmp_path / "anon.tif", band_descriptions=False)
    r = post(client, [anon.path], "How much water?")
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "band_role_missing"

    band_map = json.dumps({"green": 2, "nir": 4})
    r = post(client, [anon.path], "How much water?", band_map_1=band_map)
    assert r.status_code == 200, r.text
    layer = r.json()["trace"]["inputs"]["layers"][0]
    assert layer["band_sources"] == {"green": "override", "nir": "override"}

    for bad in ({"band_map_1": "{not json"}, {"date_1": "03/01/2024"}, {"modality_1": "lidar"}):
        r = post(client, [anon.path], "How much water?", **bad)
        assert r.status_code == 422
        assert r.json()["error"]["code"] == "invalid_override"


def test_upload_limit(client: TestClient, tmp_path: Path) -> None:
    big = tmp_path / "big.tif"
    big.write_bytes(b"\0" * (2 << 20))  # 2 MB > 1 MB test limit
    r = post(client, [big], "Any water?")
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "upload_too_large"


@pytest.mark.parametrize(
    "url",
    [
        "/requests/../../etc/files/passwd",
        "/requests/0123456789abcdef0123456789abcdef/files/..%5C..%5Ctrace.json",
        "/requests/not-a-uuid/files/trace.json",
        "/requests/0123456789abcdef0123456789abcdef/files/trace.json",
    ],
)
def test_file_route_rejects_traversal_and_unknown(client: TestClient, url: str) -> None:
    assert client.get(url).status_code == 404
