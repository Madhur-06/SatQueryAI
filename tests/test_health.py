from fastapi.testclient import TestClient

from satquery import __version__
from satquery.api.app import app


def test_health() -> None:
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}
