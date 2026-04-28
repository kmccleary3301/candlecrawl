from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_retired_hermes_compose_returns_410_shape() -> None:
    resp = client.post("/v1/hermes/compose", json={"prompt": "hello"})
    assert resp.status_code == 410
    payload = resp.json()
    assert payload["success"] is False
    assert payload["code"] == "HERMES_COMPAT_RETIRED"
    assert payload["legacyPath"] == "/v1/hermes/compose"


def test_retired_hermes_external_scrape_returns_410_shape() -> None:
    resp = client.post("/v1/hermes/external-scrape", json={"url": "https://example.com"})
    assert resp.status_code == 410
    payload = resp.json()
    assert payload["success"] is False
    assert payload["code"] == "HERMES_COMPAT_RETIRED"
    assert payload["legacyPath"] == "/v1/hermes/external-scrape"


def test_retired_hermes_catch_all_handles_other_methods() -> None:
    resp = client.get("/v1/hermes/anything")
    assert resp.status_code == 410
    payload = resp.json()
    assert payload["success"] is False
    assert payload["code"] == "HERMES_COMPAT_RETIRED"
    assert payload["legacyPath"] == "/v1/hermes/anything"
