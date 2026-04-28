from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_hermes_compat_routes_are_retired_with_410() -> None:
    resp = client.post("/v1/hermes/leads/search", json={"query": "test", "limit": 2})
    assert resp.status_code == 410
    payload = resp.json()
    assert payload["success"] is False
    assert payload["code"] == "HERMES_COMPAT_RETIRED"
    assert payload["migrationTarget"] == "/v2"


def test_hermes_compat_routes_are_hidden_from_openapi() -> None:
    schema = client.get("/openapi.json").json()
    paths = schema.get("paths", {})
    assert isinstance(paths, dict)
    assert all(not str(path).startswith("/v1/hermes") for path in paths.keys())
