from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_hermes_search_route_schema():
    resp = client.post("/v1/hermes/leads/search", json={"query": "test", "limit": 2})
    # Without real key it may error, but schema should be consistent
    data = resp.json()
    assert "success" in data
    assert "results" in data or "error" in data


def test_hermes_compose_route_schema():
    resp = client.post("/v1/hermes/compose", json={"prompt": "Say hi"})
    data = resp.json()
    assert "success" in data
    assert ("text" in data) or ("error" in data)


def test_hermes_external_scrape_route_schema():
    resp = client.post("/v1/hermes/external-scrape", json={"url": "https://example.com"})
    data = resp.json()
    assert "success" in data
    assert ("content" in data) or ("error" in data)
