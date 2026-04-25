from __future__ import annotations

from pathlib import Path

import yaml


def test_strict_openapi_artifact_exists_and_has_v2_surface() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    artifact = repo_root / "contracts" / "openapi-v2-strict.yaml"
    assert artifact.exists(), f"Missing OpenAPI artifact: {artifact}"

    spec = yaml.safe_load(artifact.read_text(encoding="utf-8"))
    assert isinstance(spec, dict)
    paths = spec.get("paths")
    assert isinstance(paths, dict)

    expected_v2_paths = {
        "/v2/actions/capabilities",
        "/v2/scrape",
        "/v2/map",
        "/v2/search",
        "/v2/crawl",
        "/v2/crawl/{job_id}",
        "/v2/crawl/{job_id}/errors",
        "/v2/batch/scrape",
        "/v2/batch/scrape/{job_id}",
        "/v2/batch/scrape/{job_id}/errors",
        "/v2/extract",
        "/v2/extract/{job_id}",
    }
    for path in expected_v2_paths:
        assert path in paths

    # CandleCrawl no longer publishes Hermes compatibility routes.
    assert all(not str(path).startswith("/v1/hermes") for path in paths.keys())
