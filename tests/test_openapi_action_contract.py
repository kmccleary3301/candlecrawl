from __future__ import annotations

from pathlib import Path

import yaml


def _load_spec() -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    artifact = repo_root / "contracts" / "openapi-v2-strict.yaml"
    assert artifact.exists(), f"Missing OpenAPI artifact: {artifact}"
    spec = yaml.safe_load(artifact.read_text(encoding="utf-8"))
    assert isinstance(spec, dict)
    return spec


def test_openapi_includes_v2_actions_capabilities_path() -> None:
    spec = _load_spec()
    paths = spec.get("paths")
    assert isinstance(paths, dict)
    assert "/v2/actions/capabilities" in paths

    op = paths["/v2/actions/capabilities"].get("get")
    assert isinstance(op, dict)
    responses = op.get("responses")
    assert isinstance(responses, dict)
    ok = responses.get("200")
    assert isinstance(ok, dict)
    content = ok.get("content")
    assert isinstance(content, dict)
    app_json = content.get("application/json")
    assert isinstance(app_json, dict)
    schema = app_json.get("schema")
    assert isinstance(schema, dict)
    assert schema.get("$ref") == "#/components/schemas/ActionsCapabilitiesResponse"


def test_openapi_action_capability_schemas_are_stable() -> None:
    spec = _load_spec()
    components = spec.get("components")
    assert isinstance(components, dict)
    schemas = components.get("schemas")
    assert isinstance(schemas, dict)

    for name in ("ActionFieldCapability", "ActionCapability", "ActionsCapabilitiesResponse"):
        assert name in schemas

    field_schema = schemas["ActionFieldCapability"]
    assert set(field_schema.get("properties", {}).keys()) == {"name", "valueType", "required", "description"}
    assert field_schema.get("required") == ["name", "valueType"]

    action_schema = schemas["ActionCapability"]
    assert set(action_schema.get("properties", {}).keys()) == {
        "type",
        "aliases",
        "requestFields",
        "outputs",
        "notes",
    }
    assert action_schema.get("required") == ["type"]

    response_schema = schemas["ActionsCapabilitiesResponse"]
    assert set(response_schema.get("properties", {}).keys()) == {
        "success",
        "contractVersion",
        "actions",
        "documentedRequestFields",
        "documentedResponseFields",
    }
    assert response_schema.get("required") in (None, [])
