# CandleCrawl Release Runbook

This runbook is for package and contract release candidates consumed by Hermes
or other downstream systems.

## Preflight

Run from a clean CandleCrawl checkout:

```bash
git status --short
python -m pip install --upgrade pip
pip install -e ".[test,service,browser,pdf,ocr]"
```

Do not publish if unreviewed secrets, local logs, browser dumps, or legacy
Hermes bridge code appear in the release artifact.

## Required Gates

```bash
pytest -q tests -m "not performance"
python -m build
python -m twine check dist/*
python scripts/check_package_boundaries.py
candlecrawl export-openapi --strict-v2 --output /tmp/openapi-v2-strict.yaml
diff -u contracts/openapi-v2-strict.yaml /tmp/openapi-v2-strict.yaml
```

Then validate a clean SDK-only install:

```bash
tmpdir="$(mktemp -d)"
python -m venv "${tmpdir}/venv"
"${tmpdir}/venv/bin/python" -m pip install --upgrade pip
"${tmpdir}/venv/bin/python" -m pip install dist/candlecrawl-*.whl
"${tmpdir}/venv/bin/python" - <<'PY'
import sys
from candlecrawl import AsyncCandleCrawlClient, CandleCrawlClient
import candlecrawl.client
import candlecrawl.schemas
import candlecrawl.errors
import candlecrawl.trace

assert AsyncCandleCrawlClient
assert CandleCrawlClient
assert "app.main" not in sys.modules
PY
rm -rf "${tmpdir}"
```

Validate a service install when the release includes runtime changes:

```bash
tmpdir="$(mktemp -d)"
python -m venv "${tmpdir}/venv"
"${tmpdir}/venv/bin/python" -m pip install --upgrade pip
"${tmpdir}/venv/bin/python" -m pip install 'dist/candlecrawl-*.whl[service,browser,pdf,ocr]'
"${tmpdir}/venv/bin/candlecrawl" doctor
"${tmpdir}/venv/bin/candlecrawl" export-openapi --strict-v2 --output "${tmpdir}/openapi.yaml"
rm -rf "${tmpdir}"
```

## Versioning

- Alpha package versions are acceptable while the top-level `app` server package
  is still being migrated behind `candlecrawl._server`.
- Do not cut a stable public PyPI release until:
  - the server package relocation is complete or explicitly deferred,
  - Hermes candidate compatibility passes against the accepted v2 contract,
  - release artifacts are free of legacy/private material,
  - rollback instructions are current.

## Hermes Candidate Compatibility

Hermes accepts `contracts/openapi-v2-strict.yaml` as the candidate producer
artifact. Before publishing a CandleCrawl release candidate for Hermes:

```bash
cd ../hermes
./scripts/check_candlecrawl_candidate_compat.sh ../candlecrawl/contracts/openapi-v2-strict.yaml
```

The compatibility check fails on removed operations, removed required fields,
removed required parameters, or removed response codes relative to Hermes'
accepted contract.

## Rollback

If a release candidate breaks Hermes:

1. Stop promoting the candidate artifact.
2. Keep Hermes pinned to the last known accepted contract.
3. Re-run `check_candlecrawl_candidate_compat.sh` against the reverted artifact.
4. File the incompatibility with the operation, field, and response-code diff.
