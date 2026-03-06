# Contributing

## Development flow
1. Branch from `main`.
2. Add tests for behavior changes.
3. Run `pytest -q` and contract checks before opening PR.
4. Keep contract-impacting changes explicitly documented.

Recommended commands:
```bash
uv run pytest -q
uv run python -m app.contract_check
```

## Contract changes
- Update `contracts/openapi-v1.yaml`.
- Classify change as major/minor/patch.
- Add migration notes in PR description.
