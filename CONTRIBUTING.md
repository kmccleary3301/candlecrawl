# Contributing

CandleCrawl is small enough that contribution discipline matters more than ceremony. The project moves fastest when endpoint changes, runtime changes, and contract changes stay explicit.

## Development Flow

1. Branch from `main`.
2. Keep the change scoped and explain the behavioral intent clearly in the PR.
3. Add or update tests whenever runtime behavior changes.
4. Run the relevant test suite locally before opening a PR.
5. If the HTTP contract changes, update the contract artifact and call that out explicitly.

## Minimum Quality Bar

Run at least:

```bash
pytest -q
```

If you touched contract-facing behavior, also verify:

- `contracts/openapi-v1.yaml` still reflects reality,
- any migration implications are called out in the PR description.

## Contract Changes

For contract-impacting changes:

- update [`contracts/openapi-v1.yaml`](./contracts/openapi-v1.yaml),
- classify the change as major, minor, or patch under [semantic versioning](./VERSIONING.md),
- add migration notes in the PR description,
- avoid "silent" compatibility drift.

## Useful Reference Docs

- [README](./README.md)
- [docs/GETTING_STARTED.md](./docs/GETTING_STARTED.md)
- [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md)
- [docs/API_AND_OPERATIONS.md](./docs/API_AND_OPERATIONS.md)
- [docs/BRANCH_PROTECTION_POLICY.md](./docs/BRANCH_PROTECTION_POLICY.md)
