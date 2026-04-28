# CandleCrawl Trusted Publishing Setup

This is the dead-simple setup path for publishing CandleCrawl to TestPyPI and
PyPI without API keys.

## What This Does

Trusted Publishing lets GitHub Actions publish packages to PyPI using GitHub
OIDC identity. No `PYPI_API_TOKEN`, `TWINE_PASSWORD`, or `.env` secret is needed
for normal releases.

The GitHub side is already prepared:

- GitHub repo: `kmccleary3301/candlecrawl`
- GitHub environments:
  - `testpypi`
  - `pypi`
- TestPyPI workflow:
  - `.github/workflows/python-package-publish-dry-run.yml`
- PyPI workflow:
  - `.github/workflows/python-package-publish.yml`

You only need to configure the PyPI/TestPyPI side.

## Step 1: Configure TestPyPI

1. Sign in to TestPyPI:
   - `https://test.pypi.org/`

2. Open Trusted Publishers:
   - If the `candlecrawl` project exists on TestPyPI, go to the project
     publishing settings.
   - If it does not exist yet, use the account publishing settings to add a
     pending trusted publisher for a new project.

3. Add this trusted publisher exactly:

```text
PyPI project name: candlecrawl
Owner: kmccleary3301
Repository name: candlecrawl
Workflow name: python-package-publish-dry-run.yml
Environment name: testpypi
```

4. Save it.

## Step 2: Run TestPyPI Dry Run

1. Open GitHub Actions:
   - `https://github.com/kmccleary3301/candlecrawl/actions`

2. Select workflow:
   - `CandleCrawl Python Package Publish Dry Run`

3. Click:
   - `Run workflow`

4. Wait for it to pass.

The workflow should:

- build the package,
- run `twine check`,
- run package boundary checks,
- publish to TestPyPI,
- install `candlecrawl[service]` from TestPyPI,
- run `candlecrawl version`,
- export OpenAPI from the installed package.

## Step 3: Configure Real PyPI

1. Sign in to PyPI:
   - `https://pypi.org/`

2. Open Trusted Publishers:
   - If the `candlecrawl` project exists on PyPI, go to the project publishing
     settings.
   - If it does not exist yet, use the account publishing settings to add a
     pending trusted publisher for a new project.

3. Add this trusted publisher exactly:

```text
PyPI project name: candlecrawl
Owner: kmccleary3301
Repository name: candlecrawl
Workflow name: python-package-publish.yml
Environment name: pypi
```

4. Save it.

## Step 4: Publish The Next Real Version

Do not republish an existing version. PyPI versions are immutable.

Before publishing, bump CandleCrawl to a new version, for example:

```text
0.1.0a5
```

Update both:

- `pyproject.toml`
- `src/candlecrawl/__init__.py`

Then run local gates:

```bash
pytest -q tests -m "not performance"
python -m build
python -m twine check dist/*
python scripts/check_package_boundaries.py
```

Then open GitHub Actions:

- `https://github.com/kmccleary3301/candlecrawl/actions`

Run workflow:

- `CandleCrawl Python Package Publish`

Manual workflow inputs:

```text
version: 0.1.0a5
confirm_public_pypi: publish
```

Wait for it to pass.

The workflow should publish to real PyPI, then install
`candlecrawl[service]==0.1.0a5` back from PyPI and export OpenAPI.

## If It Fails

If TestPyPI or PyPI says the publisher is not trusted, re-check these fields:

```text
Owner: kmccleary3301
Repository name: candlecrawl
Workflow name: python-package-publish-dry-run.yml
Environment name: testpypi
```

or:

```text
Owner: kmccleary3301
Repository name: candlecrawl
Workflow name: python-package-publish.yml
Environment name: pypi
```

The workflow filename and environment name must match exactly.

## What Not To Do

Do not add a PyPI token to `.env` for normal publishing.

Do not commit any PyPI token.

Do not use manual `twine upload` unless Trusted Publishing is blocked and you
are intentionally doing an emergency fallback.

