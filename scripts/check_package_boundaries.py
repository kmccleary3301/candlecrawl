#!/usr/bin/env python3
from __future__ import annotations

import ast
import sys
import tarfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = [ROOT / "src" / "candlecrawl", ROOT / "app"]
FORBIDDEN_IMPORT_ROOTS = {
    "bcas_original_tool_calling",
    "contact_extraction_specification",
    "firecrawl_compact",
}
FORBIDDEN_ARTIFACT_SUBSTRINGS = {
    ".env",
    "__pycache__",
    "docs_tmp",
    "firecrawl.out",
    "legacy/hermes_bcas.py",
    "app/hermes_bcas.py",
    "bcas_original_tool_calling",
    "contact_extraction_specification",
    "firecrawl_compact",
}


def _import_root(name: str) -> str:
    return name.split(".", 1)[0]


def _iter_python_files() -> list[Path]:
    files: list[Path] = []
    for root in SOURCE_ROOTS:
        if root.exists():
            files.extend(sorted(root.rglob("*.py")))
    return files


def check_source_imports() -> list[str]:
    errors: list[str] = []
    for path in _iter_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _import_root(alias.name) in FORBIDDEN_IMPORT_ROOTS:
                        errors.append(f"{path.relative_to(ROOT)} imports {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                if _import_root(node.module) in FORBIDDEN_IMPORT_ROOTS:
                    errors.append(f"{path.relative_to(ROOT)} imports {node.module}")
    return errors


def _artifact_members(path: Path) -> list[str]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return archive.namelist()
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path) as archive:
            return archive.getnames()
    return []


def check_artifacts() -> list[str]:
    errors: list[str] = []
    for path in sorted((ROOT / "dist").glob("*")):
        members = _artifact_members(path)
        for member in members:
            normalized = member.replace("\\", "/").lower()
            if normalized.startswith("app/") or "/app/" in normalized:
                errors.append(f"{path.name} contains top-level legacy app package member {member}")
            for forbidden in FORBIDDEN_ARTIFACT_SUBSTRINGS:
                if forbidden.lower() in normalized:
                    errors.append(f"{path.name} contains forbidden artifact member {member}")
    return errors


def main() -> int:
    errors = check_source_imports() + check_artifacts()
    if errors:
        print("CandleCrawl package boundary check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("CandleCrawl package boundary check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
