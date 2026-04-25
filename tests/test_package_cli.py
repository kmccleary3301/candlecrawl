from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_public_package_imports_are_lightweight() -> None:
    import candlecrawl
    import candlecrawl.client
    import candlecrawl.errors
    import candlecrawl.schemas
    import candlecrawl.trace

    assert candlecrawl.__version__


def test_public_import_does_not_import_legacy_server_app() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src") + os.pathsep + str(repo_root)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, candlecrawl; raise SystemExit(1 if 'app.main' in sys.modules else 0)",
        ],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_cli_service_import_targets_private_server_package() -> None:
    from candlecrawl import cli

    assert cli.collect_doctor_checks()["service_import"] is True
    assert cli._load_server_app()[0] is not None


def test_cli_version(capsys) -> None:
    from candlecrawl.cli import main

    assert main(["version"]) == 0
    assert capsys.readouterr().out.strip()


def test_cli_help_surfaces_core_commands() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src") + os.pathsep + str(repo_root)

    result = subprocess.run(
        [sys.executable, "-m", "candlecrawl", "--help"],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "serve" in result.stdout
    assert "export-openapi" in result.stdout
    assert "doctor" in result.stdout


def test_cli_subcommand_help() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src") + os.pathsep + str(repo_root)

    for command in ("serve", "health", "export-openapi", "doctor"):
        result = subprocess.run(
            [sys.executable, "-m", "candlecrawl", command, "--help"],
            cwd=str(repo_root),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert command in result.stdout
