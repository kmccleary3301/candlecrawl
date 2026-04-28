from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _repo_env() -> dict[str, str]:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src") + os.pathsep + str(repo_root)
    return env


def test_doctor_reports_service_import_without_printing_env_values(monkeypatch, capsys) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-secret-test-value")

    from candlecrawl.cli import command_doctor

    assert command_doctor(None) == 0
    output = capsys.readouterr().out
    assert "service_import" in output
    assert "provider_keys_present" in output
    assert "sk-secret-test-value" not in output


def test_serve_health_and_cli_health_smoke() -> None:
    port = _free_port()
    env = _repo_env()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "candlecrawl",
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=str(Path(__file__).resolve().parents[1]),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        health_url = f"http://127.0.0.1:{port}/health"
        last_error = None
        for _ in range(60):
            try:
                response = httpx.get(health_url, timeout=1.0)
                if response.status_code == 200:
                    payload = response.json()
                    assert payload["status"] == "healthy"
                    assert "browserReady" in payload
                    break
            except Exception as exc:  # pragma: no cover - diagnostic only
                last_error = exc
            time.sleep(0.25)
        else:
            output = proc.stdout.read() if proc.stdout else ""
            raise AssertionError(f"CandleCrawl service did not become healthy: {last_error}\n{output}")

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "candlecrawl",
                "health",
                "--base-url",
                f"http://127.0.0.1:{port}",
                "--json",
            ],
            cwd=str(Path(__file__).resolve().parents[1]),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert '"status": "healthy"' in result.stdout
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
