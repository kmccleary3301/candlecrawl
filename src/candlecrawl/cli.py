from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx
import yaml

from candlecrawl import __version__


def _missing_service_extra(exc: Exception) -> SystemExit:
    return SystemExit(
        "CandleCrawl service dependencies are not available. "
        "Install with `pip install 'candlecrawl[service]'` or run from a dev environment "
        f"with the service requirements installed. Original error: {exc}"
    )


def _load_server_app() -> tuple[Any, Any]:
    try:
        module = importlib.import_module("app.main")
    except Exception as exc:  # pragma: no cover - exact optional dependency varies by environment
        raise _missing_service_extra(exc) from exc
    return module.app, getattr(module, "settings", None)


def command_version(_args: argparse.Namespace) -> int:
    print(__version__)
    return 0


def command_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except Exception as exc:  # pragma: no cover - exact optional dependency varies by environment
        raise _missing_service_extra(exc) from exc

    _load_server_app()
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )
    return 0


async def _health_async(args: argparse.Namespace) -> int:
    url = args.base_url.rstrip("/") + "/health"
    try:
        async with httpx.AsyncClient(timeout=args.timeout) as client:
            response = await client.get(url)
            response.raise_for_status()
    except Exception as exc:
        print(f"CandleCrawl health check failed for {url}: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(response.json(), sort_keys=True))
    else:
        payload = response.json()
        status = payload.get("status", "unknown")
        version = payload.get("version", "unknown")
        print(f"CandleCrawl health: {status} version={version}")
    return 0


def command_health(args: argparse.Namespace) -> int:
    return asyncio.run(_health_async(args))


def command_export_openapi(args: argparse.Namespace) -> int:
    app, settings = _load_server_app()
    if settings is not None and hasattr(settings, "strict_firecrawl_v2"):
        settings.strict_firecrawl_v2 = bool(args.strict_v2)

    spec = app.openapi()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(spec, sort_keys=True, allow_unicode=False),
        encoding="utf-8",
    )
    print(f"Wrote CandleCrawl OpenAPI artifact: {output_path}")
    return 0


def command_doctor(_args: argparse.Namespace) -> int:
    checks = collect_doctor_checks()
    print(json.dumps(checks, indent=2, sort_keys=True))
    return 0 if checks["httpx"] and checks["pydantic"] else 1


def collect_doctor_checks() -> dict[str, bool]:
    checks = {
        "package": True,
        "httpx": _can_import("httpx"),
        "pydantic": _can_import("pydantic"),
        "fastapi": _can_import("fastapi"),
        "uvicorn": _can_import("uvicorn"),
        "playwright": _can_import("playwright.async_api"),
        "pypdf": _can_import("pypdf"),
        "PIL": _can_import("PIL"),
        "redis": _can_import("redis"),
    }
    checks["service_import"] = _can_import("app.main")
    checks["provider_keys_present"] = any(
        bool(os.getenv(name))
        for name in ("SERPER_DEV_API_KEY", "SCRAPE_DO_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY")
    )
    return checks


def _can_import(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
    except Exception:
        return False
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="candlecrawl", description="CandleCrawl CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    version = sub.add_parser("version", help="Print CandleCrawl package version")
    version.set_defaults(func=command_version)

    serve = sub.add_parser("serve", help="Run the CandleCrawl API service")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=3010)
    serve.add_argument("--log-level", default="warning")
    serve.set_defaults(func=command_serve)

    health = sub.add_parser("health", help="Check a running CandleCrawl service")
    health.add_argument("--base-url", default="http://127.0.0.1:3010")
    health.add_argument("--timeout", type=float, default=5.0)
    health.add_argument("--json", action="store_true")
    health.set_defaults(func=command_health)

    export_openapi = sub.add_parser("export-openapi", help="Export CandleCrawl OpenAPI")
    export_openapi.add_argument("--output", required=True)
    export_openapi.add_argument("--strict-v2", action="store_true")
    export_openapi.set_defaults(func=command_export_openapi)

    doctor = sub.add_parser("doctor", help="Diagnose installed CandleCrawl runtime support")
    doctor.set_defaults(func=command_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
