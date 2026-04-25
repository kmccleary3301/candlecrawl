from __future__ import annotations

import base64
import hashlib
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


SUPPORTED_ARTIFACT_MODES = {"inline", "local", "querylake_files"}


@dataclass(frozen=True)
class ArtifactRef:
    mode: str
    uri: str
    content_type: str | None = None
    size_bytes: int | None = None
    inline_base64: str | None = None


class ArtifactSink(Protocol):
    mode: str

    async def put_bytes(
        self,
        *,
        content: bytes,
        filename: str | None = None,
        content_type: str | None = None,
        trace_headers: dict[str, str] | None = None,
    ) -> ArtifactRef:
        ...


@dataclass(frozen=True)
class ArtifactSinkSelectorConfig:
    default_mode: str = "inline"
    allow_fallback_to_inline: bool = True
    local_root: str = "/tmp/candlecrawl_artifacts"
    inline_max_bytes: int = 2 * 1024 * 1024
    querylake_enabled: bool = False
    querylake_sink: ArtifactSink | None = None


@dataclass(frozen=True)
class ArtifactSinkSelection:
    requested_mode: str
    resolved_mode: str
    sink: ArtifactSink
    fallback_reason: str | None = None


def normalize_artifact_mode(raw_mode: str | None, *, default_mode: str = "inline") -> str:
    candidate = (raw_mode or default_mode or "inline").strip().lower()
    aliases = {
        "querylake": "querylake_files",
        "querylake-files": "querylake_files",
        "ql_files": "querylake_files",
        "filesystem": "local",
    }
    normalized = aliases.get(candidate, candidate)
    if normalized not in SUPPORTED_ARTIFACT_MODES:
        raise ValueError(
            f"Unsupported artifact_mode '{raw_mode}'. "
            f"Supported modes: {sorted(SUPPORTED_ARTIFACT_MODES)}"
        )
    return normalized


class InlineArtifactSink:
    mode = "inline"

    def __init__(self, *, max_inline_bytes: int = 2 * 1024 * 1024):
        self.max_inline_bytes = int(max_inline_bytes)

    async def put_bytes(
        self,
        *,
        content: bytes,
        filename: str | None = None,
        content_type: str | None = None,
        trace_headers: dict[str, str] | None = None,
    ) -> ArtifactRef:
        _ = filename, trace_headers
        if len(content) > self.max_inline_bytes:
            raise ValueError(
                f"Inline artifact too large ({len(content)} bytes > {self.max_inline_bytes} bytes)"
            )
        digest = hashlib.sha256(content).hexdigest()
        encoded = base64.b64encode(content).decode("ascii")
        return ArtifactRef(
            mode=self.mode,
            uri=f"inline://sha256/{digest}",
            content_type=content_type,
            size_bytes=len(content),
            inline_base64=encoded,
        )


class LocalArtifactSink:
    mode = "local"

    def __init__(self, *, root: str = "/tmp/candlecrawl_artifacts"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    async def put_bytes(
        self,
        *,
        content: bytes,
        filename: str | None = None,
        content_type: str | None = None,
        trace_headers: dict[str, str] | None = None,
    ) -> ArtifactRef:
        _ = trace_headers
        safe_name = filename or f"{uuid.uuid4().hex}.bin"
        safe_name = os.path.basename(safe_name)
        file_path = self.root / safe_name
        file_path.write_bytes(content)
        return ArtifactRef(
            mode=self.mode,
            uri=str(file_path),
            content_type=content_type,
            size_bytes=len(content),
        )


class QueryLakeArtifactSink:
    mode = "querylake_files"

    async def put_bytes(
        self,
        *,
        content: bytes,
        filename: str | None = None,
        content_type: str | None = None,
        trace_headers: dict[str, str] | None = None,
    ) -> ArtifactRef:
        _ = content, filename, content_type, trace_headers
        raise NotImplementedError(
            "QueryLake artifact sink adapter is not configured. "
            "Implement C02 to wire a real upload adapter."
        )


def select_artifact_sink(
    requested_mode: str | None,
    *,
    config: ArtifactSinkSelectorConfig | None = None,
) -> ArtifactSinkSelection:
    cfg = config or ArtifactSinkSelectorConfig()
    mode = normalize_artifact_mode(requested_mode, default_mode=cfg.default_mode)

    if mode == "inline":
        sink = InlineArtifactSink(max_inline_bytes=cfg.inline_max_bytes)
        return ArtifactSinkSelection(requested_mode=mode, resolved_mode="inline", sink=sink)

    if mode == "local":
        sink = LocalArtifactSink(root=cfg.local_root)
        return ArtifactSinkSelection(requested_mode=mode, resolved_mode="local", sink=sink)

    # querylake_files mode
    if cfg.querylake_sink is not None:
        return ArtifactSinkSelection(
            requested_mode=mode,
            resolved_mode="querylake_files",
            sink=cfg.querylake_sink,
        )
    if cfg.querylake_enabled:
        return ArtifactSinkSelection(
            requested_mode=mode,
            resolved_mode="querylake_files",
            sink=QueryLakeArtifactSink(),
        )
    if cfg.allow_fallback_to_inline:
        return ArtifactSinkSelection(
            requested_mode=mode,
            resolved_mode="inline",
            sink=InlineArtifactSink(max_inline_bytes=cfg.inline_max_bytes),
            fallback_reason="querylake_unavailable",
        )
    raise ValueError("artifact_mode=querylake_files requested but QueryLake sink is unavailable")
