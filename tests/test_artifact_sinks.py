from __future__ import annotations

from pathlib import Path

import pytest

from app.artifacts import (
    ArtifactRef,
    ArtifactSinkSelectorConfig,
    InlineArtifactSink,
    normalize_artifact_mode,
    select_artifact_sink,
)


def test_normalize_artifact_mode_aliases_and_validation() -> None:
    assert normalize_artifact_mode("querylake") == "querylake_files"
    assert normalize_artifact_mode("querylake-files") == "querylake_files"
    assert normalize_artifact_mode("filesystem") == "local"
    with pytest.raises(ValueError):
        normalize_artifact_mode("unsupported")


def test_select_artifact_sink_defaults_to_inline_mode() -> None:
    selection = select_artifact_sink(None, config=ArtifactSinkSelectorConfig(default_mode="inline"))
    assert selection.requested_mode == "inline"
    assert selection.resolved_mode == "inline"
    assert selection.fallback_reason is None
    assert isinstance(selection.sink, InlineArtifactSink)


@pytest.mark.asyncio
async def test_local_sink_selection_and_write(tmp_path: Path) -> None:
    selection = select_artifact_sink(
        "local",
        config=ArtifactSinkSelectorConfig(local_root=str(tmp_path), default_mode="inline"),
    )
    assert selection.resolved_mode == "local"
    ref = await selection.sink.put_bytes(content=b"hello", filename="sample.txt", content_type="text/plain")
    assert isinstance(ref, ArtifactRef)
    assert ref.mode == "local"
    assert ref.size_bytes == 5
    assert Path(ref.uri).exists()


def test_querylake_mode_falls_back_to_inline_when_unavailable() -> None:
    selection = select_artifact_sink(
        "querylake_files",
        config=ArtifactSinkSelectorConfig(
            querylake_enabled=False,
            allow_fallback_to_inline=True,
        ),
    )
    assert selection.requested_mode == "querylake_files"
    assert selection.resolved_mode == "inline"
    assert selection.fallback_reason == "querylake_unavailable"
    assert isinstance(selection.sink, InlineArtifactSink)


def test_querylake_mode_without_fallback_raises_when_unavailable() -> None:
    with pytest.raises(ValueError):
        select_artifact_sink(
            "querylake_files",
            config=ArtifactSinkSelectorConfig(
                querylake_enabled=False,
                allow_fallback_to_inline=False,
            ),
        )
