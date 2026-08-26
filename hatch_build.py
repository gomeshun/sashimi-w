"""Embed the SASHIMI-W source revision in build artifacts."""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

_SOURCE_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
_TARGET = "_sashimi_w_build_provenance.py"


def _valid_revision(value: str | None) -> str | None:
    """Return a full lowercase source SHA or ``None``."""
    if value is not None:
        value = value.strip()
    return value if value and _SOURCE_REVISION_PATTERN.fullmatch(value) else None


class CustomBuildHook(BuildHookInterface):
    """Add an exact source revision to each Hatchling artifact."""

    _generated_path: Path | None = None

    def _revision_from_existing_artifact(self) -> str | None:
        """Preserve a revision when rebuilding from an exported source archive."""
        path = Path(self.root) / _TARGET
        if not path.is_file():
            return None
        match = re.search(
            r"SOURCE_REVISION\s*=\s*['\"]([0-9a-f]{40})['\"]",
            path.read_text(encoding="utf-8"),
        )
        return match.group(1) if match else None

    def _resolve_revision(self) -> str:
        """Resolve the exact revision used to create the artifact."""
        revision = _valid_revision(os.environ.get("SASHIMI_W_SOURCE_REVISION"))
        if revision is None:
            try:
                result = subprocess.run(
                    ["git", "-C", self.root, "rev-parse", "HEAD"],
                    check=False,
                    capture_output=True,
                    text=True,
                )
            except OSError:
                result = None
            revision = _valid_revision(
                result.stdout if result is not None and result.returncode == 0 else None
            )
        if revision is None:
            revision = self._revision_from_existing_artifact()
        if revision is None:
            raise RuntimeError(
                "SASHIMI-W artifact builds require an exact source revision from "
                "the source archive, SASHIMI_W_SOURCE_REVISION, or Git."
            )
        return revision

    def initialize(self, version: str, build_data: dict[str, object]) -> None:
        """Generate and force-include the build provenance module."""
        revision = self._resolve_revision()
        descriptor_fd, descriptor_name = tempfile.mkstemp(
            prefix="sashimi-w-build-provenance-", suffix=".py"
        )
        os.close(descriptor_fd)
        descriptor = Path(descriptor_name)
        descriptor.write_text(f"SOURCE_REVISION = {revision!r}\n", encoding="utf-8")
        self._generated_path = descriptor
        force_include = build_data.setdefault("force_include", {})
        force_include[str(descriptor)] = _TARGET

    def finalize(
        self,
        version: str,
        build_data: dict[str, object],
        artifact_path: str,
    ) -> None:
        """Remove the temporary source file after the artifact is written."""
        if self._generated_path is not None:
            self._generated_path.unlink(missing_ok=True)
            self._generated_path = None
