"""An archive keeps its source identity regardless of its build directory."""

import importlib.util
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("source_build_hook", ROOT / "hatch_build.py")
hook_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hook_module)
TARGET = hook_module._TARGET
ENV = (
    "ITAMAE"
    if TARGET.startswith("itamae/")
    else TARGET.removeprefix("_").removesuffix("_build_provenance.py").upper()
) + "_SOURCE_REVISION"
ARCHIVE_SHA = "1234567890abcdef1234567890abcdef12345678"


def hook(root):
    return hook_module.CustomBuildHook(str(root), {}, None, None, str(root / "dist"), "wheel")


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


@pytest.fixture
def repository(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    git(tmp_path, "init", "--quiet")
    (tmp_path / "unrelated.txt").write_text("parent source")
    git(tmp_path, "add", "unrelated.txt")
    git(
        tmp_path,
        "-c",
        "user.name=Build test",
        "-c",
        "user.email=build@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "Parent source",
    )
    return tmp_path


def archive(root):
    root.mkdir(parents=True)
    descriptor = root / TARGET
    descriptor.parent.mkdir(parents=True, exist_ok=True)
    descriptor.write_text(f"SOURCE_REVISION = {ARCHIVE_SHA!r}\n")
    return root


def test_archive_in_unrelated_repository_keeps_embedded_revision(repository):
    source = archive(repository / "unpacked")
    assert hook(source)._resolve_revision() == ARCHIVE_SHA


def test_unidentified_export_rejects_enclosing_repository(repository):
    source = repository / "export"
    source.mkdir()
    with pytest.raises(RuntimeError, match="source revision"):
        hook(source)._resolve_revision()


def test_exact_source_checkout_uses_own_revision(repository):
    assert hook(repository)._resolve_revision() == git(repository, "rev-parse", "HEAD")


def test_archive_rejects_conflicting_explicit_revision(repository, monkeypatch):
    source = archive(repository / "unpacked")
    monkeypatch.setenv(ENV, "f" * 40)
    with pytest.raises(RuntimeError, match="conflict"):
        hook(source)._resolve_revision()


def test_exact_checkout_rejects_conflicting_explicit_revision(repository, monkeypatch):
    monkeypatch.setenv(ENV, "f" * 40)
    with pytest.raises(RuntimeError, match="conflict"):
        hook(repository)._resolve_revision()


def test_malformed_explicit_revision_does_not_silently_fall_back(repository, monkeypatch):
    monkeypatch.setenv(ENV, "short")
    with pytest.raises(RuntimeError, match="40"):
        hook(repository)._resolve_revision()


def test_matching_explicit_archive_revision_is_allowed(repository, monkeypatch):
    source = archive(repository / "unpacked")
    monkeypatch.setenv(ENV, ARCHIVE_SHA)
    assert hook(source)._resolve_revision() == ARCHIVE_SHA
