"""Relocatable, content-addressed builder for project-owned native kernels."""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, Sequence
from execution_accounting import native_setup


_PACKAGE_NATIVE = Path(__file__).resolve().parent / "_native_src"
_SOURCE_TREE_NATIVE = Path(__file__).resolve().parents[2] / "native"
_CACHE_DIRECTORY = Path(__file__).resolve().parents[2] / ".native_cache"


def native_source(*parts: str) -> Path:
    """Resolve a kernel source in either a wheel or the source checkout."""

    packaged = _PACKAGE_NATIVE.joinpath(*parts)
    if packaged.is_file():
        return packaged
    checkout = _SOURCE_TREE_NATIVE.joinpath(*parts)
    if checkout.is_file():
        return checkout
    raise FileNotFoundError(
        f"GCFR-Scale native source is missing from package and checkout: {parts!r}"
    )


@native_setup
def build_shared_library(
    name: str,
    source: Path,
    *,
    dependencies: Iterable[Path] = (),
    compiler_flags: Sequence[str],
) -> Path:
    """Compile a content-addressed shared library outside the installed package."""

    compiler_flags = ([flag for flag in compiler_flags if flag not in ('-static-libgcc','-static-libstdc++')] + ['-fPIC', '-fvisibility-inlines-hidden', '-D__declspec(x)=__attribute__((visibility("default")))']) if os.name != 'nt' else list(compiler_flags)
    primary = Path(source).resolve()
    inputs = tuple(Path(value).resolve() for value in (primary, *dependencies))
    import json
    root = Path(__file__).resolve().parents[3]
    manifest_path = root / 'PINNED_NATIVE_LINUX.json'
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_bytes())
        for path in inputs:
            key = str(path.relative_to(root))
            assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['source_inputs'][key], key
        selected = manifest['libraries'][name]
        library = root / selected['relative']
        assert hashlib.sha256(library.read_bytes()).hexdigest() == selected['sha256'], name
        return library
    digest = hashlib.sha256()
    digest.update(platform.platform().encode("utf-8"))
    digest.update(platform.machine().encode("utf-8"))
    for flag in compiler_flags:
        digest.update(flag.encode("utf-8"))
        digest.update(b"\0")
    for path in inputs:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    suffix = ".dll" if os.name == "nt" else ".so"
    _CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    target = _CACHE_DIRECTORY / f"{name}-{digest.hexdigest()[:24]}{suffix}"
    if target.is_file():
        return target
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{name}.", suffix=suffix, dir=_CACHE_DIRECTORY
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        completed = subprocess.run(
            [os.environ.get("CXX", "g++"), *compiler_flags, "-o", str(temporary), str(primary)],
            cwd=primary.parent,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode:
            raise RuntimeError(
                f"clean {name} build failed:\n"
                + (completed.stdout or "")
                + (completed.stderr or "")
            )
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


__all__ = ("build_shared_library", "native_source")
