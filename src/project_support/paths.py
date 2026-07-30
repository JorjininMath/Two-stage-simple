"""Stable path resolution for project scripts.

The repository uses a ``src`` layout.  Install it in editable mode with
``python -m pip install -e .`` before running scripts from arbitrary working
directories.  During a lightweight checkout test, ``PYTHONPATH=src`` provides
the same imports without changing the Python environment.

Path rules
----------
* Absolute user paths remain absolute.
* Relative user paths resolve from the project root by default.
* Paths owned by a specific experiment can resolve from that experiment's
  directory with :func:`resolve_from`.
* ``CKME_PROJECT_ROOT`` can explicitly identify a checkout when code is invoked
  from outside the repository.

No helper in this module creates files or directories.  Callers remain in
control of all filesystem writes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

PathLike = Union[str, os.PathLike[str]]

PROJECT_ROOT_ENV_VAR = "CKME_PROJECT_ROOT"
_PROJECT_MARKERS = ("pyproject.toml", "PROTOCOL.md")


class ProjectRootNotFoundError(RuntimeError):
    """Raised when a CKME project checkout cannot be located."""


def _as_search_directory(path: Path) -> Path:
    """Return a directory suitable for walking toward filesystem root."""

    resolved = path.expanduser().resolve()
    return resolved if resolved.is_dir() else resolved.parent


def _candidate_directories(start: Path) -> tuple[Path, ...]:
    """Return ``start`` and all of its parents, without duplicates."""

    directory = _as_search_directory(start)
    return (directory, *directory.parents)


def _is_project_root(path: Path) -> bool:
    """Identify this repository using two intentionally stable marker files."""

    return all((path / marker).is_file() for marker in _PROJECT_MARKERS)


def find_project_root(start: PathLike | None = None) -> Path:
    """Locate and return the CKME project root.

    Resolution order is:

    1. ``CKME_PROJECT_ROOT`` when explicitly configured;
    2. ``start`` (or the current working directory) and its parents;
    3. this installed module's location and its parents.

    The third step is what makes editable installs independent of the current
    working directory.  A regular wheel does not contain the research
    repository, so wheel users should set ``CKME_PROJECT_ROOT`` whenever they
    need repository files such as experiment configurations or artifacts.

    Parameters
    ----------
    start:
        Optional file or directory from which to begin the search.

    Raises
    ------
    ProjectRootNotFoundError
        If no checkout can be found, or if an explicit environment override is
        not an existing directory.
    """

    configured_root = os.environ.get(PROJECT_ROOT_ENV_VAR)
    if configured_root:
        root = Path(configured_root).expanduser().resolve()
        if not root.is_dir():
            raise ProjectRootNotFoundError(
                f"{PROJECT_ROOT_ENV_VAR} points to a missing directory: {root}"
            )
        return root

    search_starts = (
        Path(start) if start is not None else Path.cwd(),
        Path(__file__),
    )
    searched: list[Path] = []
    for search_start in search_starts:
        for candidate in _candidate_directories(search_start):
            if candidate in searched:
                continue
            searched.append(candidate)
            if _is_project_root(candidate):
                return candidate

    searched_text = ", ".join(str(path) for path in searched)
    raise ProjectRootNotFoundError(
        "Could not locate the CKME project root. Run from a project checkout "
        f"or set {PROJECT_ROOT_ENV_VAR}. Searched: {searched_text}"
    )


def resolve_project_path(
    path: PathLike,
    *,
    root: PathLike | None = None,
    must_exist: bool = False,
) -> Path:
    """Resolve ``path`` relative to the project root.

    Parameters
    ----------
    path:
        Absolute path or project-root-relative path.
    root:
        Explicit project root, primarily useful for tests and tooling.  When
        omitted, :func:`find_project_root` performs discovery.
    must_exist:
        Raise ``FileNotFoundError`` when the resolved path is absent.
    """

    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        base = (
            Path(root).expanduser().resolve()
            if root is not None
            else find_project_root()
        )
        candidate = base / candidate

    resolved = candidate.resolve()
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Resolved project path does not exist: {resolved}")
    return resolved


def resolve_from(
    base_directory: PathLike,
    path: PathLike,
    *,
    must_exist: bool = False,
) -> Path:
    """Resolve ``path`` from an owning directory, unless it is absolute.

    Experiment scripts should pass their own directory as ``base_directory``
    for default config and output locations.  User-supplied project paths should
    generally use :func:`resolve_project_path` instead.
    """

    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path(base_directory).expanduser().resolve() / candidate

    resolved = candidate.resolve()
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Resolved path does not exist: {resolved}")
    return resolved
