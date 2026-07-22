"""Shared infrastructure for repository-level scripts.

The scientific packages (``CKME``, ``CP``, and ``Two_stage``) deliberately do
not depend on this module.  Experiment and maintenance scripts can use these
helpers to locate files without modifying ``sys.path`` or depending on the
current working directory.
"""

from .paths import (
    PROJECT_ROOT_ENV_VAR,
    ProjectRootNotFoundError,
    find_project_root,
    resolve_from,
    resolve_project_path,
)

__all__ = [
    "PROJECT_ROOT_ENV_VAR",
    "ProjectRootNotFoundError",
    "find_project_root",
    "resolve_from",
    "resolve_project_path",
]
