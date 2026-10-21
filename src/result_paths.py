"""Resolve historical result paths without rewriting hashed evidence files."""

import json
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def _moved_paths():
    manifest = ROOT / "docs/repository_layout_migration.json"
    return json.loads(manifest.read_text())["moves"]


def resolve_result_path(path):
    path = Path(path)
    if path.exists():
        return path
    absolute = path if path.is_absolute() else ROOT / path
    if absolute.exists():
        return absolute
    try:
        old = absolute.relative_to(ROOT).as_posix()
    except ValueError:
        return path
    moved = _moved_paths().get(old)
    return ROOT / moved["to"] if moved else path
