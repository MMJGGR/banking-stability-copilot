"""Runtime fallback for Git LFS serving artifacts.

Streamlit Community Cloud deployments can occasionally see Git LFS pointer
files instead of the resolved binary artifacts. The active serving artifacts
are public, so when a pointer is detected this module downloads the real object
from GitHub's media endpoint and replaces the pointer in place.
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote

import requests


LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"
DEFAULT_MEDIA_BASE = (
    "https://media.githubusercontent.com/media/"
    "MMJGGR/banking-stability-copilot/master"
)


def is_lfs_pointer(path: Path) -> bool:
    path = Path(path)
    if not path.exists() or path.stat().st_size > 1024:
        return False
    with path.open("rb") as source:
        return source.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX


def ensure_lfs_file(path, repository_root=None, timeout=120, force=False) -> Path:
    """Replace a Git LFS pointer with the real public media file if needed.

    When ``force`` is true the object is re-downloaded even if the on-disk file
    is not a pointer. This self-heals a stale resolved artifact: some hosts
    (e.g. Streamlit Community Cloud) persist a working tree where a previous run
    overwrote the pointer with now-outdated content, so a plain pointer check
    would never refresh it after the tracked object changes.
    """
    path = Path(path)
    if not force and not is_lfs_pointer(path):
        return path

    root = Path(repository_root) if repository_root is not None else Path(__file__).resolve().parents[1]
    rel_path = path.resolve().relative_to(root.resolve()).as_posix()
    media_base = os.getenv("BANKING_COPILOT_MEDIA_BASE", DEFAULT_MEDIA_BASE).rstrip("/")
    url = f"{media_base}/{quote(rel_path)}"

    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()

    temporary = path.with_suffix(path.suffix + ".download")
    with temporary.open("wb") as output:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                output.write(chunk)

    if is_lfs_pointer(temporary) or temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded LFS media is invalid for {rel_path}")

    temporary.replace(path)
    return path
