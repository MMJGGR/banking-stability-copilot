"""Resilient retrieval primitives for weak or changing source APIs."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterable

import pandas as pd
import requests


class SourceUnavailableError(RuntimeError):
    pass


@dataclass
class SourceResult:
    source: str
    retrieval_method: str
    path: str
    bytes: int
    sha256: str
    retrieved_at: str
    remote_version: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class SourceAdapter:
    """Retrieve an export file with API, bulk, and local fallbacks."""

    def __init__(
        self,
        name: str,
        local_patterns: Iterable[str],
        required_columns: Iterable[str],
        api_url_env: str,
        bulk_url_env: str,
        timeout_seconds: int = 60,
    ):
        self.name = name
        self.local_patterns = tuple(local_patterns)
        self.required_columns = set(required_columns)
        self.api_url_env = api_url_env
        self.bulk_url_env = bulk_url_env
        self.timeout_seconds = timeout_seconds

    def configured_urls(self) -> list[tuple[str, str]]:
        urls = []
        api_url = os.getenv(self.api_url_env)
        bulk_url = os.getenv(self.bulk_url_env)
        if api_url:
            urls.append(("api", api_url))
        if bulk_url and bulk_url != api_url:
            urls.append(("bulk", bulk_url))
        return urls

    def check_remote_version(self) -> dict:
        errors = []
        for method, url in self.configured_urls():
            try:
                response = requests.head(
                    url,
                    allow_redirects=True,
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                return {
                    "source": self.name,
                    "configured": True,
                    "available": True,
                    "method": method,
                    "url": url,
                    "etag": response.headers.get("ETag"),
                    "last_modified": response.headers.get("Last-Modified"),
                    "content_length": response.headers.get("Content-Length"),
                }
            except requests.RequestException as error:
                errors.append(f"{method}: {error}")
        return {
            "source": self.name,
            "configured": bool(self.configured_urls()),
            "available": False,
            "errors": errors,
        }

    def fetch(self, destination_dir, local_search_root=None) -> SourceResult:
        destination_dir = Path(destination_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)
        errors = []

        for method, url in self.configured_urls():
            try:
                result = self._download(url, destination_dir, method)
                self.validate(result.path)
                return result
            except (requests.RequestException, OSError, ValueError) as error:
                errors.append(f"{method}: {error}")

        local = self.find_local_fallback(local_search_root or Path.cwd())
        if local is not None:
            self.validate(local)
            return SourceResult(
                source=self.name,
                retrieval_method="local_fallback",
                path=str(local.resolve()),
                bytes=local.stat().st_size,
                sha256=self._sha256(local),
                retrieved_at=datetime.now(timezone.utc).isoformat(),
            )

        detail = "; ".join(errors) if errors else "no source URL configured"
        raise SourceUnavailableError(
            f"{self.name} is unavailable and has no valid local fallback: {detail}"
        )

    def find_local_fallback(self, root) -> Path | None:
        root = Path(root)
        matches = []
        for pattern in self.local_patterns:
            matches.extend(root.glob(pattern))
        files = [path for path in matches if path.is_file()]
        return max(files, key=lambda path: path.stat().st_mtime) if files else None

    def validate(self, path) -> None:
        path = Path(path)
        if not path.exists() or path.stat().st_size == 0:
            raise ValueError(f"{self.name} source file is missing or empty")

        suffix = path.suffix.lower()
        if suffix == ".csv":
            columns = set(pd.read_csv(path, nrows=0).columns)
        elif suffix in {".xlsx", ".xls"}:
            columns = set(pd.read_excel(path, nrows=0).columns)
        else:
            raise ValueError(f"Unsupported source format for {self.name}: {suffix}")

        missing = self.required_columns.difference(columns)
        if missing:
            raise ValueError(
                f"{self.name} source is missing required columns: {sorted(missing)}"
            )

    def _download(self, url: str, destination_dir: Path, method: str) -> SourceResult:
        response = requests.get(
            url,
            stream=True,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        filename = self._filename_from_response(response, url)
        destination = destination_dir / filename

        with tempfile.NamedTemporaryFile(
            dir=destination_dir,
            prefix=f".{filename}.",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            try:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        temporary.write(chunk)
            except Exception:
                temporary_path.unlink(missing_ok=True)
                raise

        shutil.move(str(temporary_path), destination)
        return SourceResult(
            source=self.name,
            retrieval_method=method,
            path=str(destination.resolve()),
            bytes=destination.stat().st_size,
            sha256=self._sha256(destination),
            retrieved_at=datetime.now(timezone.utc).isoformat(),
            remote_version=(
                response.headers.get("ETag")
                or response.headers.get("Last-Modified")
            ),
        )

    @staticmethod
    def _filename_from_response(response, url: str) -> str:
        disposition = response.headers.get("Content-Disposition", "")
        if "filename=" in disposition:
            return disposition.split("filename=", 1)[1].strip().strip('"')
        candidate = Path(url.split("?", 1)[0]).name
        return candidate or "source-download.csv"

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()
