from pathlib import Path

from src.lfs_resolver import ensure_lfs_file, is_lfs_pointer


class FakeResponse:
    def __init__(self, payload: bytes):
        self.payload = payload

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        yield self.payload


def test_ensure_lfs_file_replaces_pointer(tmp_path, monkeypatch):
    artifact = tmp_path / "cache" / "risk_model.pkl"
    artifact.parent.mkdir()
    artifact.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:abc\n"
        "size 10\n",
        encoding="utf-8",
    )

    captured = {}

    def fake_get(url, stream, timeout):
        captured["url"] = url
        captured["stream"] = stream
        captured["timeout"] = timeout
        return FakeResponse(b"real-binary")

    monkeypatch.setattr("src.lfs_resolver.requests.get", fake_get)
    monkeypatch.setenv(
        "BANKING_COPILOT_MEDIA_BASE",
        "https://media.example/repo/master",
    )

    assert is_lfs_pointer(artifact)
    ensure_lfs_file(artifact, repository_root=tmp_path)

    assert artifact.read_bytes() == b"real-binary"
    assert captured["url"] == "https://media.example/repo/master/cache/risk_model.pkl"
    assert captured["stream"] is True
