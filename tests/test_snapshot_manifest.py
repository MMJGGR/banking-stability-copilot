import hashlib

from src.snapshot_manifest import sha256_file, write_snapshot_manifest


def test_sha256_file(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"banking-copilot")

    assert sha256_file(source) == hashlib.sha256(
        b"banking-copilot"
    ).hexdigest()


def test_write_snapshot_manifest(tmp_path):
    output = tmp_path / "nested" / "manifest.json"
    result = write_snapshot_manifest(
        {"snapshot_id": "2025-12-31"},
        output,
    )

    assert result == output
    assert '"snapshot_id": "2025-12-31"' in output.read_text(
        encoding="utf-8"
    )
