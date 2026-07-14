import hashlib

from src.scripts.smoke_test_artifacts import manifest_artifact_failures


def test_manifest_artifact_check_accepts_valid_rooted_file(tmp_path):
    artifact = tmp_path / "data" / "reference" / "government.parquet"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"compact-reference")
    manifest = {
        "artifacts": {
            "data/reference/government.parquet": {
                "sha256": hashlib.sha256(b"compact-reference").hexdigest()
            }
        }
    }

    assert manifest_artifact_failures(manifest, tmp_path) == []


def test_manifest_artifact_check_rejects_traversal_and_bad_checksum(tmp_path):
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}", encoding="utf-8")
    manifest = {
        "artifacts": {
            "../outside.json": {"sha256": "0" * 64},
            "artifact.json": {"sha256": "1" * 64},
        }
    }

    assert manifest_artifact_failures(manifest, tmp_path) == [
        "unsafe path: ../outside.json",
        "checksum: artifact.json",
    ]
