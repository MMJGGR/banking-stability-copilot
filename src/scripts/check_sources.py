"""Check configured remote sources without replacing production data."""

import argparse
import json

from src.sources import build_source_adapters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-configured",
        action="store_true",
        help="Fail if any source has no configured API or bulk URL.",
    )
    args = parser.parse_args()

    reports = [
        adapter.check_remote_version()
        for adapter in build_source_adapters().values()
    ]
    print(json.dumps(reports, indent=2))

    unavailable = [
        report
        for report in reports
        if (args.require_configured and not report["configured"])
        or (report["configured"] and not report["available"])
    ]
    if unavailable:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
