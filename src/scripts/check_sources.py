"""Check configured remote sources without replacing production data."""

import argparse
import json

from src.sources import build_source_adapters
from src.sources.sdmx import build_sdmx_sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("official", "legacy"),
        default="official",
        help="official checks IMF SDMX/World Bank freshness; legacy checks env URLs.",
    )
    parser.add_argument(
        "--require-configured",
        action="store_true",
        help="Fail if any source has no configured API or bulk URL.",
    )
    args = parser.parse_args()

    if args.mode == "official":
        reports = []
        for source in build_sdmx_sources().values():
            try:
                report = source.check_version()
                report["configured"] = True
                report["available"] = True
            except Exception as error:
                report = {
                    "source": getattr(source, "name", "unknown"),
                    "configured": True,
                    "available": False,
                    "errors": [str(error)],
                }
            reports.append(report)
    else:
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
