#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from codoxear.page_state_sqlite import import_legacy_app_dir_to_db


def main() -> int:
    parser = argparse.ArgumentParser(description="Import Codoxear legacy page metadata JSON files into SQLite")
    parser.add_argument("--source-app-dir", required=True, help="Legacy Codoxear app dir containing JSON state files and socks/")
    parser.add_argument("--db-path", required=True, help="Target sqlite database path")
    args = parser.parse_args()

    report = import_legacy_app_dir_to_db(
        source_app_dir=Path(args.source_app_dir),
        db_path=Path(args.db_path),
    )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
