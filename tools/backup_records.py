"""Create a consistent SQLite backup, including committed WAL transactions."""

import argparse
import os
from pathlib import Path
import sqlite3


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--database", type=Path, default=Path(os.environ.get("REHAB_DB", root / "data/rehab.sqlite3")))
    args = parser.parse_args()
    if not args.database.is_file():
        parser.error("源数据库不存在")
    if args.destination.exists():
        parser.error("备份目标已存在，请使用新的文件名")
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(args.database.resolve().as_uri() + "?mode=ro", uri=True) as source:
        with sqlite3.connect(args.destination) as destination:
            source.backup(destination)
    print("备份完成：", args.destination.resolve())


if __name__ == "__main__":
    main()
