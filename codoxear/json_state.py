from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any


def load_json_file(path: Path, default: Any = None) -> Any:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return default
    return json.loads(raw)


def atomic_write_json(path: Path, obj: Any, *, sort_keys: bool = True, indent: int | None = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        tmp.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=sort_keys, indent=indent) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass
