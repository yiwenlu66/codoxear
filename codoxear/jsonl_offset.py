from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


LogException = Callable[[str, BaseException], None]


def read_jsonl_from_offset(
    path: Path,
    offset: int,
    *,
    max_bytes: int,
    advance_on_oversized_unterminated: bool = True,
    log_exception: LogException,
) -> tuple[list[dict[str, Any]], int]:
    try:
        with path.open("rb") as f:
            f.seek(offset)
            target = max(1, int(max_bytes))
            chunk_size = max(64 * 1024, min(target, 1024 * 1024))
            data = f.read(target)
            if b"\n" not in data:
                if advance_on_oversized_unterminated:
                    # Read at most one overflow chunk so a live, unterminated JSONL
                    # record cannot make every poll read the rest of a huge file.
                    # Complete oversized records with a nearby newline still advance;
                    # fragments with no newline in this bounded window are skipped.
                    data += f.read(chunk_size)
                else:
                    # Live broker tailing must not advance over an incomplete row,
                    # but it also must not get stuck once an oversized row is
                    # completed beyond the bounded poll window. In no-advance mode,
                    # keep scanning until a newline proves at least one full record
                    # is available, or EOF proves the row is still incomplete.
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        data += chunk
                        if b"\n" in chunk:
                            break
    except Exception as e:
        log_exception(f"read jsonl {path} from offset {offset}", e)
        raise

    if not data:
        return [], int(offset)

    # When tailing a live JSONL file, we can read a chunk that ends in the middle
    # of the last record, including the middle of a multibyte UTF-8 sequence.
    # Only parse newline-terminated records, and do not advance the offset past
    # the last newline we observed.
    last_nl = data.rfind(b"\n")
    if last_nl < 0:
        read_cap = max(1, int(max_bytes)) + max(64 * 1024, min(max(1, int(max_bytes)), 1024 * 1024))
        if advance_on_oversized_unterminated and len(data) >= read_cap:
            return [], int(offset) + len(data)
        return [], int(offset)
    data = data[: last_nl + 1]
    new_off = int(offset) + int(last_nl) + 1

    lines = data.splitlines()
    out: list[dict[str, Any]] = []
    for line in lines:
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        except Exception as e:
            log_exception(f"decode jsonl line from {path}", e)
            raise
        if isinstance(obj, dict):
            out.append(obj)
    return out, new_off
