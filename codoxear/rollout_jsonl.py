from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Iterator


@dataclass(frozen=True)
class JsonlRecord:
    start: int
    end: int
    obj: dict[str, Any]


def _parse_jsonl_line(raw_line: bytes | str) -> dict[str, Any] | None:
    if isinstance(raw_line, bytes):
        try:
            line = raw_line.decode("utf-8")
        except UnicodeDecodeError:
            return None
    else:
        line = raw_line
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _read_jsonl_tail(path: Path, max_bytes: int) -> list[dict[str, Any]]:
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        start = max(0, size - max_bytes)
        starts_at_line_boundary = start == 0
        if start > 0:
            f.seek(start - 1)
            starts_at_line_boundary = f.read(1) == b"\n"
        f.seek(start)
        data = f.read()

    if not data:
        return []
    if start > 0 and not starts_at_line_boundary:
        nl = data.find(b"\n")
        if nl >= 0:
            data = data[nl + 1 :]

    out: list[dict[str, Any]] = []
    for line in data.splitlines():
        obj = _parse_jsonl_line(line)
        if obj is not None:
            out.append(obj)
    return out


def _read_jsonl_records_from_offset(path: Path, offset: int, *, max_bytes: int) -> tuple[list[JsonlRecord], int]:
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        start = max(0, min(int(offset), size))
        f.seek(start)
        target = max(1, int(max_bytes))
        chunk_size = max(64 * 1024, min(target, 1024 * 1024))
        data = f.read(target)
        if b"\n" not in data:
            # Bound overflow work for live, unterminated JSONL records while
            # still allowing complete oversized records with a nearby newline.
            # Fragments with no newline in this bounded window are skipped.
            data += f.read(chunk_size)

    if not data:
        return [], start

    last_nl = data.rfind(b"\n")
    if last_nl < 0:
        read_cap = max(1, int(max_bytes)) + max(64 * 1024, min(max(1, int(max_bytes)), 1024 * 1024))
        if len(data) >= read_cap:
            return [], start + len(data)
        return [], start
    data = data[: last_nl + 1]
    new_off = start + last_nl + 1

    out: list[JsonlRecord] = []
    pos = start
    for raw_line in data.splitlines(keepends=True):
        end = pos + len(raw_line)
        line = raw_line.rstrip(b"\r\n")
        obj = _parse_jsonl_line(line)
        if obj is not None:
            out.append(JsonlRecord(start=pos, end=end, obj=obj))
        pos = end
    return out, new_off


def _iter_jsonl_objects_reverse(path: Path, *, block_bytes: int = 64 * 1024) -> Iterator[dict[str, Any]]:
    if block_bytes <= 0:
        raise ValueError("block_bytes must be positive")
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        offset = f.tell()
        carry = b""
        while offset > 0:
            read_size = min(block_bytes, offset)
            offset -= read_size
            f.seek(offset)
            chunk = f.read(read_size)
            data = chunk + carry
            parts = data.split(b"\n")
            if offset > 0:
                carry = parts[0]
                parts = parts[1:]
            else:
                carry = b""
            for raw_line in reversed(parts):
                line = raw_line.rstrip(b"\r")
                if not line:
                    continue
                obj = _parse_jsonl_line(line)
                if obj is not None:
                    yield obj
        if carry:
            line = carry.rstrip(b"\r")
            if line:
                obj = _parse_jsonl_line(line)
                if obj is not None:
                    yield obj


def _iter_jsonl_records_reverse(path: Path, *, before: int | None = None, block_bytes: int = 64 * 1024) -> Iterator[JsonlRecord]:
    if block_bytes <= 0:
        raise ValueError("block_bytes must be positive")
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        end = size if before is None else max(0, min(int(before), size))
        offset = end
        carry = b""
        drop_trailing_partial = False
        if end > 0:
            f.seek(end - 1)
            drop_trailing_partial = f.read(1) != b"\n"
        while offset > 0:
            read_size = min(block_bytes, offset)
            offset -= read_size
            f.seek(offset)
            chunk = f.read(read_size)
            data = chunk + carry
            parts = data.split(b"\n")
            if drop_trailing_partial and parts:
                parts = parts[:-1]
                drop_trailing_partial = False
            if offset > 0:
                leading = parts[0] if parts else b""
                carry = leading
                parts = parts[1:] if parts else []
                pos = offset + len(leading) + 1
            else:
                carry = b""
                pos = 0
            batch: list[JsonlRecord] = []
            for raw_line in parts:
                start = pos
                end_off = start + len(raw_line) + 1
                pos = end_off
                line = raw_line.rstrip(b"\r")
                if not line:
                    continue
                obj = _parse_jsonl_line(line)
                if obj is not None:
                    batch.append(JsonlRecord(start=start, end=end_off, obj=obj))
            for record in reversed(batch):
                yield record
