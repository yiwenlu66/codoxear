import json

from codoxear.rollout_log import _read_jsonl_records_from_offset
from codoxear.util import read_jsonl_from_offset


def test_read_jsonl_from_offset_does_not_parse_truncated_utf8_tail(tmp_path):
    line1 = json.dumps({"a": 1}).encode("utf-8") + b"\n"
    obj2 = {"text": "汉"}
    line2 = json.dumps(obj2, ensure_ascii=False).encode("utf-8") + b"\n"
    p = tmp_path / "rollout.jsonl"
    p.write_bytes(line1 + line2)

    ubytes = "汉".encode("utf-8")
    pos = line2.index(ubytes)
    # Read into the second line but cut inside a multibyte character so
    # json.loads(bytes) would raise UnicodeDecodeError if we tried to parse it.
    max_bytes = len(line1) + pos + 1

    objs, off = read_jsonl_from_offset(p, 0, max_bytes=max_bytes)
    assert objs == [{"a": 1}]
    assert off == len(line1)

    objs2, off2 = read_jsonl_from_offset(p, off, max_bytes=4096)
    assert objs2 == [obj2]
    assert off2 == len(line1) + len(line2)


def test_read_jsonl_from_offset_skips_non_object_json_rows(tmp_path):
    line0 = json.dumps({"ready": 1}).encode("utf-8") + b"\n"
    p = tmp_path / "rollout.jsonl"
    p.write_bytes(line0 + b"[]\nnull\n1\n\"text\"\n" + json.dumps({"after": 2}).encode("utf-8") + b"\n")

    objs, off = read_jsonl_from_offset(p, 0, max_bytes=4096)

    assert objs == [{"ready": 1}, {"after": 2}]
    assert off == p.stat().st_size


def test_read_jsonl_from_offset_ignores_partial_appended_json_line(tmp_path):
    line0 = json.dumps({"ready": 1}).encode("utf-8") + b"\n"
    partial = b'{"partial": '
    p = tmp_path / "rollout.jsonl"
    p.write_bytes(line0 + partial)

    objs, off = read_jsonl_from_offset(p, 0, max_bytes=4096)

    assert objs == [{"ready": 1}]
    assert off == len(line0)

    line1 = json.dumps({"partial": 2}).encode("utf-8") + b"\n"
    p.write_bytes(line0 + line1)
    objs2, off2 = read_jsonl_from_offset(p, off, max_bytes=4096)
    assert objs2 == [{"partial": 2}]
    assert off2 == len(line0) + len(line1)


def test_read_jsonl_from_offset_skips_oversized_unterminated_fragment(tmp_path):
    fragment = b'{"partial":"' + (b"x" * 70000)
    next_line = json.dumps({"after": 1}).encode("utf-8") + b"\n"
    p = tmp_path / "rollout.jsonl"
    p.write_bytes(fragment)

    objs, off = read_jsonl_from_offset(p, 0, max_bytes=1024)

    assert objs == []
    assert off == 1024 + 64 * 1024

    p.write_bytes(fragment + b"\n" + next_line)
    objs2, off2 = read_jsonl_from_offset(p, off, max_bytes=4096)
    assert objs2 == [{"after": 1}]
    assert off2 == len(fragment) + 1 + len(next_line)


def test_read_jsonl_from_offset_recovers_when_skip_lands_inside_utf8(tmp_path):
    target = 1024
    overflow = 64 * 1024
    cap = target + overflow
    prefix = b'{"partial":"'
    char = "汉".encode("utf-8")
    pad_len = (cap - len(prefix) - 1) % len(char)
    fragment = prefix + (b"a" * pad_len) + (char * 30000)
    next_line = json.dumps({"after": 1}).encode("utf-8") + b"\n"
    p = tmp_path / "rollout.jsonl"
    p.write_bytes(fragment)

    objs, off = read_jsonl_from_offset(p, 0, max_bytes=target)

    assert objs == []
    assert off == cap

    p.write_bytes(fragment + b"\n" + next_line)
    objs2, off2 = read_jsonl_from_offset(p, off, max_bytes=4096)
    assert objs2 == [{"after": 1}]
    assert off2 == len(fragment) + 1 + len(next_line)


def test_rollout_record_reader_skips_oversized_unterminated_fragment(tmp_path):
    fragment = b'{"partial":"' + (b"x" * 70000)
    next_line = json.dumps({"type": "event", "payload": {"after": 1}}).encode("utf-8") + b"\n"
    p = tmp_path / "rollout.jsonl"
    p.write_bytes(fragment)

    records, off = _read_jsonl_records_from_offset(p, 0, max_bytes=1024)

    assert records == []
    assert off == 1024 + 64 * 1024

    p.write_bytes(fragment + b"\n" + next_line)
    records2, off2 = _read_jsonl_records_from_offset(p, off, max_bytes=4096)
    assert [record.obj for record in records2] == [{"type": "event", "payload": {"after": 1}}]
    assert off2 == len(fragment) + 1 + len(next_line)


def test_read_jsonl_from_offset_advances_over_oversized_record(tmp_path):
    line0 = json.dumps({"prefix": 1}).encode("utf-8") + b"\n"
    obj1 = {"text": "x" * (2 * 1024 * 1024 + 256)}
    line1 = json.dumps(obj1).encode("utf-8") + b"\n"
    p = tmp_path / "rollout.jsonl"
    p.write_bytes(line0 + line1)

    objs, off = read_jsonl_from_offset(p, len(line0), max_bytes=2 * 1024 * 1024)

    assert objs == [obj1]
    assert off == len(line0) + len(line1)
