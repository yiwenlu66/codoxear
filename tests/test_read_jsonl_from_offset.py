import json

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


def test_read_jsonl_from_offset_advances_over_oversized_record(tmp_path):
    line0 = json.dumps({"prefix": 1}).encode("utf-8") + b"\n"
    obj1 = {"text": "x" * (2 * 1024 * 1024 + 256)}
    line1 = json.dumps(obj1).encode("utf-8") + b"\n"
    p = tmp_path / "rollout.jsonl"
    p.write_bytes(line0 + line1)

    objs, off = read_jsonl_from_offset(p, len(line0), max_bytes=2 * 1024 * 1024)

    assert objs == [obj1]
    assert off == len(line0) + len(line1)
def test_read_jsonl_from_offset_skips_partial_utf8_line_at_nonzero_offset(tmp_path):
    obj1 = {"text": "prefix 你好 world"}
    obj2 = {"a": 2}
    line1 = json.dumps(obj1, ensure_ascii=False).encode("utf-8") + b"\n"
    line2 = json.dumps(obj2).encode("utf-8") + b"\n"
    p = tmp_path / "rollout.jsonl"
    p.write_bytes(line1 + line2)

    ubytes = "你".encode("utf-8")
    offset = line1.index(ubytes) + 1

    objs, off = read_jsonl_from_offset(p, offset, max_bytes=4096)
    assert objs == [obj2]
    assert off == len(line1) + len(line2)
