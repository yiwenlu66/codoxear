from pathlib import Path

from codoxear import file_types
from codoxear import server


def test_file_type_helpers_classify_content_and_paths() -> None:
    assert file_types.image_content_type(Path("img.png"), b"\x89PNG\r\n\x1a\nrest") == "image/png"
    assert file_types.image_content_type(Path("img.jpg"), b"\xff\xd8\xffrest") == "image/jpeg"
    assert file_types.image_content_type(Path("vector.svg"), b"<svg/>") == "image/svg+xml; charset=utf-8"
    assert file_types.pdf_content_type(Path("doc.bin"), b"%PDF-1.7") == "application/pdf"
    assert file_types.video_content_type(Path("movie.unknown"), b"\x00\x00\x00\x18ftypmp42") == "video/mp4"
    assert file_types.video_content_type(Path("clip.webm"), b"") == "video/webm"
    assert file_types.file_kind(Path("note.txt"), b"hello") == ("text", None)


def test_server_file_kind_facade_uses_file_types_runtime_function() -> None:
    assert server._file_kind(Path("img.png"), b"\x89PNG\r\n\x1a\nrest") == ("image", "image/png")
