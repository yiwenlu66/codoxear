from pathlib import Path
import tempfile

from codoxear import file_types
from codoxear import server
from codoxear.file_view import read_client_file_view


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


def test_client_file_view_classifies_html_documents_and_preserves_other_kinds() -> None:
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        files = {
            "page.html": (b"<h1>HTML</h1>", "html"),
            "page.htm": (b"<p>HTM</p>", "html"),
            "page.HTML": (b"<p>uppercase</p>", "html"),
            "notes.md": (b"# notes\n", "markdown"),
            "notes.txt": (b"plain\n", "text"),
            "icon.png": (b"\x89PNG\r\n\x1a\nbytes", "image"),
        }
        for name, (raw, expected_kind) in files.items():
            path = base / name
            path.write_bytes(raw)
            view = read_client_file_view(path)
            assert view.kind == expected_kind
            if expected_kind == "html":
                assert view.text == raw.decode("utf-8")
                assert view.editable is True
                assert view.version is not None


def test_raw_file_content_type_covers_documents_assets_and_text_fallback() -> None:
    assert file_types.raw_file_content_type(Path("page.html")) == "text/html; charset=utf-8"
    assert file_types.raw_file_content_type(Path("page.XHTML")) == "application/xhtml+xml; charset=utf-8"
    assert file_types.raw_file_content_type(Path("app.mjs")) == "text/javascript; charset=utf-8"
    assert file_types.raw_file_content_type(Path("font.woff2")) == "font/woff2"
    assert file_types.raw_file_content_type(Path("note.py")) == "text/plain; charset=utf-8"
    assert file_types.raw_file_content_type(Path("blob.bin")) == "application/octet-stream"


def test_audio_classification_and_ogg_disambiguation() -> None:
    for name, raw, expected in [
        ("song.mp3", b"", ("audio", "audio/mpeg")),
        ("song.m4a", b"", ("audio", "audio/mp4")),
        ("song.unknown", b"ID3abc", ("audio", "audio/mpeg")),
        ("song.unknown", b"\xff\xfb", ("audio", "audio/mpeg")),
        ("song.unknown", b"RIFFxxxxWAVE", ("audio", "audio/wav")),
        ("song.unknown", b"fLaCabc", ("audio", "audio/flac")),
        ("song.unknown", b"FORMxxxxAIFF", ("audio", "audio/aiff")),
        ("song.unknown", b"OggSxxxvorbis", ("audio", "audio/ogg")),
        ("movie.unknown", b"OggSxxx\x80theora", ("video", "video/ogg")),
        ("movie.ogg", b"OggSxxx\x80theora", ("video", "video/ogg")),
        ("movie.ogv", b"OggSxxxvorbis", ("video", "video/ogg")),
        ("movie.mp4", b"ID3", ("video", "video/mp4")),
    ]:
        assert file_types.file_kind(Path(name), raw) == expected


def test_ogg_audio_view_is_passthrough(tmp_path: Path) -> None:
    path = tmp_path / "voice.ogg"
    path.write_bytes(b"OggS\x00vorbis\x00")
    view = read_client_file_view(path)
    assert (view.kind, view.content_type, view.text) == ("audio", "audio/ogg", None)
