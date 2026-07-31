from __future__ import annotations

from pathlib import Path

MARKDOWN_EXTENSIONS = frozenset({"md", "markdown", "mdown", "mkd"})
VIDEO_CONTENT_TYPES = {
    ".3gp": "video/3gpp",
    ".avi": "video/x-msvideo",
    ".flv": "video/x-flv",
    ".m4v": "video/mp4",
    ".mkv": "video/x-matroska",
    ".mov": "video/quicktime",
    ".mp4": "video/mp4",
    ".mpeg": "video/mpeg",
    ".mpg": "video/mpeg",
    ".ogv": "video/ogg",
    ".webm": "video/webm",
    ".wmv": "video/x-ms-wmv",
}
TEXTUAL_EXTENSIONS = frozenset(
    {
        "bash",
        "c",
        "cc",
        "cfg",
        "conf",
        "cpp",
        "css",
        "csv",
        "diff",
        "go",
        "h",
        "hpp",
        "htm",
        "html",
        "ini",
        "java",
        "js",
        "json",
        "jsonl",
        "log",
        "md",
        "markdown",
        "mdown",
        "mkd",
        "patch",
        "py",
        "rs",
        "scss",
        "sh",
        "sql",
        "svg",
        "toml",
        "ts",
        "tsx",
        "txt",
        "xml",
        "yaml",
        "yml",
        "zsh",
    }
)
TEXTUAL_FILENAMES = frozenset({"dockerfile", "license", "makefile", "readme"})


def sniff_image_ext(raw: bytes) -> str | None:
    if len(raw) >= 8 and raw[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if len(raw) >= 3 and raw[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return ".webp"
    return None


def image_content_type(path: Path, raw: bytes) -> str | None:
    if path.suffix.lower() == ".svg":
        return "image/svg+xml; charset=utf-8"
    ext = sniff_image_ext(raw)
    if ext == ".png":
        return "image/png"
    if ext == ".jpg":
        return "image/jpeg"
    if ext == ".webp":
        return "image/webp"
    return None


def pdf_content_type(path: Path, raw: bytes) -> str | None:
    if path.suffix.lower() == ".pdf" or raw.startswith(b"%PDF-"):
        return "application/pdf"
    return None


def video_content_type(path: Path, raw: bytes) -> str | None:
    ctype = VIDEO_CONTENT_TYPES.get(path.suffix.lower())
    if ctype is not None:
        return ctype
    if len(raw) >= 12 and raw[4:8] == b"ftyp":
        return "video/mp4"
    if raw.startswith(b"\x1a\x45\xdf\xa3"):
        return "video/webm"
    if raw.startswith(b"OggS"):
        return "video/ogg"
    return None


def file_kind(path: Path, raw: bytes) -> tuple[str, str | None]:
    ctype = image_content_type(path, raw)
    if ctype is not None:
        return "image", ctype
    ctype = pdf_content_type(path, raw)
    if ctype is not None:
        return "pdf", ctype
    ctype = video_content_type(path, raw)
    if ctype is not None:
        return "video", ctype
    return "text", None
