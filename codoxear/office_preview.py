"""Lazy PDF derivatives of Office documents; original files are never rewritten.

Calc exports the used area of every visible worksheet after a disposable XLSX
copy has had its sheet visibility, print ranges, and forced scaling normalized.
We deliberately do not use SinglePageSheets: it creates unbounded page geometry.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import signal
import stat
import subprocess
import tempfile
import threading
import time
import zipfile
from pathlib import Path
from typing import Any
from xml.dom import minidom
from xml.parsers.expat import ExpatError

from .file_text import open_regular_file_no_symlink, stat_path_no_symlink
from .file_types import OFFICE_CONTENT_TYPES
from .git_ops import path_json_text
from .system_dependencies import resolve_executable

# Bump whenever rendering/normalization/profile settings change. This is part of
# the cache identity, so a deploy never reuses derivatives with old semantics.
OFFICE_RENDERER_VERSION = "lo-pdf-v2-all-sheets-normal-pagination-keep-titles"
OFFICE_PREVIEW_TIMEOUT_SECONDS = 120
OFFICE_PREVIEW_MAX_CONCURRENT = 2
OFFICE_PREVIEW_CACHE_MAX_FILES = 128
OFFICE_PREVIEW_CACHE_MAX_BYTES = 1024 * 1024 * 1024
OFFICE_PREVIEW_FAILURE_TTL_SECONDS = 15
OFFICE_PREVIEW_FAILURE_MAX_ENTRIES = 128
_XLSX_MAX_EXPANDED_BYTES = 512 * 1024 * 1024
_XLSX_MAX_XML_BYTES = 32 * 1024 * 1024
_XLSX_MAX_MEMBERS = 10000
_CONVERSIONS = threading.BoundedSemaphore(OFFICE_PREVIEW_MAX_CONCURRENT)
_CACHE_GUARD = threading.Lock()
_LOCKS_GUARD = threading.Lock()
_FAILURES: dict[Path, tuple[float, str]] = {}


class _PreviewLock:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.refs = 0


_LOCKS: dict[Path, _PreviewLock] = {}


def office_response_payload(
    *, path_obj: Path, size: int, content_type: str | None,
    source_url: str, preview_url: str, rel: str | None = None,
) -> dict[str, Any]:
    source_type = content_type or OFFICE_CONTENT_TYPES.get(path_obj.suffix.lower(), "application/octet-stream")
    payload: dict[str, Any] = {
        "ok": True, "kind": "office", "path": path_json_text(path_obj),
        "size": int(size), "content_type": source_type,
        "source_name": path_json_text(path_obj.name), "source_content_type": source_type,
        "source_url": source_url, "preview_url": preview_url,
        "preview_content_type": "application/pdf",
    }
    if rel is not None:
        payload["rel"] = rel
    return payload


def _source_stat(path: Path) -> os.stat_result:
    st = stat_path_no_symlink(path)
    if stat.S_ISLNK(st.st_mode):
        raise ValueError("symlink file not supported")
    if not stat.S_ISREG(st.st_mode):
        raise ValueError("path is not a file")
    return st


def _identity(st: os.stat_result) -> tuple[int, int, int, int, int]:
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns


def _output_path(path: Path, st: os.stat_result, preview_dir: Path) -> Path:
    key = f"{path.resolve()}\0{_identity(st)}\0{OFFICE_RENDERER_VERSION}"
    return preview_dir / (hashlib.sha256(key.encode("utf-8", errors="surrogateescape")).hexdigest() + ".pdf")


def office_preview_path(path: Path, *, preview_dir: Path) -> Path:
    return _output_path(path, _source_stat(path), preview_dir)


def _ready(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            return fh.read(5) == b"%PDF-" and os.fstat(fh.fileno()).st_size > 5
    except OSError:
        return False


def _write_profile(profile: Path) -> None:
    user = profile / "user"
    user.mkdir(parents=True)
    # These are application configuration properties, not UNO load arguments.
    # Calc and Writer have DIFFERENT enum values for 'never update links'.
    # Fresh profile => no trusted locations, user extensions, or shared lock.
    settings = {
        "/org.openoffice.Office.Common/Security/Scripting": {
            "MacroSecurityLevel": "3", "DisableMacrosExecution": "true",
            "BlockUntrustedRefererLinks": "true",
        },
        "/org.openoffice.Office.Calc/Content/Update": {"Link": "1"},
        "/org.openoffice.Office.Writer/Content/Update": {"Link": "2", "Field": "false", "Chart": "false"},
    }
    items = "".join(
        f'<item oor:path="{path}">' + "".join(
            f'<prop oor:name="{key}" oor:op="fuse"><value>{value}</value></prop>'
            for key, value in props.items()
        ) + "</item>" for path, props in settings.items()
    )
    (user / "registrymodifications.xcu").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<oor:items xmlns:oor="http://openoffice.org/2001/registry">' + items + '</oor:items>',
        encoding="utf-8",
    )


def _normalize_workbook_xml(raw: bytes, *, workbook: bool) -> bytes:
    # DOM retains namespace declarations used only in mc:Ignorable values;
    # ElementTree's automatic prefix rewriting can leave those undeclared.
    with minidom.parseString(raw) as doc:
        if workbook:
            for sheet in doc.getElementsByTagNameNS("*", "sheet"):
                sheet.setAttribute("state", "visible")
            for name in list(doc.getElementsByTagNameNS("*", "definedName")):
                if name.getAttribute("name") == "_xlnm.Print_Area":
                    name.parentNode.removeChild(name)
            for props in doc.getElementsByTagNameNS("*", "workbookPr"):
                props.setAttribute("updateLinks", "never")
        else:
            for props in doc.getElementsByTagNameNS("*", "pageSetUpPr"):
                props.setAttribute("fitToPage", "0")
            for setup in doc.getElementsByTagNameNS("*", "pageSetup"):
                for key in ("fitToWidth", "fitToHeight"):
                    if setup.hasAttribute(key):
                        setup.removeAttribute(key)
                setup.setAttribute("scale", "100")
        return doc.toxml(encoding="UTF-8")


def normalize_spreadsheet(source: Path, destination: Path) -> None:
    """Rewrite only a disposable XLSX ZIP, never the original Office file."""
    try:
        with zipfile.ZipFile(source) as archive, zipfile.ZipFile(destination, "w") as out:
            members = archive.infolist()
            if len(members) > _XLSX_MAX_MEMBERS or sum(x.file_size for x in members) > _XLSX_MAX_EXPANDED_BYTES:
                raise RuntimeError("spreadsheet is too large to normalize for preview")
            if "xl/workbook.xml" not in archive.namelist():
                raise RuntimeError("spreadsheet has no XLSX workbook")
            for entry in members:
                name = entry.filename
                is_workbook = name == "xl/workbook.xml"
                is_worksheet = name.startswith("xl/worksheets/") and name.endswith(".xml") and "/_rels/" not in name
                if is_workbook or is_worksheet:
                    if entry.file_size > _XLSX_MAX_XML_BYTES:
                        raise RuntimeError("spreadsheet XML is too large to normalize for preview")
                    raw = _normalize_workbook_xml(archive.read(entry), workbook=is_workbook)
                    out.writestr(entry, raw)
                else:
                    # Stream media and other unchanged ZIP members, avoiding a
                    # second in-memory copy of potentially large embedded data.
                    with archive.open(entry) as src, out.open(entry, "w") as dst:
                        shutil.copyfileobj(src, dst)
    except (zipfile.BadZipFile, ExpatError, KeyError, NotImplementedError) as exc:
        raise RuntimeError(f"cannot read spreadsheet for preview: {exc}") from exc


def _kill_conversion_group(proc: subprocess.Popen[bytes]) -> None:
    # start_new_session makes this exact child's pid its private pgid. Never
    # inspect/match other soffice processes (including the user's desktop LO).
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def _run_conversion(
    executable: str, source: Path, *, output_dir: Path, profile: Path,
    target: str, deadline: float,
) -> Path:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise RuntimeError("LibreOffice preview conversion timed out")
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        executable, f"-env:UserInstallation={profile.as_uri()}",
        "--headless", "--nologo", "--nodefault", "--nofirststartwizard", "--norestore",
        "--convert-to", target, "--outdir", str(output_dir), str(source),
    ]
    # Keep output bounded in memory even if a broken converter is very noisy.
    with tempfile.TemporaryFile() as log:
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        except OSError as exc:
            raise RuntimeError(f"cannot start LibreOffice: {exc}") from exc
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired as exc:
            _kill_conversion_group(proc)
            raise RuntimeError("LibreOffice preview conversion timed out") from exc
        except BaseException:
            _kill_conversion_group(proc)
            raise
        expected = output_dir / (source.stem + "." + target.split(":", 1)[0])
        if proc.returncode != 0 or not expected.is_file() or expected.stat().st_size == 0:
            log.seek(0)
            detail = log.read(4096).decode("utf-8", errors="replace").strip()
            raise RuntimeError("LibreOffice could not convert this document (it may be encrypted or unsupported)"
                               + (f": {detail}" if detail else ""))
    return expected


def _generate_office_preview(path: Path, out: Path, *, preview_dir: Path, source_stat: os.stat_result) -> Path:
    executable = resolve_executable("soffice")
    if executable is None:
        raise RuntimeError("LibreOffice (soffice) is required for Office previews; run codoxear-doctor to check dependencies")
    deadline = time.monotonic() + OFFICE_PREVIEW_TIMEOUT_SECONDS
    if not _CONVERSIONS.acquire(timeout=max(0, deadline - time.monotonic())):
        raise RuntimeError("Office preview converters are busy; try again shortly")
    try:
        preview_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".office-", dir=preview_dir) as temp:
            work = Path(temp).resolve()
            profile = work / "profile"
            _write_profile(profile)
            source = work / ("source" + path.suffix.lower())
            with open_regular_file_no_symlink(path) as (fh, opened_stat), source.open("wb") as dest:
                if _identity(opened_stat) != _identity(source_stat):
                    raise RuntimeError("source changed before Office preview conversion; reopen the file")
                shutil.copyfileobj(fh, dest)
                if _identity(os.fstat(fh.fileno())) != _identity(source_stat):
                    raise RuntimeError("source changed while copying Office preview; reopen the file")
            suffix = path.suffix.lower()
            if suffix in {".xls", ".xlsx"}:
                if suffix == ".xls":
                    source = _run_conversion(executable, source, output_dir=work / "xlsx", profile=profile,
                                             target="xlsx:Calc MS Excel 2007 XML", deadline=deadline)
                normalized = work / "workbook.xlsx"
                normalize_spreadsheet(source, normalized)
                source = normalized
                pdf_filter = "calc_pdf_Export"
            elif suffix in {".doc", ".docx"}:
                pdf_filter = "writer_pdf_Export"
            else:
                pdf_filter = "impress_pdf_Export"
            pdf = _run_conversion(executable, source, output_dir=work / "pdf", profile=profile,
                                  target=f"pdf:{pdf_filter}", deadline=deadline)
            if not _ready(pdf):
                raise RuntimeError("LibreOffice produced an invalid or empty PDF preview")
            if pdf.stat().st_size > OFFICE_PREVIEW_CACHE_MAX_BYTES:
                raise RuntimeError("Office PDF preview exceeds the cache size limit")
            if _identity(_source_stat(path)) != _identity(source_stat):
                raise RuntimeError("source changed during Office preview conversion; reopen the file")
            os.replace(pdf, out)
        return out
    finally:
        _CONVERSIONS.release()


def ensure_office_preview(path: Path, *, preview_dir: Path) -> Path:
    if path.suffix.lower() not in OFFICE_CONTENT_TYPES:
        raise ValueError("file is not a supported Office document")
    source_stat = _source_stat(path)
    out = _output_path(path, source_stat, preview_dir)
    with _LOCKS_GUARD:
        entry = _LOCKS.setdefault(out, _PreviewLock())
        entry.refs += 1
    try:
        with entry.lock:
            if _ready(out):
                prune_office_preview_cache(preview_dir, keep=out)
                return out
            now = time.monotonic()
            with _LOCKS_GUARD:
                for key, (expiry, _message) in list(_FAILURES.items()):
                    if expiry <= now:
                        _FAILURES.pop(key, None)
                failure = _FAILURES.get(out)
            if failure:
                raise RuntimeError(f"recent Office preview conversion failed: {failure[1]}")
            try:
                _generate_office_preview(path, out, preview_dir=preview_dir, source_stat=source_stat)
            except RuntimeError as exc:
                with _LOCKS_GUARD:
                    _FAILURES[out] = (time.monotonic() + OFFICE_PREVIEW_FAILURE_TTL_SECONDS, str(exc))
                    while len(_FAILURES) > OFFICE_PREVIEW_FAILURE_MAX_ENTRIES:
                        _FAILURES.pop(next(iter(_FAILURES)))
                raise
            prune_office_preview_cache(preview_dir, keep=out)
            return out
    finally:
        with _LOCKS_GUARD:
            entry.refs -= 1
            if entry.refs == 0:
                _LOCKS.pop(out, None)


def clear_office_preview_cache(preview_dir: Path) -> int:
    """Delete only this renderer's SHA-256-named PDFs, never source files."""
    removed = 0
    with _CACHE_GUARD:
        for path in preview_dir.glob("*.pdf"):
            if len(path.stem) != 64 or any(c not in "0123456789abcdef" for c in path.stem):
                continue
            try:
                if not stat.S_ISREG(path.stat(follow_symlinks=False).st_mode):
                    continue
                path.unlink()
                removed += 1
            except FileNotFoundError:
                continue
    return removed


def prune_office_preview_cache(
    preview_dir: Path, *, keep: Path | None = None,
    max_files: int = OFFICE_PREVIEW_CACHE_MAX_FILES, max_bytes: int = OFFICE_PREVIEW_CACHE_MAX_BYTES,
) -> None:
    with _CACHE_GUARD:
        entries = []
        for path in preview_dir.glob("*.pdf"):
            try:
                st = path.stat()
            except OSError:
                continue
            if stat.S_ISREG(st.st_mode):
                entries.append((st.st_mtime_ns, path.name, st.st_size, path))
        count, total = len(entries), sum(x[2] for x in entries)
        for _mtime, _name, size, path in sorted(entries):
            if count <= max_files and total <= max_bytes:
                break
            if path == keep:
                continue
            try:
                path.unlink()
            except OSError:
                continue
            count -= 1
            total -= size
