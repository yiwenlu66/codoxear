from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from codoxear import office_preview as office
from codoxear.file_types import OFFICE_CONTENT_TYPES, file_kind, raw_file_content_type
from codoxear.file_view import read_client_file_view, read_text_or_image


@pytest.fixture(autouse=True)
def clear_failure_cache():
    with office._LOCKS_GUARD:
        office._FAILURES.clear()
    yield
    assert office._LOCKS == {}


@pytest.fixture
def converter(monkeypatch):
    calls = []
    guard = threading.Lock()
    activity = {"current": 0, "maximum": 0}

    def convert(executable, source, *, output_dir, profile, target, deadline):
        with guard:
            activity["current"] += 1
            activity["maximum"] = max(activity["maximum"], activity["current"])
        try:
            settings = ET.parse(profile / "user/registrymodifications.xcu")
            calls.append({"source": source, "profile": profile, "settings": settings, "target": target})
            assert source.name.startswith(("source.", "workbook."))
            assert deadline > time.monotonic()
            time.sleep(0.03)
            output_dir.mkdir()
            out = output_dir / (source.stem + "." + target.split(":")[0])
            out.write_bytes(b"%PDF-1.7\n" + source.read_bytes())
            return out
        finally:
            with guard:
                activity["current"] -= 1

    monkeypatch.setattr(office, "resolve_executable", lambda _name: "/fake/soffice")
    monkeypatch.setattr(office, "_run_conversion", convert)
    return calls, activity


@pytest.mark.parametrize("extension,mime", OFFICE_CONTENT_TYPES.items())
def test_office_identity_is_original_and_classification_is_lazy(tmp_path, extension, mime):
    source = tmp_path / ("文档" + extension.upper())
    source.write_bytes(b"PK\x03\x04binary\0" + b"x" * 4096)
    view = read_client_file_view(source)
    assert (view.kind, view.content_type, view.size, view.text, view.editable) == ("office", mime, source.stat().st_size, None, False)
    assert file_kind(source, b"binary") == ("office", mime)
    assert raw_file_content_type(source) == mime
    assert read_text_or_image(source) == ("office", source.stat().st_size, mime, None)
    assert list(tmp_path.iterdir()) == [source]


def test_cache_and_safe_snapshot_preserve_original(tmp_path, converter):
    source = tmp_path / "--headless strange 文.docx"
    source.write_bytes(b"source bytes")
    initial = source.stat()
    cache = tmp_path / "cache"
    first = office.ensure_office_preview(source, preview_dir=cache)
    assert office.ensure_office_preview(source, preview_dir=cache) == first
    calls, _activity = converter
    assert len(calls) == 1
    assert first.read_bytes() == b"%PDF-1.7\nsource bytes"
    assert source.read_bytes() == b"source bytes"
    assert source.stat().st_mtime_ns == initial.st_mtime_ns
    assert not calls[0]["profile"].exists()
    assert list(cache.iterdir()) == [first]
    props = {prop.attrib['{http://openoffice.org/2001/registry}name']: prop.findtext("value")
             for prop in calls[0]["settings"].iter("prop")}
    assert props["DisableMacrosExecution"] == "true"
    assert props["MacroSecurityLevel"] == "3"
    assert props["BlockUntrustedRefererLinks"] == "true"
    items = {item.attrib['{http://openoffice.org/2001/registry}path']: item for item in calls[0]["settings"].iter("item")}
    assert items["/org.openoffice.Office.Calc/Content/Update"].find("prop/value").text == "1"
    assert items["/org.openoffice.Office.Writer/Content/Update"].find("prop/value").text == "2"


def test_cache_key_invalidates_for_source_and_renderer_settings(tmp_path, converter, monkeypatch):
    source = tmp_path / "source.docx"
    source.write_bytes(b"first")
    cache = tmp_path / "cache"
    old = office.ensure_office_preview(source, preview_dir=cache)
    source.write_bytes(b"second version")
    newer = office.ensure_office_preview(source, preview_dir=cache)
    assert old != newer
    assert newer.read_bytes().endswith(b"second version")
    monkeypatch.setattr(office, "OFFICE_RENDERER_VERSION", "new-settings")
    assert office.office_preview_path(source, preview_dir=cache) != newer


def test_same_size_replacement_with_saved_mtime_invalidates_cache(tmp_path, converter):
    source = tmp_path / "source.docx"
    source.write_bytes(b"before")
    original_stat = source.stat()
    cache = tmp_path / "cache"
    before = office.ensure_office_preview(source, preview_dir=cache)
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"after!")
    os.utime(replacement, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    replacement.replace(source)
    after = office.ensure_office_preview(source, preview_dir=cache)
    assert after != before
    assert after.read_bytes().endswith(b"after!")


def test_clear_cache_deletes_only_owned_regular_pdfs(tmp_path):
    owned = tmp_path / ("a" * 64 + ".pdf")
    owned.write_bytes(b"cached PDF")
    source = tmp_path / "report.pdf"
    source.write_bytes(b"original PDF")
    link = tmp_path / ("b" * 64 + ".pdf")
    link.symlink_to(source)
    temporary = tmp_path / ".office-work"
    temporary.mkdir()
    assert office.clear_office_preview_cache(tmp_path) == 1
    assert not owned.exists()
    assert source.read_bytes() == b"original PDF"
    assert link.is_symlink()
    assert temporary.is_dir()


def test_same_source_parallel_requests_convert_once(tmp_path, converter):
    source = tmp_path / "source.doc"
    source.write_bytes(b"content")
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda _: office.ensure_office_preview(source, preview_dir=tmp_path / "cache"), range(6)))
    assert len(set(results)) == 1
    assert len(converter[0]) == 1


def test_distinct_sources_use_bounded_unique_profiles(tmp_path, converter):
    sources = [tmp_path / f"doc-{n}.docx" for n in range(6)]
    for source in sources:
        source.write_bytes(b"content")
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda src: office.ensure_office_preview(src, preview_dir=tmp_path / "cache"), sources))
    calls, activity = converter
    assert len(set(results)) == len(sources)
    assert len({call["profile"] for call in calls}) == len(sources)
    assert activity["maximum"] == office.OFFICE_PREVIEW_MAX_CONCURRENT


def test_changed_source_never_publishes_stale_derivative(tmp_path, converter, monkeypatch):
    source = tmp_path / "source.docx"
    source.write_bytes(b"before")
    original_convert = office._run_conversion

    def rewrite(*args, **kwargs):
        result = original_convert(*args, **kwargs)
        source.write_bytes(b"after")
        return result

    monkeypatch.setattr(office, "_run_conversion", rewrite)
    cache = tmp_path / "cache"
    with pytest.raises(RuntimeError, match="source changed"):
        office.ensure_office_preview(source, preview_dir=cache)
    assert list(cache.iterdir()) == []


def test_invalid_pdf_is_explicit_and_short_lived_failure_cached(tmp_path, converter, monkeypatch):
    source = tmp_path / "bad.docx"
    source.write_bytes(b"corrupt")
    convert = office._run_conversion

    def corrupt(*args, **kwargs):
        out = convert(*args, **kwargs)
        out.write_bytes(b"not a PDF")
        return out

    monkeypatch.setattr(office, "_run_conversion", corrupt)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="invalid or empty PDF"):
            office.ensure_office_preview(source, preview_dir=tmp_path / "cache")
    assert len(converter[0]) == 1
    assert list((tmp_path / "cache").iterdir()) == []


def test_missing_converter_is_actionable_and_never_touches_original(tmp_path, monkeypatch):
    source = tmp_path / "source.pptx"
    source.write_bytes(b"source")
    monkeypatch.setattr(office, "resolve_executable", lambda name: None)
    with pytest.raises(RuntimeError, match="LibreOffice.*codoxear-doctor"):
        office.ensure_office_preview(source, preview_dir=tmp_path / "cache")
    assert source.read_bytes() == b"source"
    assert not (tmp_path / "cache").exists()


def test_symlinks_and_wrong_kinds_are_rejected(tmp_path):
    source = tmp_path / "source.doc"
    source.write_bytes(b"content")
    link = tmp_path / "link.doc"
    link.symlink_to(source)
    with pytest.raises(ValueError, match="symlink"):
        office.ensure_office_preview(link, preview_dir=tmp_path / "cache")
    with pytest.raises(ValueError, match="supported Office"):
        office.ensure_office_preview(tmp_path / "no.txt", preview_dir=tmp_path / "cache")


def test_normalization_unhides_all_sheets_removes_print_limits_and_preserves_cells(tmp_path):
    source = tmp_path / "source.xlsx"
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    workbook = f'''<workbook xmlns="{ns}" xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" xmlns:xr="urn:revision" mc:Ignorable="xr"><workbookPr updateLinks="always"/><sheets><sheet name="A" state="visible"/><sheet name="B" state="hidden"/><sheet name="C" state="veryHidden"/></sheets><definedNames><definedName name="_xlnm.Print_Area">A!$A$1:$B$2</definedName><definedName name="_xlnm.Print_Titles">A!$1:$2</definedName><definedName name="KeepMe">42</definedName></definedNames></workbook>'''
    worksheet = f'''<worksheet xmlns="{ns}"><sheetPr><pageSetUpPr fitToPage="1"/></sheetPr><sheetData><row r="100"><c r="H100" t="inlineStr"><is><t>outside 中文</t></is></c></row></sheetData><pageSetup scale="10" fitToWidth="1" fitToHeight="1"/></worksheet>'''
    with zipfile.ZipFile(source, "w") as z:
        z.writestr("xl/workbook.xml", workbook)
        z.writestr("xl/worksheets/sheet1.xml", worksheet)
        z.writestr("xl/media/image.bin", b"media")
    original = source.read_bytes()
    dest = tmp_path / "normalized.xlsx"
    office.normalize_spreadsheet(source, dest)
    with zipfile.ZipFile(dest) as z:
        wb = ET.fromstring(z.read("xl/workbook.xml"))
        assert all(x.attrib["state"] == "visible" for x in wb.findall(f".//{{{ns}}}sheet"))
        assert [x.attrib["name"] for x in wb.findall(f".//{{{ns}}}definedName")] == ["_xlnm.Print_Titles", "KeepMe"]
        assert wb.find(f"{{{ns}}}workbookPr").attrib["updateLinks"] == "never"
        sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
        assert sheet.find(f".//{{{ns}}}pageSetUpPr").attrib["fitToPage"] == "0"
        assert sheet.find(f"{{{ns}}}pageSetup").attrib == {"scale": "100"}
        assert sheet.find(f".//{{{ns}}}t").text == "outside 中文"
        assert z.read("xl/media/image.bin") == b"media"
    assert source.read_bytes() == original


def test_corrupt_spreadsheet_is_readable_failure(tmp_path):
    source = tmp_path / "bad.xlsx"
    source.write_bytes(b"not a zip")
    with pytest.raises(RuntimeError, match="cannot read spreadsheet"):
        office.normalize_spreadsheet(source, tmp_path / "normalized.xlsx")


def test_cache_pruning_preserves_current_pdf_and_ignores_temporary_dirs(tmp_path):
    files = [tmp_path / f"{n}.pdf" for n in range(4)]
    for n, file in enumerate(files):
        file.write_bytes(b"x" * 10)
        os.utime(file, (n + 1, n + 1))
    temp = tmp_path / ".office-work"
    temp.mkdir()
    office.prune_office_preview_cache(tmp_path, keep=files[0], max_files=2, max_bytes=20)
    assert {x.name for x in tmp_path.glob("*.pdf")} == {"0.pdf", "3.pdf"}
    assert temp.is_dir()


def test_timeout_kills_only_spawned_process_group_including_descendant(tmp_path):
    marker = tmp_path / "escaped"
    script = tmp_path / "soffice"
    child_code = f"import time; from pathlib import Path; time.sleep(.5); Path({str(marker)!r}).write_text('escaped')"
    script.write_text(f"#!{sys.executable}\nimport subprocess,sys,time\nsubprocess.Popen([sys.executable,'-c',{child_code!r}])\ntime.sleep(30)\n")
    script.chmod(0o755)
    source = tmp_path / "source.docx"
    source.write_bytes(b"content")
    profile = tmp_path / "profile"
    office._write_profile(profile)
    with pytest.raises(RuntimeError, match="timed out"):
        office._run_conversion(str(script), source, output_dir=tmp_path / "out", profile=profile,
                               target="pdf:writer_pdf_Export", deadline=time.monotonic() + .15)
    time.sleep(.6)
    assert not marker.exists()


def test_success_exit_without_output_is_failure(tmp_path):
    script = tmp_path / "soffice"
    script.write_text(f"#!{sys.executable}\nprint('password required')\n")
    script.chmod(0o755)
    source = tmp_path / "encrypted.docx"
    source.touch()
    with pytest.raises(RuntimeError, match="encrypted.*password required"):
        office._run_conversion(str(script), source, output_dir=tmp_path / "out", profile=tmp_path,
                               target="pdf:writer_pdf_Export", deadline=time.monotonic() + 5)
