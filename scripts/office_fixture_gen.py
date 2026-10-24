#!/usr/bin/env python3
"""Generate the Office behavioral-verification fixture inside a container.

This tool is stdlib-only for authoring: .docx/.xlsx/.pptx are written as ZIP
packages with hand-built XML so the fixture needs no host libraries. Legacy
.doc/.xls/.ppt files are produced by LibreOffice itself (``soffice
--convert-to``) so the binary formats are guaranteed real. An encrypted .xlsx
fixture is produced with the image-only test dependency msoffcrypto-tool when
available. Nothing here runs against the host; the verify harness copies this
script into the container and executes it there.

Fixture properties that the backend preview must honor:
  * data.xlsx sheet "HiddenSheet" is hidden (state="hidden").
  * data.xlsx sheet "PrintCut" has a saved print area A1:B6 while its marker
    lives at H20, outside that range.
  * data.xlsx sheet "Main" forces fit-to-1x1-page scaling over 30x160 cells,
    so a raw conversion is one unreadable micro-scaled page; a correct preview
    resets the scaling and paginates normally across several pages.
  * every file carries unique ASCII markers plus CJK marker text.

The generator also records a *baseline*: a raw ``soffice --convert-to pdf``
of each OOXML source, with per-marker presence. The baseline proves the raw
converter alone does NOT surface the hidden-sheet / outside-print-range
markers, so their presence in the product preview is attributable to the
backend, not to LibreOffice defaults.

Writes manifest.json into the output directory describing every artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

# --- markers -------------------------------------------------------------
# Distinctive tokens: no whitespace inside the ASCII token so PDF text
# extraction cannot split them, and a CJK phrase per file for font coverage.
MARKERS = {
    "docx": ("DOCXMARK7F41", "文档标记词七"),
    "xlsx_main": ("XLSXMAIN7F41", "主表标记词"),
    "xlsx_tail": ("XLSXTAIL7F41", "尾页标记词"),
    "xlsx_hidden": ("XLSXHIDDEN7F41", "隐藏表标记词"),
    "xlsx_print_in": ("XLSXPRINTIN7F41", "打印范围内标记词"),
    "xlsx_print_out": ("XLSXPRINTCUT7F41", "打印范围外标记词"),
    "pptx": ("PPTXMARK7F41", "幻灯片标记词"),
    "xlsx_veryhidden": ("XLSXVERYHID7F41", "极隐藏标记词"),
}
LEGACY_NOTE = "legacy 二进制格式"

ZIP_DATE = (2024, 1, 1, 0, 0, 0)


def _writestr(zf: zipfile.ZipFile, name: str, data: str) -> None:
    info = zipfile.ZipInfo(name, date_time=ZIP_DATE)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    zf.writestr(info, data.encode("utf-8"))


# --- DOCX ----------------------------------------------------------------
def build_docx(path: Path) -> None:
    ascii_mark, cjk_mark = MARKERS["docx"]
    body = "".join(
        f"<w:p><w:r><w:t xml:space=\"preserve\">{escape(text)}</w:t></w:r></w:p>"
        for text in (
            f"{ascii_mark} {cjk_mark} first paragraph.",
            "Second paragraph: preview must render ordinary prose.",
            f"第三段：中文内容需要 Noto CJK 字体覆盖。{cjk_mark}",
        )
    )
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}"
        '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/></w:sectPr>'
        "</w:body></w:document>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
        "</Relationships>"
    )
    with zipfile.ZipFile(path, "w") as zf:
        _writestr(zf, "[Content_Types].xml", content_types)
        _writestr(zf, "_rels/.rels", rels)
        _writestr(zf, "word/document.xml", document)


# --- XLSX ----------------------------------------------------------------
def _col_name(index: int) -> str:  # 1-based
    name = ""
    while index:
        index, rem = divmod(index - 1, 26)
        name = chr(65 + rem) + name
    return name


def _inline_cell(ref: str, text: str) -> str:
    return f'<c r="{ref}" t="inlineStr"><is><t xml:space="preserve">{escape(text)}</t></is></c>'


def _number_cell(ref: str, value: int) -> str:
    return f'<c r="{ref}"><v>{value}</v></c>'


def _sheet_xml(*, rows: list[str], fit_to_one_page: bool = False) -> str:
    sheet_pr = '<sheetPr><pageSetUpPr fitToPage="1"/></sheetPr>' if fit_to_one_page else ""
    page_setup = (
        '<pageSetup paperSize="9" orientation="portrait" fitToWidth="1" fitToHeight="1"/>'
        if fit_to_one_page
        else '<pageSetup paperSize="9" orientation="portrait"/>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"{sheet_pr}"
        f'<sheetData>{"".join(rows)}</sheetData>'
        '<pageMargins left="0.7" right="0.7" top="0.75" bottom="0.75" header="0.3" footer="0.3"/>'
        f"{page_setup}"
        "</worksheet>"
    )


def _row(index: int, cells: list[str]) -> str:
    return f'<row r="{index}">{"".join(cells)}</row>'


def build_xlsx(path: Path, *, include_print_area: bool = True) -> None:
    main_mark, main_cjk = MARKERS["xlsx_main"]
    tail_mark, tail_cjk = MARKERS["xlsx_tail"]
    hidden_mark, hidden_cjk = MARKERS["xlsx_hidden"]
    print_in_mark, print_in_cjk = MARKERS["xlsx_print_in"]
    print_out_mark, print_out_cjk = MARKERS["xlsx_print_out"]
    veryhidden_mark, veryhidden_cjk = MARKERS["xlsx_veryhidden"]

    # Main: marker rows first (empty right neighbors so cell text is never
    # clipped), then 30 columns x 160 rows of data, forced fit-to-1x1 scaling.
    wide_cols, tall_rows = 30, 160
    header_row = 3
    last_row = header_row + tall_rows
    main_rows = [
        _row(1, [_inline_cell("A1", main_mark)]),
        _row(2, [_inline_cell("A2", main_cjk)]),
        _row(header_row, [_inline_cell("A3", "行列")]
             + [_inline_cell(f"{_col_name(c)}{header_row}", f"列标题{c}") for c in range(2, wide_cols + 1)]),
    ]
    for r in range(header_row + 1, last_row + 1):
        cells = []
        for c in range(1, wide_cols + 1):
            ref = f"{_col_name(c)}{r}"
            if c == 1:
                cells.append(_inline_cell(ref, f"行标记{r}填充"))
            else:
                cells.append(_number_cell(ref, r * 100 + c))
        main_rows.append(_row(r, cells))
    # Tail markers in dedicated rows with empty right neighbors: they prove the
    # last page rendered, and cell text cannot be clipped by a neighbor or by
    # the page's right edge.
    main_rows.append(_row(last_row + 1, [_inline_cell(f"A{last_row + 1}", tail_mark)]))
    main_rows.append(_row(last_row + 2, [_inline_cell(f"A{last_row + 2}", tail_cjk)]))
    sheet_main = _sheet_xml(rows=main_rows, fit_to_one_page=True)

    sheet_hidden = _sheet_xml(rows=[
        _row(1, [_inline_cell("A1", hidden_mark)]),
        _row(2, [_inline_cell("A2", hidden_cjk)]),
        _row(3, [_inline_cell("A3", "hidden sheet body 隐藏内容")]),
    ])

    sheet_printcut = _sheet_xml(rows=[
        _row(1, [_inline_cell("A1", print_in_mark)]),
        _row(2, [_inline_cell("A2", print_in_cjk)]),
        *[_row(r, [_inline_cell(f"A{r}", f"print row {r}")]) for r in range(3, 7)],
        _row(20, [_inline_cell("H20", print_out_mark)]),
        _row(21, [_inline_cell("H21", print_out_cjk)]),
    ])

    sheet_veryhidden = _sheet_xml(rows=[
        _row(1, [_inline_cell("A1", veryhidden_mark)]),
        _row(2, [_inline_cell("A2", veryhidden_cjk)]),
        _row(3, [_inline_cell("A3", "veryHidden sheet body 极隐藏内容")]),
    ])

    defined_names = ""
    if include_print_area:
        # Sheet order below is Main(0), HiddenSheet(1), PrintCut(2), VeryHidden(3).
        defined_names = (
            "<definedNames>"
            '<definedName name="_xlnm.Print_Area" localSheetId="2">\'PrintCut\'!$A$1:$B$6</definedName>'
            "</definedNames>"
        )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        "<sheets>"
        '<sheet name="Main" sheetId="1" r:id="rId1"/>'
        '<sheet name="HiddenSheet" sheetId="2" state="hidden" r:id="rId2"/>'
        '<sheet name="PrintCut" sheetId="3" r:id="rId3"/>'
        '<sheet name="VeryHiddenSheet" sheetId="4" state="veryHidden" r:id="rId4"/>'
        "</sheets>"
        f"{defined_names}"
        "</workbook>"
    )
    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/>'
        '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet3.xml"/>'
        '<Relationship Id="rId4" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet4.xml"/>'
        '<Relationship Id="rId5" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
        "</Relationships>"
    )
    styles = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
        '<fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills>'
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>'
        "</styleSheet>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '<Override PartName="/xl/worksheets/sheet2.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '<Override PartName="/xl/worksheets/sheet3.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '<Override PartName="/xl/worksheets/sheet4.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        "</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        "</Relationships>"
    )
    with zipfile.ZipFile(path, "w") as zf:
        _writestr(zf, "[Content_Types].xml", content_types)
        _writestr(zf, "_rels/.rels", rels)
        _writestr(zf, "xl/workbook.xml", workbook)
        _writestr(zf, "xl/_rels/workbook.xml.rels", workbook_rels)
        _writestr(zf, "xl/styles.xml", styles)
        _writestr(zf, "xl/worksheets/sheet1.xml", sheet_main)
        _writestr(zf, "xl/worksheets/sheet2.xml", sheet_hidden)
        _writestr(zf, "xl/worksheets/sheet3.xml", sheet_printcut)
        _writestr(zf, "xl/worksheets/sheet4.xml", sheet_veryhidden)


# --- PPTX ----------------------------------------------------------------
_P_NS = 'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"'

_PPTX_THEME = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="Fixture">'
    "<a:themeElements>"
    '<a:clrScheme name="Fixture">'
    '<a:dk1><a:sysClr val="windowText" lastClr="000000"/></a:dk1>'
    '<a:lt1><a:sysClr val="window" lastClr="FFFFFF"/></a:lt1>'
    '<a:dk2><a:srgbClr val="1F497D"/></a:dk2><a:lt2><a:srgbClr val="EEECE1"/></a:lt2>'
    '<a:accent1><a:srgbClr val="4F81BD"/></a:accent1><a:accent2><a:srgbClr val="C0504D"/></a:accent2>'
    '<a:accent3><a:srgbClr val="9BBB59"/></a:accent3><a:accent4><a:srgbClr val="8064A2"/></a:accent4>'
    '<a:accent5><a:srgbClr val="4BACC6"/></a:accent5><a:accent6><a:srgbClr val="F79646"/></a:accent6>'
    '<a:hlink><a:srgbClr val="0000FF"/></a:hlink><a:folHlink><a:srgbClr val="800080"/></a:folHlink>'
    "</a:clrScheme>"
    '<a:fontScheme name="Fixture">'
    '<a:majorFont><a:latin typeface="Calibri"/><a:ea typeface=""/><a:cs typeface=""/></a:majorFont>'
    '<a:minorFont><a:latin typeface="Calibri"/><a:ea typeface=""/><a:cs typeface=""/></a:minorFont>'
    "</a:fontScheme>"
    '<a:fmtScheme name="Fixture">'
    '<a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'
    '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'
    '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst>'
    '<a:lnStyleLst><a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln>'
    '<a:ln w="25400"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln>'
    '<a:ln w="38100"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst>'
    '<a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle>'
    '<a:effectStyle><a:effectLst/></a:effectStyle>'
    '<a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst>'
    '<a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'
    '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'
    '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst>'
    "</a:fmtScheme>"
    "</a:themeElements>"
    "</a:theme>"
)


def _pptx_slide(texts: list[str]) -> str:
    paras = "".join(
        f'<a:p><a:r><a:rPr lang="en-US" sz="2400"/><a:t>{escape(t)}</a:t></a:r></a:p>' for t in texts
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f"<p:sld {_P_NS}><p:cSld><p:spTree>"
        '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
        "<p:grpSpPr/>"
        '<p:sp><p:nvSpPr><p:cNvPr id="2" name="FixtureText"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>'
        "<p:spPr>"
        '<a:xfrm><a:off x="685800" y="685800"/><a:ext cx="7772400" cy="5486400"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
        "</p:spPr>"
        f"<p:txBody><a:bodyPr wrap=\"square\"/><a:lstStyle/>{paras}</p:txBody>"
        "</p:sp>"
        "</p:spTree></p:cSld><p:clrMapOvr><a:overrideClrMapping/></p:clrMapOvr></p:sld>"
    )


def build_pptx(path: Path) -> None:
    ascii_mark, cjk_mark = MARKERS["pptx"]
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>'
        '<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>'
        '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>'
        '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>'
        '<Override PartName="/ppt/slides/slide1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        '<Override PartName="/ppt/slides/slide2.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        "</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>'
        "</Relationships>"
    )
    presentation = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f"<p:presentation {_P_NS}>"
        '<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>'
        '<p:sldIdLst><p:sldId id="256" r:id="rId2"/><p:sldId id="257" r:id="rId3"/></p:sldIdLst>'
        '<p:sldSz cx="9144000" cy="6858000"/><p:notesSz cx="6858000" cy="9144000"/>'
        "</p:presentation>"
    )
    presentation_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide1.xml"/>'
        '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide2.xml"/>'
        "</Relationships>"
    )
    master = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f"<p:sldMaster {_P_NS}><p:cSld><p:spTree>"
        '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
        "<p:grpSpPr/></p:spTree></p:cSld>"
        '<p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>'
        '<p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>'
        "<p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles>"
        "</p:sldMaster>"
    )
    master_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>'
        "</Relationships>"
    )
    layout = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f"<p:sldLayout {_P_NS} type=\"blank\" preserve=\"1\"><p:cSld name=\"Blank\"><p:spTree>"
        '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
        "<p:grpSpPr/></p:spTree></p:cSld>"
        '<p:clrMapOvr><a:overrideClrMapping/></p:clrMapOvr>'
        "</p:sldLayout>"
    )
    layout_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>'
        "</Relationships>"
    )
    slide_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>'
        "</Relationships>"
    )
    with zipfile.ZipFile(path, "w") as zf:
        _writestr(zf, "[Content_Types].xml", content_types)
        _writestr(zf, "_rels/.rels", rels)
        _writestr(zf, "ppt/presentation.xml", presentation)
        _writestr(zf, "ppt/_rels/presentation.xml.rels", presentation_rels)
        _writestr(zf, "ppt/slideMasters/slideMaster1.xml", master)
        _writestr(zf, "ppt/slideMasters/_rels/slideMaster1.xml.rels", master_rels)
        _writestr(zf, "ppt/slideLayouts/slideLayout1.xml", layout)
        _writestr(zf, "ppt/slideLayouts/_rels/slideLayout1.xml.rels", layout_rels)
        _writestr(zf, "ppt/theme/theme1.xml", _PPTX_THEME)
        _writestr(zf, "ppt/slides/slide1.xml", _pptx_slide([
            f"{ascii_mark} {cjk_mark} slide one",
            "第一页：中文幻灯片内容。",
        ]))
        _writestr(zf, "ppt/slides/_rels/slide1.xml.rels", slide_rels)
        _writestr(zf, "ppt/slides/slide2.xml", _pptx_slide([
            f"second slide {ascii_mark}",
            f"第二页 {cjk_mark}。",
        ]))
        _writestr(zf, "ppt/slides/_rels/slide2.xml.rels", slide_rels)


# --- LibreOffice / checks --------------------------------------------------
def run(cmd: list[str], *, log: list[str]) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    log.append(f"$ {' '.join(cmd)}\nexit={proc.returncode}\n{proc.stdout}{proc.stderr}")
    return proc


def soffice_convert(soffice: str, src: Path, fmt: str, outdir: Path, log: list[str], profile: Path) -> Path:
    proc = run([
        soffice, "--headless", "--norestore", "--nolockcheck",
        f"-env:UserInstallation=file://{profile}",
        "--convert-to", fmt, "--outdir", str(outdir), str(src),
    ], log=log)
    out = outdir / (src.stem + "." + fmt)
    if proc.returncode != 0 or not out.exists() or out.stat().st_size == 0:
        raise SystemExit(f"soffice conversion failed: {src} -> {fmt}\n{proc.stdout}{proc.stderr}")
    return out


def pdf_text(soffice: str, src: Path, workdir: Path, log: list[str], profile: Path) -> tuple[str, int, list[str]]:
    """Convert to PDF and return (text, page_count, page_size_lines)."""
    pdf = soffice_convert(soffice, src, "pdf", workdir, log, profile)
    text_proc = run(["pdftotext", str(pdf), "-"], log=log)
    info_proc = run(["pdfinfo", "-f", "1", "-l", "1000", str(pdf)], log=log)
    pages = 0
    size_lines: list[str] = []
    for line in info_proc.stdout.splitlines():
        if line.startswith("Pages:"):
            pages = int(line.split(":", 1)[1].strip())
        if line.startswith("Page") and "size:" in line:
            size_lines.append(line.strip())
    return text_proc.stdout, pages, size_lines


def page_max_dimension_pts(size_line: str) -> float:
    """Parse a pdfinfo page-size line into its larger dimension in points."""
    dims = re.findall(r"(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*pts", size_line)
    if not dims:
        return 0.0
    width, height = float(dims[0][0]), float(dims[0][1])
    return max(width, height)


def xlsx_workbook_properties(xlsx: Path) -> dict:
    with zipfile.ZipFile(xlsx) as zf:
        workbook = zf.read("xl/workbook.xml").decode("utf-8")
    return {
        "has_hidden_sheet": 'state="hidden"' in workbook,
        "has_veryhidden_sheet": 'state="veryHidden"' in workbook,
        "has_print_area": "_xlnm.Print_Area" in workbook,
        "sheet_names_present": all(name in workbook for name in ("Main", "HiddenSheet", "PrintCut", "VeryHiddenSheet")),
    }


def xlsx_marker_text(xlsx: Path) -> str:
    chunks: list[str] = []
    with zipfile.ZipFile(xlsx) as zf:
        for name in zf.namelist():
            if (name.startswith("xl/worksheets/") and name.endswith(".xml")) or name == "xl/sharedStrings.xml":
                chunks.append(zf.read(name).decode("utf-8"))
    return "\n".join(chunks)


def encrypt_xlsx(src: Path, dst: Path, password: str, log: list[str]) -> bool:
    try:
        import msoffcrypto  # type: ignore
    except Exception:
        return False
    try:
        with src.open("rb") as fin:
            office = msoffcrypto.OfficeFile(fin)
            with dst.open("wb") as fout:
                office.encrypt(password, fout)
        return True
    except Exception as exc:  # pragma: no cover - environment dependent
        log.append(f"msoffcrypto encrypt failed: {exc!r}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--soffice", default=shutil.which("soffice") or "/usr/bin/soffice")
    ap.add_argument("--password", default="codoxear-verify", help="password for the encrypted fixture")
    args = ap.parse_args()

    outdir: Path = args.outdir
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True)
    log: list[str] = []
    profile = Path(tempfile.mkdtemp(prefix="lo-fixture-profile-"))
    baseline_dir = Path(tempfile.mkdtemp(prefix="lo-fixture-baseline-"))

    manifest: dict = {"markers": MARKERS, "files": {}, "baseline_pdf": {}, "notes": []}

    def record(path: Path, markers: list[str], **extra) -> None:
        data = path.read_bytes()
        manifest["files"][path.name] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "size": len(data),
            "markers": markers,
            **extra,
        }

    # OOXML sources.
    build_docx(outdir / "report.docx")
    record(outdir / "report.docx", list(MARKERS["docx"]), role="word-ooxml")

    build_xlsx(outdir / "data.xlsx")
    props = xlsx_workbook_properties(outdir / "data.xlsx")
    if not all(props.values()):
        raise SystemExit(f"generated data.xlsx lost required properties: {props}")
    record(
        outdir / "data.xlsx",
        [m for pair in (MARKERS[k] for k in ("xlsx_main", "xlsx_tail", "xlsx_hidden", "xlsx_veryhidden", "xlsx_print_in", "xlsx_print_out")) for m in pair],
        role="excel-ooxml", workbook_properties=props,
    )

    build_pptx(outdir / "deck.pptx")
    record(outdir / "deck.pptx", list(MARKERS["pptx"]), role="powerpoint-ooxml")

    # Corrupt fixture: a truncated ZIP cannot be a valid document.
    whole = (outdir / "report.docx").read_bytes()
    (outdir / "broken.docx").write_bytes(whole[: len(whole) // 3])
    record(outdir / "broken.docx", [], role="corrupt")

    # Encrypted fixture (test-only dependency, container image only).
    encrypted_ok = encrypt_xlsx(outdir / "data.xlsx", outdir / "secret.xlsx", args.password, log)
    if encrypted_ok:
        record(outdir / "secret.xlsx", [], role="encrypted", password=args.password)
    else:
        manifest["notes"].append("msoffcrypto-tool unavailable; encrypted fixture skipped")

    # Legacy binary formats via LibreOffice.
    doc = soffice_convert(args.soffice, outdir / "report.docx", "doc", outdir, log, profile)
    record(doc, list(MARKERS["docx"]), role="word-legacy")
    xls = soffice_convert(args.soffice, outdir / "data.xlsx", "xls", outdir, log, profile)
    record(xls, [m for pair in (MARKERS[k] for k in ("xlsx_main", "xlsx_tail", "xlsx_hidden", "xlsx_veryhidden", "xlsx_print_in", "xlsx_print_out")) for m in pair], role="excel-legacy")
    ppt = soffice_convert(args.soffice, outdir / "deck.pptx", "ppt", outdir, log, profile)
    record(ppt, list(MARKERS["pptx"]), role="powerpoint-legacy")

    # Round-trip self-check: the legacy XLS must still carry the hidden sheet
    # and the print area, or the fixture cannot prove the backend requirement.
    rt_dir = Path(tempfile.mkdtemp(prefix="lo-fixture-roundtrip-"))
    rt_xlsx = soffice_convert(args.soffice, xls, "xlsx", rt_dir, log, profile)
    rt_props = xlsx_workbook_properties(rt_xlsx)
    rt_text = xlsx_marker_text(rt_xlsx)
    rt_markers = {key: all(m in rt_text for m in MARKERS[key]) for key in
                  ("xlsx_main", "xlsx_tail", "xlsx_hidden", "xlsx_veryhidden", "xlsx_print_in", "xlsx_print_out")}
    manifest["xls_roundtrip"] = {"properties": rt_props, "markers": rt_markers}
    manifest["files"]["data.xls"]["roundtrip_properties"] = rt_props
    if not (rt_props["has_hidden_sheet"] and rt_props["has_print_area"]):
        raise SystemExit(f"legacy data.xls lost hidden sheet / print area: {rt_props}")
    if not rt_props["has_veryhidden_sheet"]:
        # LibreOffice downgrades veryHidden -> hidden through the XLS path; the
        # sheet content survives, so the .xls fixture still proves the
        # originally-veryHidden sheet is surfaced, just via the hidden rule.
        manifest["notes"].append(
            f"data.xls round-trip downgraded veryHidden to hidden: {rt_props}")
    if not rt_markers.get("xlsx_veryhidden"):
        raise SystemExit(f"legacy data.xls lost the veryHidden sheet content: {rt_markers}")

    # Baseline raw conversions: what plain `soffice --convert-to pdf` shows.
    for name in ("report.docx", "data.xlsx", "deck.pptx", "report.doc", "data.xls", "deck.ppt"):
        text, pages, sizes = pdf_text(args.soffice, outdir / name, baseline_dir, log, profile)
        marker_presence = {key: all(m in text for m in pair) for key, pair in MARKERS.items()}
        manifest["baseline_pdf"][name] = {
            "pages": pages,
            "markers": marker_presence,
            "page_sizes_first3": sizes[:3],
            "max_page_dimension_pts_first3": [page_max_dimension_pts(line) for line in sizes[:3]],
        }

    (outdir / "fixture-build.log").write_text("\n\n".join(log), encoding="utf-8")
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    shutil.rmtree(profile, ignore_errors=True)
    shutil.rmtree(baseline_dir, ignore_errors=True)
    shutil.rmtree(rt_dir, ignore_errors=True)
    print(json.dumps({"outdir": str(outdir), "files": sorted(manifest["files"]), "baseline_pages": {k: v["pages"] for k, v in manifest["baseline_pdf"].items()}}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
