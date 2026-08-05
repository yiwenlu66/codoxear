"""Behavioral PDF.js pipeline coverage for the vendored viewer runtime."""

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PDF_MODULE = ROOT / "codoxear" / "static" / "vendor" / "pdf.mjs"


def test_vendored_pdfjs_parses_page_and_extracts_text_in_node() -> None:
    """The same ESM bundle loaded by the viewer can parse a real PDF document."""
    program = f"""
import * as pdfjs from {json.dumps(PDF_MODULE.as_uri())};

function onePagePdf() {{
  const objects = [
    "<< /Type /Catalog /Pages 2 0 R >>",
    "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
    "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
    "<< /Length 40 >>\\nstream\\nBT /F1 12 Tf 72 72 Td (Hello PDF) Tj ET\\nendstream",
    "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
  ];
  let source = "%PDF-1.4\\n";
  const offsets = [0];
  for (let index = 0; index < objects.length; index += 1) {{
    offsets.push(source.length);
    source += `${{index + 1}} 0 obj\\n${{objects[index]}}\\nendobj\\n`;
  }}
  const startXref = source.length;
  source += `xref\\n0 ${{objects.length + 1}}\\n0000000000 65535 f \\n`;
  for (const offset of offsets.slice(1)) {{
    source += `${{String(offset).padStart(10, "0")}} 00000 n \\n`;
  }}
  source += `trailer\\n<< /Size ${{objects.length + 1}} /Root 1 0 R >>\\nstartxref\\n${{startXref}}\\n%%EOF\\n`;
  return new TextEncoder().encode(source);
}}

const document = await pdfjs.getDocument({{
  data: onePagePdf(),
  disableWorker: true,
  useSystemFonts: true,
}}).promise;
try {{
  const page = await document.getPage(1);
  const textContent = await page.getTextContent();
  process.stdout.write(JSON.stringify({{
    numPages: document.numPages,
    pageNumber: page.pageNumber,
    text: textContent.items.map((item) => item.str).join(""),
  }}));
}} finally {{
  await document.destroy();
}}
"""
    completed = subprocess.run(
        ["node", "--input-type=module"],
        input=program,
        check=True,
        capture_output=True,
        text=True,
        timeout=8,
    )

    result = json.loads(completed.stdout)
    assert result == {"numPages": 1, "pageNumber": 1, "text": "Hello PDF"}
