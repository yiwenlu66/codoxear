import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"
APP_CSS = ROOT / "codoxear" / "static" / "app.css"


def _extract_ctx_chip_attrs(source: str) -> str:
    match = re.search(r'const ctxChip = el\("button", \{(?P<attrs>.*?)\}\);', source, re.S)
    if not match:
        raise AssertionError("ctxChip is not constructed as a native button")
    return match.group("attrs")


def _extract_block_after(source: str, marker: str) -> str:
    start = source.index(marker)
    end = source.index("\n        };", start)
    return source[start:end]


def _extract_css_rule(source: str, selector: str) -> str:
    pattern = re.compile(rf"{re.escape(selector)}\s*\{{(?P<body>.*?)\n\s*\}}", re.S)
    match = pattern.search(source)
    if not match:
        raise AssertionError(f"missing CSS rule for {selector}")
    return match.group("body")


if __name__ == "__main__":
    unittest.main()
