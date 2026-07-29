import re
import unittest
from pathlib import Path


APP_CSS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.css"

SHELL_TARGET_SELECTORS = [
    ".pill > .icon-btn",
    ".topActions .icon-btn",
    ".sidebar header .icon-btn",
    ".chatNavRail .icon-btn",
    ".agentBackendTab",
]


COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
RULE_RE = re.compile(r"(?P<selectors>[^{}]+)\{(?P<body>[^{}]*)\}", re.MULTILINE)


def media_block(css: str, marker: str) -> str:
    start = css.index(marker)
    open_brace = css.index("{", start)
    depth = 1
    pos = open_brace + 1
    while pos < len(css) and depth:
        if css[pos] == "{":
            depth += 1
        elif css[pos] == "}":
            depth -= 1
        pos += 1
    return css[open_brace + 1 : pos - 1]


def selector_bodies(css: str, selector: str) -> list[str]:
    css = COMMENT_RE.sub("", css)
    bodies: list[str] = []
    for match in RULE_RE.finditer(css):
        selectors = [part.strip() for part in match.group("selectors").split(",")]
        if selector in selectors:
            bodies.append(match.group("body"))
    return bodies


def last_selector_body(css: str, selector: str) -> str:
    bodies = selector_bodies(css, selector)
    if not bodies:
        raise AssertionError(f"missing CSS rule for {selector}")
    return bodies[-1]


if __name__ == "__main__":
    unittest.main()
