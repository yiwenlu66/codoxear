#!/usr/bin/env python3
"""Reject unresolved bare function calls in Codoxear's app shell.

This is deliberately a small deployment guard, not a JavaScript type checker. It
checks calls of the form ``name(...)`` in app.js because those are the calls
that become browser ``ReferenceError`` exceptions when a refactor leaves a
render closure pointing at an undeclared variable. Property calls are excluded;
their receiver owns their lookup semantics.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

IDENTIFIER = r"[A-Za-z_$][A-Za-z0-9_$]*"
CALL_PATTERN = re.compile(rf"(?<![.$\w])({IDENTIFIER})\s*\(")
FUNCTION_DECLARATION = re.compile(rf"\b(?:async\s+)?function\s+({IDENTIFIER})\s*\(")
CLASS_DECLARATION = re.compile(rf"\bclass\s+({IDENTIFIER})\b")
VARIABLE_DECLARATION = re.compile(rf"\b(?:const|let|var)\s+({IDENTIFIER})\b")
DESTRUCTURE_PATTERN = re.compile(rf"\b(?:const|let|var)\s*\{{\s*([^}}]+)\s*\}}\s*=")
FUNCTION_PARAMETERS = re.compile(r"\b(?:async\s+)?function(?:\s+[A-Za-z_$][A-Za-z0-9_$]*)?\s*\(([^)]*)\)")
ARROW_PARAMETERS = re.compile(r"\(([^)]*)\)\s*=>|\b([A-Za-z_$][A-Za-z0-9_$]*)\s*=>")
WINDOW_GLOBAL = re.compile(rf"\b(?:window|globalThis)\.({IDENTIFIER})\s*=")
SCRIPT_SRC = re.compile(r'<script\b[^>]*\bsrc=["\']([^"\']+)["\']', re.IGNORECASE)

KEYWORDS = frozenset(
    {
        "as",
        "async",
        "await",
        "break",
        "case",
        "catch",
        "class",
        "const",
        "continue",
        "debugger",
        "default",
        "delete",
        "do",
        "else",
        "export",
        "extends",
        "finally",
        "for",
        "from",
        "function",
        "if",
        "import",
        "in",
        "instanceof",
        "let",
        "new",
        "of",
        "return",
        "static",
        "super",
        "switch",
        "this",
        "throw",
        "try",
        "typeof",
        "var",
        "void",
        "while",
        "with",
        "yield",
    }
)

BUILTINS = frozenset(
    {
        "AggregateError",
        "Array",
        "ArrayBuffer",
        "BigInt",
        "Blob",
        "Boolean",
        "BroadcastChannel",
        "CSS",
        "DataView",
        "Date",
        "DOMException",
        "DOMParser",
        "Error",
        "EvalError",
        "Event",
        "EventSource",
        "File",
        "FileReader",
        "FormData",
        "Function",
        "Headers",
        "Image",
        "Int8Array",
        "Int16Array",
        "Int32Array",
        "Map",
        "MutationObserver",
        "Number",
        "Object",
        "Promise",
        "Proxy",
        "RangeError",
        "ReferenceError",
        "RegExp",
        "Request",
        "ResizeObserver",
        "Response",
        "Set",
        "String",
        "Symbol",
        "SyntaxError",
        "TextDecoder",
        "TextEncoder",
        "TypeError",
        "URIError",
        "URL",
        "URLSearchParams",
        "Uint8Array",
        "Uint8ClampedArray",
        "Uint16Array",
        "Uint32Array",
        "WeakMap",
        "WeakSet",
        "XMLSerializer",
        "alert",
        "atob",
        "btoa",
        "cancelAnimationFrame",
        "cancelIdleCallback",
        "clearInterval",
        "clearTimeout",
        "confirm",
        "decodeURI",
        "decodeURIComponent",
        "encodeURI",
        "encodeURIComponent",
        "escape",
        "eval",
        "fetch",
        "getComputedStyle",
        "isFinite",
        "isNaN",
        "matchMedia",
        "parseFloat",
        "parseInt",
        "queueMicrotask",
        "requestAnimationFrame",
        "requestIdleCallback",
        "setInterval",
        "setTimeout",
        "structuredClone",
        "unescape",
    }
)


def _blank(character: str) -> str:
    return "\n" if character == "\n" else " "


def mask_non_code(source: str) -> str:
    """Replace comments and string literals with spaces while retaining code.

    Template interpolation is code, so it is recursively retained. Keeping line
    breaks makes diagnostics point to the source line that a reviewer sees.
    """

    output: list[str] = []
    length = len(source)

    def consume_quoted(index: int, quote: str) -> int:
        output.append(" ")
        index += 1
        while index < length:
            character = source[index]
            output.append(_blank(character))
            index += 1
            if character == "\\" and index < length:
                output.append(_blank(source[index]))
                index += 1
            elif character == quote:
                break
        return index

    def consume_regex(index: int) -> int:
        """Mask a regular-expression literal, including its character classes."""
        output.append(" ")
        index += 1
        in_character_class = False
        while index < length:
            character = source[index]
            output.append(_blank(character))
            index += 1
            if character == "\\" and index < length:
                output.append(_blank(source[index]))
                index += 1
            elif character == "[":
                in_character_class = True
            elif character == "]":
                in_character_class = False
            elif character == "/" and not in_character_class:
                while index < length and source[index].isalpha():
                    output.append(" ")
                    index += 1
                break
        return index

    def regex_can_start() -> bool:
        prior = "".join(output).rstrip()
        if not prior:
            return True
        if prior[-1] in "([{=,:;!&|?":
            return True
        return bool(re.search(r"\b(?:case|delete|return|throw|typeof|void|yield)$", prior))

    def consume_template(index: int) -> int:
        output.append(" ")
        index += 1
        while index < length:
            character = source[index]
            if character == "\\":
                output.append(" ")
                index += 1
                if index < length:
                    output.append(_blank(source[index]))
                    index += 1
            elif character == "`":
                output.append(" ")
                return index + 1
            elif character == "$" and index + 1 < length and source[index + 1] == "{":
                output.extend("${")
                index = consume_code(index + 2, template_brace_depth=0)
            else:
                output.append(_blank(character))
                index += 1
        return index

    def consume_code(index: int, template_brace_depth: int | None = None) -> int:
        while index < length:
            character = source[index]
            next_character = source[index + 1] if index + 1 < length else ""
            if character == "/" and next_character == "/":
                output.extend("  ")
                index += 2
                while index < length and source[index] != "\n":
                    output.append(" ")
                    index += 1
            elif character == "/" and next_character == "*":
                output.extend("  ")
                index += 2
                while index < length:
                    block_character = source[index]
                    output.append(_blank(block_character))
                    index += 1
                    if block_character == "*" and index < length and source[index] == "/":
                        output.append(" ")
                        index += 1
                        break
            elif character == "/" and regex_can_start():
                index = consume_regex(index)
            elif character in {"'", '"'}:
                index = consume_quoted(index, character)
            elif character == "`":
                index = consume_template(index)
            elif template_brace_depth is not None and character == "{":
                output.append(character)
                template_brace_depth += 1
                index += 1
            elif template_brace_depth is not None and character == "}":
                output.append(character)
                index += 1
                if template_brace_depth == 0:
                    return index
                template_brace_depth -= 1
            else:
                output.append(character)
                index += 1
        return index

    consume_code(0)
    return "".join(output)


def _names_in_parameters(parameter_text: str) -> set[str]:
    return set(re.findall(IDENTIFIER, parameter_text))


def app_defined_names(app_source: str) -> set[str]:
    code = mask_non_code(app_source)
    names = set(FUNCTION_DECLARATION.findall(code))
    names.update(CLASS_DECLARATION.findall(code))
    names.update(VARIABLE_DECLARATION.findall(code))
    for match in FUNCTION_PARAMETERS.finditer(code):
        names.update(_names_in_parameters(match.group(1)))
    for match in ARROW_PARAMETERS.finditer(code):
        names.update(_names_in_parameters(match.group(1) or match.group(2) or ""))
    return names


def referenced_module_globals(static_dir: Path, index_path: Path) -> set[str]:
    globals_: set[str] = set()
    for source_ref in SCRIPT_SRC.findall(index_path.read_text()):
        script_path = static_dir / source_ref.split("?", 1)[0]
        if script_path.suffix != ".js" or not script_path.is_file():
            continue
        globals_.update(WINDOW_GLOBAL.findall(mask_non_code(script_path.read_text())))
    return globals_


def undefined_call_references(app_path: Path, static_dir: Path, index_path: Path) -> list[tuple[str, int]]:
    source = app_path.read_text()
    code = mask_non_code(source)
    defined = app_defined_names(source) | referenced_module_globals(static_dir, index_path) | BUILTINS | KEYWORDS
    undefined: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for match in CALL_PATTERN.finditer(code):
        name = match.group(1)
        if name in defined:
            continue
        occurrence = (name, code.count("\n", 0, match.start(1)) + 1)
        if occurrence not in seen:
            undefined.append(occurrence)
            seen.add(occurrence)
    return undefined


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("static_dir", type=Path, help="directory containing app.js and index.html")
    args = parser.parse_args(argv)
    static_dir = args.static_dir.resolve()
    app_path = static_dir / "app.js"
    index_path = static_dir / "index.html"
    if not app_path.is_file() or not index_path.is_file():
        parser.error(f"expected app.js and index.html in {static_dir}")

    undefined = undefined_call_references(app_path, static_dir, index_path)
    if not undefined:
        return 0
    for name, line in undefined:
        print(f"{app_path}:{line}: undefined function reference: {name}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
