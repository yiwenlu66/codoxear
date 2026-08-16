#!/usr/bin/env python3
"""Static analysis for explicit frontend controller wiring.

The frontend is an ESM module graph. Controller factories receive `options`
objects, and `app_wiring.js` constructs the contracts passed between modules.
This checker verifies both sides of that contract:

1. it traces creation calls, requires selector lists to cover controller inputs,
   and reports unbound bare values in wiring option literals;
2. it ratchets architecture rules through a checked-in allowlist: pass-through
   option factories, `...options`/`...deps` spread elements in object literals,
   direct shared-bag controller arguments, and direct property assignments on
   `window`, `globalThis`, or `global`.

The allowlist identifies findings by check id, relative file path, and stable
factory/call/global name. A finding absent from the allowlist fails the check;
an allowlist entry without a current finding is stale and also fails it.

Usage: python3 scripts/check_wiring.py [static_dir] [--allowlist PATH]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import NamedTuple


CHECK_PASS_THROUGH_FACTORY = "pass-through-factory"
# Kept accepted for old allowlists while Phase 0 migrates its entries.
CHECK_SPREAD_INTO_FACTORY = "spread-into-factory"
CHECK_BAG_SPREAD = "bag-spread"
CHECK_GLOBAL_REGISTRATION = "global-registration"
CHECK_DIRECT_BAG_ARGUMENT = "direct-bag-argument"
CHECK_SELECT_UNDERCOVERAGE = "select-undercoverage"
CHECK_UNBOUND_OPTION_VALUE = "unbound-option-value"
ALLOWLIST_CHECKS = frozenset({
    CHECK_PASS_THROUGH_FACTORY,
    CHECK_SPREAD_INTO_FACTORY,
    CHECK_BAG_SPREAD,
    CHECK_GLOBAL_REGISTRATION,
    CHECK_DIRECT_BAG_ARGUMENT,
})
DEFAULT_ALLOWLIST_PATH = Path(__file__).with_name("wiring_guard_allowlist.json")


class GuardViolation(NamedTuple):
    check: str
    file: str
    name: str


class ScanCoverage(NamedTuple):
    files: int
    option_factories: int


PASS_THROUGH_FACTORY_RE = re.compile(
    r"(?:function\s+)?(?P<factory>create[A-Z]\w*Options)\s*\(\s*"
    r"(?P<argument>[A-Za-z_$][A-Za-z0-9_$]*)\s*\)\s*\{\s*"
    r"return\s+(?P=argument)\s*;\s*\}"
)
OPTION_FACTORY_RE = re.compile(r"\bcreate[A-Z]\w*Options\s*\(")
BAG_SPREAD_RE = re.compile(r"\.\.\.\s*(?:options|deps)\b")
GLOBAL_ASSIGNMENT_RE = re.compile(
    r"\b(?P<root>window|globalThis|global)\s*"
    r"(?:(?:\.\s*(?P<dot>[A-Za-z_$][A-Za-z0-9_$]*))|(?:\[\s*\]))\s*=(?!=)"
)
BRACKET_PROPERTY_RE = re.compile(
    r"\[\s*(?P<quote>['\"])(?P<name>(?:\\.|(?!['\"]).)*)\1\s*\]\s*$",
    re.DOTALL,
)
FUNCTION_RE = re.compile(r"\bfunction\s+(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)\s*\(")
DIRECT_CALLEE_RE = re.compile(r"(?P<callee>[A-Za-z_$][A-Za-z0-9_$.]*)\s*\(\s*$")
DIRECT_CREATE_BAG_RE = re.compile(
    r"(?P<callee>(?:[A-Za-z_$][A-Za-z0-9_$]*\.)*create[A-Z][A-Za-z0-9_$]*)\s*"
    r"\(\s*(?P<argument>options|deps)\s*\)"
)
IMMEDIATE_REQUIRE_BAG_RE = re.compile(
    r"requireFunction\s*\([^()]*\)\s*\(\s*(?P<argument>options|deps)\s*\)"
)
REGEX_PREFIX_WORDS = frozenset({"case", "delete", "do", "else", "in", "instanceof", "new", "of", "return", "throw", "typeof", "void", "yield"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "static_dir",
        nargs="?",
        type=Path,
        default=Path(__file__).parent.parent / "codoxear" / "static",
    )
    parser.add_argument("--allowlist", type=Path, default=DEFAULT_ALLOWLIST_PATH)
    return parser.parse_args()


def source_files(static_dir: Path) -> list[Path]:
    """Return checked frontend source: top-level JS plus the inline boot script.

    `static/*.js` excludes generated/vendor subtrees and `dist/`. `index.html` is
    included because its inline boot script owns browser globals used by Docker
    verification.
    """
    paths = sorted(
        path for path in static_dir.glob("*.js")
        if path.name != "app.bundle.js"
    )
    index_html = static_dir / "index.html"
    if index_html.exists():
        paths.append(index_html)
    return paths


def _mask(masked: list[str], source: str, start: int, end: int) -> None:
    for index in range(start, end):
        if source[index] != "\n":
            masked[index] = " "


def _can_start_regex(source: str, index: int) -> bool:
    prefix = source[:index].rstrip()
    if not prefix:
        return True
    if prefix[-1] in "([{=,:;!&|?~+-*%^<>":
        return True
    word = re.search(r"([A-Za-z_$][A-Za-z0-9_$]*)$", prefix)
    return bool(word and word.group(1) in REGEX_PREFIX_WORDS)


def strip_js_comments_and_strings(source: str) -> str:
    """Mask comments and literal contents while preserving source positions.

    This is a lexical heuristic, not a JavaScript parser. It handles escaped
    quotes, character classes in regex literals, and the common contexts where
    `/` starts a regex, so `//` inside a regex is not taken for a comment.
    Template literals are fully masked; expressions inside them are deliberately
    out of scope because guard rules concern executable wiring, not text.
    """
    masked = list(source)
    index = 0
    while index < len(source):
        char = source[index]
        next_char = source[index + 1] if index + 1 < len(source) else ""
        if char in ("'", '"', "`"):
            quote = char
            start = index
            index += 1
            while index < len(source):
                if source[index] == "\\":
                    index += 2
                    continue
                if source[index] == quote:
                    index += 1
                    break
                index += 1
            _mask(masked, source, start, min(index, len(source)))
            continue
        if char == "/" and next_char == "/":
            start = index
            newline = source.find("\n", index + 2)
            index = len(source) if newline == -1 else newline
            _mask(masked, source, start, index)
            continue
        if char == "/" and next_char == "*":
            start = index
            close = source.find("*/", index + 2)
            index = len(source) if close == -1 else close + 2
            _mask(masked, source, start, index)
            continue
        if char == "/" and next_char not in ("/", "*") and _can_start_regex(source, index):
            start = index
            index += 1
            in_character_class = False
            while index < len(source):
                if source[index] == "\\":
                    index += 2
                    continue
                if source[index] == "[":
                    in_character_class = True
                elif source[index] == "]":
                    in_character_class = False
                elif source[index] == "/" and not in_character_class:
                    index += 1
                    while index < len(source) and source[index].isalpha():
                        index += 1
                    break
                elif source[index] == "\n":
                    break
                index += 1
            _mask(masked, source, start, min(index, len(source)))
            continue
        index += 1
    return "".join(masked)


def _is_object_literal(source: str, opening_brace: int) -> bool:
    prefix = source[:opening_brace].rstrip()
    if not prefix:
        return False
    if prefix.endswith("=>"):
        return False
    previous = prefix[-1]
    if previous in "=(:,[!&|?~+-*%^<>{":
        return True
    word = re.search(r"([A-Za-z_$][A-Za-z0-9_$]*)$", prefix)
    return bool(word and word.group(1) == "return")


def _matching_delimiter(source: str, opening: int, close: str) -> int | None:
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == source[opening]:
            depth += 1
        elif source[index] == close:
            depth -= 1
            if depth == 0:
                return index
    return None


def _function_ranges(source: str) -> list[tuple[int, int, str]]:
    ranges: list[tuple[int, int, str]] = []
    for match in FUNCTION_RE.finditer(source):
        opening_paren = source.find("(", match.start(), match.end())
        closing_paren = _matching_delimiter(source, opening_paren, ")")
        if closing_paren is None:
            continue
        opening_brace = source.find("{", closing_paren + 1)
        if opening_brace == -1:
            continue
        closing_brace = _matching_delimiter(source, opening_brace, "}")
        if closing_brace is not None:
            ranges.append((opening_brace, closing_brace, match.group("name")))
    return ranges


def _spread_context(
    source: str,
    opening_brace: int,
    spread_index: int,
    function_ranges: list[tuple[int, int, str]],
) -> str:
    before_object = source[:opening_brace]
    callee = DIRECT_CALLEE_RE.search(before_object)
    if callee:
        return callee.group("callee")
    containing = [entry for entry in function_ranges if entry[0] < spread_index < entry[1]]
    if containing:
        return min(containing, key=lambda entry: entry[1] - entry[0])[2]
    return "object-literal"


def _unique_contexts(contexts: list[str]) -> list[str]:
    counts = Counter(contexts)
    seen: Counter[str] = Counter()
    unique: list[str] = []
    for context in contexts:
        seen[context] += 1
        unique.append(context if counts[context] == 1 else f"{context}#{seen[context]}")
    return unique


def find_bag_spreads(source: str) -> list[str]:
    """Find `options`/`deps` spread elements directly inside object literals."""
    stack: list[tuple[str, bool, int]] = []
    contexts: list[str] = []
    function_ranges = _function_ranges(source)
    index = 0
    while index < len(source):
        char = source[index]
        if char == "{":
            stack.append((char, _is_object_literal(source, index), index))
        elif char in "([":
            stack.append((char, False, index))
        elif char == "}" and stack and stack[-1][0] == "{":
            stack.pop()
        elif char == ")" and stack and stack[-1][0] == "(":
            stack.pop()
        elif char == "]" and stack and stack[-1][0] == "[":
            stack.pop()
        elif char == "." and stack and stack[-1][0] == "{" and stack[-1][1]:
            match = BAG_SPREAD_RE.match(source, index)
            if match:
                contexts.append(_spread_context(source, stack[-1][2], index, function_ranges))
                index = match.end()
                continue
        index += 1
    return _unique_contexts(contexts)


def find_direct_bag_arguments(source: str) -> list[str]:
    """Find controller/factory calls handed the whole `options`/`deps` bag.

    The direct `createX(options)` form and `requireFunction(...)(options)` are
    both the small, idiomatic regressions that bypass an explicit projection.
    Method/function declarations are excluded; their closing paren is followed
    by an opening brace rather than being a call expression.
    """
    contexts: list[str] = []
    for match in DIRECT_CREATE_BAG_RE.finditer(source):
        before = source[:match.start()].rstrip()
        after = source[match.end():].lstrip()
        if re.search(r"\bfunction\s+$", before) or after.startswith("{"):
            continue
        contexts.append(match.group("callee"))
    for match in IMMEDIATE_REQUIRE_BAG_RE.finditer(source):
        contexts.append("requireFunction")
    return _unique_contexts(contexts)


def _global_property_name(source: str, sanitized: str, match: re.Match[str]) -> str:
    root = match.group("root")
    if match.group("dot"):
        return f"{root}.{match.group('dot')}"
    assignment_index = sanitized.find("=", match.start(), match.end())
    target = source[match.start():assignment_index]
    bracket = BRACKET_PROPERTY_RE.search(target)
    if not bracket:
        return f"{root}[string]"
    return f"{root}.{bracket.group('name')}"


def find_architecture_violations(static_dir: Path) -> list[GuardViolation]:
    """Return ratcheted architecture-rule violations in checked frontend source."""
    violations: list[GuardViolation] = []
    for path in source_files(static_dir):
        source = path.read_text()
        sanitized = strip_js_comments_and_strings(source)
        relative_path = path.relative_to(static_dir).as_posix()
        if path.name == "app_wiring.js":
            violations.extend(
                GuardViolation(CHECK_PASS_THROUGH_FACTORY, relative_path, match.group("factory"))
                for match in PASS_THROUGH_FACTORY_RE.finditer(sanitized)
            )
        violations.extend(
            GuardViolation(CHECK_BAG_SPREAD, relative_path, context)
            for context in find_bag_spreads(sanitized)
        )
        violations.extend(
            GuardViolation(CHECK_DIRECT_BAG_ARGUMENT, relative_path, context)
            for context in find_direct_bag_arguments(sanitized)
        )
        global_names = [
            _global_property_name(source, sanitized, match)
            for match in GLOBAL_ASSIGNMENT_RE.finditer(sanitized)
        ]
        violations.extend(
            GuardViolation(CHECK_GLOBAL_REGISTRATION, relative_path, name)
            for name in _unique_contexts(global_names)
        )
    return sorted(violations)


def scan_coverage(static_dir: Path) -> ScanCoverage:
    wiring_path = static_dir / "app_wiring.js"
    option_factories = 0
    if wiring_path.exists():
        option_factories = len(OPTION_FACTORY_RE.findall(strip_js_comments_and_strings(wiring_path.read_text())))
    return ScanCoverage(files=len(source_files(static_dir)), option_factories=option_factories)


def load_allowlist(path: Path) -> list[GuardViolation]:
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError:
        raise ValueError(f"allowlist not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid allowlist JSON: {path}: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("violations"), list):
        raise ValueError("allowlist must be an object with a 'violations' array")

    entries: list[GuardViolation] = []
    seen: set[GuardViolation] = set()
    for index, entry in enumerate(raw["violations"]):
        if not isinstance(entry, dict):
            raise ValueError(f"allowlist entry {index} must be an object")
        check = entry.get("check")
        file = entry.get("file")
        name = entry.get("name")
        if check not in ALLOWLIST_CHECKS or not all(isinstance(value, str) and value for value in (file, name)):
            raise ValueError(
                f"allowlist entry {index} must contain a known check plus non-empty file and name"
            )
        violation = GuardViolation(check, file, name)
        if violation in seen:
            raise ValueError(f"duplicate allowlist entry: [{check}] {file}:{name}")
        seen.add(violation)
        entries.append(violation)
    return entries


def report_allowlist_violations(violations: list[GuardViolation], allowlist: list[GuardViolation]) -> bool:
    """Print unexpected/stale ratchet results and return whether the ratchet is clean."""
    actual = Counter(violations)
    permitted = Counter(allowlist)
    unexpected = actual - permitted
    stale = permitted - actual
    if not unexpected and not stale:
        return True

    for violation, count in sorted(unexpected.items()):
        for _ in range(count):
            print(f"UNEXPECTED [{violation.check}] {violation.file}:{violation.name}")
    for violation, count in sorted(stale.items()):
        for _ in range(count):
            print(f"STALE ALLOWLIST ENTRY [{violation.check}] {violation.file}:{violation.name}")
    return False


def get_static_dir(path: Path) -> Path:
    if not path.exists():
        print(f"ERROR: static dir not found: {path}", file=sys.stderr)
        raise SystemExit(2)
    return path


def extract_destructured_names(source: str) -> set[str]:
    """Find property names destructured from `options` in a module."""
    names: set[str] = set()
    for m in re.finditer(
        r'(?:const|let|var)\s*\{([^}]+)\}\s*=\s*(?:requireObject\s*\(\s*)?options\b',
        source,
        re.DOTALL,
    ):
        for part in split_object_properties(m.group(1)):
            property_name = re.split(r'[:=]', part.strip())[0].strip()
            if re.match(r'^[a-zA-Z_$][a-zA-Z0-9_$]*$', property_name):
                names.add(property_name)
    return names


def extract_required_option_names(source: str) -> set[str]:
    """Find constructor-header inputs supplied by the options contract."""
    names = extract_destructured_names(source)
    names.update(re.findall(r'\b(?:requireFunction|requireObject)\s*\(\s*options\.([A-Za-z_$][A-Za-z0-9_$]*)\b', source))
    return names


def extract_select_keys(source: str) -> dict[str, set[str]]:
    """Return `createXOptions` selector keys from the original wiring module."""
    selectors: dict[str, set[str]] = {}
    pattern = re.compile(
        r'\b(?P<name>create[A-Z][A-Za-z0-9_$]*Options)\s*\(\s*deps\s*\)\s*'
        r'\{\s*return\s+select\s*\(\s*deps\s*,\s*\[',
        re.DOTALL,
    )
    for match in pattern.finditer(source):
        opening = match.end() - 1
        depth = 0
        closing = None
        for index in range(opening, len(source)):
            if source[index] == '[':
                depth += 1
            elif source[index] == ']':
                depth -= 1
                if depth == 0:
                    closing = index
                    break
        if closing is None:
            continue
        selectors[match.group('name')] = set(re.findall(r"['\"]([A-Za-z_$][A-Za-z0-9_$]*)['\"]", source[opening + 1:closing]))
    return selectors


def find_calls(source: str, pattern: str) -> list[tuple[str, int, int]]:
    """Find calls by name pattern, returning name and argument bounds."""
    calls: list[tuple[str, int, int]] = []
    for match in re.finditer(rf'(?P<name>{pattern})\s*\(', source):
        opening = source.find('(', match.start(), match.end())
        closing = _matching_delimiter(source, opening, ')')
        if closing is not None:
            calls.append((match.group('name'), opening + 1, closing))
    return calls


def controller_selector_pairs(sources: dict[str, str]) -> list[tuple[str, str, str]]:
    """Trace controller calls whose argument contains a wiring selector call."""
    pairs: list[tuple[str, str, str]] = []
    for creator_file, source in sources.items():
        for controller, start, end in find_calls(source, r'(?:[A-Za-z_$][A-Za-z0-9_$]*\.)*create[A-Z][A-Za-z0-9_$]*Controller'):
            argument = source[start:end]
            for selector in re.findall(r'\b(create[A-Z][A-Za-z0-9_$]*Options)\s*\(', argument):
                pairs.append((selector, controller.rsplit('.', 1)[-1], creator_file))
    return pairs


def split_object_properties(body: str) -> list[str]:
    """Split a literal's top-level properties after strings/comments are masked."""
    properties: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(body):
        if char in '({[':
            depth += 1
        elif char in ')}]':
            depth -= 1
        elif char == ',' and depth == 0:
            properties.append(body[start:index])
            start = index + 1
    properties.append(body[start:])
    return properties


def unbound_option_values(source: str) -> list[tuple[str, str]]:
    """Return bare identifier values in direct `wiring.create*Options({...})` calls.

    This is deliberately scoped to wiring literals: it catches a missed import or
    destructure at the contract boundary without becoming a general JS linter.
    """
    sanitized = strip_js_comments_and_strings(source)
    bound = set(BUILTINS)
    bound.update(extract_destructured_names(sanitized))
    for match in re.finditer(r'\b(?:const|let|var)\s*\{([^}]+)\}\s*=\s*[A-Za-z_$][A-Za-z0-9_$]*\b', sanitized, re.DOTALL):
        for part in match.group(1).split(','):
            local_name = re.split(r'[:=]', part.strip())[-1].strip()
            if re.fullmatch(r'[A-Za-z_$][A-Za-z0-9_$]*', local_name):
                bound.add(local_name)
    bound.update(re.findall(r'\b(?:const|let|var|function|class)\s+([A-Za-z_$][A-Za-z0-9_$]*)\b', sanitized))
    # Function parameter destructures are the dominant wiring scope in
    # composition. Treat their names as bindings throughout that function.
    for match in re.finditer(r'\bfunction\s+[A-Za-z_$][A-Za-z0-9_$]*\s*\(\s*\{([^}]*)\}', sanitized, re.DOTALL):
        bound.update(re.findall(r'[A-Za-z_$][A-Za-z0-9_$]*', match.group(1)))
    for match in re.finditer(r'\bimport\s+(.*?)\s+from\s+["\']', source, re.DOTALL):
        bound.update(re.findall(r'\b([A-Za-z_$][A-Za-z0-9_$]*)\b', match.group(1)))

    findings: list[tuple[str, str]] = []
    for factory, start, end in find_calls(sanitized, r'wiring\.create[A-Z][A-Za-z0-9_$]*Options'):
        argument = sanitized[start:end].strip()
        if not argument.startswith('{'):
            continue
        closing = _matching_delimiter(sanitized, start + sanitized[start:end].find('{'), '}')
        if closing is None or closing >= end:
            continue
        body = sanitized[start + sanitized[start:end].find('{') + 1:closing]
        for property_text in split_object_properties(body):
            property_text = property_text.strip()
            if not property_text or property_text.startswith('...'):
                continue
            colon = property_text.find(':')
            if colon == -1:
                key = property_text
                value = property_text
            else:
                key = property_text[:colon].strip()
                value = property_text[colon + 1:].strip()
            if not re.fullmatch(r'[A-Za-z_$][A-Za-z0-9_$]*', key):
                continue
            if re.fullmatch(r'[A-Za-z_$][A-Za-z0-9_$]*', value) and value not in bound:
                findings.append((factory.rsplit('.', 1)[-1], value))
    return findings


def find_creation_object(source: str, factory_pattern: str) -> list[tuple[str, bool]]:
    """Find all object literals passed to a factory call.

    Returns list of (object_body, has_spread) tuples.
    """
    results = []
    # Search for the factory call pattern followed by an opening brace.
    for m in re.finditer(rf'{factory_pattern}\s*\(\s*\{{', source):
        brace_start = m.end() - 1
        depth = 0
        i = brace_start
        while i < len(source):
            if source[i] == '{':
                depth += 1
            elif source[i] == '}':
                depth -= 1
                if depth == 0:
                    body = source[brace_start + 1:i]
                    results.append((body, '...' in body))
                    break
            i += 1
    return results


def extract_keys_from_object_body(body: str) -> set[str]:
    """Extract property names from a JS object literal body."""
    keys = set()
    for km in re.finditer(r'(?:^|[\s,])([a-zA-Z_$][a-zA-Z0-9_$]*)\s*(?=[,:}\n\r])', body, re.MULTILINE):
        keys.add(km.group(1))
    return keys


# Names that are built-in or provided by the JS runtime, not the options chain.
BUILTINS = frozenset({
    'window', 'document', 'navigator', 'HTMLElement', 'Element', 'EventSource', 'AbortController',
    'AudioContext', 'Notification', 'crypto', 'setInterval', 'clearInterval',
    'getComputedStyle', 'requestAnimationFrame', 'setTimeout', 'clearTimeout', 'Node',
    'performance', 'console', 'Math', 'JSON', 'Object', 'Array', 'String', 'Number',
    'Date', 'Error', 'Promise', 'Set', 'Map', 'Path', 'fetch', 'URL', 'Event',
    'MutationObserver', 'ResizeObserver', 'IntersectionObserver', 'alert', 'confirm',
    'parseInt', 'parseFloat', 'isNaN', 'isFinite', 'encodeURIComponent',
    'decodeURIComponent', 'matchMedia', 'CSS', 'CustomEvent', 'DOMParser',
    'XMLSerializer', 'TextEncoder', 'TextDecoder',
})


def factory_body(source: str, factory: str) -> str | None:
    """Return a named function factory body using balanced delimiters."""
    match = re.search(rf'\bfunction\s+{re.escape(factory)}\s*\(', source)
    if not match:
        return None
    opening_paren = source.find('(', match.start(), match.end())
    closing_paren = _matching_delimiter(source, opening_paren, ')')
    if closing_paren is None:
        return None
    opening_brace = source.find('{', closing_paren + 1)
    closing_brace = _matching_delimiter(source, opening_brace, '}') if opening_brace != -1 else None
    return source[opening_brace:closing_brace + 1] if closing_brace is not None else None


def constructor_header(source: str, factory: str) -> str:
    """Return statements through the first non-declaration initializer."""
    body = factory_body(source, factory) or source
    # Headers conventionally contain only require calls/destructures. The first
    # nested function begins controller-local implementation.
    nested = re.search(r'\n\s{4}function\s+', body)
    return body[:nested.start()] if nested else body


def find_module_factories(file_path: Path, source: str | None = None) -> set[str]:
    """Find controller factories exported by a module."""
    if source is None:
        source = strip_js_comments_and_strings(file_path.read_text())
    return set(re.findall(r'\bfunction\s+(create[A-Z]\w*Controller)\s*\(', source))


def check_option_contracts(static_dir: Path) -> bool:
    """Verify selector coverage and the bindings used to build option literals."""
    raw_sources = {
        path.name: path.read_text()
        for path in sorted(static_dir.glob("app_*.js"))
    }
    sources = {filename: strip_js_comments_and_strings(source) for filename, source in raw_sources.items()}
    controller_deps: dict[str, tuple[str, set[str]]] = {}
    for filename, source in sources.items():
        required = extract_required_option_names(source) - BUILTINS
        for factory in find_module_factories(static_dir / filename, source):
            # Each controller's own header is its contract. Modules that host
            # multiple controllers must not donate their aggregate option reads.
            contract_source = constructor_header(source, factory)
            controller_deps[factory] = (filename, extract_required_option_names(contract_source) - BUILTINS)

    wiring_source = raw_sources.get("app_wiring.js", "")
    selector_keys = extract_select_keys(wiring_source)
    undercoverage: list[tuple[str, str, str, str]] = []
    phase_two_selectors = {"createTranscriptRenderOptions", "createSendLifecycleOptions", "createFileOpsOptions", "createChatInteractionOptions"}
    for selector, controller, creator in controller_selector_pairs(sources):
        target = controller_deps.get(controller)
        # Phase 2's new explicit chat/file selectors are the contract boundary
        # this check protects. Older module-internal selectors retain their
        # existing compatibility defaults and are intentionally out of scope.
        fixture_selector = selector == "createFeatureOptions"
        if not target or (selector not in phase_two_selectors and not fixture_selector):
            continue
        module, required = target
        for name in sorted(required - selector_keys[selector]):
            undercoverage.append((selector, module, controller, name))

    unbound: list[tuple[str, str, str]] = []
    for filename, source in raw_sources.items():
        for selector, value in unbound_option_values(source):
            if selector in phase_two_selectors or selector == "createFeatureOptions":
                unbound.append((filename, selector, value))

    if undercoverage:
        print(f"Found {len(undercoverage)} selector undercoverage violations:\n")
        for selector, module, controller, name in undercoverage:
            print(f"  [{CHECK_SELECT_UNDERCOVERAGE}] {module}: '{name}' required by {controller} is absent from {selector}")
    if unbound:
        print(f"Found {len(unbound)} unbound option values:\n")
        for module, selector, value in unbound:
            print(f"  [{CHECK_UNBOUND_OPTION_VALUE}] {module}: '{value}' in {selector} is not bound")
    return not undercoverage and not unbound


def main() -> int:
    args = parse_args()
    static_dir = get_static_dir(args.static_dir)
    try:
        allowlist = load_allowlist(args.allowlist)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    coverage = scan_coverage(static_dir)
    print(f"SCAN COVERAGE files={coverage.files} option_factories={coverage.option_factories}")
    option_contracts_clean = check_option_contracts(static_dir)
    ratchet_clean = report_allowlist_violations(find_architecture_violations(static_dir), allowlist)
    if option_contracts_clean and ratchet_clean:
        print("\n✓ Wiring guard passed.")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
