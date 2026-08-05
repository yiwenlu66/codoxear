"""Structural reachability checks for the focused frontend extraction modules.

These checks execute every module in a minimal browser-like VM to discover its
public API, then analyze JavaScript tokens (rather than comments or strings).
They protect the extraction boundary: a public entry point needs a consumer in
the app composition graph, and a named helper needs a live code reference.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "codoxear" / "static"

EXTRACTED_MODULES = {
    "app_message_flow.js": "CodoxearMessageFlow",
    "app_session_edit.js": "CodoxearSessionEdit",
    "app_unattended.js": "CodoxearUnattended",
    "app_attachments.js": "CodoxearAttachments",
}


def _javascript_tokens(source: str) -> list[str]:
    """Return JavaScript identifiers and punctuation outside comments/strings."""
    tokens: list[str] = []
    index = 0
    length = len(source)
    while index < length:
        char = source[index]
        next_char = source[index + 1] if index + 1 < length else ""
        if char.isspace():
            index += 1
        elif char == "/" and next_char == "/":
            newline = source.find("\n", index + 2)
            index = length if newline < 0 else newline + 1
        elif char == "/" and next_char == "*":
            close = source.find("*/", index + 2)
            index = length if close < 0 else close + 2
        elif char in "'\"`":
            quote = char
            index += 1
            while index < length:
                if source[index] == "\\":
                    index += 2
                elif source[index] == quote:
                    index += 1
                    break
                else:
                    index += 1
        elif char.isalpha() or char in "_$":
            end = index + 1
            while end < length and (source[end].isalnum() or source[end] in "_$"):
                end += 1
            tokens.append(source[index:end])
            index = end
        else:
            tokens.append(char)
            index += 1
    return tokens


def _named_function_definitions(tokens: list[str]) -> set[str]:
    definitions = set()
    for index, token in enumerate(tokens[:-1]):
        if token == "function" and tokens[index + 1] != "(":
            definitions.add(tokens[index + 1])
    return definitions


def _load_exports(path: Path, global_name: str) -> dict[str, str]:
    source = json.dumps(path.read_text(encoding="utf-8"))
    program = f"""
        const vm = require("vm");
        const context = {{
          window: {{
            CodoxearPolling: {{
              messagePollDelayMs() {{}}, normalizeMessagePollKickDelay() {{}}, browserOffline() {{}},
            }},
            CodoxearTranscript: {{
              startsTypingCountWindow() {{}}, hasHumanOriginatedUserEvent() {{}},
              thinkingModeForTokens() {{}}, transcriptSnapshotFromData() {{}},
            }},
            CodoxearSessionHelpers: {{ sessionLaunchFailed() {{ return false; }} }},
            CodoxearModal: {{ restoreModalFocus() {{}} }},
          }},
        }};
        vm.createContext(context);
        vm.runInContext({source}, context, {{ filename: {json.dumps(path.name)} }});
        const exported = context.window[{json.dumps(global_name)}];
        const api = Object.fromEntries(
          Object.getOwnPropertyNames(exported).sort().map((name) => [name, typeof exported[name]])
        );
        process.stdout.write(JSON.stringify(api));
    """
    completed = subprocess.run(
        ["node", "-e", program],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_extracted_modules_publish_callable_public_apis() -> None:
    for filename, global_name in EXTRACTED_MODULES.items():
        exports = _load_exports(STATIC / filename, global_name)
        assert exports, f"{filename} did not publish {global_name}"
        assert set(exports.values()) == {"function"}, (
            f"{filename} has a non-callable public API: {exports}"
        )


def test_extracted_module_public_apis_have_composition_consumers() -> None:
    sources = {
        path.name: _javascript_tokens(path.read_text(encoding="utf-8"))
        for path in STATIC.glob("*.js")
    }
    for filename, global_name in EXTRACTED_MODULES.items():
        exports = _load_exports(STATIC / filename, global_name)
        consumer_tokens = [
            token
            for other_filename, tokens in sources.items()
            if other_filename != filename
            for token in tokens
        ]
        for export_name in exports:
            assert export_name in consumer_tokens, (
                f"{filename} exports {export_name}, but no other static module references it"
            )


def test_extracted_modules_have_no_orphaned_named_helpers() -> None:
    for filename in EXTRACTED_MODULES:
        tokens = _javascript_tokens((STATIC / filename).read_text(encoding="utf-8"))
        token_counts = {token: tokens.count(token) for token in set(tokens)}
        orphans = sorted(
            name
            for name in _named_function_definitions(tokens)
            if token_counts[name] == 1
        )
        assert not orphans, f"{filename} has orphaned named helpers: {', '.join(orphans)}"
