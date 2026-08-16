from __future__ import annotations

import json
from collections import Counter
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check_wiring.py"
REAL_STATIC = ROOT / "codoxear" / "static"
REAL_ALLOWLIST = ROOT / "scripts" / "wiring_guard_allowlist.json"


def _run_guard(static_dir: Path, allowlist: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER), str(static_dir), "--allowlist", str(allowlist)],
        text=True,
        capture_output=True,
        check=False,
    )


def _write_fixture(static_dir: Path, files: dict[str, str], allowlist_entries: list[dict[str, str]] | None = None) -> Path:
    static_dir.mkdir(parents=True)
    for name, source in files.items():
        (static_dir / name).write_text(source)
    allowlist = static_dir.parent / "allowlist.json"
    allowlist.write_text(json.dumps({"violations": allowlist_entries or []}))
    return allowlist


def _allowlist_entries(path: Path) -> list[dict[str, str]]:
    return json.loads(path.read_text())["violations"]


def _head_allowlist_entries() -> list[dict[str, str]] | None:
    result = subprocess.run(
        ["git", "show", "HEAD:scripts/wiring_guard_allowlist.json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        return None
    return json.loads(result.stdout)["violations"]


def test_wiring_guard_accepts_the_real_frontend_tree_and_reports_coverage() -> None:
    result = _run_guard(REAL_STATIC, REAL_ALLOWLIST)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Wiring guard passed" in result.stdout
    coverage = next(line for line in result.stdout.splitlines() if line.startswith("SCAN COVERAGE "))
    assert "files=0" not in coverage
    assert "option_factories=0" not in coverage


def test_wiring_guard_allowlist_only_shrinks_from_head() -> None:
    head_entries = _head_allowlist_entries()
    if head_entries is None:
        return

    current = Counter(
        (entry["check"], entry["file"], entry["name"])
        for entry in _allowlist_entries(REAL_ALLOWLIST)
    )
    head = Counter((entry["check"], entry["file"], entry["name"]) for entry in head_entries)

    assert not (current - head), "working-tree allowlist may only remove HEAD entries"


def test_wiring_guard_rejects_each_planted_architecture_violation(tmp_path: Path) -> None:
    cases = [
        (
            "pass-through-factory",
            {"app_wiring.js": "function createPlantedOptions(deps) { return deps; }\n"},
        ),
        (
            "bag-spread",
            {"app_feature.js": "function feature() { return { ...options, ready: true }; }\n"},
        ),
        (
            "global-registration",
            {"app_feature.js": "globalThis[\"planted\"] = {};\n"},
        ),
        (
            "direct-bag-argument",
            {"app_feature.js": "createPlantedController(options);\n"},
        ),
        (
            "direct-bag-argument",
            {"app_feature.js": "requireFunction(factory, 'factory')(deps);\n"},
        ),
    ]

    for index, (check_id, files) in enumerate(cases):
        fixture_root = tmp_path / f"{check_id}-{index}"
        static_dir = fixture_root / "static"
        allowlist = _write_fixture(static_dir, files)

        result = _run_guard(static_dir, allowlist)

        assert result.returncode == 1
        assert f"UNEXPECTED [{check_id}]" in result.stdout


def test_wiring_guard_rejects_planted_selector_undercoverage_and_unbound_values(tmp_path: Path) -> None:
    static_dir = tmp_path / "static"
    allowlist = _write_fixture(
        static_dir,
        {
            "app_wiring.js": """
            function select(deps, keys) { return {}; }
            function createFeatureOptions(deps) { return select(deps, ['provided']); }
            """,
            "app_feature.js": """
            function createFeatureController(options = {}) {
              const required = requireFunction(options.required, 'required');
              return required;
            }
            createFeatureController(wiring.createFeatureOptions({ provided: unboundValue }));
            """,
        },
    )

    result = _run_guard(static_dir, allowlist)

    assert result.returncode == 1
    assert "[select-undercoverage] app_feature.js: 'required' required by createFeatureController is absent from createFeatureOptions" in result.stdout
    assert "[unbound-option-value] app_feature.js: 'unboundValue' in createFeatureOptions is not bound" in result.stdout


def test_wiring_guard_ignores_non_object_bag_spreads_and_literal_text(tmp_path: Path) -> None:
    static_dir = tmp_path / "static"
    allowlist = _write_fixture(
        static_dir,
        {
            "app_feature.js": """
            const ignored = fn(...options);
            const values = [...deps];
            const documentation = "window.fake = {}; ...options";
            // return { ...options };
            const regex = /https?:\\/\\/example\\.test/;
            """,
        },
    )

    result = _run_guard(static_dir, allowlist)

    assert result.returncode == 0, result.stdout + result.stderr


def test_wiring_guard_rejects_a_stale_allowlist_entry_and_duplicates(tmp_path: Path) -> None:
    static_dir = tmp_path / "static"
    stale_allowlist = _write_fixture(
        static_dir,
        {"app_feature.js": "const ready = true;\n"},
        [{"check": "global-registration", "file": "app_feature.js", "name": "window.gone"}],
    )

    stale = _run_guard(static_dir, stale_allowlist)

    assert stale.returncode == 1
    assert "STALE ALLOWLIST ENTRY [global-registration] app_feature.js:window.gone" in stale.stdout

    duplicate_allowlist = static_dir.parent / "duplicate-allowlist.json"
    entry = {"check": "global-registration", "file": "app_feature.js", "name": "window.gone"}
    duplicate_allowlist.write_text(json.dumps({"violations": [entry, entry]}))
    duplicate = _run_guard(static_dir, duplicate_allowlist)

    assert duplicate.returncode == 2
    assert "duplicate allowlist entry" in duplicate.stderr


def test_wiring_guard_allowlist_counter_handles_duplicate_head_entries(tmp_path: Path, monkeypatch) -> None:
    entry = {"check": "global-registration", "file": "app_feature.js", "name": "window.gone"}
    allowlist = _write_fixture(
        tmp_path / "static",
        {"app_feature.js": "const ready = true;\n"},
        [entry],
    )
    monkeypatch.setattr(sys.modules[__name__], "REAL_ALLOWLIST", allowlist)
    monkeypatch.setattr(sys.modules[__name__], "_head_allowlist_entries", lambda: [entry, entry])

    test_wiring_guard_allowlist_only_shrinks_from_head()
