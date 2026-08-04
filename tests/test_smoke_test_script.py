from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SMOKE_TEST = ROOT / "scripts" / "smoke_test.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def _run_smoke_test(tmp_path: Path, eval_responses: list[str]) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    operations = tmp_path / "operations.log"
    responses = tmp_path / "eval-responses"
    responses.write_text("\n".join(eval_responses) + "\n")
    _write_executable(
        fake_bin / "agent-browser",
        f"""#!/usr/bin/env bash
set -eu
printf '%s\\n' "$*" >> {operations!s}
case "$1" in
  eval)
    response="$(head -n 1 {responses!s})"
    tail -n +2 {responses!s} > {responses!s}.next
    mv {responses!s}.next {responses!s}
    printf '%s\\n' "$response"
    ;;
esac
""",
    )
    environment = os.environ | {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "CODOXEAR_SMOKE_ATTEMPTS": "1",
        "CODOXEAR_SMOKE_SETTLE_MS": "0",
    }
    completed = subprocess.run(
        [str(SMOKE_TEST), "http://127.0.0.1:9876", "test-password"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
    )
    return completed, operations.read_text().splitlines()


def test_smoke_test_requires_clean_initial_load_and_reload(tmp_path: Path) -> None:
    completed, operations = _run_smoke_test(tmp_path, ['"OK"', '"OK"'])

    assert completed.returncode == 0, completed.stderr
    assert "smoke test passed: initial load and reload rendered a session card" in completed.stdout
    assert sum(operation.startswith("eval ") for operation in operations) == 2
    assert any(operation.startswith("reload") for operation in operations)
    assert any(operation.startswith("close") for operation in operations)


def test_smoke_test_rejects_a_reload_failure_after_clean_initial_load(tmp_path: Path) -> None:
    completed, operations = _run_smoke_test(tmp_path, ['"OK"', '"FAIL: sessionCardsRendered; cards=0"'])

    assert completed.returncode != 0
    assert "smoke test failed during reload" in completed.stderr
    assert "smoke test failed: app did not render cleanly after 1 attempt(s)" in completed.stderr
    assert sum(operation.startswith("eval ") for operation in operations) == 2
    assert any(operation.startswith("reload") for operation in operations)
