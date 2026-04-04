import json
import subprocess
import textwrap
import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"


def _extract_named_function(name: str) -> str:
    source = APP_JS.read_text(encoding="utf-8")
    anchor = f"function {name}("
    start = source.index(anchor)
    func_start = source.index("{", start)
    depth = 0
    end = func_start
    for index in range(func_start, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = index + 1
                break
    else:
        raise AssertionError(f"failed to extract function {name}")
    return source[start:end]


def eval_timestamp_helper(ts: float) -> bool:
    helper = _extract_named_function("isDisplayableEpochTs")
    injected = json.dumps(helper + "\nglobalThis.__test_isDisplayableEpochTs = isDisplayableEpochTs;\n")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{}};
        vm.createContext(ctx);
        vm.runInContext({injected}, ctx);
        process.stdout.write(JSON.stringify(ctx.__test_isDisplayableEpochTs({json.dumps(ts)})));
        """
    )
    proc = subprocess.run(
        ["node", "-e", js],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(proc.stdout)


class TestMessageTimestampUi(unittest.TestCase):
    def test_displayable_epoch_ts_rejects_synthetic_offset_seconds(self) -> None:
        self.assertFalse(eval_timestamp_helper(1_650_000.0))

    def test_displayable_epoch_ts_accepts_real_recent_epoch_seconds(self) -> None:
        self.assertTrue(eval_timestamp_helper(1_744_000_000.0))
