import json
import subprocess
import textwrap
import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"


def eval_parse_local_file_ref(raw_value: str) -> object:
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index("      function filePathExtension(path) {")
    end = source.index("      function normalizePathLike(rawPath) {", start)
    snippet = source[start:end]
    injected = json.dumps(snippet + "\nglobalThis.__test_parseLocalFileRef = parseLocalFileRef;\n")
    js = textwrap.dedent(
        f"""
        const vm = require('vm');
        const ctx = {{ console }};
        vm.createContext(ctx);
        vm.runInContext({injected}, ctx);
        process.stdout.write(JSON.stringify(ctx.__test_parseLocalFileRef({json.dumps(raw_value)})));
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


class TestFileRefParsingUi(unittest.TestCase):
    def test_parse_local_file_ref_rejects_trailing_colon_path(self) -> None:
        self.assertIsNone(eval_parse_local_file_ref("/tmp/pi.sock:"))

    def test_parse_local_file_ref_keeps_real_line_suffix(self) -> None:
        self.assertEqual(
            eval_parse_local_file_ref("codoxear/server.py:42"),
            {"path": "codoxear/server.py", "line": 42},
        )


if __name__ == "__main__":
    unittest.main()
