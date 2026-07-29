import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"


def eval_sidebar_model_text_cases() -> dict[str, str]:
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index("function sidebarModelText(s)")
    end = source.index("\n\n      function sessionIdFromHash", start)
    helper_source = source[start:end]
    js = textwrap.dedent(
        f"""
        {helper_source}
        const cases = {{
          nullValue: sidebarModelText(null),
          missing: sidebarModelText({{}}),
          nullModel: sidebarModelText({{ model: null }}),
          empty: sidebarModelText({{ model: "" }}),
          whitespace: sidebarModelText({{ model: "   " }}),
          defaultLower: sidebarModelText({{ model: "default" }}),
          defaultMixedWhitespace: sidebarModelText({{ model: "  DeFaUlT  " }}),
          gpt: sidebarModelText({{ model: "  gpt-5.4  " }}),
          claude: sidebarModelText({{ model: "claude-sonnet-4-5" }}),
          longProviderModel: sidebarModelText({{ model: "provider/very-long-model-name-for-ellipsis-proof" }}),
        }};
        process.stdout.write(JSON.stringify(cases));
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


class TestSidebarModelDisplaySource(unittest.TestCase):
    def test_sidebar_model_text_trims_and_suppresses_noise_values(self) -> None:
        result = eval_sidebar_model_text_cases()
        self.assertEqual(result["nullValue"], "")
        self.assertEqual(result["missing"], "")
        self.assertEqual(result["nullModel"], "")
        self.assertEqual(result["empty"], "")
        self.assertEqual(result["whitespace"], "")
        self.assertEqual(result["defaultLower"], "")
        self.assertEqual(result["defaultMixedWhitespace"], "")
        self.assertEqual(result["gpt"], "gpt-5.4")
        self.assertEqual(result["claude"], "claude-sonnet-4-5")
        self.assertEqual(result["longProviderModel"], "provider/very-long-model-name-for-ellipsis-proof")

if __name__ == "__main__":
    unittest.main()
