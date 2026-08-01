import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"


def eval_sidebar_display_cases() -> dict[str, str]:
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index("const SIDEBAR_REASONING_EFFORT_CODES")
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
          glm: sidebarModelText({{ model: "glm-5.2" }}),
          gpt: sidebarModelText({{ model: "gpt-5.6-sol" }}),
          kimi: sidebarModelText({{ model: "kimi-k3" }}),
          sixteenCharacters: sidebarModelText({{ model: "abcdefghijklmnop" }}),
          deepseek: sidebarModelText({{ model: "deepseek-v4-flash" }}),
          claude: sidebarModelText({{ model: "claude-sonnet-4-5" }}),
          longProviderModel: sidebarModelText({{ model: "provider/very-long-model-name-for-ellipsis-proof", model_provider: "ignored-provider" }}),
          effortNull: sidebarEffortCode(null),
          effortUnknown: sidebarEffortCode("ultra"),
          effortOff: sidebarEffortCode("off"),
          effortMinimal: sidebarEffortCode("minimal"),
          effortLow: sidebarEffortCode("low"),
          effortMedium: sidebarEffortCode("medium"),
          effortHigh: sidebarEffortCode("high"),
          effortXhigh: sidebarEffortCode("xhigh"),
          effortMax: sidebarEffortCode("max"),
        }};
        process.stdout.write(JSON.stringify(cases));
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


class TestSidebarModelDisplaySource(unittest.TestCase):
    def test_sidebar_model_text_uses_middle_ellipsis_and_preserves_common_names(self) -> None:
        result = eval_sidebar_display_cases()
        self.assertEqual(result["nullValue"], "")
        self.assertEqual(result["missing"], "")
        self.assertEqual(result["nullModel"], "")
        self.assertEqual(result["empty"], "")
        self.assertEqual(result["whitespace"], "")
        self.assertEqual(result["defaultLower"], "")
        self.assertEqual(result["defaultMixedWhitespace"], "")
        self.assertEqual(result["glm"], "glm-5.2")
        self.assertEqual(result["gpt"], "gpt-5.6-sol")
        self.assertEqual(result["kimi"], "kimi-k3")
        self.assertEqual(result["sixteenCharacters"], "abcdefghijklmnop")
        self.assertEqual(result["deepseek"], "deepse…v4-flash")
        self.assertEqual(result["claude"], "claude…nnet-4-5")
        self.assertEqual(result["longProviderModel"], "provid…is-proof")

    def test_sidebar_reasoning_effort_codes_have_compact_known_values(self) -> None:
        result = eval_sidebar_display_cases()
        self.assertEqual(result["effortNull"], "")
        self.assertEqual(result["effortUnknown"], "")
        self.assertEqual(
            {key: result[key] for key in ("effortOff", "effortMinimal", "effortLow", "effortMedium", "effortHigh", "effortXhigh", "effortMax")},
            {
                "effortOff": "off",
                "effortMinimal": "min",
                "effortLow": "low",
                "effortMedium": "med",
                "effortHigh": "hi",
                "effortXhigh": "xh",
                "effortMax": "max",
            },
        )

if __name__ == "__main__":
    unittest.main()
