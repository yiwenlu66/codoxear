import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TestPackagingStaticAssets(unittest.TestCase):
    def test_wheel_includes_nested_logo_assets(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            outdir = Path(td)
            subprocess.run(
                ["python3", "-m", "pip", "wheel", str(ROOT), "-w", str(outdir), "--no-deps"],
                check=True,
                cwd=ROOT,
            )
            wheel = next(outdir.glob("codoxear-*.whl"))
            with zipfile.ZipFile(wheel) as zf:
                names = set(zf.namelist())
        self.assertIn("codoxear/static/logos/codex.svg", names)
        self.assertIn("codoxear/static/logos/pi.svg", names)


if __name__ == "__main__":
    unittest.main()
