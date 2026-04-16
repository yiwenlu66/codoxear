import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import codoxear.server as server


class TestViteAssetVersioning(unittest.TestCase):
    def test_asset_version_uses_manifest_hashes_when_present(self) -> None:
        manifest = {
            "index.html": {
                "file": "assets/main-abcd1234.js",
                "css": ["assets/main-efgh5678.css"],
            }
        }

        version = server._asset_version_from_manifest(manifest)

        self.assertEqual("abcd1234-efgh5678", version)

    def test_current_app_version_prefers_vite_manifest_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dist = root / "web" / "dist"
            manifest_dir = dist / ".vite"
            legacy = root / "codoxear" / "static"
            manifest_dir.mkdir(parents=True)
            legacy.mkdir(parents=True)
            (dist / "index.html").write_text("<html><body>vite</body></html>", encoding="utf-8")
            (legacy / "app.js").write_text("console.log('legacy');\n", encoding="utf-8")
            (legacy / "app.css").write_text("body { color: black; }\n", encoding="utf-8")
            (manifest_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "index.html": {
                            "file": "assets/main-abcd1234.js",
                            "css": ["assets/main-efgh5678.css"],
                        }
                    }
                ),
                encoding="utf-8",
            )

            with (
                mock.patch.object(server, "WEB_DIST_DIR", dist),
                mock.patch.object(server, "STATIC_DIR", legacy),
                mock.patch.object(server, "USE_LEGACY_WEB", False),
            ):
                version = server._current_app_version()

        self.assertEqual("abcd1234-efgh5678", version)

    def test_current_app_version_uses_packaged_dist_manifest_when_checkout_dist_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            checkout_dist = root / "web" / "dist"
            packaged_dist = root / "codoxear" / "static" / "dist"
            manifest_dir = packaged_dist / ".vite"
            legacy = root / "codoxear" / "static"
            manifest_dir.mkdir(parents=True)
            legacy.mkdir(parents=True, exist_ok=True)
            (packaged_dist / "index.html").write_text("<html><body>packaged</body></html>", encoding="utf-8")
            (legacy / "app.js").write_text("console.log('legacy');\n", encoding="utf-8")
            (legacy / "app.css").write_text("body { color: black; }\n", encoding="utf-8")
            (manifest_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "index.html": {
                            "file": "assets/main-zxyw9876.js",
                            "css": ["assets/main-vuts5432.css"],
                        }
                    }
                ),
                encoding="utf-8",
            )

            with (
                mock.patch.object(server, "WEB_DIST_DIR", checkout_dist),
                mock.patch.object(server, "PACKAGED_WEB_DIST_DIR", packaged_dist),
                mock.patch.object(server, "STATIC_DIR", legacy),
                mock.patch.object(server, "USE_LEGACY_WEB", False),
            ):
                version = server._current_app_version()

        self.assertEqual("zxyw9876-vuts5432", version)

    def test_cache_control_marks_dist_assets_immutable(self) -> None:
        cache_control = server._cache_control_for_path(Path("web/dist/assets/main-abcd1234.js"))

        self.assertEqual("public, max-age=31536000, immutable", cache_control)

    def test_cache_control_keeps_non_asset_files_uncached(self) -> None:
        cache_control = server._cache_control_for_path(Path("codoxear/static/index.html"))

        self.assertEqual("no-store", cache_control)

    def test_asset_version_uses_stable_fallback_for_unhashed_manifest_entries(self) -> None:
        manifest = {
            "index.html": {
                "file": "assets/main.js",
                "css": ["assets/main.css"],
            }
        }

        version = server._asset_version_from_manifest(manifest)

        self.assertNotEqual("dev", version)

    def test_current_app_version_uses_manifest_from_served_dist_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            checkout_dist = root / "web" / "dist"
            packaged_dist = root / "codoxear" / "static" / "dist"
            checkout_manifest_dir = checkout_dist / ".vite"
            packaged_manifest_dir = packaged_dist / ".vite"
            checkout_manifest_dir.mkdir(parents=True)
            packaged_manifest_dir.mkdir(parents=True)
            (packaged_dist / "index.html").write_text("<html><body>packaged</body></html>", encoding="utf-8")
            (checkout_manifest_dir / "manifest.json").write_text(
                json.dumps({"index.html": {"file": "assets/main-checkout1111.js"}}),
                encoding="utf-8",
            )
            (packaged_manifest_dir / "manifest.json").write_text(
                json.dumps({"index.html": {"file": "assets/main-packaged2222.js"}}),
                encoding="utf-8",
            )
            legacy = root / "codoxear" / "static"
            legacy.mkdir(parents=True, exist_ok=True)
            (legacy / "app.js").write_text("console.log('legacy');\n", encoding="utf-8")
            (legacy / "app.css").write_text("body { color: black; }\n", encoding="utf-8")

            with (
                mock.patch.object(server, "WEB_DIST_DIR", checkout_dist),
                mock.patch.object(server, "PACKAGED_WEB_DIST_DIR", packaged_dist),
                mock.patch.object(server, "STATIC_DIR", legacy),
                mock.patch.object(server, "USE_LEGACY_WEB", False),
            ):
                version = server._current_app_version()

        self.assertEqual("packaged2222", version)


if __name__ == "__main__":
    unittest.main()
