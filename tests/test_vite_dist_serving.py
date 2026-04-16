import tempfile
import unittest
from pathlib import Path
from unittest import mock

import codoxear.server as server


class TestViteDistServing(unittest.TestCase):
    def test_root_prefers_vite_dist_index_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            dist = Path(td) / "web" / "dist"
            dist.mkdir(parents=True)
            (dist / "index.html").write_text("<html><body>vite</body></html>", encoding="utf-8")

            with mock.patch.object(server, "WEB_DIST_DIR", dist):
                body, ctype = server._read_web_index()

        self.assertIn("vite", body)
        self.assertEqual("text/html; charset=utf-8", ctype)

    def test_public_asset_prefers_dist_copy_before_legacy_static(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dist = root / "web" / "dist"
            legacy = root / "codoxear" / "static"
            dist.mkdir(parents=True)
            legacy.mkdir(parents=True)
            (dist / "index.html").write_text("<html><body>vite</body></html>", encoding="utf-8")
            (dist / "manifest.webmanifest").write_text('{"name":"vite"}', encoding="utf-8")
            (legacy / "manifest.webmanifest").write_text('{"name":"legacy"}', encoding="utf-8")

            with (
                mock.patch.object(server, "WEB_DIST_DIR", dist),
                mock.patch.object(server, "LEGACY_STATIC_DIR", legacy),
                mock.patch.object(server, "USE_LEGACY_WEB", False),
            ):
                resolved = server._resolve_public_web_asset("manifest.webmanifest")

        self.assertEqual((dist / "manifest.webmanifest").resolve(), resolved)

    def test_path_within_rejects_prefix_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dist = root / "dist"
            sibling = root / "dist-secret"
            dist.mkdir()
            sibling.mkdir()

            candidate = (sibling / "asset.js").resolve()

            self.assertFalse(server._is_path_within(dist.resolve(), candidate))

    def test_public_asset_falls_back_to_legacy_when_no_dist_shell_is_active(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dist = root / "web" / "dist"
            legacy = root / "codoxear" / "static"
            dist.mkdir(parents=True)
            legacy.mkdir(parents=True)
            (legacy / "service-worker.js").write_text("self.addEventListener('install', () => {});", encoding="utf-8")

            with (
                mock.patch.object(server, "WEB_DIST_DIR", dist),
                mock.patch.object(server, "PACKAGED_WEB_DIST_DIR", root / "pkg-dist"),
                mock.patch.object(server, "LEGACY_STATIC_DIR", legacy),
                mock.patch.object(server, "USE_LEGACY_WEB", False),
            ):
                resolved = server._resolve_public_web_asset("service-worker.js")

        self.assertEqual((legacy / "service-worker.js").resolve(), resolved)

    def test_packaged_dist_is_used_when_checkout_dist_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            checkout_dist = root / "web" / "dist"
            packaged_dist = root / "codoxear" / "static" / "dist"
            packaged_dist.mkdir(parents=True)
            (packaged_dist / "index.html").write_text("<html><body>packaged</body></html>", encoding="utf-8")

            with (
                mock.patch.object(server, "WEB_DIST_DIR", checkout_dist),
                mock.patch.object(server, "PACKAGED_WEB_DIST_DIR", packaged_dist),
            ):
                body, _ = server._read_web_index()

        self.assertIn("packaged", body)

    def test_packaged_dist_is_used_when_checkout_dist_is_partial(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            checkout_dist = root / "web" / "dist"
            packaged_dist = root / "codoxear" / "static" / "dist"
            checkout_dist.mkdir(parents=True)
            packaged_dist.mkdir(parents=True)
            (packaged_dist / "index.html").write_text("<html><body>packaged</body></html>", encoding="utf-8")

            with (
                mock.patch.object(server, "WEB_DIST_DIR", checkout_dist),
                mock.patch.object(server, "PACKAGED_WEB_DIST_DIR", packaged_dist),
            ):
                body, _ = server._read_web_index()

        self.assertIn("packaged", body)

    def test_root_rewrite_applies_url_prefix_to_dist_assets(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dist = root / "web" / "dist"
            dist.mkdir(parents=True)
            (dist / "index.html").write_text(
                '<html><head><link rel="manifest" href="/manifest.webmanifest" /></head>'
                '<body><script type="module" src="/assets/main.js"></script></body></html>',
                encoding="utf-8",
            )

            with (
                mock.patch.object(server, "WEB_DIST_DIR", dist),
                mock.patch.object(server, "PACKAGED_WEB_DIST_DIR", root / "pkg-dist"),
                mock.patch.object(server, "URL_PREFIX", "/codoxear"),
            ):
                body, _ = server._read_web_index()

        self.assertIn('href="/codoxear/manifest.webmanifest"', body)
        self.assertIn('src="/codoxear/assets/main.js"', body)

    def test_root_rewrite_does_not_double_prefix_existing_prefixed_assets(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dist = root / "web" / "dist"
            dist.mkdir(parents=True)
            (dist / "index.html").write_text(
                '<html><head><link rel="manifest" href="/codoxear/manifest.webmanifest" /></head>'
                '<body><script type="module" src="/codoxear/assets/main.js"></script></body></html>',
                encoding="utf-8",
            )

            with (
                mock.patch.object(server, "WEB_DIST_DIR", dist),
                mock.patch.object(server, "PACKAGED_WEB_DIST_DIR", root / "pkg-dist"),
                mock.patch.object(server, "URL_PREFIX", "/codoxear"),
            ):
                body, _ = server._read_web_index()

        self.assertNotIn('/codoxear/codoxear/assets/', body)
        self.assertIn('src="/codoxear/assets/main.js"', body)

    def test_public_root_file_prefers_dist_copy(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dist = root / "web" / "dist"
            legacy = root / "codoxear" / "static"
            dist.mkdir(parents=True)
            legacy.mkdir(parents=True)
            (dist / "index.html").write_text("<html><body>vite</body></html>", encoding="utf-8")
            (dist / "robots.txt").write_text("User-agent: *\nDisallow:\n", encoding="utf-8")
            (legacy / "robots.txt").write_text("legacy\n", encoding="utf-8")

            with (
                mock.patch.object(server, "WEB_DIST_DIR", dist),
                mock.patch.object(server, "PACKAGED_WEB_DIST_DIR", root / "pkg-dist"),
                mock.patch.object(server, "LEGACY_STATIC_DIR", legacy),
                mock.patch.object(server, "USE_LEGACY_WEB", False),
            ):
                resolved = server._resolve_public_web_asset("robots.txt")

        self.assertEqual((dist / "robots.txt").resolve(), resolved)

    def test_public_asset_stays_on_served_packaged_dist_when_checkout_is_partial(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            checkout_dist = root / "web" / "dist"
            packaged_dist = root / "codoxear" / "static" / "dist"
            checkout_dist.mkdir(parents=True)
            packaged_dist.mkdir(parents=True)
            (packaged_dist / "index.html").write_text("<html><body>packaged</body></html>", encoding="utf-8")
            (checkout_dist / "robots.txt").write_text("checkout\n", encoding="utf-8")
            (packaged_dist / "robots.txt").write_text("packaged\n", encoding="utf-8")

            with (
                mock.patch.object(server, "WEB_DIST_DIR", checkout_dist),
                mock.patch.object(server, "PACKAGED_WEB_DIST_DIR", packaged_dist),
                mock.patch.object(server, "USE_LEGACY_WEB", False),
            ):
                resolved = server._resolve_public_web_asset("robots.txt")

        self.assertEqual((packaged_dist / "robots.txt").resolve(), resolved)

    def test_public_asset_returns_none_without_active_dist_shell(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            checkout_dist = root / "web" / "dist"
            packaged_dist = root / "codoxear" / "static" / "dist"
            checkout_dist.mkdir(parents=True)
            packaged_dist.mkdir(parents=True)
            (checkout_dist / "robots.txt").write_text("checkout\n", encoding="utf-8")
            (packaged_dist / "robots.txt").write_text("packaged\n", encoding="utf-8")

            with (
                mock.patch.object(server, "WEB_DIST_DIR", checkout_dist),
                mock.patch.object(server, "PACKAGED_WEB_DIST_DIR", packaged_dist),
                mock.patch.object(server, "USE_LEGACY_WEB", False),
            ):
                resolved = server._resolve_public_web_asset("robots.txt")

        self.assertIsNone(resolved)


if __name__ == "__main__":
    unittest.main()
