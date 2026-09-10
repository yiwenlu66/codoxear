from __future__ import annotations

import json
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlparse
from xml.etree import ElementTree

from PIL import Image

from codoxear.static_routes import CONTENT_SECURITY_POLICY
from codoxear.static_routes import STATIC_DIR
from codoxear.static_routes import STATIC_ASSET_VERSION_FILES
from codoxear.static_routes import StaticRouteDeps
from codoxear.static_routes import handle_static_get_route
from codoxear.static_routes import read_static_bytes
from codoxear.static_routes import static_asset_version
from codoxear.static_routes import static_cache_control_headers


class _Headers(list[tuple[str, str]]):
    def __init__(self) -> None:
        super().__init__()
        self.request: dict[str, str] = {}

    def get(self, name: str, default: str | None = None) -> str | None:
        return self.request.get(name, default)


class _Handler:
    def __init__(self) -> None:
        self.status: int | None = None
        self.errors: list[int] = []
        self.headers = _Headers()
        self.wfile = BytesIO()

    def send_response(self, status: int) -> None:
        self.status = status

    def send_header(self, name: str, value: str) -> None:
        self.headers.append((name, value))

    def end_headers(self) -> None:
        pass

    def send_error(self, status: int) -> None:
        self.errors.append(status)


class _HeadLinks(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "link":
            self.links.append({key: value or "" for key, value in attrs})


def _deps() -> StaticRouteDeps:
    return StaticRouteDeps(
        static_dir=STATIC_DIR,
        top_level_static_assets=(
            ("/favicon.png", "favicon.png"),
            ("/apple-touch-icon.png", "apple-touch-icon.png"),
            ("/manifest.webmanifest", "manifest.webmanifest"),
            ("/", "index.html"),
        ),
        read_static_bytes=lambda path: read_static_bytes(path, attach_upload_max_bytes=123),
        static_cache_control_headers=lambda *, versioned=False, is_html=False: static_cache_control_headers(
            versioned=versioned,
            is_html=is_html,
        ),
        content_security_policy=CONTENT_SECURITY_POLICY,
    )


def _get(path: str) -> tuple[_Handler, bytes]:
    parsed = urlparse(path)
    handler = _Handler()
    assert handle_static_get_route(handler, path=parsed.path, query=parsed.query, deps=_deps()) is True
    assert handler.status == 200
    assert handler.errors == []
    return handler, handler.wfile.getvalue()


def _link(links: list[dict[str, str]], rel: str) -> dict[str, str]:
    return next(link for link in links if link.get("rel") == rel)


def test_favicon_is_full_bleed_and_distinct_from_padded_pwa_icons(tmp_path: Path) -> None:
    _html_headers, html = _get("/")
    parser = _HeadLinks()
    parser.feed(html.decode("utf-8"))
    version = static_asset_version(STATIC_DIR)
    base = "http://codoxear.test/"

    favicon_link = _link(parser.links, "icon")
    touch_link = _link(parser.links, "apple-touch-icon")
    manifest_link = _link(parser.links, "manifest")
    assert favicon_link["type"] == "image/svg+xml"
    assert "sizes" not in favicon_link
    assert parse_qs(urlparse(favicon_link["href"]).query) == {"v": [version]}

    favicon_url = urlparse(urljoin(base, favicon_link["href"]))
    favicon_headers, favicon_svg = _get(favicon_url.path + f"?{favicon_url.query}")
    assert ("Content-Type", "image/svg+xml; charset=utf-8") in favicon_headers.headers
    favicon_document = ElementTree.fromstring(favicon_svg)
    assert favicon_document.attrib["viewBox"] == "130 92 284 328"
    assert not favicon_document.findall("{http://www.w3.org/2000/svg}rect")

    _png_headers, favicon_png = _get(f"/favicon.png?v={version}")
    favicon_path = tmp_path / "favicon.png"
    favicon_path.write_bytes(favicon_png)
    with Image.open(favicon_path) as image:
        favicon = image.convert("RGBA")
        assert favicon.size == (64, 64)
        alpha = favicon.getchannel("A")
        painted = alpha.getbbox()
        assert painted == (4, 0, 60, 64)
        assert alpha.getextrema()[0] == 0
        svg_width, svg_height = (float(value) for value in favicon_document.attrib["viewBox"].split()[2:])
        painted_width = painted[2] - painted[0]
        painted_height = painted[3] - painted[1]
        assert abs((painted_width / painted_height) - (svg_width / svg_height)) < 0.02
        assert not any(pixel[:3] == (234, 228, 216) for _count, pixel in favicon.getcolors(64 * 64) or [])

    touch_url = urlparse(urljoin(base, touch_link["href"]))
    manifest_url = urlparse(urljoin(base, manifest_link["href"]))
    _touch_headers, touch_png = _get(touch_url.path + f"?{touch_url.query}")
    _manifest_headers, manifest_body = _get(manifest_url.path + f"?{manifest_url.query}")
    manifest = json.loads(manifest_body)
    pwa_url = urlparse(urljoin(base + "manifest.webmanifest", manifest["icons"][0]["src"]))
    _pwa_headers, pwa_png = _get(pwa_url.path)

    for name, data, expected_size in (("touch", touch_png, (180, 180)), ("pwa", pwa_png, (512, 512))):
        image_path = tmp_path / f"{name}.png"
        image_path.write_bytes(data)
        with Image.open(image_path) as image:
            rgba = image.convert("RGBA")
            assert rgba.size == expected_size
            assert rgba.getpixel((0, 0)) == (234, 228, 216, 255)
            assert rgba.getpixel((expected_size[0] - 1, expected_size[1] - 1)) == (234, 228, 216, 255)


def test_installed_brand_assets_are_versioned_opaque_and_resolve_from_served_shell(tmp_path: Path) -> None:
    _html_headers, html = _get("/")
    parser = _HeadLinks()
    parser.feed(html.decode("utf-8"))
    version = static_asset_version(STATIC_DIR)

    favicon_link = _link(parser.links, "icon")
    touch_link = _link(parser.links, "apple-touch-icon")
    manifest_link = _link(parser.links, "manifest")
    assert favicon_link["type"] == "image/svg+xml"
    assert touch_link["sizes"] == "180x180"

    base = "http://codoxear.test/"
    for link in (favicon_link, touch_link, manifest_link):
        parsed = urlparse(urljoin(base, link["href"]))
        assert parse_qs(parsed.query) == {"v": [version]}
        _headers, body = _get(parsed.path + (f"?{parsed.query}" if parsed.query else ""))
        assert body

    source_headers, source_body = _get(f"/codoxear-icon.svg?v={version}")
    assert ("Content-Type", "image/svg+xml; charset=utf-8") in source_headers.headers
    assert source_body

    _manifest_headers, manifest_body = _get(urlparse(urljoin(base, manifest_link["href"])).path)
    manifest = json.loads(manifest_body)
    assert manifest["background_color"] == "#eae4d8"
    assert manifest["theme_color"] == "#eae4d8"
    assert len(manifest["icons"]) == 1
    assert "purpose" not in manifest["icons"][0]

    icon_url = urlparse(urljoin(base + "manifest.webmanifest", manifest["icons"][0]["src"]))
    _icon_headers, icon_body = _get(icon_url.path)
    assets = {
        "apple-touch": (urlparse(urljoin(base, touch_link["href"])).path, (180, 180)),
        "manifest": (icon_url.path, (512, 512)),
    }
    for name, (path, expected_size) in assets.items():
        _headers, body = _get(path)
        image_path = tmp_path / f"{name}.png"
        image_path.write_bytes(body)
        with Image.open(image_path) as image:
            rgba = image.convert("RGBA")
            assert rgba.size == expected_size
            assert rgba.getpixel((0, 0)) == (234, 228, 216, 255)
            assert rgba.getpixel((expected_size[0] - 1, expected_size[1] - 1)) == (234, 228, 216, 255)
            assert rgba.getextrema()[3] == (255, 255)

    assert icon_body


def test_canonical_brand_source_and_installed_derivatives_are_package_version_inputs() -> None:
    required = {
        "codoxear-icon.svg",
        "codoxear-icon.png",
        "favicon.svg",
        "favicon.png",
        "apple-touch-icon.png",
    }
    assert required.issubset(STATIC_ASSET_VERSION_FILES)
    for name in required:
        assert (STATIC_DIR / name).is_file()
