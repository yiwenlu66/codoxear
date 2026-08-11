from frontend_module_loader import module_path
import json
import subprocess
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_MARKDOWN_JS = module_path("app_markdown.js")


def run_image_layout_probe(program: str) -> dict[str, object]:
    source = APP_MARKDOWN_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const source = {json.dumps(source)};
        const values = new Map();
        const localStorage = {{
          getItem(key) {{ return values.has(key) ? values.get(key) : null; }},
          setItem(key, value) {{ values.set(key, String(value)); }},
        }};
        function makeImage() {{
          const attributes = new Map();
          return {{
            style: {{}},
            dataset: {{}},
            naturalWidth: 0,
            naturalHeight: 0,
            setAttribute(name, value) {{ attributes.set(name, String(value)); }},
            removeAttribute(name) {{ attributes.delete(name); }},
            getAttribute(name) {{ return attributes.get(name) || ""; }},
            matches(selector) {{ return selector === "img[data-codoxear-image-key]" && Boolean(this.dataset.codoxearImageKey); }},
          }};
        }}
        function makeContext() {{
          const listeners = {{}};
          const document = {{
            addEventListener(name, listener) {{ listeners[name] = listener; }},
          }};
          const ctx = {{
            URL,
            location: {{ origin: "http://localhost", href: "http://localhost/" }},
            document,
            window: {{ localStorage, CodoxearUrls: {{ resolveAppUrl: (path) => path }} }},
          }};
          vm.createContext(ctx);
          vm.runInContext(source, ctx);
          return {{ api: ctx.window.CodoxearMarkdown, listeners }};
        }}
        {program}
        """
    )
    proc = subprocess.run(["node", "-e", js], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


def test_image_dimension_cache_reserves_default_then_persists_natural_dimensions_across_renders() -> None:
    result = run_image_layout_probe(
        """
        const first = makeContext();
        const unknown = makeImage();
        first.api.prepareImageForDisplay(unknown, "/api/sessions/s/file/blob?path=%2Fuploads%2Fs%2Fchart.png");
        unknown.naturalWidth = 640;
        unknown.naturalHeight = 480;
        first.listeners.load({ target: unknown });

        const second = makeContext();
        const known = makeImage();
        second.api.prepareImageForDisplay(known, "/api/sessions/s/file/blob?path=%2Fuploads%2Fs%2Fchart.png");
        process.stdout.write(JSON.stringify({
          unknown: { aspect: unknown.style.aspectRatio, width: unknown.style.width, height: unknown.style.height },
          known: { aspect: known.style.aspectRatio, width: known.style.width, height: known.style.height, attrWidth: known.getAttribute("width"), attrHeight: known.getAttribute("height") },
        }));
        """
    )

    assert result["unknown"] == {"aspect": "640 / 480", "width": "100%", "height": "auto"}
    assert result["known"] == {"aspect": "640 / 480", "width": "100%", "height": "auto", "attrWidth": "640", "attrHeight": "480"}


def test_cached_natural_dimensions_are_pinned_before_hydration_assigns_image_src() -> None:
    result = run_image_layout_probe(
        """
        const runtime = makeContext();
        const blobSource = "/api/sessions/s/file/blob?path=%2Fuploads%2Fs%2Fchart.png";
        localStorage.setItem("codoxear.image-dimensions.v1", JSON.stringify({
          [blobSource]: { width: 640, height: 480, usedAt: 1 },
        }));
        const image = makeImage();
        image.dataset.codoxearImageSrc = blobSource;
        image.dataset.codoxearImageDimensionsUrl = "/api/sessions/s/file/image-dimensions?path=%2Fuploads%2Fs%2Fchart.png";
        image.removeAttribute = () => {};
        const srcAssignments = [];
        Object.defineProperty(image, "src", {
          set(value) {
            const entry = JSON.parse(localStorage.getItem("codoxear.image-dimensions.v1"))[blobSource];
            srcAssignments.push({
              value,
              cache: { width: entry.width, height: entry.height },
              aspect: image.style.aspectRatio,
              width: image.getAttribute("width"),
              height: image.getAttribute("height"),
            });
          },
        });
        runtime.api.hydrateMarkedImages({
          querySelectorAll(selector) {
            return selector === "img[data-codoxear-image-src][data-codoxear-image-dimensions-url]" ? [image] : [];
          },
        });
        process.stdout.write(JSON.stringify(srcAssignments));
        """
    )

    assert result == [
        {
            "value": "/api/sessions/s/file/blob?path=%2Fuploads%2Fs%2Fchart.png",
            "cache": {"width": 640, "height": 480},
            "aspect": "640 / 480",
            "width": "640",
            "height": "480",
        }
    ]


def test_image_dimension_cache_uses_default_box_and_evicts_least_recently_used_entry() -> None:
    result = run_image_layout_probe(
        """
        const runtime = makeContext();
        const unknown = makeImage();
        runtime.api.prepareImageForDisplay(unknown, "https://example.test/new.png");
        for (let i = 0; i <= 500; i++) {
          const image = makeImage();
          runtime.api.prepareImageForDisplay(image, `https://example.test/${i}.png`);
          image.naturalWidth = 100 + i;
          image.naturalHeight = 50;
          runtime.listeners.load({ target: image });
        }
        const entries = JSON.parse(localStorage.getItem("codoxear.image-dimensions.v1"));
        process.stdout.write(JSON.stringify({
          defaultAspect: unknown.style.aspectRatio,
          defaultWidth: unknown.style.width,
          defaultHeight: unknown.style.height,
          size: Object.keys(entries).length,
          containsOldest: Boolean(entries["https://example.test/0.png"]),
          containsNewest: Boolean(entries["https://example.test/500.png"]),
        }));
        """
    )

    assert result == {
        "defaultAspect": "16 / 9",
        "defaultWidth": "100%",
        "defaultHeight": "auto",
        "size": 500,
        "containsOldest": False,
        "containsNewest": True,
    }
