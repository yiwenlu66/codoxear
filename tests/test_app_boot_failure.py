from __future__ import annotations

import json
import subprocess
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "codoxear" / "static" / "app.js"


def test_post_login_render_failure_replaces_live_root_and_logs_error() -> None:
    app_source = APP_JS.read_text(encoding="utf-8")
    app_source = app_source.replace(
        'import { createApplicationController } from "./app_application.js";\n',
        """
        const createApplicationController = () => ({
          api: async () => { throw { status: 401 }; },
          renderApp: () => { throw new ReferenceError("missing render dependency"); },
          renderLogin: (onAuthed) => {
            const login = document.createElement("form");
            login.setAttribute("id", "loginForm");
            document.getElementById("root").appendChild(login);
            onAuthed();
          },
        });
        """,
    ).replace(
        "  (async function boot() {",
        "  globalThis.bootPromise = (async function boot() {",
    )
    program = textwrap.dedent(
        f"""
        const vm = require("vm");
        class Element {{
          constructor(tagName) {{
            this.tagName = tagName;
            this.attributes = {{}};
            this.children = [];
            this.style = {{}};
            this.textContent = "";
          }}
          setAttribute(name, value) {{ this.attributes[name] = String(value); }}
          appendChild(child) {{ this.children.push(child); return child; }}
          replaceChildren(...children) {{ this.children = children; this.textContent = ""; }}
          set innerHTML(value) {{ this.children = []; this.textContent = String(value || ""); }}
          get innerHTML() {{ return this.textContent; }}
        }}
        const root = new Element("main");
        root.setAttribute("id", "root");
        const body = new Element("body");
        body.appendChild(root);
        const errors = [];
        const context = {{
          window: {{}},
          document: {{
            body,
            createElement: (tagName) => new Element(tagName),
            getElementById: (id) => id === "root" ? root : null,
          }},
          console: {{ error: (...args) => errors.push(args) }},
          Promise,
        }};
        context.globalThis = context;
        vm.createContext(context);
        vm.runInContext({json.dumps(app_source)}, context);
        context.bootPromise.then(() => {{
          const panel = root.children[0];
          process.stdout.write(JSON.stringify({{
            errors: errors.map((args) => args.map((value) => value && value.message ? value.message : String(value))),
            panel: panel && {{
              marker: panel.attributes["data-codoxear-boot-error"],
              role: panel.attributes.role,
              text: panel.textContent,
            }},
          }}));
        }});
        """
    )

    result = subprocess.run(
        ["node", "-e", program],
        check=True,
        capture_output=True,
        text=True,
    )
    observed = json.loads(result.stdout)

    assert observed["errors"] == [["application bootstrap failed after login", "missing render dependency"]]
    assert observed["panel"] == {
        "marker": "true",
        "role": "alert",
        "text": "Codoxear failed to start\n\nmissing render dependency\n\nReload to retry.",
    }
