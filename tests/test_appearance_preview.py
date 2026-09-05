from frontend_module_loader import module_path
import json
import subprocess
import textwrap


def run_preview_harness() -> dict:
    sources = {
        name: module_path(name).read_text(encoding="utf-8")
        for name in [
            "app_session_state.js",
            "app_sessions.js",
            "app_message_rows.js",
            "app_shell.js",
            "app_appearance_preview.js",
        ]
    }
    script = textwrap.dedent(
        f"""
        const vm = require("vm");
        const sources = {json.dumps(sources)};

        function makeClassList(node, initial) {{
          const values = new Set(String(initial || "").split(/\\s+/).filter(Boolean));
          return {{
            add(...names) {{ names.forEach((name) => values.add(name)); }},
            remove(...names) {{ names.forEach((name) => values.delete(name)); }},
            toggle(name, force) {{ const next = force === undefined ? !values.has(name) : Boolean(force); if (next) values.add(name); else values.delete(name); return next; }},
            contains(name) {{ return values.has(name); }},
            values,
          }};
        }}
        class Node {{
          constructor(tag, attrs = {{}}) {{
            this.tag = tag; this.attrs = {{ ...attrs }}; this.children = []; this.parentNode = null;
            this.dataset = {{}};
            for (const [name, value] of Object.entries(attrs)) {{
              if (name.startsWith("data-")) this.dataset[name.slice(5).replace(/-([a-z])/g, (_, letter) => letter.toUpperCase())] = String(value);
            }}
            this.style = {{}}; this.classList = makeClassList(this, attrs.class); this.onclick = null;
            this.textContent = attrs.text || ""; this.innerHTML = attrs.html || "";
          }}
          appendChild(child) {{ if (!child) return child; child.parentNode = this; this.children.push(child); return child; }}
          append(...children) {{ children.forEach((child) => this.appendChild(child)); }}
          setAttribute(name, value) {{ this.attrs[name] = String(value); }}
          getAttribute(name) {{ return this.attrs[name]; }}
          removeAttribute(name) {{ delete this.attrs[name]; }}
          querySelectorAll() {{ return []; }}
          get childElementCount() {{ return this.children.length; }}
          set innerHTML(value) {{ this._innerHTML = value; if (value === "") this.children = []; }}
          get innerHTML() {{ return this._innerHTML || ""; }}
        }}
        const el = (tag, attrs = {{}}, children = []) => {{
          const node = new Node(tag, attrs);
          for (const child of children) node.appendChild(child);
          return node;
        }};
        function walk(root, predicate, found = []) {{
          if (predicate(root)) found.push(root);
          for (const child of root.children) walk(child, predicate, found);
          return found;
        }}
        const ctx = {{ window: {{}}, console }};
        vm.createContext(ctx);
        for (const name of ["app_session_state.js", "app_sessions.js", "app_message_rows.js", "app_shell.js", "app_appearance_preview.js"]) {{
          vm.runInContext(sources[name], ctx);
        }}
        const preview = ctx.window.CodoxearAppearancePreview.createAppearancePreview({{
          el,
          iconSvg: (name) => `<svg data-icon="${{name}}"></svg>`,
          chatMarkdownHtmlCached: (text) => `<p>${{text}}</p>`,
        const root = preview.element;
        const canvas = walk(root, (node) => node.classList.contains("appearancePreviewCanvas"))[0];
        const cards = walk(root, (node) => node.classList.contains("session"));
        const rows = walk(root, (node) => node.classList.contains("msg-row"));
        const composer = walk(root, (node) => node.classList.contains("composer"))[0];
        const buttons = walk(root, (node) => node.classList.contains("choiceChip"));
        const before = canvas.dataset.appearance;
        const warm = buttons.find((node) => node.textContent === "Warm Editorial");
        warm.onclick();
        const afterWarm = {{
          canvas: canvas.dataset.appearance,
          warmPressed: warm.attrs["aria-pressed"],
          paperPressed: buttons.find((node) => node.textContent === "Paper").attrs["aria-pressed"],
        }};
        const terminal = buttons.find((node) => node.textContent === "Terminal Night");
        terminal.onclick();
        const afterTerminal = {{
          canvas: canvas.dataset.appearance,
          terminalPressed: terminal.attrs["aria-pressed"],
          activeButtons: buttons.filter((node) => node.classList.contains("active")).map((node) => node.textContent),
        }};
        preview.dispose();
        process.stdout.write(JSON.stringify({{
          before, cardCount: cards.length, rowRoles: rows.map((node) => node.dataset.role),
          composerTag: composer && composer.tag, composerTextareaCount: walk(composer, (node) => node.tag === "textarea").length,
          choices: buttons.map((node) => node.textContent), afterWarm, afterTerminal,
          handlersCleared: buttons.every((node) => node.onclick === null),
        }}));
        """
    )
    result = subprocess.run(["node", "-e", script], check=True, text=True, capture_output=True)
    return json.loads(result.stdout)


def test_appearance_preview_reuses_fixture_components_and_scopes_switching() -> None:
    result = run_preview_harness()

    assert result["before"] == "paper"
    assert result["cardCount"] == 2
    assert result["rowRoles"] == ["assistant", "user"]
    assert result["composerTag"] == "div"
    assert result["composerTextareaCount"] == 1
    assert result["choices"] == ["Paper", "Soft Light", "Warm Editorial", "Terminal Night"]
    assert result["afterWarm"] == {"canvas": "warm-editorial", "warmPressed": "true", "paperPressed": "false"}
    assert result["afterTerminal"] == {"canvas": "terminal-night", "terminalPressed": "true", "activeButtons": ["Terminal Night"]}
    assert result["handlersCleared"] is True
