"""Cwd dropdown suffix-preserving labels: DOM structure and stylesheet contract.

Long working-directory paths used to clip with a trailing ellipsis, hiding the
distinguishing leaf behind shared ``/home/...`` prefixes. The fix is a
presentation-only label shape: ``.fileMenuPath`` keeps its flex-item role, an
inline-block ``.cwdPathClip`` becomes the RTL overflow box (inline-end edge =
left, so the ellipsis leads and the path's end stays visible and short paths
shrink to content width on the left), and the full path text lives in a
``bdi[dir=ltr]`` child so the Unicode bidi algorithm resolves it in natural
LTR order exactly as rendered everywhere else.

These tests pin behavior, not source text: the DOM tests execute the real
``renderRecentCwdMenu`` through the real element factory against a stub
document, and the CSS tests parse the stylesheet and assert computed
declarations per selector.
"""

from __future__ import annotations

import json
import subprocess
import textwrap
import unittest
from pathlib import Path

from tinycss2 import parse_rule_list

from css_tokens import APP_CSS, THEMES_DIR, declarations, parse_stylesheet, selector_text

from frontend_module_loader import module_path

ROOT = Path(__file__).resolve().parents[1]
DISPLAY = module_path("app_display.js")
LAUNCH = module_path("app_launch.js")
SESSION_CATALOG = module_path("app_session_catalog.js")
APPLICATION = module_path("app_application.js")
NEW_SESSION = module_path("app_new_session.js")

MIXED_PATH = "/tmp/shared-prefix/共享-ws/проект-42/عربي-leaf/project-final"


def eval_cwd_menu_dom(recent_cwds: list[str], *, browse_anchor: str | None = None) -> dict:
    """Render the real cwd menu against a stub document and return its tree.

    ``browse_anchor`` mirrors what the dialog reports after the picker cleared a
    committed path (browse mode); ``None`` is the plain empty-field menu.
    """
    fixtures = [str(path) for path in (DISPLAY, LAUNCH, SESSION_CATALOG, APPLICATION, NEW_SESSION)]
    script = textwrap.dedent(
        f"""
        const vm = require("vm");
        const sources = {json.dumps(fixtures)}.map((path) => require("fs").readFileSync(path, "utf8"));
        function makeNode(tag) {{
          return {{
            tag, className: "", textContent: "", innerHTML: "", attrs: {{}}, children: [], value: "", placeholder: "", style: {{}},
            onclick: null, onmousedown: null,
            setAttribute(k, v) {{ this.attrs[k] = String(v); }},
            getAttribute(k) {{ return Object.prototype.hasOwnProperty.call(this.attrs, k) ? this.attrs[k] : null; }},
            removeAttribute(k) {{ delete this.attrs[k]; }},
            appendChild(child) {{ this.children.push(child); return child; }},
            classList: {{ toggle() {{}}, remove() {{}} }},
          }};
        }}
        const document = {{ createElement: makeNode }};
        const storage = {{ data:new Map(), getItem(k) {{ return this.data.get(k)||null; }}, setItem(k,v) {{this.data.set(k,String(v));}}, removeItem(k) {{this.data.delete(k);}} }};
        const ctx = {{
          URL, document, setTimeout: () => 0, clearTimeout: () => {{}},
          window: {{ location: {{ href: "http://localhost/", origin: "http://localhost", pathname: "/" }},
                     CodoxearUrls: {{resolveAppUrl:(p)=>String(p)}}, CodoxearStorage: storage }},
        }};
        vm.createContext(ctx); for (const source of sources) vm.runInContext(source,ctx);
        const noop=()=>{{}};
        const cwdMenu=makeNode("div"), cwdField=makeNode("div"), cwdHint=makeNode("div");
        const inputEvents={{focus:0,selection:null,aria:null}};
        const cwdInput={{value:"",focus(){{inputEvents.focus++;}},setSelectionRange(a,b){{inputEvents.selection=[a,b];}},
          setAttribute(k,v){{ if(k==="aria-activedescendant") inputEvents.aria=String(v); }}, removeAttribute(){{}}}};
        let menuFocus=-1;
        const sessionCatalog=ctx.window.CodoxearSessionCatalog.createSessionCatalog({{consoleError:()=>{{}}}});
        sessionCatalog.set("newSessionDefaults",{{model:"m",models:[],provider_choices:["chatgpt"],model_providers:["chatgpt"],provider_models:{{}},reasoning_efforts:["high"],reasoning_efforts_by_model:{{}}}});
        sessionCatalog.set("latestSessions",[]); sessionCatalog.set("recentCwds",{json.dumps(recent_cwds)}); sessionCatalog.set("tmuxAvailable",true);
        const c=ctx.window.CodoxearNewSession.createNewSessionController({{
          backend:()=>"codex", provider:()=>"chatgpt", reasoningEffort:()=>"high", literalModelInputValue:()=>"", launchPresetProviderAbsent:()=>false,
          sessionCatalog,
          assignProvider:noop, assignReasoningEffort:noop, assignLiteralModelInputValue:noop, assignLaunchPresetProviderAbsent:noop,
          modelInput:{{value:""}}, modelField:makeNode("div"), status:makeNode("div"), reasoningBtn:makeNode("div"),
          setPickerButtonContent:noop, renderReasoningMenu:noop, renderModelMenu:noop, setFast:noop, setBackend:noop, setTmuxChecked:noop,
          applyDialogMenus:noop, closeModelMenu:noop,
          cwdInput, cwdMenu, cwdField, cwdHint, nameInput:makeNode("input"),
          cwdMenuFocus:()=>menuFocus, assignCwdMenuFocus:(v)=>{{menuFocus=v;}}, closeCwdMenu:noop,
          cwdBrowseAnchor:()=>{json.dumps(browse_anchor)},
          el:(tag, attrs=[], children=[]) => ctx.window.CodoxearDom.createElement(tag, attrs, children, null),
          resumeMenu:makeNode("div"), resumeBtn:makeNode("div"), closeResumeMenu:noop, fetchResumeCandidates:async()=>({{sessions:[]}}),
          tmuxToggle:makeNode("input"), tmuxField:makeNode("div"), worktreeToggle:makeNode("input"),
          worktreeInput:makeNode("input"), worktreeField:makeNode("div"), startBtn:makeNode("button"),
        }});
        const items=c.renderRecentCwdMenu();
        const serialize=(node)=>({{tag:node.tag, className:node.className, attrs:{{...node.attrs}}, text:node.textContent,
          children:node.children.map(serialize), hasClick:typeof node.onclick==="function", hasMouseDown:typeof node.onmousedown==="function"}});
        // Selection semantics ride along with the label: click the mixed-script row.
        const mixedIndex=items.findIndex((item)=>item.cwd==={json.dumps(MIXED_PATH)});
        if (mixedIndex>=0) cwdMenu.children[mixedIndex].onclick();
        process.stdout.write(JSON.stringify({{
          rows:cwdMenu.children.map(serialize), itemCount:items.length, focusAssigned:menuFocus,
          inputAfterClick:cwdInput.value, inputEvents,
        }}));
        """
    )
    proc = subprocess.run(["node", "-e", script], check=True, text=True, capture_output=True, cwd=ROOT, timeout=30)
    return json.loads(proc.stdout)


class TestCwdMenuLabelDomBehavior(unittest.TestCase):
    def test_long_and_mixed_script_paths_render_inside_an_ltr_bdi_clip_box(self) -> None:
        recent = [MIXED_PATH, "/tmp/short", "/tmp/shared-prefix/averylongworkingdirectoryname-with-suffix-alpha"]
        result = eval_cwd_menu_dom(recent, browse_anchor=MIXED_PATH)
        rows = [row for row in result["rows"] if row["className"].startswith("fileMenuItem")]
        self.assertEqual(len(rows), len(recent))
        seen_paths = []
        for row in rows:
            path = row["children"][0]["children"][0]["children"][0]
            seen_paths.append(path["text"])
            # Label shape: fileMenuPath flex item > cwdPathClip RTL overflow box > bdi[dir=ltr].
            label, clip = row["children"][0], row["children"][0]["children"][0]
            self.assertEqual(label["tag"], "span")
            self.assertEqual(label["className"], "fileMenuPath")
            self.assertEqual(label["text"], "")
            self.assertEqual(len(label["children"]), 1)
            self.assertEqual(clip["tag"], "span")
            self.assertEqual(clip["className"], "cwdPathClip")
            self.assertEqual(len(clip["children"]), 1)
            self.assertEqual(path["tag"], "bdi")
            self.assertEqual(path["attrs"].get("dir"), "ltr")
        # The full path reaches the bdi verbatim for every row: no reversal,
        # slicing, or JS-side truncation of mixed-script strings.
        self.assertEqual(set(seen_paths), set(recent))
        # Full path survives on the hover target.
        for row, expected in zip(rows, recent):
            self.assertEqual(row["attrs"]["title"], expected)
            self.assertEqual(row["attrs"]["role"], "option")

    def test_clicking_a_label_selects_the_full_mixed_script_path(self) -> None:
        result = eval_cwd_menu_dom([MIXED_PATH, "/tmp/short"], browse_anchor=MIXED_PATH)
        self.assertEqual(result["inputAfterClick"], MIXED_PATH)
        self.assertEqual(result["inputEvents"]["selection"], [len(MIXED_PATH), len(MIXED_PATH)])
        self.assertEqual(result["inputEvents"]["focus"], 1)
        # Browse mode focuses the restored anchor row and reports it to the input.
        self.assertEqual(result["focusAssigned"], 0)
        self.assertEqual(result["inputEvents"]["aria"], "newSessionCwdOption-0")
        anchor_row = result["rows"][0]
        self.assertIn("active", anchor_row["className"])
        self.assertEqual(anchor_row["attrs"]["aria-selected"], "true")

    def test_recent_badge_stays_a_sibling_of_the_path_label(self) -> None:
        result = eval_cwd_menu_dom(["/tmp/short"])
        row = next(r for r in result["rows"] if r["className"].startswith("fileMenuItem"))
        label, badge = row["children"][0], row["children"][1]
        self.assertEqual(label["className"], "fileMenuPath")
        self.assertEqual(badge["className"], "cwdSuggestionSource")
        self.assertEqual(badge["text"], "Recent")
        self.assertEqual(badge["children"], [])
        self.assertTrue(row["hasClick"])
        self.assertTrue(row["hasMouseDown"])

    def test_empty_menu_stays_plain_without_clip_boxes(self) -> None:
        result = eval_cwd_menu_dom([])
        self.assertEqual(len(result["rows"]), 1)
        empty = result["rows"][0]
        self.assertEqual(empty["className"], "pickerEmpty")
        self.assertEqual(empty["text"], "No recent directories")
        self.assertEqual(empty["children"], [])


def _rule_entries(path: Path) -> list[tuple[str, dict[str, str], bool]]:
    """(selector, declarations, inside_at_rule) for every qualified rule."""
    entries: list[tuple[str, dict[str, str], bool]] = []

    def walk(rules: list[object], nested: bool) -> None:
        for rule in rules:
            if getattr(rule, "type", None) == "qualified-rule":
                entries.append((selector_text(rule), declarations(rule), nested))
            elif getattr(rule, "type", None) == "at-rule" and rule.content:
                walk(parse_rule_list(rule.content, skip_comments=True, skip_whitespace=True), True)

    walk(parse_stylesheet(path), nested=False)
    return entries


class TestCwdMenuLabelStylesheetContract(unittest.TestCase):
    def test_clip_box_carries_the_full_leading_ellipsis_mechanism(self) -> None:
        entries = [entry for entry in _rule_entries(APP_CSS) if entry[0] == ".cwdPathClip"]
        self.assertEqual(len(entries), 1, f"expected exactly one .cwdPathClip rule, got {entries}")
        _, decls, nested = entries[0]
        self.assertFalse(nested, "clip rule must apply at every viewport, not inside a media query")
        self.assertEqual(decls["direction"], "rtl")
        self.assertEqual(decls["display"], "inline-block")
        self.assertEqual(decls["vertical-align"], "top")
        self.assertEqual(decls["max-width"], "100%")
        self.assertEqual(decls["overflow"], "hidden")
        self.assertEqual(decls["text-overflow"], "ellipsis")
        self.assertEqual(decls["white-space"], "nowrap")

    def test_flex_path_label_never_becomes_the_rtl_box(self) -> None:
        # direction:rtl directly on .fileMenuPath would right-align short paths
        # (RTL line-box start edge) and break the "short paths unchanged" rule;
        # the RTL box must be the width-capped inline-block clip span instead.
        entries = [entry for entry in _rule_entries(APP_CSS) if entry[0] == ".fileMenuPath"]
        self.assertTrue(entries)
        computed: dict[str, str] = {}
        for _, decls, _ in entries:
            computed.update(decls)
        self.assertNotIn("direction", computed)
        # The flex item must keep shrinking so the badge stays pinned and the
        # clip box has a bounded width to clip against.
        self.assertEqual(computed["min-width"], "0")
        self.assertEqual(computed["flex"], "1 1 auto")

    def test_no_theme_rule_restyles_the_cwd_label_mechanism(self) -> None:
        for theme in sorted(THEMES_DIR.glob("*.css")):
            for selector, decls, _ in _rule_entries(theme):
                self.assertNotIn(
                    "cwdPathClip",
                    selector,
                    f"{theme.name} styles .cwdPathClip; the clip mechanism is family-invariant geometry",
                )
                if ".fileMenuPath" in selector:
                    self.assertNotIn("direction", decls, f"{theme.name} overrides .fileMenuPath direction")
