import json
import subprocess
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_DISPLAY_JS = ROOT / "codoxear" / "static" / "app_display.js"
APP_FILE_HELPERS_JS = ROOT / "codoxear" / "static" / "app_file_helpers.js"
APP_FILE_PICKER_JS = ROOT / "codoxear" / "static" / "app_file_picker.js"


def run_picker_module_probe() -> dict[str, object]:
    display_source = APP_DISPLAY_JS.read_text(encoding="utf-8")
    helper_source = APP_FILE_HELPERS_JS.read_text(encoding="utf-8")
    picker_source = APP_FILE_PICKER_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}} }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(display_source)}, ctx);
        vm.runInContext({json.dumps(helper_source)}, ctx);
        vm.runInContext({json.dumps(picker_source)}, ctx);
        const picker = ctx.window.CodoxearFilePicker;
        let missingError = "";
        const missingCtx = {{ window: {{}} }};
        vm.createContext(missingCtx);
        try {{
          vm.runInContext({json.dumps(picker_source)}, missingCtx);
        }} catch (err) {{
          missingError = err && err.message ? err.message : String(err);
        }}
        let hostError = "";
        try {{
          picker.createSearchState({{}});
        }} catch (err) {{
          hostError = err && err.message ? err.message : String(err);
        }}
        let menuHostError = "";
        try {{
          picker.createMenuState({{}});
        }} catch (err) {{
          menuHostError = err && err.message ? err.message : String(err);
        }}
        const menu = picker.createMenuState({{
          normalizeLineNumber: (value) => {{
            const n = Number(value);
            return Number.isFinite(n) && n >= 1 ? Math.floor(n) : null;
          }},
        }});
        const opened = menu.openSearchQuery("src/app.js", {{ line: "42", suppressDraft: true }});
        const selectionBeforeInput = menu.selectionLine("src/app.js");
        const draftSuppressedBeforeInput = menu.draftSuppressed("src/app.js");
        menu.setPreserveSearchOnFocus(true);
        const preservedFirst = menu.takePreservedSearchOnFocus();
        const preservedSecond = menu.takePreservedSearchOnFocus();
        menu.handleInput("src/other.js");
        const selectionAfterInput = menu.selectionLine("src/other.js");
        const draftSuppressedAfterInput = menu.draftSuppressed("src/other.js");
        const visibleQueryAfterInput = menu.visibleQuery(" src/other.js ");
        menu.setFocus(10);
        const clampedFocus = menu.clampFocus(2);
        const movedFocus = menu.moveFocus(2, 1);
        const enterIndex = menu.enterIndex();
        const closed = menu.close();
        const domEvents = [];
        function classNode(name) {{
          return {{
            name,
            classes: {{}},
            classList: {{ toggle(className, enabled) {{ domEvents.push(["toggle", name, className, Boolean(enabled)]); if (enabled) this.owner.classes[className] = true; else delete this.owner.classes[className]; }} }},
          }};
        }}
        const field = classNode("field");
        const menuNode = classNode("menu");
        field.classList.owner = field;
        menuNode.classList.owner = menuNode;
        const input = {{
          value: "typed",
          attrs: {{}},
          setAttribute(name, value) {{ this.attrs[name] = String(value); domEvents.push(["setAttr", name, String(value)]); }},
          removeAttribute(name) {{ delete this.attrs[name]; domEvents.push(["removeAttr", name]); }},
        }};
        const domMenu = picker.createMenuState({{ normalizeLineNumber: (value) => Number(value) || null }});
        const domRuntime = picker.createMenuDomRuntime({{ field, menu: menuNode, input, menuState: domMenu }});
        domMenu.openSearchQuery("src/app.js", {{ line: 12 }});
        const openDomState = domRuntime.apply();
        const openDomSnapshot = {{ fieldActive: Boolean(field.classes.active), menuOpen: Boolean(menuNode.classes.open), expanded: input.attrs["aria-expanded"] }};
        const resetDomState = domRuntime.resetInput("src/current.py");
        const resetDomSnapshot = {{ value: input.value, activeDescendant: input.attrs["aria-activedescendant"] || "", focus: resetDomState.focus }};
        domMenu.openSearchQuery("next.js", {{ line: 9 }});
        input.setAttribute("aria-activedescendant", "filePickerOption-0");
        const closeDomState = domRuntime.close({{ restoreInput: true, inputValue: "restored.py" }});
        const closeDomSnapshot = {{ fieldActive: Boolean(field.classes.active), menuOpen: Boolean(menuNode.classes.open), expanded: input.attrs["aria-expanded"], activeDescendant: input.attrs["aria-activedescendant"] || "", value: input.value, focus: closeDomState.focus }};
        const syncPositive = domRuntime.syncActiveDescendant(3);
        const syncPositiveValue = input.attrs["aria-activedescendant"] || "";
        const syncNegative = domRuntime.syncActiveDescendant(-1);
        const syncNegativeValue = input.attrs["aria-activedescendant"] || "";
        let domHostError = "";
        try {{ picker.createMenuDomRuntime({{ field, menu: menuNode, input }}); }} catch (err) {{ domHostError = err && err.message ? err.message : String(err); }}
        function el(tag, attrs = {{}}) {{
          return {{ tag, attrs, textContent: "", children: [], appendChild(child) {{ this.children.push(child); return child; }} }};
        }}
        function createTextNode(text) {{ return {{ tag: "#text", text: String(text) }}; }}
        const highlightParent = {{ children: [], appendChild(child) {{ this.children.push(child); return child; }} }};
        const highlighted = picker.appendHighlightedFileMenuPath(highlightParent, "src/foo_bar.py", "foo", {{ el, createTextNode }});
        const plainParent = {{ children: [], appendChild(child) {{ this.children.push(child); return child; }} }};
        const plain = picker.appendHighlightedFileMenuPath(plainParent, "README.md", "", {{ el, createTextNode }});
        let highlightHostError = "";
        try {{ picker.appendHighlightedFileMenuPath(highlightParent, "x", "x", {{ el }}); }} catch (err) {{ highlightHostError = err && err.message ? err.message : String(err); }}
        const menuParent = {{ children: [], appendChild(child) {{ this.children.push(child); return child; }} }};
        const sectionResult = picker.appendFilePickerSection(menuParent, "Changed files", {{ el }});
        const emptySectionResult = picker.appendFilePickerSection(menuParent, "", {{ el }});
        const statusParent = {{ children: [], appendChild(child) {{ this.children.push(child); return child; }} }};
        const statusRow = picker.appendFilePickerStatusRow(statusParent, "Searching full project...", {{ el }});
        let statusHostError = "";
        try {{ picker.appendFilePickerStatusRow(statusParent, "x", {{}}); }} catch (err) {{ statusHostError = err && err.message ? err.message : String(err); }}
        const draftEvents = [];
        const draftItem = picker.appendDraftFileMenuItem(menuParent, "draft/new.txt", 2, true, {{ el, openDraftFilePath: (path) => draftEvents.push(["openDraft", path]) }});
        let draftPrevented = 0;
        draftItem.onmousedown({{ preventDefault() {{ draftPrevented += 1; }} }});
        draftItem.onclick();
        let draftHostError = "";
        try {{ picker.appendDraftFileMenuItem(menuParent, "x", 0, false, {{ el }}); }} catch (err) {{ draftHostError = err && err.message ? err.message : String(err); }}
        const entryParent = {{ children: [], appendChild(child) {{ this.children.push(child); return child; }} }};
        const entryEvents = [];
        const entryHost = {{
          el,
          createTextNode,
          openDraftFilePath: (path) => entryEvents.push(["openDraft", path]),
          openEntry: (entry) => entryEvents.push(["openEntry", entry.path, Boolean(entry.changed), Boolean(entry.gitPath), entry.apiPath || ""]),
        }};
        const createEntryItem = picker.appendFilePickerEntryItem(entryParent, {{ path: "draft/new.txt", createNew: true }}, 0, false, "draft", "", "Create title", entryHost);
        const changedEntryItem = picker.appendFilePickerEntryItem(entryParent, {{ path: "src/app.py", changed: true, additions: null, deletions: 2, gitPath: true, apiPath: "tok" }}, 1, true, "app", "git version", "Changed title", entryHost);
        const regularEntryItem = picker.appendFilePickerEntryItem(entryParent, {{ path: "README.md" }}, 2, false, "", "recent file", "Regular title", entryHost);
        const workbenchEntryParent = {{ children: [], appendChild(child) {{ this.children.push(child); return child; }} }};
        const untrackedEntryItem = picker.appendFilePickerEntryItem(workbenchEntryParent, {{ path: "new.txt", untracked: true, gitPath: true, apiPath: "ut-tok" }}, 3, false, "", "", "Untracked title", entryHost);
        const renameEntryItem = picker.appendFilePickerEntryItem(workbenchEntryParent, {{ path: "moved.md", changed: true, additions: 0, deletions: 0, rename: true, oldPath: "orig.md", gitPath: true, apiPath: "rn-tok" }}, 4, false, "", "", "Rename title", entryHost);
        let entryPrevented = 0;
        changedEntryItem.onmousedown({{ preventDefault() {{ entryPrevented += 1; }} }});
        createEntryItem.onclick();
        changedEntryItem.onclick();
        regularEntryItem.onclick();
        let entryHostError = "";
        try {{ picker.appendFilePickerEntryItem(entryParent, {{ path: "x" }}, 0, false, "", "", "", {{ el, createTextNode }}); }} catch (err) {{ entryHostError = err && err.message ? err.message : String(err); }}
        // Non-repo git-state notice must render as an explicit status row in the
        // candidate (no-query) view rather than a silently empty changed-files list.
        const gitStatusParent = {{ children: [], appendChild(child) {{ this.children.push(child); return child; }} }};
        const gitStatusRow = picker.appendFilePickerGitStatusRow(gitStatusParent, "Not a git repository \u2014 no changed files", {{ el }});
        let gitStatusHostError = "";
        try {{ picker.appendFilePickerGitStatusRow(gitStatusParent, "x", {{}}); }} catch (err) {{ gitStatusHostError = err && err.message ? err.message : String(err); }}
        const renderMenuHost = {{ children: [], innerHTML: "", appendChild(child) {{ this.children.push(child); return child; }} }};
        const renderRuntime = picker.createMenuRenderRuntime({{
          menu: renderMenuHost,
          menuState: {{ visibleQuery: () => "", focusIndex: () => -1, clampFocus: () => -1 }},
          inputValue: () => "",
          visibleEntries: () => [],
          searchSnapshot: () => ({{}}),
          normalizeDraftFilePath: () => "",
          draftSuppressed: () => false,
          draftEntry: () => ({{ createNew: true }}),
          syncActiveDescendant: () => null,
          sectionLabel: () => "",
          duplicatePaths: () => new Set(),
          identityHint: () => "",
          titleForEntry: () => "",
          normalizeFileApiPath: (v) => v || "",
          activeIdentity: () => ({{ path: "", gitPath: false, apiPath: "" }}),
          gitStatusMessage: () => "Not a git repository \u2014 no changed files",
          openDraftFilePath: () => null,
          openEntry: () => null,
          el,
          createTextNode: (t) => ({{ tag: "#text", text: String(t) }}),
        }});
        renderRuntime.render();
        const renderedGitStatus = renderMenuHost.children.find((node) => (node.attrs && node.attrs.class || "").includes("fileMenuGitStatus"));
        function summarizeNode(node) {{
          const attrs = node.attrs || {{}};
          return {{
            tag: node.tag,
            cls: attrs.class || "",
            text: attrs.text || node.text || node.textContent || "",
            id: attrs.id || "",
            title: attrs.title || "",
            aria: attrs["aria-selected"] || "",
            children: (node.children || []).map(summarizeNode),
          }};
        }}
        process.stdout.write(JSON.stringify({{
          frozen: Object.isFrozen(picker),
          exports: Object.keys(picker).sort(),
          missingError,
          hostError,
          menuHostError,
          domHostError,
          highlightHostError,
          draftHostError,
          entryHostError,
          gitStatusHostError,
          untrackedEntry: summarizeNode(untrackedEntryItem),
          renameEntry: summarizeNode(renameEntryItem),
          gitStatusRow: summarizeNode(gitStatusRow),
          renderedGitStatus: renderedGitStatus ? summarizeNode(renderedGitStatus) : null,
          statusHostError,
          menuState: {{
            opened,
            selectionBeforeInput,
            draftSuppressedBeforeInput,
            preservedFirst,
            preservedSecond,
            selectionAfterInput,
            draftSuppressedAfterInput,
            visibleQueryAfterInput,
            clampedFocus,
            movedFocus,
            enterIndex,
            closed,
          }},
          domState: {{
            frozen: Object.isFrozen(domRuntime),
            openDomState,
            openDomSnapshot,
            resetDomSnapshot,
            closeDomSnapshot,
            syncPositive,
            syncPositiveValue,
            syncNegative,
            syncNegativeValue,
            domEvents,
          }},
          highlightState: {{
            parentChildCount: highlightParent.children.length,
            spanClass: highlighted.attrs.class,
            highlightedChildren: highlighted.children.map((child) => child.tag === "#text" ? ["text", child.text] : [child.tag, child.attrs.class || "", child.attrs.text || ""]),
            plainText: plain.textContent,
            plainParentChildCount: plainParent.children.length,
          }},
          menuRenderState: {{
            sectionResult,
            emptySectionResult,
            parentChildren: menuParent.children.map((child) => [child.tag, child.attrs.class || "", child.attrs.text || "", child.attrs.id || "", child.attrs["aria-selected"] || ""]),
            draftItemChildren: draftItem.children.map((child) => [child.tag, child.attrs.class || "", child.attrs.text || ""]),
            draftPrevented,
            draftEvents,
            statusRow: [statusRow.tag, statusRow.attrs.class || "", statusRow.attrs.text || ""],
            statusParentChildCount: statusParent.children.length,
          }},
          entryRenderState: {{
            rows: entryParent.children.map(summarizeNode),
            entryPrevented,
            entryEvents,
          }},
        }}));
        """
    )
    proc = subprocess.run(["node"], input=js, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


def run_picker_render_runtime_probe() -> dict[str, object]:
    display_source = APP_DISPLAY_JS.read_text(encoding="utf-8")
    helper_source = APP_FILE_HELPERS_JS.read_text(encoding="utf-8")
    picker_source = APP_FILE_PICKER_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}} }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(display_source)}, ctx);
        vm.runInContext({json.dumps(helper_source)}, ctx);
        vm.runInContext({json.dumps(picker_source)}, ctx);
        const picker = ctx.window.CodoxearFilePicker;
        function el(tag, attrs = {{}}) {{
          return {{ tag, attrs, textContent: "", children: [], onclick: null, onmousedown: null, appendChild(child) {{ this.children.push(child); return child; }} }};
        }}
        function createTextNode(text) {{ return {{ tag: "#text", text: String(text) }}; }}
        const menu = {{ innerHTML: "stale", children: [], appendChild(child) {{ this.children.push(child); return child; }} }};
        let query = "";
        let entries = [];
        let focus = -1;
        let suppressed = false;
        let snapshot = {{}};
        let active = {{ path: "", gitPath: false, apiPath: "" }};
        const syncs = [];
        const opened = [];
        const menuState = {{
          visibleQuery: () => query,
          focusIndex: () => focus,
          clampFocus(count) {{ if (focus >= count) focus = count ? count - 1 : -1; return focus; }},
        }};
        const runtime = picker.createMenuRenderRuntime({{
          menu,
          menuState,
          inputValue: () => query,
          visibleEntries: () => entries,
          searchSnapshot: () => snapshot,
          normalizeDraftFilePath: (value) => String(value || "").trim(),
          draftSuppressed: () => suppressed,
          draftEntry: (path) => ({{ path, createNew: true }}),
          syncActiveDescendant: (idx) => syncs.push(idx),
          sectionLabel: (source) => source === "changed" ? "Changed files" : source === "recent" ? "Recent files" : "",
          duplicatePaths: (items) => new Set(items.map((item) => item.path).filter((path, _idx, arr) => arr.indexOf(path) !== arr.lastIndexOf(path))),
          identityHint: (entry, duplicatePaths, options) => duplicatePaths.has(entry.path) ? (options.showSourceSections ? "duplicate" : "same path") : "",
          titleForEntry: (entry, hint) => hint ? `${{entry.path}} (${{hint}})` : entry.path,
          normalizeFileApiPath: (value) => String(value || ""),
          activeIdentity: () => active,
          openDraftFilePath: (path) => opened.push(["draft", path]),
          openEntry: (entry) => opened.push(["entry", entry.path, Boolean(entry.gitPath), entry.apiPath || ""]),
          el,
          createTextNode,
        }});
        function summarize(node) {{
          return {{
            tag: node.tag,
            cls: (node.attrs && node.attrs.class) || "",
            id: (node.attrs && node.attrs.id) || "",
            text: (node.attrs && node.attrs.text) || node.text || node.textContent || "",
            aria: (node.attrs && node.attrs["aria-selected"]) || "",
            title: (node.attrs && node.attrs.title) || "",
            children: (node.children || []).map(summarize),
          }};
        }}
        function resetMenu() {{ menu.innerHTML = "stale"; menu.children = []; syncs.length = 0; opened.length = 0; }}
        function runCase(next) {{
          resetMenu();
          Object.assign({{}}, next);
          query = next.query;
          entries = next.entries;
          focus = next.focus;
          suppressed = Boolean(next.suppressed);
          snapshot = next.snapshot || {{}};
          active = next.active || {{ path: "", gitPath: false, apiPath: "" }};
          const returned = runtime.render();
          return {{ returned, innerHTML: menu.innerHTML, rows: menu.children.map(summarize), syncs: syncs.slice(), opened: opened.slice() }};
        }}
        const pending = runCase({{ query: "draft/new.txt", entries: null, focus: 0, suppressed: false, snapshot: {{ pendingQuery: "draft/new.txt" }} }});
        const empty = runCase({{ query: "missing", entries: [], focus: -1, suppressed: true, snapshot: {{ errorQuery: "missing", error: "search failed" }} }});
        const populated = runCase({{
          query: "",
          focus: -1,
          active: {{ path: "README.md", gitPath: false, apiPath: "" }},
          entries: [
            {{ path: "src/app.py", source: "changed", changed: true, additions: 1, deletions: 2, gitPath: true, apiPath: "tok" }},
            {{ path: "README.md", source: "recent", gitPath: false, apiPath: "" }},
          ],
          snapshot: {{}},
        }});
        const footer = runCase({{ query: "app", focus: 0, entries: [{{ path: "src/app.py", source: "search" }}], snapshot: {{ pendingQuery: "app" }} }});
        process.stdout.write(JSON.stringify({{ pending, empty, populated, footer, frozen: Object.isFrozen(runtime) }}));
        """
    )
    proc = subprocess.run(["node"], input=js, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


def run_picker_raw_byte_collision_render_probe() -> dict[str, object]:
    display_source = APP_DISPLAY_JS.read_text(encoding="utf-8")
    helper_source = APP_FILE_HELPERS_JS.read_text(encoding="utf-8")
    picker_source = APP_FILE_PICKER_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}} }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(display_source)}, ctx);
        vm.runInContext({json.dumps(helper_source)}, ctx);
        vm.runInContext({json.dumps(picker_source)}, ctx);
        const picker = ctx.window.CodoxearFilePicker;
        const helpers = ctx.window.CodoxearFileHelpers;
        function el(tag, attrs = {{}}) {{
          return {{ tag, attrs, textContent: "", children: [], onclick: null, onmousedown: null, appendChild(child) {{ this.children.push(child); return child; }} }};
        }}
        function createTextNode(text) {{ return {{ tag: "#text", text: String(text) }}; }}
        const menu = {{ innerHTML: "", children: [], appendChild(child) {{ this.children.push(child); return child; }} }};
        // Two non-git entries share the JSON-safe display path; one carries a
        // tokenized apiPath (raw bytes), the other is a literal/display name.
        const entries = [
          {{ path: "bad\\xffname.txt", source: "recent", gitPath: false, apiPath: "tok-raw" }},
          {{ path: "bad\\xffname.txt", source: "mentioned", gitPath: false, apiPath: "" }},
        ];
        const menuState = {{
          visibleQuery: () => "",
          focusIndex: () => -1,
          clampFocus: () => -1,
        }};
        const runtime = picker.createMenuRenderRuntime({{
          menu,
          menuState,
          inputValue: () => "",
          visibleEntries: () => entries,
          searchSnapshot: () => ({{}}),
          normalizeDraftFilePath: () => "",
          draftSuppressed: () => false,
          draftEntry: (path) => ({{ path, createNew: true }}),
          syncActiveDescendant: () => null,
          sectionLabel: (source) => source === "mentioned" ? "Mentioned in chat" : source === "recent" ? "Recently opened" : "",
          // Real helpers exercise the collision-aware hint path end to end.
          duplicatePaths: (items) => helpers.duplicateFilePickerPaths(items),
          rawByteDuplicatePaths: (items) => helpers.rawByteDuplicatePaths(items),
          identityHint: (entry, duplicatePaths, options) => helpers.filePickerIdentityHint(entry, duplicatePaths, options),
          titleForEntry: (entry, hint) => helpers.filePickerTitle(entry, hint),
          normalizeFileApiPath: (value) => String(value || ""),
          activeIdentity: () => ({{ path: "", gitPath: false, apiPath: "" }}),
          openDraftFilePath: () => null,
          openEntry: () => null,
          el,
          createTextNode,
        }});
        runtime.render();
        function summarize(node) {{
          return {{
            tag: node.tag,
            cls: (node.attrs && node.attrs.class) || "",
            text: (node.attrs && node.attrs.text) || node.text || node.textContent || "",
            title: (node.attrs && node.attrs.title) || "",
            children: (node.children || []).map(summarize),
          }};
        }}
        const rows = menu.children.filter((node) => (node.attrs && node.attrs.class || "").startsWith("fileMenuItem")).map(summarize);
        const hints = rows.map((row) => row.children.find((child) => child.cls === "fileMenuHint fileMenuIdentity"));
        process.stdout.write(JSON.stringify({{
          rowCount: rows.length,
          titles: rows.map((row) => row.title),
          hints: hints.map((hint) => (hint ? hint.text : null)),
        }}));
        """
    )
    proc = subprocess.run(["node"], input=js, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


def run_picker_input_runtime_probe() -> dict[str, object]:
    display_source = APP_DISPLAY_JS.read_text(encoding="utf-8")
    helper_source = APP_FILE_HELPERS_JS.read_text(encoding="utf-8")
    picker_source = APP_FILE_PICKER_JS.read_text(encoding="utf-8")
    js = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}} }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(display_source)}, ctx);
        vm.runInContext({json.dumps(helper_source)}, ctx);
        vm.runInContext({json.dumps(picker_source)}, ctx);
        const picker = ctx.window.CodoxearFilePicker;
        const events = [];
        let entries = [];
        let currentSession = "";
        let selectedSession = "sid-selected";
        let focusInside = false;
        let rafCallback = null;
        const input = {{
          value: "",
          attrs: {{}},
          setAttribute(name, value) {{ this.attrs[name] = String(value); }},
          removeAttribute(name) {{ delete this.attrs[name]; }},
        }};
        const menuState = picker.createMenuState({{ normalizeLineNumber: (value) => Number(value) || null }});
        const runtime = picker.createInputRuntime({{
          input,
          menuState,
          ensureCurrentSession: async () => {{ events.push(["ensure"]); return true; }},
          renderMenu: () => {{ events.push(["render"]); return entries; }},
          applyMenuState: () => events.push(["apply"]),
          resetInput: () => {{ events.push(["resetInput"]); input.value = "active.py"; }},
          closeMenu: (options) => events.push(["close", options]),
          currentSessionId: () => currentSession,
          selectedSessionId: () => selectedSession,
          resetSearchState: () => events.push(["resetSearch"]),
          setSearchSessionId: (sessionId) => events.push(["setSearchSession", sessionId]),
          scheduleSearch: (query) => events.push(["schedule", query]),
          selectionLine: () => menuState.selectionLine(input.value),
          openDraftFilePathWithGuard: (path) => events.push(["openDraft", path]),
          openFilePathWithResolvedMode: (path, options) => {{ events.push(["openFile", path, options]); return Promise.resolve(true); }},
          setStatus: (status) => events.push(["status", status]),
          optionElementById: (id) => ({{ scrollIntoView(options) {{ events.push(["scroll", id, options]); }} }}),
          isFocusInsideField: () => focusInside,
          requestAnimationFrame: (callback) => {{ rafCallback = callback; events.push(["raf"]); }},
        }});
        async function run() {{
          entries = [{{ path: "src/app.py", changed: true, gitPath: true, apiPath: "tok" }}];
          menuState.openSearchQuery("src/app.py", {{ line: 42, suppressDraft: true }});
          menuState.setPreserveSearchOnFocus(true);
          input.value = "src/app.py";
          const focusResult = await runtime.focus();
          const focusEvents = events.splice(0);
          const clickEvent = {{ stopped: 0, stopPropagation() {{ this.stopped += 1; events.push(["stop"]); }} }};
          const clickResult = await runtime.click(clickEvent);
          const clickEvents = events.splice(0);
          const openSearchResult = runtime.openSearchQuery("ambig", {{ line: 7, suppressDraft: true }});
          const openSearchEvents = events.splice(0);
          const openSearchState = {{ value: input.value, selectionLine: menuState.selectionLine("ambig"), draftSuppressed: menuState.draftSuppressed("ambig") }};
          input.value = "";
          currentSession = "";
          selectedSession = "sid-selected";
          const emptyInputResult = await runtime.input();
          const emptyInputEvents = events.splice(0);
          input.value = "abc";
          currentSession = "sid-current";
          const searchInputResult = await runtime.input();
          const searchInputEvents = events.splice(0);
          entries = [{{ path: "src/app.py", changed: true, gitPath: true, apiPath: "tok" }}];
          menuState.setOpen(true);
          const arrowEvent = {{ key: "ArrowDown", prevented: 0, preventDefault() {{ this.prevented += 1; events.push(["prevent"]); }} }};
          const arrowResult = await runtime.keydown(arrowEvent);
          const arrowEvents = events.splice(0);
          entries = [{{ path: "draft/new.txt", createNew: true }}];
          menuState.setOpen(true);
          const draftEvent = {{ key: "Enter", preventDefault() {{ events.push(["prevent"]); }} }};
          const draftResult = await runtime.keydown(draftEvent);
          const draftEvents = events.splice(0);
          entries = [{{ path: "src/app.py", changed: true, gitPath: true, apiPath: "tok" }}];
          menuState.openSearchQuery("src/app.py", {{ line: 42 }});
          input.value = "src/app.py";
          const enterEvent = {{ key: "Enter", preventDefault() {{ events.push(["prevent"]); }} }};
          const enterResult = await runtime.keydown(enterEvent);
          const enterEvents = events.splice(0);
          menuState.setOpen(true);
          const escapeEvent = {{ key: "Escape", preventDefault() {{ events.push(["prevent"]); }}, stopPropagation() {{ events.push(["stop"]); }} }};
          const escapeResult = await runtime.keydown(escapeEvent);
          const escapeEvents = events.splice(0);
          menuState.setOpen(true);
          const tabResult = await runtime.keydown({{ key: "Tab" }});
          const tabEvents = events.splice(0);
          focusInside = false;
          const blurResult = runtime.blur();
          const blurScheduledEvents = events.splice(0);
          rafCallback();
          const blurEvents = events.splice(0);
          let missingError = "";
          try {{ picker.createInputRuntime({{ input, menuState }}); }} catch (err) {{ missingError = err && err.message ? err.message : String(err); }}
          return {{ focusResult, focusEvents, clickResult, clickEvents, clickStopped: clickEvent.stopped, openSearchResult, openSearchEvents, openSearchState, emptyInputResult, emptyInputEvents, searchInputResult, searchInputEvents, arrowResult, arrowEvents, arrowPrevented: arrowEvent.prevented, draftResult, draftEvents, enterResult, enterEvents, escapeResult, escapeEvents, tabResult, tabEvents, blurResult, blurScheduledEvents, blurEvents, missingError, frozen: Object.isFrozen(runtime) }};
        }}
        run().then((result) => process.stdout.write(JSON.stringify(result))).catch((err) => {{ console.error(err && err.stack || err); process.exit(1); }});
        """
    )
    proc = subprocess.run(["node"], input=js, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


class TestFrontendFilePickerModuleBehavior(unittest.TestCase):
    def test_file_picker_module_exports_and_fails_closed(self) -> None:
        result = run_picker_module_probe()
        self.assertTrue(result["frozen"])
        self.assertEqual(
            result["exports"],
            [
                "appendDraftFileMenuItem",
                "appendFilePickerEntryItem",
                "appendFilePickerGitStatusRow",
                "appendFilePickerSection",
                "appendFilePickerStatusRow",
                "appendHighlightedFileMenuPath",
                "createEntryRuntime",
                "createInputRuntime",
                "createMenuDomRuntime",
                "createMenuRenderRuntime",
                "createMenuState",
                "createSearchState",
                "localFilePickerSearchEntries",
                "normalizeSamePathFilePickerScores",
                "pendingSessionPathEntry",
                "prependPendingSessionPathEntry",
                "visibleFilePickerEntries",
            ],
        )
        self.assertContains("Codoxear file picker helpers failed to load", result["missingError"])
        self.assertContains("Codoxear file picker host missing blocked", result["hostError"])
        self.assertContains("Codoxear file picker host missing normalizeLineNumber", result["menuHostError"])
        self.assertContains("Codoxear file picker host missing snapshot", result["domHostError"])
        self.assertContains("Codoxear file picker host missing createTextNode", result["highlightHostError"])
        self.assertContains("Codoxear file picker host missing openDraftFilePath", result["draftHostError"])
        self.assertContains("Codoxear file picker host missing openDraftFilePath", result["entryHostError"])
        self.assertContains("Codoxear file picker host missing el", result["statusHostError"])
        self.assertContains("Codoxear file picker host missing el", result["gitStatusHostError"])
        # Untracked file: distinct read-only "untracked" stat, not a +/- diff stat.
        untracked = result["untrackedEntry"]
        self.assertEqual(untracked["cls"], "fileMenuItem")
        self.assertEqual([c for c in untracked["children"] if c["cls"] == "fileMenuStat untracked"], [{"tag": "span", "cls": "fileMenuStat untracked", "text": "untracked", "id": "", "title": "", "aria": "", "children": []}])
        self.assertFalse(any(c["cls"] == "fileMenuStat changed" for c in untracked["children"]))
        # Rename: path rendered as old -> new, +/- stat still present.
        rename = result["renameEntry"]
        rename_path = next(c for c in rename["children"] if c["cls"] == "fileMenuPath")
        self.assertEqual(rename_path["text"], "orig.md \u2192 moved.md")
        self.assertTrue(any(c["cls"] == "fileMenuStat changed" for c in rename["children"]))
        # Non-repo notice renders as an explicit status row, both standalone and
        # inside the menu render (no-query candidate view).
        self.assertEqual(result["gitStatusRow"]["cls"], "fileMenuGitStatus")
        self.assertEqual(result["gitStatusRow"]["text"], "Not a git repository \u2014 no changed files")
        self.assertIsNotNone(result["renderedGitStatus"])
        self.assertEqual(result["renderedGitStatus"]["text"], "Not a git repository \u2014 no changed files")

    def test_file_picker_menu_render_runtime_behavior(self) -> None:
        result = run_picker_render_runtime_probe()
        self.assertTrue(result["frozen"])
        pending = result["pending"]
        self.assertEqual(pending["returned"], [{"path": "draft/new.txt", "createNew": True}])
        self.assertEqual([row["cls"] for row in pending["rows"]], ["fileMenuItem fileMenuCreate active", "pickerEmpty"])
        self.assertEqual(pending["rows"][1]["text"], "Searching files...")
        self.assertEqual(pending["syncs"], [])
        empty = result["empty"]
        self.assertEqual(empty["rows"], [{"tag": "div", "cls": "pickerEmpty", "id": "", "text": "search failed", "aria": "", "title": "", "children": []}])
        self.assertEqual(empty["syncs"], [-1])
        populated = result["populated"]
        self.assertEqual([row["text"] for row in populated["rows"] if row["cls"] == "fileMenuSection"], ["Changed files", "Recent files"])
        self.assertEqual(populated["rows"][1]["id"], "filePickerOption-0")
        self.assertEqual(populated["rows"][1]["aria"], "false")
        self.assertEqual(populated["rows"][3]["id"], "filePickerOption-1")
        self.assertEqual(populated["rows"][3]["aria"], "true")
        self.assertEqual(populated["syncs"], [-1])
        footer = result["footer"]
        self.assertEqual(footer["rows"][-1]["text"], "Searching full project...")
        self.assertEqual(footer["syncs"], [0])

    def test_file_picker_raw_byte_collision_rows_are_distinguishable(self) -> None:
        result = run_picker_raw_byte_collision_render_probe()
        # Two rows share the display path but must carry different visible hints
        # and titles so browser users and automation can tell them apart before
        # selecting. The tokenized (raw-byte) side reads "non-UTF bytes"; the
        # literal/display side reads "literal name".
        self.assertEqual(result["rowCount"], 2)
        self.assertEqual(sorted(result["hints"]), ["current folder · literal name", "current folder · non-UTF bytes"])
        titles = result["titles"]
        self.assertTrue(any("non-UTF bytes" in title for title in titles), titles)
        self.assertTrue(any("literal name" in title for title in titles), titles)
        self.assertNotEqual(titles[0], titles[1])

    def test_file_picker_input_runtime_behavior(self) -> None:
        result = run_picker_input_runtime_probe()
        self.assertTrue(result["frozen"])
        self.assertTrue(result["focusResult"])
        self.assertEqual(result["focusEvents"], [["ensure"], ["render"], ["apply"]])
        self.assertTrue(result["clickResult"])
        self.assertEqual(result["clickStopped"], 1)
        self.assertEqual(result["clickEvents"], [["stop"], ["ensure"], ["render"], ["apply"]])
        self.assertTrue(result["openSearchResult"])
        self.assertEqual(result["openSearchEvents"], [["schedule", "ambig"], ["render"], ["apply"]])
        self.assertEqual(result["openSearchState"], {"value": "ambig", "selectionLine": 7, "draftSuppressed": True})
        self.assertTrue(result["emptyInputResult"])
        self.assertEqual(result["emptyInputEvents"], [["ensure"], ["render"], ["apply"], ["resetSearch"], ["setSearchSession", "sid-selected"], ["render"], ["apply"]])
        self.assertTrue(result["searchInputResult"])
        self.assertEqual(result["searchInputEvents"], [["ensure"], ["render"], ["apply"], ["schedule", "abc"], ["render"], ["apply"]])
        self.assertTrue(result["arrowResult"])
        self.assertEqual(result["arrowPrevented"], 1)
        self.assertEqual(result["arrowEvents"], [["ensure"], ["render"], ["prevent"], ["render"], ["apply"], ["scroll", "filePickerOption-0", {"block": "nearest"}]])
        self.assertTrue(result["draftResult"])
        self.assertEqual(result["draftEvents"], [["ensure"], ["render"], ["prevent"], ["openDraft", "draft/new.txt"]])
        self.assertTrue(result["enterResult"])
        self.assertEqual(result["enterEvents"], [["ensure"], ["render"], ["prevent"], ["openFile", "src/app.py", {"line": 42, "changed": True, "gitPath": True, "apiPath": "tok"}]])
        self.assertTrue(result["escapeResult"])
        self.assertEqual(result["escapeEvents"], [["ensure"], ["render"], ["prevent"], ["stop"], ["close", {"restoreInput": True}]])
        self.assertTrue(result["tabResult"])
        self.assertEqual(result["tabEvents"], [["ensure"], ["render"], ["close", {"restoreInput": True}]])
        self.assertTrue(result["blurResult"])
        self.assertEqual(result["blurScheduledEvents"], [["raf"]])
        self.assertEqual(result["blurEvents"], [["close", {"restoreInput": True}]])
        self.assertContains("Codoxear file picker host missing ensureCurrentSession", result["missingError"])

    def test_file_picker_menu_state_behavior(self) -> None:
        result = run_picker_module_probe()["menuState"]
        self.assertTrue(result["opened"])
        self.assertEqual(result["selectionBeforeInput"], 42)
        self.assertTrue(result["draftSuppressedBeforeInput"])
        self.assertTrue(result["preservedFirst"])
        self.assertFalse(result["preservedSecond"])
        self.assertIsNone(result["selectionAfterInput"])
        self.assertFalse(result["draftSuppressedAfterInput"])
        self.assertEqual(result["visibleQueryAfterInput"], "src/other.js")
        self.assertEqual(result["clampedFocus"], 1)
        self.assertEqual(result["movedFocus"], 0)
        self.assertEqual(result["enterIndex"], 0)
        self.assertFalse(result["closed"]["open"])
        self.assertEqual(result["closed"]["focus"], -1)

    def test_file_picker_highlighted_path_behavior(self) -> None:
        result = run_picker_module_probe()["highlightState"]
        self.assertEqual(result["parentChildCount"], 1)
        self.assertEqual(result["spanClass"], "fileMenuPath")
        self.assertEqual(result["highlightedChildren"], [["text", "src/"], ["mark", "fileMenuMatch", "foo"], ["text", "_bar.py"]])
        self.assertEqual(result["plainText"], "README.md")
        self.assertEqual(result["plainParentChildCount"], 1)

    def test_file_picker_menu_render_helpers_behavior(self) -> None:
        result = run_picker_module_probe()["menuRenderState"]
        self.assertTrue(result["sectionResult"])
        self.assertFalse(result["emptySectionResult"])
        self.assertEqual(result["parentChildren"], [
            ["div", "fileMenuSection", "Changed files", "", ""],
            ["button", "fileMenuItem fileMenuCreate active", "", "filePickerOption-2", "true"],
        ])
        self.assertEqual(result["draftItemChildren"], [
            ["span", "fileMenuPath", "Create new file: draft/new.txt"],
            ["span", "fileMenuHint", "Creates only when you save"],
        ])
        self.assertEqual(result["draftPrevented"], 1)
        self.assertEqual(result["draftEvents"], [["openDraft", "draft/new.txt"]])
        self.assertEqual(result["statusRow"], ["div", "pickerEmpty", "Searching full project..."])
        self.assertEqual(result["statusParentChildCount"], 1)

    def test_file_picker_entry_row_renderer_behavior(self) -> None:
        result = run_picker_module_probe()["entryRenderState"]
        rows = result["rows"]
        self.assertEqual([row["cls"] for row in rows], ["fileMenuItem fileMenuCreate", "fileMenuItem active", "fileMenuItem"])
        self.assertEqual([row["id"] for row in rows], ["filePickerOption-0", "filePickerOption-1", "filePickerOption-2"])
        self.assertEqual([row["title"] for row in rows], ["Create title", "Changed title", "Regular title"])
        self.assertEqual([row["aria"] for row in rows], ["false", "true", "false"])
        self.assertEqual(rows[0]["children"][1]["text"], "Creates only when you save")
        self.assertEqual(rows[1]["children"][1]["text"], "git version")
        self.assertEqual(rows[1]["children"][2]["cls"], "fileMenuStat changed")
        self.assertEqual([child["text"] for child in rows[1]["children"][2]["children"]], ["+?", "-2"])
        self.assertEqual(rows[2]["children"][1]["text"], "recent file")
        self.assertEqual(result["entryPrevented"], 1)
        self.assertEqual(result["entryEvents"], [
            ["openDraft", "draft/new.txt"],
            ["openEntry", "src/app.py", True, True, "tok"],
            ["openEntry", "README.md", False, False, ""],
        ])

    def test_file_picker_dom_runtime_behavior(self) -> None:
        result = run_picker_module_probe()["domState"]
        self.assertTrue(result["frozen"])
        self.assertTrue(result["openDomState"]["open"])
        self.assertEqual(result["openDomSnapshot"], {"fieldActive": True, "menuOpen": True, "expanded": "true"})
        self.assertEqual(result["resetDomSnapshot"], {"value": "src/current.py", "activeDescendant": "", "focus": -1})
        self.assertEqual(result["closeDomSnapshot"], {"fieldActive": False, "menuOpen": False, "expanded": "false", "activeDescendant": "", "value": "restored.py", "focus": -1})
        self.assertTrue(result["syncPositive"])
        self.assertEqual(result["syncPositiveValue"], "filePickerOption-3")
        self.assertTrue(result["syncNegative"])
        self.assertEqual(result["syncNegativeValue"], "")
        self.assertEqual(result["domEvents"], [
            ["toggle", "field", "active", True],
            ["toggle", "menu", "open", True],
            ["setAttr", "aria-expanded", "true"],
            ["removeAttr", "aria-activedescendant"],
            ["setAttr", "aria-activedescendant", "filePickerOption-0"],
            ["removeAttr", "aria-activedescendant"],
            ["toggle", "field", "active", False],
            ["toggle", "menu", "open", False],
            ["setAttr", "aria-expanded", "false"],
            ["removeAttr", "aria-activedescendant"],
            ["setAttr", "aria-activedescendant", "filePickerOption-3"],
            ["removeAttr", "aria-activedescendant"],
        ])

if __name__ == "__main__":
    unittest.main()
