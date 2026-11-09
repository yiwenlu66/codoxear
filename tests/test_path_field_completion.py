"""Execute the path-field completion core in a Node VM.

Tab completion in a path selector is decided by two pure functions in
``app_path_completion.js``: ``completePathFragment`` (fragment + directory names
-> the fragment to insert) and ``completionPlan`` (field value + listing -> the
replacement field value). Both surfaces (New Session cwd, file viewer path)
share them, so the rules are pinned here once instead of twice through the DOM.
"""
from __future__ import annotations

import json
import subprocess
import textwrap
from pathlib import Path

from frontend_module_loader import module_path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = module_path("app_path_completion.js").read_text(encoding="utf-8")


def run_core() -> dict[str, object]:
    js = textwrap.dedent(
        f"""
        const vm = require("node:vm");
        const ctx = {{ window: {{}}, console }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(SOURCE)}, ctx);
        const core = ctx.window.CodoxearPathCompletion;
        const calls = [];
        const resolve = async (args) => core.resolveTabCompletion(Object.assign({{
          defaultDir: "",
          fetchListing: async (text) => {{ calls.push(text); return null; }},
        }}, args || {{}}));
        const fragment = (f, names, options) => core.completePathFragment(f, names, options || {{}});
        const plan = (args) => core.completionPlan(args);
        const names = (list) => list.map((entry) => ({{ name: entry[0], isDir: Boolean(entry[1]) }}));
        const out = {{
          unique_file: fragment("app_f", ["app_file.js", "index.html"]),
          unique_dir: fragment("src", ["src", "index.html"]),
          case_repair: fragment("APP_f", ["app_file.js", "index.html"]),
          shared_prefix: fragment("src", ["src/a.js", "src/b.js"]),
          no_extension: fragment("app", ["app", "apple"]),
          no_match: fragment("zzz", ["app.js"]),
          empty_fragment: fragment("", ["app.js"]),
          empty_listing: fragment("app", []),
          truncated_partial: fragment("al", ["alpha", "alpine"], {{ truncated: true }}),
          truncated_full_name: fragment("al", ["al"], {{ truncated: true }}),
          truncated_without_flag: fragment("al", ["alpha", "alpine"]),
          hidden_names: fragment(".g", [".git", ".github", "src"]),
          plan_bare_fragment: plan({{ value: "app_f", names: names([["app_file.js", false]]) }}),
          plan_relative_dir: plan({{ value: "./src/ind", names: names([["index.js", false]]) }}),
          plan_absolute_dir: plan({{ value: "/etc/ho", names: names([["hosts", false]]) }}),
          plan_home_dir: plan({{ value: "~/pro", names: names([["projects", true]]) }}),
          plan_dir_marker: plan({{ value: "~/codox", names: names([["codoxear", true]]) }}),
          plan_default_dir: plan({{ value: "codox", defaultDir: "/", names: names([["codoxear", true]]) }}),
          plan_no_default_dir: plan({{ value: "codox", names: names([["codoxear", true]]) }}),
          plan_noop: plan({{ value: "~/app", names: names([["app", false]]) }}),
          plan_partial_prefix: plan({{ value: "~/co", names: names([["codex", true], ["codoxear", true]]) }}),
          plan_trailing_slash: plan({{ value: "~/", names: names([["codoxear", true]]) }}),
          plan_empty_value: plan({{ value: "", names: names([["codoxear", true]]) }}),
          plan_truncated_partial: plan({{ value: "~/al", truncated: true, names: names([["alpha", true], ["alpine", true]]) }}),
        }};
        const listing = (query, entries, truncated) => ({{ query, names: names(entries), truncated: Boolean(truncated) }});
        const focusState = {{ active: null }};
        const node = (name, visible = true) => {{
          const element = {{
            name, focused: false, attrs: {{}},
            focus() {{ if (!visible) return; focusState.active = element; element.focused = true; }},
            getAttribute(key) {{ return Object.prototype.hasOwnProperty.call(element.attrs, key) ? element.attrs[key] : null; }},
            closest() {{ return null; }},
            getBoundingClientRect() {{ return {{ width: visible ? 10 : 0, height: visible ? 10 : 0 }}; }},
            ownerDocument: null,
          }};
          return element;
        }};
        (async () => {{
          out.tab = {{
            pickedRow: await resolve({{ value: "zzz", picked: {{ path: "lib/picked.py" }} }}),
            pickedEqualsValue: await resolve({{ value: "lib/picked.py", picked: {{ path: "lib/picked.py" }} }}),
            cachedCompletion: await resolve({{ value: "~/pro", cachedListing: listing("~/pro", [["projects", true]]) }}),
            cachedNoMatchSkipsFetch: await resolve({{ value: "~/zzz", cachedListing: listing("~/zzz", [["projects", true]]) }}),
            foreignListingIgnored: await resolve({{ value: "~/a", cachedListing: listing("../a", [["alphaOld", true]]) }}),
            coldFetchCompletes: await resolve({{ value: "~/pro", cachedListing: null, fetchListing: async (text) => {{ calls.push("cold:" + text); return listing(text, [["projects", true]]); }} }}),
            coldForeignListingIgnored: await resolve({{ value: "~/a", fetchListing: async () => listing("../a", [["alphaOld", true]]) }}),
            coldFetchNoMatch: await resolve({{ value: "~/zzz", cachedListing: null }}),
            coldFetchThrows: await resolve({{ value: "~/pro", cachedListing: null, fetchListing: async () => {{ throw new Error("boom"); }} }}),
            emptyValue: await resolve({{ value: "", cachedListing: null }}),
            firstRowSingle: await resolve({{ value: "~/rec", firstRow: () => ({{ path: "/srv/recent-project" }}) }}),
            firstRowEqualsValue: await resolve({{ value: "/srv/recent-project", firstRow: () => ({{ path: "/srv/recent-project" }}) }}),
            firstRowAmongMany: await resolve({{ value: "/srv/dir/", firstRow: () => ({{ path: "/srv/dir/first-entry" }}) }}),
            firstRowAbsent: await resolve({{ value: "/srv/dir/", firstRow: null }}),
            trailingSlashSingleEntry: await resolve({{ value: "~/solo/", firstRow: null, fetchListing: async (text) => listing(text, [["only-child", true]]) }}),
            trailingSlashTwoEntries: await resolve({{ value: "~/pair/", fetchListing: async (text) => listing(text, [["a", true], ["b", true]]) }}),
            cachedFetches: calls.filter((text) => text === "~/pro").length,
            emptyWithCachedListing: await resolve({{ value: "", cachedListing: listing("", [["alpha", true]]) }}),
            failedCachedListing: await resolve({{ value: "~/pro", cachedListing: {{ query: "~/pro", ok: false, names: [["projects", true]], truncated: false }} }}),
            failedCachedListingRefetches: calls.filter((text) => text === "~/pro").length,
          }};
          // focusNextControl: verified moves, wrap inside the container, hidden and
          // negative-tabindex candidates skipped, popups excluded.
          const first = node("first");
          const hidden = node("hidden", false);
          const tabbable = node("tabbable");
          tabbable.attrs.tabindex = "-1";
          const last = node("last");
          const container = {{ querySelectorAll: () => [first, hidden, tabbable, last] }};
          focusState.active = first;
          const doc = {{ get activeElement() {{ return focusState.active; }}, querySelectorAll: () => [first, hidden, tabbable, last] }};
          for (const element of [first, hidden, tabbable, last]) element.ownerDocument = doc;
          out.focus = {{
            skipsToNext: core.focusNextControl(first, {{ container }}),
            focusedAfterSkip: focusState.active ? focusState.active.name : null,
            wrapsToFirst: core.focusNextControl(last, {{ container }}),
            focusedAfterWrap: focusState.active ? focusState.active.name : null,
            disabledFallsBack: (() => {{
              const off = node("off");
              off.disabled = true;
              off.ownerDocument = doc;
              const host = {{ querySelectorAll: () => [focusState.active, off, last] }};
              return core.focusNextControl(focusState.active, {{ container: host }});
            }})(),
          }};
          process.stdout.write(JSON.stringify(out));
        }})().catch((error) => {{ console.error(error && error.stack || error); process.exit(1); }});
        """
    )
    proc = subprocess.run(["node"], input=js, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


def test_unique_and_shared_prefix_completions() -> None:
    result = run_core()
    assert result["unique_file"] == {"insert": "app_file.js", "unique": True}
    assert result["unique_dir"] == {"insert": "src", "unique": True}
    assert result["shared_prefix"] == {"insert": "src/", "unique": False}
    assert result["case_repair"] == {"insert": "app_file.js", "unique": True}
    assert result["hidden_names"] == {"insert": ".git", "unique": False}


def test_fragment_with_nothing_to_add_returns_nothing() -> None:
    result = run_core()
    # "app" is already the shared prefix: no extension exists, so Tab must fall
    # through to native focus movement instead of inserting a guess.
    assert result["no_extension"] is None
    assert result["no_match"] is None
    assert result["empty_fragment"] is None
    assert result["empty_listing"] is None


def test_truncated_listing_only_completes_to_a_real_name() -> None:
    result = run_core()
    # A capped listing can hide siblings, so a partial common prefix is refused...
    assert result["truncated_partial"] is None
    assert result["plan_truncated_partial"] is None
    # ...while a fragment that equals a returned name is still real.
    assert result["truncated_full_name"] == {"insert": "al", "unique": True}
    # Without the flag the same listing completes to the shared prefix.
    assert result["truncated_without_flag"] == {"insert": "alp", "unique": False}


def test_plan_rewrites_only_the_fragment_and_marks_directories() -> None:
    result = run_core()
    assert result["plan_bare_fragment"]["value"] == "app_file.js"
    assert result["plan_bare_fragment"]["caret"] == len("app_file.js")
    assert result["plan_bare_fragment"]["unique"] is True
    assert result["plan_relative_dir"]["value"] == "./src/index.js"
    assert result["plan_relative_dir"]["caret"] == len("./src/index.js")
    assert result["plan_relative_dir"]["unique"] is True
    assert result["plan_absolute_dir"]["value"] == "/etc/hosts"
    assert result["plan_absolute_dir"]["caret"] == len("/etc/hosts")
    assert result["plan_absolute_dir"]["unique"] is True
    assert result["plan_home_dir"]["value"] == "~/projects/"
    assert result["plan_home_dir"]["caret"] == len("~/projects/")
    assert result["plan_home_dir"]["unique"] is True
    # A full name that is a directory gains the trailing separator: that is the
    # drill that makes the next listing show its contents.
    assert result["plan_dir_marker"]["value"] == "~/codoxear/"
    assert result["plan_dir_marker"]["caret"] == len("~/codoxear/")
    assert result["plan_dir_marker"]["unique"] is True
    assert result["plan_partial_prefix"]["value"] == "~/cod"
    assert result["plan_partial_prefix"]["caret"] == len("~/cod")
    assert result["plan_partial_prefix"]["unique"] is False
    assert result["plan_trailing_slash"] is None
    assert result["plan_empty_value"] is None
    assert result["plan_noop"] is None


def test_slashless_value_uses_the_surface_default_directory() -> None:
    result = run_core()
    # The cwd field reads a bare fragment under "/" (its long-standing menu
    # behavior); the file picker resolves it against the session cwd.
    assert result["plan_default_dir"]["value"] == "/codoxear/"
    assert result["plan_default_dir"]["caret"] == len("/codoxear/")
    assert result["plan_default_dir"]["unique"] is True
    assert result["plan_no_default_dir"]["value"] == "codoxear/"
    assert result["plan_no_default_dir"]["caret"] == len("codoxear/")
    assert result["plan_no_default_dir"]["unique"] is True


def test_tab_resolution_never_depends_on_whether_a_listing_has_landed() -> None:
    tab = run_core()["tab"]
    # The measured bug: typing and pressing Tab immediately found no listing and
    # did nothing. The resolver asks for the listing itself in that case.
    assert tab["coldFetchCompletes"] == {"apply": "~/projects/"}
    # A listing the field already has is final: no second request for the same
    # text, because an identical request could only return the same answer.
    assert tab["cachedCompletion"] == {"apply": "~/projects/"}
    # The cached answer is used as-is: no second request for the same text.
    assert tab["cachedFetches"] == 0


def test_tab_resolution_falls_through_only_when_there_is_truly_nothing() -> None:
    tab = run_core()["tab"]
    assert tab["cachedNoMatchSkipsFetch"] == {"fallthrough": True}
    assert tab["coldFetchNoMatch"] == {"fallthrough": True}
    assert tab["coldFetchThrows"] == {"fallthrough": True}
    assert tab["emptyValue"] == {"fallthrough": True}


def test_tab_resolution_takes_picked_and_single_rows() -> None:
    tab = run_core()["tab"]
    # `fromRow` tells the caller the value names something a list offered (a
    # directory), so the caller can add the separator that lists its contents.
    assert tab["pickedRow"] == {"apply": "lib/picked.py", "fromRow": True}
    # A picked row that equals the value is not a decision: fall through to the
    # listing so a directory can still be drilled into.
    assert tab["pickedEqualsValue"] == {"fallthrough": True}
    assert tab["firstRowSingle"] == {"apply": "/srv/recent-project", "fromRow": True}
    assert tab["firstRowEqualsValue"] == {"fallthrough": True}


def test_a_listing_from_another_value_is_never_used() -> None:
    tab = run_core()["tab"]
    # The measured cross-directory insertion: a cached or fetched listing that
    # belongs to a different text must be ignored, not combined with this text.
    assert tab["foreignListingIgnored"] == {"fallthrough": True}
    assert tab["coldForeignListingIgnored"] == {"fallthrough": True}


def test_an_empty_fragment_still_uses_a_single_entry_listing() -> None:
    tab = run_core()["tab"]
    # A value that already ends with "/" has no fragment to extend, but a
    # directory with one child is unambiguous — and this must hold cold as well as
    # warm, because it comes from the listing, not from previously rendered rows.
    assert tab["trailingSlashSingleEntry"] == {"apply": "~/solo/only-child/"}
    assert tab["trailingSlashTwoEntries"] == {"fallthrough": True}


def test_focus_walker_verifies_the_move_skips_and_wraps() -> None:
    focus = run_core()["focus"]
    # The hidden node and the tabindex="-1" node are skipped; the move is verified.
    assert focus["skipsToNext"] is True
    assert focus["focusedAfterSkip"] == "last"
    # A field that is last wraps to the beginning of its container instead of
    # leaving focus stranded.
    assert focus["wrapsToFirst"] is True
    assert focus["focusedAfterWrap"] == "first"
    # A disabled candidate falls through to the next valid one.
    assert focus["disabledFallsBack"] is True


def test_an_empty_field_never_decides_anything() -> None:
    tab = run_core()["tab"]
    # Warm or cold, an empty field is a focus move: reporting a completion here
    # depended on whether a listing for "" happened to be cached.
    assert tab["emptyWithCachedListing"] == {"fallthrough": True}
    assert tab["emptyValue"] == {"fallthrough": True}


def test_a_failed_listing_is_not_an_empty_directory() -> None:
    tab = run_core()["tab"]
    # A 503 must not be cached as "this directory is empty", which would make
    # every later Tab skip the request and find nothing.
    assert tab["failedCachedListing"] == {"fallthrough": True}
    assert tab["failedCachedListingRefetches"] >= 1


def test_a_visible_list_makes_tab_a_take_key() -> None:
    """The rule the user could feel: while rows are listed, Tab takes one.

    A user typing a prefix sees rows on screen. If the typed fragment has nothing
    to complete, Tab used to close the list and move focus — indistinguishable
    from "the feature does not work". The top row (what Enter would take) is the
    answer; only an empty list falls through to focus movement.
    """
    tab = run_core()["tab"]
    assert tab["firstRowAmongMany"] == {"apply": "/srv/dir/first-entry", "fromRow": True}
    assert tab["firstRowAbsent"] == {"fallthrough": True}
