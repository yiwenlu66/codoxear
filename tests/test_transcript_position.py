"""Behavioral contract: the reader's per-session viewport position.

Two behaviors are required and they are different kinds of position:

- a reader who left a session at the bottom returns to the bottom, and keeps
  following it while the rendered content grows (an image whose header
  dimensions arrive after paint is the case that exposed this);
- a reader who left mid-history returns to the same message row, at the same
  distance into it.

The controller under test is the single owner of that position, so it must also
stand down the moment anyone else takes the viewport: the reader, another
writer, or a transcript that no longer contains the remembered row.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import textwrap

from frontend_module_loader import module_path


TRANSCRIPT_JS = module_path("app_transcript.js")
POSITION_JS = module_path("app_transcript_position.js")


def _run_node(js: str) -> dict:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".js", encoding="utf-8") as script:
        script.write(js)
        script.flush()
        proc = subprocess.run(
            ["node", script.name],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    return json.loads(proc.stdout)


def _harness(body: str) -> str:
    return textwrap.dedent(
        f"""
        const vm = require("vm");
        const observers = [];
        class FakeResizeObserver {{
          constructor(callback) {{ this.callback = callback; this.targets = []; observers.push(this); }}
          observe(target) {{ this.targets.push(target); }}
          disconnect() {{ this.targets = []; }}
          static fire() {{ for (const observer of observers) for (const _target of observer.targets) observer.callback([]); }}
        }}
        const ctx = {{ window: {{}}, console, ResizeObserver: FakeResizeObserver }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(TRANSCRIPT_JS.read_text(encoding="utf-8"))}, ctx);
        vm.runInContext({json.dumps(POSITION_JS.read_text(encoding="utf-8"))}, ctx);

        const CHAT_TOP = 100;

        // A scroller whose row rects are computed from a plain layout model, so
        // "the page grew" and "the viewport moved" are both expressible.
        function makeScroller(rows) {{
          const state = {{ top: 0, clientHeight: 600 }};
          const chat = {{
            clientHeight: state.clientHeight,
            getBoundingClientRect() {{ return {{ top: CHAT_TOP, bottom: CHAT_TOP + state.clientHeight }}; }},
            get scrollHeight() {{ return rows.reduce((total, row) => total + row.height, 0); }},
            get scrollTop() {{ return state.top; }},
            set scrollTop(next) {{
              const max = Math.max(0, chat.scrollHeight - state.clientHeight);
              state.top = Math.max(0, Math.min(Number(next) || 0, max));
            }},
            rowTop(index) {{
              let offset = 0;
              for (let i = 0; i < index; i++) offset += rows[i].height;
              return offset - state.top;
            }},
          }};
          for (const [index, row] of rows.entries()) {{
            row.getBoundingClientRect = () => ({{
              top: CHAT_TOP + chat.rowTop(index),
              bottom: CHAT_TOP + chat.rowTop(index) + row.height,
            }});
          }}
          chat.scrollToBottom = () => {{ chat.scrollTop = chat.scrollHeight - state.clientHeight; }};
          chat.atBottom = () => chat.scrollTop >= chat.scrollHeight - state.clientHeight - 1;
          chat.gap = () => chat.scrollHeight - state.clientHeight - chat.scrollTop;
          chat.state = state;
          return chat;
        }}

        function makeRows(count, height = 200) {{
          return Array.from({{ length: count }}, (_, index) => ({{
            height,
            dataset: {{ messageId: `m${{index}}`, historyCursor: `c${{index}}` }},
          }}));
        }}

        function makeRuntime(chat, options = {{}}) {{
          const rafs = [];
          const jumpButton = {{ style: {{ display: "" }} }};
          const timeChip = {{ style: {{ display: "" }}, textContent: "" }};
          const runtime = ctx.window.CodoxearTranscript.createTranscriptScrollRuntime({{
            chat,
            content: options.content || null,
            resizeObserver: options.resizeObserver || null,
            jumpButton,
            timeChip,
            requestAnimationFrame: (fn) => rafs.push(fn),
            hasSelection: () => true,
            isSearchOpen: () => false,
            firstVisibleMessageRow: () => null,
            dayLabel: () => "Day",
            time24: () => "12:34",
            shouldCancelOlderLoad: () => false,
            cancelOlderLoad: () => {{}},
            autoLoadOlder: () => {{}},
            bottomThresholdPx: 80,
            olderTopTriggerPx: 1,
            olderCancelPx: 48,
          }});
          return {{
            runtime, rafs, chat,
            jump: () => jumpButton.style.display,
            flushRafs() {{ while (rafs.length) rafs.shift()(); }},
          }};
        }}

        function makePositions(chat, rows, options = {{}}) {{
          const selected = {{ id: options.sessionId || "sid-a" }};
          const loads = [];
          const controller = ctx.window.CodoxearTranscriptPosition.createTranscriptPositionController({{
            chat,
            content: {{}},
            scrollRuntime: options.runtime,
            getSelectedSessionId: () => selected.id,
            renderedRows: () => rows,
            resolveAnchorWindow: (cursor) => {{ loads.push(cursor); return options.load ? options.load(cursor, rows) : Promise.resolve(false); }},
            resizeObserver: FakeResizeObserver,
            consoleError: () => {{}},
          }});
          return {{ controller, selected, loads }};
        }}
        const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

        (async () => {{
        {body}
        }})().then((out) => process.stdout.write(JSON.stringify(out))).catch((error) => {{
          console.error(error && error.stack || error);
          process.exit(1);
        }});
        """
    )


def test_leaving_at_the_bottom_returns_to_the_bottom_and_keeps_following_it() -> None:
    out = _run_node(
        _harness(
            """
            const out = {};
            const rows = makeRows(10);            // 10 * 200 = 2000px of content
            const chat = makeScroller(rows);
            const h = makeRuntime(chat, { content: {}, resizeObserver: FakeResizeObserver });
            chat.scrollToBottom();
            const p = makePositions(chat, rows, { runtime: h.runtime });

            out.saved = p.controller.save("sid-a");
            out.savedPosition = p.controller.positionFor("sid-a");

            // The return render lands wherever the browser puts it; restore must
            // re-establish the bottom.
            chat.scrollTop = 0;
            out.restored = p.controller.restore("sid-a");
            out.afterRestore = { atBottom: chat.atBottom(), gap: chat.gap(), jump: h.jump() };

            // Requirement 1 under late layout: an image finishes hydrating and
            // the content grows below the viewport. The runtime's pin has to
            // keep the viewport at the bottom.
            rows.push({ height: 900, dataset: { messageId: "img", historyCursor: "cimg" } });
            FakeResizeObserver.fire();
            out.afterGrowth = { atBottom: chat.atBottom(), gap: chat.gap(), jump: h.jump() };

            // A reader gesture owns the viewport again: growth must not yank it.
            chat.scrollTop = 200;
            h.runtime.noteUserScroll();
            h.runtime.handleScroll();
            rows.push({ height: 400, dataset: { messageId: "img2", historyCursor: "cimg2" } });
            FakeResizeObserver.fire();
            out.afterReaderGesture = { top: chat.scrollTop, gap: chat.gap() };
            return out;
            """
        )
    )
    assert out["saved"] is True
    assert out["savedPosition"] == {"kind": "tail"}
    assert out["restored"] is True
    assert out["afterRestore"]["atBottom"] is True
    assert out["afterRestore"]["jump"] == "none"
    # The content grew 900px below the viewport; following the tail is an
    # invariant, so the gap is closed again and the affordance stays hidden.
    assert out["afterGrowth"]["atBottom"] is True
    assert out["afterGrowth"]["gap"] == 0
    assert out["afterGrowth"]["jump"] == "none"
    # After a reader gesture the viewport is the reader's: growth leaves it.
    assert out["afterReaderGesture"]["top"] == 200


def test_anchor_restore_keeps_the_same_row_at_the_same_offset() -> None:
    out = _run_node(
        _harness(
            """
            const out = {};
            const rows = makeRows(12);
            const chat = makeScroller(rows);
            const h = makeRuntime(chat);
            chat.scrollTop = 900;                 // row 4 sits 100px above the top
            const p = makePositions(chat, rows, { runtime: h.runtime });
            p.controller.save("sid-a");
            out.savedPosition = p.controller.positionFor("sid-a");

            // Return render: different content metrics (another viewport width
            // rewraps every row), so a pixel offset would land on another row.
            for (const row of rows) row.height = 260;
            chat.scrollTop = 0;
            p.controller.restore("sid-a");
            const anchor = rows.find((row) => row.dataset.messageId === out.savedPosition.messageId);
            out.afterRestore = {
              rowTop: anchor.getBoundingClientRect().top - CHAT_TOP,
              offset: out.savedPosition.offset,
              jump: h.jump(),
            };

            // Late layout above the anchor (an image earlier in the transcript
            // reserving its box) must not move the reader.
            rows[1].height += 500;
            FakeResizeObserver.fire();
            out.afterGrowthAbove = anchor.getBoundingClientRect().top - CHAT_TOP;
            out.settling = p.controller.snapshot().settling;

            // The reader moves: the next content change stands the settle down
            // instead of dragging the viewport back.
            chat.scrollTop += 300;
            h.runtime.noteUserScroll();
            h.runtime.handleScroll();
            const readerTop = chat.scrollTop;
            rows[2].height += 500;
            FakeResizeObserver.fire();
            out.afterReaderGesture = { settling: p.controller.snapshot().settling, top: chat.scrollTop, readerTop };
            out.afterGrowthAfterGesture = anchor.getBoundingClientRect().top - CHAT_TOP;
            return out;
            """
        )
    )
    assert out["savedPosition"]["kind"] == "anchor"
    assert out["savedPosition"]["messageId"] == "m4"
    # 900px scrolled with 200px rows: row 4 starts at 800px, i.e. 100px above
    # the viewport top.
    assert out["savedPosition"]["offset"] == -100
    # After a rewrap the row is placed at the same viewport offset.
    assert out["afterRestore"] == {"rowTop": -100, "offset": -100, "jump": "inline-flex"}
    assert out["afterGrowthAbove"] == -100
    assert out["settling"] is True
    assert out["afterReaderGesture"]["settling"] is False
    # The reader's own scroll is not undone by later growth.
    assert out["afterReaderGesture"]["top"] == out["afterReaderGesture"]["readerTop"]
    assert out["afterGrowthAfterGesture"] != -100


def test_a_competing_writer_or_session_change_ends_the_restore() -> None:
    out = _run_node(
        _harness(
            """
            const out = {};
            const rows = makeRows(12);
            const chat = makeScroller(rows);
            const h = makeRuntime(chat);
            chat.scrollTop = 900;
            const p = makePositions(chat, rows, { runtime: h.runtime });
            p.controller.save("sid-a");

            chat.scrollTop = 0;
            p.controller.restore("sid-a");
            const anchorTop = rows[4].getBoundingClientRect().top - CHAT_TOP;
            // Another writer (older-page prepend, navigation, a searched row)
            // moves the viewport; the settle must not drag it back.
            h.runtime.setScrollTop(200);
            out.writerSeqChanged = h.runtime.writeSeq() > 0;
            rows[1].height += 400;
            FakeResizeObserver.fire();
            out.afterForeignWrite = {
              top: chat.scrollTop,
              settling: p.controller.snapshot().settling,
              anchorTop,
            };

            // A session switch ends any settle that is still open.
            chat.scrollTop = 900;
            p.controller.restore("sid-a");
            p.selected.id = "sid-b";
            rows[1].height += 400;
            FakeResizeObserver.fire();
            out.afterSwitch = { settling: p.controller.snapshot().settling };
            return out;
            """
        )
    )
    assert out["writerSeqChanged"] is True
    assert out["afterForeignWrite"]["top"] == 200
    assert out["afterForeignWrite"]["settling"] is False
    assert out["afterSwitch"]["settling"] is False


def test_a_queued_bottom_write_cannot_override_an_anchor_restore() -> None:
    out = _run_node(
        _harness(
            """
            const out = {};
            const rows = makeRows(12);
            const chat = makeScroller(rows);
            const h = makeRuntime(chat);
            // The render that just completed queued its own double bottom write.
            h.runtime.scheduleScrollToBottom({ double: true });
            out.queued = h.rafs.length;

            chat.scrollTop = 900;
            const p = makePositions(chat, rows, { runtime: h.runtime });
            p.controller.save("sid-a");
            chat.scrollTop = 0;
            p.controller.restore("sid-a");
            const afterRestore = chat.scrollTop;

            h.flushRafs();
            out.afterFlush = { top: chat.scrollTop, restored: afterRestore };
            return out;
            """
        )
    )
    assert out["queued"] >= 1
    # The queued bottom write does not fire; the reader stays on the row.
    assert out["afterFlush"]["top"] == out["afterFlush"]["restored"]
    assert out["afterFlush"]["top"] != 0


def test_a_remembered_row_outside_the_rendered_window_is_materialized_or_dropped() -> None:
    out = _run_node(
        _harness(
            """
            const out = {};
            const rows = makeRows(12);
            const chat = makeScroller(rows);
            const h = makeRuntime(chat);
            chat.scrollTop = 900;
            const p = makePositions(chat, rows, {
              runtime: h.runtime,
              load: async (cursor, currentRows) => {
                // The window that contains the remembered row is rendered.
                currentRows.splice(0, currentRows.length,
                  { height: 200, dataset: { messageId: "old0", historyCursor: "cold0" } },
                  { height: 200, dataset: { messageId: "m4", historyCursor: "c4" } },
                  { height: 200, dataset: { messageId: "old2", historyCursor: "cold2" } },
                  { height: 200, dataset: { messageId: "old3", historyCursor: "cold3" } },
                  { height: 200, dataset: { messageId: "old4", historyCursor: "cold4" } },
                  { height: 200, dataset: { messageId: "old5", historyCursor: "cold5" } });
                for (const [index, row] of currentRows.entries()) {
                  row.getBoundingClientRect = () => ({
                    top: CHAT_TOP + chat.rowTop(index),
                    bottom: CHAT_TOP + chat.rowTop(index) + row.height,
                  });
                }
                return true;
              },
            });
            p.controller.save("sid-a");
            out.saved = p.controller.positionFor("sid-a");
            // The return render is the tail window: the remembered row is gone.
            const tailRows = makeRows(6).map((row) => ({ height: 200, dataset: { messageId: `t${row.dataset.messageId}`, historyCursor: `t${row.dataset.historyCursor}` } }));
            rows.splice(0, rows.length, ...tailRows);
            for (const [index, row] of rows.entries()) {
              row.getBoundingClientRect = () => ({
                top: CHAT_TOP + chat.rowTop(index),
                bottom: CHAT_TOP + chat.rowTop(index) + row.height,
              });
            }
            chat.scrollTop = 0;
            out.restored = p.controller.restore("sid-a");
            await flush();
            out.loads = p.loads;
            const anchor = rows.find((row) => row.dataset.messageId === "m4");
            out.materialized = {
              rowTop: anchor ? anchor.getBoundingClientRect().top - CHAT_TOP : null,
              offset: out.saved.offset,
              rows: rows.length,
            };

            // No cursor at all: nothing identifies the row, so the tail is the
            // only honest answer.
            const rows2 = makeRows(12);
            const chat2 = makeScroller(rows2);
            const h2 = makeRuntime(chat2);
            chat2.scrollTop = 900;
            const p2 = makePositions(chat2, rows2, { runtime: h2.runtime });
            p2.controller.save("sid-a");
            rows2.splice(0, rows2.length);
            chat2.scrollTop = 0;
            out.droppedRestore = p2.controller.restore("sid-a");
            out.droppedPosition = p2.controller.positionFor("sid-a");
            out.droppedAtBottom = chat2.atBottom();
            return out;
            """
        )
    )
    assert out["saved"]["kind"] == "anchor"
    assert out["saved"]["messageId"] == "m4"
    assert out["restored"] is True
    assert out["loads"] == ["c4"]
    # The row is placed in the materialized window at its remembered offset.
    assert out["materialized"]["rows"] == 6
    assert out["materialized"]["rowTop"] == out["materialized"]["offset"]
    assert out["droppedRestore"] is True
    # The remembered anchor survives (it names a row in this log, which an older
    # page could still hold); the render falls back to the tail because the row
    # cannot be resolved right now.
    assert out["droppedPosition"]["kind"] == "anchor"
    assert out["droppedAtBottom"] is True


def test_save_falls_back_to_the_tail_when_nothing_identifiable_is_rendered() -> None:
    out = _run_node(
        _harness(
            """
            const out = {};
            const chat = makeScroller([]);
            const h = makeRuntime(chat);
            const p = makePositions(chat, [], { runtime: h.runtime });
            p.controller.save("sid-a");
            out.position = p.controller.positionFor("sid-a");
            out.restoreWithoutMemory = p.controller.restore("sid-other");
            out.cleared = p.controller.clear("sid-a");
            out.restoreAfterClear = p.controller.restore("sid-a");
            out.remembered = p.controller.snapshot().remembered;
            return out;
            """
        )
    )
    assert out["position"] == {"kind": "tail"}
    assert out["restoreWithoutMemory"] is False
    assert out["cleared"] is True
    assert out["restoreAfterClear"] is False
    assert out["remembered"] == 0
