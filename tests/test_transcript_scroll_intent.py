"""Behavioral contract: who may move the transcript's "away from the live tail" state.

The date chip (`#chatTimeChip`) and the jump button both render
`shouldStickToBottom()`, so a scroll event that nobody asked for must not be
able to flip that predicate. A phone's soft keyboard produces exactly those
events (the platform clamping the container when the composer or the keyboard
changes its height, and its caret-reveal pan settling), which is why typing
used to make the chip strobe.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import textwrap

from frontend_module_loader import module_path


ROOT_TRANSCRIPT_JS = module_path("app_transcript.js")


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


def _harness() -> str:
    source = ROOT_TRANSCRIPT_JS.read_text(encoding="utf-8")
    return textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}} }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(source)}, ctx);

        // A scroll container whose writes are counted, so "the app wrote a
        // position it already held" is observable.
        function makeChat({{ top, height, client }}) {{
          let value = top;
          const writes = [];
          const chat = {{
            scrollHeight: height,
            clientHeight: client,
            writes,
            get scrollTop() {{ return value; }},
            set scrollTop(next) {{
              const clamped = Math.max(0, Math.min(Number(next) || 0, chat.scrollHeight - chat.clientHeight));
              writes.push(clamped);
              value = clamped;
            }},
            // What the platform does when it moves the viewport itself: no
            // application write, no gesture.
            platformMove(next) {{
              value = Math.max(0, Math.min(Number(next) || 0, chat.scrollHeight - chat.clientHeight));
            }},
            platformResize(nextHeight, nextClient) {{
              chat.scrollHeight = nextHeight;
              if (nextClient !== undefined) chat.clientHeight = nextClient;
              value = Math.max(0, Math.min(value, chat.scrollHeight - chat.clientHeight));
            }},
          }};
          return chat;
        }}

        function makeRuntime(chat) {{
          const rafs = [];
          const jumpButton = {{ style: {{ display: "" }} }};
          const timeChip = {{ style: {{ display: "" }}, textContent: "" }};
          const runtime = ctx.window.CodoxearTranscript.createTranscriptScrollRuntime({{
            chat,
            jumpButton,
            timeChip,
            requestAnimationFrame: (fn) => rafs.push(fn),
            hasSelection: () => true,
            isSearchOpen: () => false,
            firstVisibleMessageRow: () => ({{ dataset: {{ ts: "1794504177" }} }}),
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
            runtime,
            rafs,
            chip: () => ({{ display: timeChip.style.display, text: timeChip.textContent }}),
            jump: () => jumpButton.style.display,
          }};
        }}

        const out = {{}};

        // 1. A platform-initiated upward move (keyboard pan / caret reveal)
        //    with no reader gesture must not read as "the reader left the tail".
        {{
          const chat = makeChat({{ top: 900, height: 1000, client: 100 }});
          const h = makeRuntime(chat);
          h.runtime.syncJumpButton();
          chat.platformMove(600);
          h.runtime.handleScroll();
          out.unprompted = {{ snap: h.runtime.snapshot(), chip: h.chip(), jump: h.jump() }};
        }}

        // 2. The same movement after a reader gesture is the reader leaving:
        //    chip and jump button appear once the reader is beyond the window.
        {{
          const chat = makeChat({{ top: 900, height: 1000, client: 100 }});
          const h = makeRuntime(chat);
          h.runtime.syncJumpButton();
          h.runtime.handleWheel({{ deltaY: 120 }});
          chat.platformMove(600);
          h.runtime.handleScroll();
          out.gesture = {{ snap: h.runtime.snapshot(), chip: h.chip(), jump: h.jump() }};
        }}

        // 3. A scroll shortcut and a scrollbar grab tag the same way.
        {{
          const chat = makeChat({{ top: 900, height: 1000, client: 100 }});
          const h = makeRuntime(chat);
          h.runtime.noteUserScroll();
          chat.platformMove(600);
          h.runtime.handleScroll();
          out.tagged = {{ snap: h.runtime.snapshot(), chip: h.chip() }};
        }}

        // 4. A clamp that lands exactly on the (new) tail is not leaving, even
        //    with a gesture in play: the phone's keyboard growing the chat
        //    shrinks the bottom offset and the browser clamps to it.
        {{
          const chat = makeChat({{ top: 900, height: 1000, client: 100 }});
          const h = makeRuntime(chat);
          h.runtime.noteUserScroll();
          chat.platformResize(1000, 400);
          h.runtime.handleScroll();
          out.clamped = {{ snap: h.runtime.snapshot(), chip: h.chip(), jump: h.jump(), top: chat.scrollTop }};
        }}

        // 5. Returning into the window re-attaches with no gesture (momentum
        //    ending at the bottom, a scrollbar release, a clamp back).
        {{
          const chat = makeChat({{ top: 900, height: 1000, client: 100 }});
          const h = makeRuntime(chat);
          h.runtime.handleWheel({{ deltaY: -1 }});
          chat.platformMove(600);
          h.runtime.handleScroll();
          chat.platformMove(860);
          h.runtime.handleScroll();
          out.returned = {{ snap: h.runtime.snapshot(), chip: h.chip() }};
        }}

        // 6. The app asserting the tail ends the reader-attribution window:
        //    after it places the viewport, a platform jiggle is not reader
        //    evidence any more.
        {{
          const chat = makeChat({{ top: 900, height: 1000, client: 100 }});
          const h = makeRuntime(chat);
          h.runtime.noteUserScroll();
          h.runtime.scrollToBottom();
          chat.platformMove(700);
          h.runtime.handleScroll();
          out.reasserted = {{ snap: h.runtime.snapshot(), chip: h.chip(), jump: h.jump() }};
        }}

        // 7. scrollToBottom writes nothing when the viewport already holds the
        //    tail, and still reaches the tail after the content grew.
        {{
          const chat = makeChat({{ top: 900, height: 1000, client: 100 }});
          const h = makeRuntime(chat);
          h.runtime.scrollToBottom();
          const idleWrites = chat.writes.length;
          chat.platformResize(1400, 100);
          h.runtime.scrollToBottom();
          out.writes = {{ idleWrites, afterGrowth: chat.writes.slice(), top: chat.scrollTop }};
        }}

        process.stdout.write(JSON.stringify(out));
        """
    )


class TestTranscriptScrollIntent:
    def test_only_a_reader_gesture_can_leave_the_live_tail(self) -> None:
        out = _run_node(_harness())

        # The reported symptom: typing on a phone moves the scroller, and the
        # chip used to appear. Platform movement alone now changes nothing.
        assert out["unprompted"]["snap"]["autoScroll"] is True
        assert out["unprompted"]["snap"]["renderedAtLiveTail"] is True
        assert out["unprompted"]["chip"] == {"display": "none", "text": ""}
        assert out["unprompted"]["jump"] == "none"

        # A wheel spin (or touch drag, or scrollbar grab, or scroll shortcut)
        # still detaches and shows both surfaces.
        assert out["gesture"]["snap"]["autoScroll"] is False
        assert out["gesture"]["chip"]["display"] == "inline-flex"
        assert out["gesture"]["chip"]["text"] == "Day · 12:34"
        assert out["gesture"]["jump"] == "inline-flex"

        assert out["tagged"]["snap"]["autoScroll"] is False
        assert out["tagged"]["chip"]["display"] == "inline-flex"

    def test_tail_clamp_and_return_keep_following(self) -> None:
        out = _run_node(_harness())

        # A clamp lands exactly on the new tail: still following, chip hidden.
        assert out["clamped"]["top"] == 600
        assert out["clamped"]["snap"]["autoScroll"] is True
        assert out["clamped"]["chip"] == {"display": "none", "text": ""}
        assert out["clamped"]["jump"] == "none"

        # Coming back inside the 80px window re-attaches without a gesture.
        assert out["returned"]["snap"]["autoScroll"] is True
        assert out["returned"]["chip"]["display"] == "none"

    def test_app_scroll_write_rearms_nothing_and_is_idempotent(self) -> None:
        out = _run_node(_harness())

        assert out["reasserted"]["snap"]["autoScroll"] is True
        assert out["reasserted"]["chip"] == {"display": "none", "text": ""}
        assert out["reasserted"]["jump"] == "none"

        # Already at the tail: no write at all (the per-keystroke composer
        # auto-grow used to re-issue it on every input).
        assert out["writes"]["idleWrites"] == 0
        assert out["writes"]["afterGrowth"] == [1300]
        assert out["writes"]["top"] == 1300
