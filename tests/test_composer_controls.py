"""Execute the composer and queue owners; assert their displayed control state."""
from test_composer_height_mode import run_scenario
from test_queue_button_source import _bootstrap_controller, run_node_json


def test_clear_visibility_tracks_the_grabber_condition_and_nonempty_text():
    result = run_scenario('''
      const h = harness();
      const probe = () => [h.composer.dataset.controls || null, h.clearBtn.style.display];
      h.textarea.value = "short"; h.textarea.emit("input"); const short = probe();
      h.textarea.scrollHeight = 400; h.textarea.emit("input"); const overflow = probe();
      h.expandBtn.emit("click"); h.textarea.value = ""; h.textarea.emit("input"); const emptyTall = probe();
      h.textarea.value = " "; h.textarea.emit("input"); const tallText = probe();
      process.stdout.write(JSON.stringify({short, overflow, emptyTall, tallText}));
    ''')
    assert result == {"short": [None, "none"], "overflow": ["stack", ""],
                      "emptyTall": ["stack", "none"], "tallText": ["stack", ""]}


def test_clear_undo_round_trip_uses_draft_sync_and_preserves_height():
    result = run_scenario('''
      const h = harness(); h.expandBtn.emit("click");
      const text = "first line\\nsecond line ";
      h.controller.setDraftFromServer("sid", text);
      h.clearBtn.emit("click");
      const cleared = {value: h.textarea.value, chip: h.draftUndoBtn.style.display,
        strip: h.composerStrip.style.display, grab: h.expandBtn.style.display,
        cache: h.storage.get("codexweb.draft.sid") || null, mode: h.snapshot().mode};
      h.draftUndoBtn.emit("click");
      process.stdout.write(JSON.stringify({cleared, restored: h.textarea.value, edits: h.edits,
        cache: h.storage.get("codexweb.draft.sid"), caret: h.textarea.selectionStart,
        focus: h.calls, chip: h.draftUndoBtn.style.display, grab: h.expandBtn.style.display}));
    ''')
    assert result["cleared"] == {"value": "", "chip": "inline-flex", "strip": "flex", "grab": "none", "cache": None, "mode": "tall"}
    assert result["restored"] == "first line\nsecond line " == result["cache"]
    assert result["edits"] == ["", result["restored"]]
    assert result["caret"] == len(result["restored"])
    assert result["focus"] == [["focus"]]
    assert result["chip"] == "none" and result["grab"] == ""


def test_undo_dismissals_are_permanent_and_never_clobber_a_new_draft():
    result = run_scenario('''
      const h = harness();
      const clear = () => { h.controller.setDraftFromServer("sid", "saved"); h.clearBtn.emit("click"); };
      const chip = () => h.draftUndoBtn.style.display;
      clear(); h.textarea.value = "pasted"; h.textarea.emit("input"); const typed = chip();
      h.textarea.value = ""; h.textarea.emit("input"); const erased = chip();
      h.draftUndoBtn.emit("click"); const staleUndo = h.textarea.value;
      clear(); h.controller.setDraftFromServer("sid", "remote"); h.draftUndoBtn.emit("click");
      const remote = h.textarea.value, noClobber = chip();
      clear(); h.sessionState.set("selected", "other"); const switched = chip();
      h.sessionState.set("selected", "sid"); const back = chip();
      clear(); await h.controller.sendText("steering"); const sent = chip();
      clear(); h.controller.clearComposer(); const drained = chip();
      clear(); h.controller.showSendChoice("queued"); await h.sendChoiceLaterBtn.onclick(); const queued = chip();
      process.stdout.write(JSON.stringify({typed, erased, staleUndo, remote, noClobber, switched, back, sent, drained, queued}));
    ''')
    assert result == {"typed": "none", "erased": "none", "staleUndo": "", "remote": "remote",
                      "noClobber": "none", "switched": "none", "back": "none", "sent": "none", "drained": "none", "queued": "none"}


def test_queue_visibility_follows_depth_and_keeps_orphan_recovery_reachable():
    result = run_node_json(_bootstrap_controller('''
      const state = deps.sessionState;
      const probe = () => [queueBtn.style.display, queueBtn.children[0].textContent, queueBtn.disabled, queueBtn.title];
      const noSession = probe();
      sessions.set("sid-1", session()); state.set("selected", "sid-1"); const empty = probe();
      state.set("queueLen", 2); const queued = probe();
      state.set("queueLen", 0); const drained = probe();
      sessions.set("sid-1", session({orphan_recovery: true, queue_recovery: true, queue_len: 1}));
      controller.syncQueueSubmitState(); const recovery = probe();
      sessions.set("sid-1", session({commit_unknown_send: true}));
      controller.syncQueueSubmitState(); const unknown = probe();
      globalThis.__result = {noSession, empty, queued, drained, recovery, unknown};
    '''))
    assert result["noSession"][0] == result["empty"][0] == result["drained"][0] == "none"
    assert result["queued"][:3] == ["", "2", False]
    assert result["recovery"] == ["", "", False, "Review preserved queued recovery items"]
    assert result["unknown"][0] == "none" and result["unknown"][2] is True


def test_failed_submissions_keep_undo_and_clear_never_mutates_attachments():
    result = run_scenario('''
      const staged = [{id: "attachment", path: "photo.png"}];
      const h = harness({staged, sendOk: false, queueOk: false});
      h.controller.setDraftFromServer("sid", "saved"); h.clearBtn.emit("click");
      const sent = await h.controller.sendText("steer");
      const afterSend = h.draftUndoBtn.style.display;
      const q = harness({queueOk: false});
      q.controller.setDraftFromServer("sid", "saved"); q.clearBtn.emit("click");
      q.controller.showSendChoice("queue"); await q.sendChoiceLaterBtn.onclick();
      const afterQueue = q.draftUndoBtn.style.display;
      h.draftUndoBtn.emit("click");
      process.stdout.write(JSON.stringify({staged, sent, afterSend, afterQueue, text: h.textarea.value}));
    ''')
    assert result == {"staged": [{"id": "attachment", "path": "photo.png"}], "sent": False,
                      "afterSend": "inline-flex", "afterQueue": "inline-flex", "text": "saved"}


def test_clear_control_arrives_with_row_room_not_with_the_grabber():
    result = run_scenario('''
      const probe = (h) => ({controls: h.composer.dataset.controls || null, clear: h.clearBtn.style.display,
        strip: h.composerStrip.style.display, grab: h.expandBtn.style.display});
      const twoLines = harness({ textareaHeight: 52 });
      twoLines.textarea.value = "a\\nb"; twoLines.textarea.emit("input"); const cramped = probe(twoLines);
      const fourLines = harness({ textareaHeight: 92 });
      fourLines.textarea.value = "a\\nb\\nc\\nd"; fourLines.textarea.emit("input"); const roomy = probe(fourLines);
      fourLines.textarea.value = ""; fourLines.textarea.emit("input"); const emptied = probe(fourLines);
      process.stdout.write(JSON.stringify({cramped, roomy, emptied}));
    ''')
    # Two icons need 2*32px + the 12px stack gap plus a 4px margin each side (84px):
    # a two-line draft has no room, a four-line draft does, and the height
    # control (grabber strip) only appears when the draft overflows the cap.
    assert result == {
        "cramped": {"controls": None, "clear": "none", "strip": "none", "grab": "none"},
        "roomy": {"controls": "stack", "clear": "", "strip": "none", "grab": "none"},
        "emptied": {"controls": "stack", "clear": "none", "strip": "none", "grab": "none"},
    }
