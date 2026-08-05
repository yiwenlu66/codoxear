from test_frontend_attachments_module import run_attachments


def test_attach_button_reflects_session_selection_and_recovery_blockers() -> None:
    result = run_attachments(
        """
        const states = [];
        function record(label, info) {
          selected = label === "none" ? null : "sid";
          if (info) sessions.set("sid", info); else sessions.delete("sid");
          controller.syncAttachButtonState();
          states.push({ label, disabled: attachBtn.disabled, title: attachBtn.title, aria: attachBtn.attrs["aria-label"] });
        }
        record("none", null);
        record("idle", { session_id: "sid", launch_state: "ready" });
        record("busy", { session_id: "sid", launch_state: "ready", busy: true });
        record("failed", { session_id: "sid", launch_state: "failed" });
        record("unknown", { session_id: "sid", launch_state: "ready", commit_unknown_send: true });
        record("orphan", { session_id: "sid", launch_state: "ready", orphan_recovery: true });
        record("queue-recovery", { session_id: "sid", launch_state: "ready", queue_recovery: true });
        process.stdout.write(JSON.stringify({ states }));
        """
    )
    states = {row["label"]: row for row in result["states"]}
    assert states["none"] == {
        "label": "none",
        "disabled": True,
        "title": "Select a session to attach a file",
        "aria": "Select a session to attach a file",
    }
    assert states["idle"] == {
        "label": "idle",
        "disabled": False,
        "title": "Attach file (max 1024 B)",
        "aria": "Attach file (max 1024 B)",
    }
    assert states["busy"]["disabled"] is False
    assert states["busy"]["title"] == "Attach file (max 1024 B)"
    for label, title in {
        "failed": "Failed launch cannot receive file attachments",
        "unknown": "Resolve the unknown send before attaching a file",
        "orphan": "Missing session can only be reviewed",
        "queue-recovery": "Review preserved queued recovery items before attaching a file",
    }.items():
        assert states[label]["disabled"] is True
        assert states[label]["title"] == title
        assert states[label]["aria"] == title


def test_attach_button_blocks_client_send_in_progress() -> None:
    result = run_attachments(
        """
        sending = true;
        controller.syncAttachButtonState();
        process.stdout.write(JSON.stringify({ disabled: attachBtn.disabled, title: attachBtn.title, aria: attachBtn.attrs["aria-label"] }));
        """
    )
    assert result == {
        "disabled": True,
        "title": "Wait for the current send to finish before attaching a file",
        "aria": "Wait for the current send to finish before attaching a file",
    }
