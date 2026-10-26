"""Execute the production controller against deterministic upload/status peers."""
from test_frontend_attachments_module import run_attachments


def test_acknowledged_resource_survives_old_snapshot_and_newer_delete_wins() -> None:
    result = run_attachments(
        """
        (async () => {
          const file = { name: "report.txt", size: 3 };
          apiImpl = async () => ({ ok: true, attachment: { id: "a" }, attachments: [{ id: "a", display_name: file.name }], pending_attachment: true, attachments_revision: 2 });
          await controller.stageFiles([file]);
          sessionCatalog.set("latestSessions", [{ session_id: "sid", busy: false, attachments_revision: 1, staged_attachments: [], pending_attachment: false }]);
          controller.syncStagedAttachmentsFromSelectedSession();
          const stale = { staged: controller.getStagedAttachments(), row: sessionCatalog.get("sessionIndex").get("sid"), badge: attachBtn.children[0].textContent };
          sessionCatalog.set("latestSessions", [{ session_id: "sid", attachments_revision: 3, staged_attachments: [], pending_attachment: false }]);
          controller.syncStagedAttachmentsFromSelectedSession();
          process.stdout.write(JSON.stringify({ stale, deleted: controller.getStagedAttachments(), badge: attachBtn.children[0].textContent }));
        })();
        """
    )
    assert result["stale"]["staged"][0]["id"] == "a"
    assert result["stale"]["row"]["busy"] is False
    assert result["stale"]["row"]["pending_attachment"] is True
    assert result["stale"]["badge"] == "1"
    assert result["deleted"] == []
    assert result["badge"] == ""


def test_lost_commit_response_recovers_by_uncached_identity_without_duplicate_post() -> None:
    result = run_attachments(
        """
        (async () => {
          const file = { name: "中文 report.txt", size: 3, arrayBuffer: () => { throw new Error("must not buffer/base64"); } };
          const requests = [];
          apiImpl = async (path, options) => {
            requests.push({ path, cache: options.cache, timeout: options.timeoutMs, rawSame: options.rawBody === file });
            if (path.endsWith("inject_file")) throw new Error("response body lost");
            return { ok: true, state: "committed", active: true, attachment: { id: "a" }, attachments: [{ id: "a" }], pending_attachment: true, attachments_revision: 4 };
          };
          const staged = await controller.stageFiles([file]);
          process.stdout.write(JSON.stringify({ staged, requests, apiCalls, list: controller.getStagedAttachments(), pending: controller.hasPendingUploads() }));
        })();
        """
    )
    assert result["staged"] is True
    assert len(result["requests"]) == 2
    assert result["requests"][0]["rawSame"] is True
    upload_id = result["apiCalls"][0]["headers"]["X-Codoxear-Upload-Id"]
    assert result["apiCalls"][0]["headers"]["X-Codoxear-Filename"] == "%E4%B8%AD%E6%96%87%20report.txt"
    assert result["requests"][1]["path"].endswith("upload_id=" + upload_id)
    assert result["requests"][1]["cache"] == "no-store"
    assert result["requests"][1]["timeout"] == 15000
    assert [item["id"] for item in result["list"]] == ["a"]
    assert result["pending"] is False


def test_http_200_invalid_commit_json_reconciles_instead_of_definite_failure() -> None:
    result = run_attachments(
        """
        (async () => {
          apiImpl = async (path) => {
            if (path.endsWith("inject_file")) throw Object.assign(new SyntaxError("HTTP 200 · response was not JSON"), { status: 200, rawText: "<html>gateway</html>" });
            return { ok: true, state: "committed", active: true, attachment: { id: "a" }, attachments: [{ id: "a" }], pending_attachment: true, attachments_revision: 2 };
          };
          const staged = await controller.stageFiles([{ name: "committed.txt", size: 3 }]);
          process.stdout.write(JSON.stringify({ staged, calls: apiCalls, list: controller.getStagedAttachments(), pending: controller.hasPendingUploads() }));
        })();
        """
    )
    assert result["staged"] is True
    assert len(result["calls"]) == 2
    assert result["calls"][0]["path"].endswith("inject_file")
    assert "attachments/upload-status?upload_id=" in result["calls"][1]["path"]
    assert result["calls"][1]["cache"] == "no-store"
    assert [item["id"] for item in result["list"]] == ["a"]
    assert result["pending"] is False


def test_unknown_outcome_retains_blob_and_explicit_retry_checks_before_same_id_post() -> None:
    result = run_attachments(
        """
        (async () => {
          const file = { name: "report.txt", size: 3 };
          let recovering = false;
          const blobs = [];
          apiImpl = async (path, options) => {
            if (path.endsWith("inject_file")) {
              blobs.push(options.rawBody === file);
              if (!recovering) throw new Error("interrupted");
              return { ok: true, attachment: { id: "a" }, attachments: [{ id: "a" }], pending_attachment: true, attachments_revision: 2 };
            }
            if (!recovering) throw new Error("status unreachable");
            return { ok: true, state: "absent", attachments: [], pending_attachment: false, attachments_revision: 0 };
          };
          await controller.stageFiles([file]);
          const failed = tray.children[0];
          const before = { chips: tray.children.length, pending: controller.hasPendingUploads(), text: failed.children.find((c) => c.className === "stagedAttachmentPhase").textContent };
          recovering = true;
          await failed.children.find((c) => c.className === "stagedAttachmentRetry").onclick();
          process.stdout.write(JSON.stringify({ before, apiCalls, blobs, list: controller.getStagedAttachments(), pending: controller.hasPendingUploads() }));
        })();
        """
    )
    assert result["before"]["chips"] == 1
    assert result["before"]["pending"] is True
    assert "Upload outcome unknown: status unreachable" in result["before"]["text"]
    paths = [call["path"] for call in result["apiCalls"]]
    assert paths[0].endswith("inject_file") and paths[1] == paths[2] and paths[3].endswith("inject_file")
    assert result["apiCalls"][0]["headers"]["X-Codoxear-Upload-Id"] == result["apiCalls"][3]["headers"]["X-Codoxear-Upload-Id"]
    assert result["blobs"] == [True, True]
    assert result["pending"] is False
    assert len(result["list"]) == 1


def test_committed_inactive_status_is_terminal_and_never_restages_removed_file() -> None:
    result = run_attachments(
        """
        (async () => {
          apiImpl = async (path) => {
            if (path.endsWith("inject_file")) throw new Error("lost response");
            return { ok: true, state: "committed", active: false, attachments: [], pending_attachment: false, attachments_revision: 8 };
          };
          const staged = await controller.stageFiles([{ name: "removed.txt", size: 3 }]);
          process.stdout.write(JSON.stringify({ staged, calls: apiCalls.length, list: controller.getStagedAttachments(), chips: tray.children.length, toasts }));
        })();
        """
    )
    assert result == {"staged": False, "calls": 2, "list": [], "chips": 0, "toasts": ["upload already removed or sent"]}


def test_late_upload_ack_cannot_reverse_newer_deletion() -> None:
    result = run_attachments(
        """
        (async () => {
          let complete;
          apiImpl = async () => new Promise((resolve) => { complete = resolve; });
          const pending = controller.stageFiles([{ name: "late.txt", size: 3 }]);
          while (!complete) await new Promise((resolve) => setImmediate(resolve));
          sessionCatalog.patchSession("sid", { attachments_revision: 5, staged_attachments: [], pending_attachment: false });
          complete({ ok: true, attachment: { id: "a" }, attachments: [{ id: "a" }], pending_attachment: true, attachments_revision: 4 });
          await pending;
          process.stdout.write(JSON.stringify({ list: controller.getStagedAttachments(), row: sessionCatalog.get("sessionIndex").get("sid"), chips: tray.children.length }));
        })();
        """
    )
    assert result["list"] == []
    assert result["row"]["attachments_revision"] == 5
    assert result["row"]["pending_attachment"] is False
    assert result["chips"] == 0


def test_bounded_retries_always_reconcile_and_do_not_send_a_prompt() -> None:
    result = run_attachments(
        """
        (async () => {
          apiImpl = async (path) => {
            if (path.endsWith("inject_file")) throw new Error("interrupted");
            return { ok: true, state: "absent", attachments: [], pending_attachment: false, attachments_revision: 0 };
          };
          await controller.stageFiles([{ name: "retry.txt", size: 3 }]);
          process.stdout.write(JSON.stringify({ apiCalls, chips: tray.children.length, pending: controller.hasPendingUploads(), reason: tray.children[0].children.find((c) => c.className === "stagedAttachmentPhase").textContent }));
        })();
        """
    )
    assert len(result["apiCalls"]) == 4
    assert [call["path"].split("/")[-1].split("?")[0] for call in result["apiCalls"]] == ["inject_file", "upload-status", "inject_file", "upload-status"]
    assert result["apiCalls"][0]["headers"] == result["apiCalls"][2]["headers"]
    assert result["chips"] == 1
    assert result["pending"] is False  # two authoritative absence reads
    assert "after two attempts" in result["reason"]


def test_explicit_cancel_wins_over_late_commit_and_never_retries_upload() -> None:
    result = run_attachments(
        """
        (async () => {
          let complete;
          apiImpl = async (path, options) => {
            if (path.endsWith("inject_file")) return new Promise((resolve) => { complete = resolve; });
            return { ok: true, state: "cancelled", attachments: [], pending_attachment: false, attachments_revision: 6 };
          };
          const pending = controller.stageFiles([{ name: "cancel.txt", size: 3 }]);
          while (!complete) await new Promise((resolve) => setImmediate(resolve));
          await tray.children[0].children.find((c) => c.className === "stagedAttachmentRemove").onclick();
          complete({ ok: true, attachment: { id: "a" }, attachments: [{ id: "a" }], attachments_revision: 5 });
          await pending;
          process.stdout.write(JSON.stringify({ apiCalls, chips: tray.children.length, row: sessionCatalog.get("sessionIndex").get("sid"), pending: controller.hasPendingUploads(), toasts }));
        })();
        """
    )
    assert len(result["apiCalls"]) == 2
    assert result["apiCalls"][1]["path"].endswith("upload-cancel")
    assert result["apiCalls"][1]["body"]["upload_id"] == result["apiCalls"][0]["headers"]["X-Codoxear-Upload-Id"]
    assert result["chips"] == 0
    assert result["row"]["attachments_revision"] == 6
    assert result["row"]["staged_attachments"] == []
    assert result["pending"] is False
    assert result["toasts"] == []


def test_uncertain_cancel_retains_visible_recovery_and_retry_only_cancels() -> None:
    result = run_attachments(
        """
        (async () => {
          let cancelWorks = false;
          apiImpl = async (path) => {
            if (path.endsWith("upload-cancel") && cancelWorks) return { ok: true, state: "cancelled", attachments: [], pending_attachment: false, attachments_revision: 2 };
            throw new Error("unreachable");
          };
          await controller.stageFiles([{ name: "cancel.txt", size: 3 }]);
          await tray.children[0].children.find((c) => c.className === "stagedAttachmentRemove").onclick();
          const phase = tray.children[0].children.find((c) => c.className === "stagedAttachmentPhase").textContent;
          const pending = controller.hasPendingUploads();
          cancelWorks = true;
          await tray.children[0].children.find((c) => c.className === "stagedAttachmentRetry").onclick();
          process.stdout.write(JSON.stringify({ phase, pending, paths: apiCalls.map((call) => call.path), chips: tray.children.length }));
        })();
        """
    )
    assert "Cancellation outcome unknown: unreachable" == result["phase"]
    assert result["pending"] is True
    assert result["paths"][-1].endswith("upload-cancel")
    assert result["paths"][-2].endswith("upload-cancel")
    assert result["chips"] == 0


def test_validation_rejection_does_not_blind_retry_or_lookup() -> None:
    result = run_attachments(
        """
        (async () => {
          apiImpl = async () => { throw Object.assign(new Error("file rejected"), { status: 413 }); };
          await controller.stageFiles([{ name: "bad.txt", size: 3 }]);
          process.stdout.write(JSON.stringify({ calls: apiCalls.length, chips: tray.children.length, pending: controller.hasPendingUploads(), phase: tray.children[0].children.find((c) => c.className === "stagedAttachmentPhase").textContent }));
        })();
        """
    )
    assert result == {"calls": 1, "chips": 1, "pending": False, "phase": "uploading: file rejected"}


def test_refresh_sidebar_and_selected_projection_use_reconciled_catalog_rows() -> None:
    result = run_attachments(
        """
        (async () => {
          apiImpl = async () => ({ ok: true, attachment: { id: "a" }, attachments: [{ id: "a" }], pending_attachment: true, attachments_revision: 2 });
          await controller.stageFiles([{ name: "report.txt", size: 3 }]);
          let sidebar, runtime;
          const noop = () => {};
          const refresh = ctx.window.CodoxearSessionRefresh.createSessionRefreshController({
            api: async () => ({ sessions: [{ session_id: "sid", busy: false, attachments_revision: 0, staged_attachments: [], pending_attachment: false }] }),
            isDisposed: () => false, apiResponseNotModified: () => false, sessionCatalog,
            emptyDefaults: () => ({}), clearFileDiscoveryCaches: noop, useDesktopSessionActions: () => true,
            sessionState, clearSelectedSessionAfterRemoval: noop, applySessionListTranscriptIdentity: noop,
            syncAttachments: () => controller.syncStagedAttachmentsFromSelectedSession(), clearAttachments: noop,
            renderSessions: (rows) => { sidebar = rows; return true; }, hasDeferredRefresh: () => false,
            updateTypingStats: (row) => { runtime = row; }, maybeSelectPendingHashSession: noop,
          });
          const rows = await refresh.refreshSessions();
          process.stdout.write(JSON.stringify({ sidebar, runtime, rows, list: controller.getStagedAttachments() }));
        })();
        """
    )
    for row in [result["sidebar"][0], result["runtime"], result["rows"][0]]:
        assert row["attachments_revision"] == 2
        assert row["pending_attachment"] is True
        assert row["busy"] is False
    assert len(result["list"]) == 1


def test_pending_status_is_polled_with_backoff_then_retained_without_duplicate_upload() -> None:
    result = run_attachments(
        """
        (async () => {
          apiImpl = async (path) => {
            if (path.endsWith("inject_file")) throw new Error("lost response");
            return { ok: true, state: "pending", attachments: [], pending_attachment: false, attachments_revision: 0 };
          };
          let done = false;
          const staging = controller.stageFiles([{ name: "slow.txt", size: 3 }]).then(() => { done = true; });
          const waits = [];
          while (!done) {
            for (const timer of scheduled.filter((timer) => !timer.cleared)) {
              timer.cleared = true; waits.push(timer.ms); timer.fn();
            }
            await new Promise((resolve) => setImmediate(resolve));
          }
          await staging;
          process.stdout.write(JSON.stringify({ paths: apiCalls.map((call) => call.path), waits, chips: tray.children.length, pending: controller.hasPendingUploads(), phase: tray.children[0].children.find((c) => c.className === "stagedAttachmentPhase").textContent }));
        })();
        """
    )
    assert len(result["paths"]) == 4
    assert sum(path.endswith("inject_file") for path in result["paths"]) == 1
    assert result["waits"] == [500, 1000]
    assert result["chips"] == 1
    assert result["pending"] is True
    assert "still processing" in result["phase"]


def test_send_cleanup_verifies_resource_without_unversioned_clear_or_count_rollback() -> None:
    result = run_attachments(
        """
        (async () => {
          sessionCatalog.patchSession("sid", { attachments_revision: 2, staged_attachments: [{ id: "a" }], pending_attachment: true });
          controller.syncStagedAttachmentsFromSelectedSession();
          let resolveRead;
          apiImpl = async (path) => new Promise((resolve) => { resolveRead = resolve; });
          controller.setSelectedSessionPendingAttachment("sid", false);
          controller.setAttachCount(0);
          const before = controller.getStagedAttachments();
          resolveRead({ ok: true, attachments: [], pending_attachment: false, attachments_revision: 3 });
          await new Promise((resolve) => setImmediate(resolve));
          process.stdout.write(JSON.stringify({ before, apiCalls, list: controller.getStagedAttachments(), row: sessionCatalog.get("sessionIndex").get("sid") }));
        })();
        """
    )
    assert len(result["before"]) == 1
    assert result["apiCalls"][0]["path"].endswith("/attachments")
    assert result["apiCalls"][0]["cache"] == "no-store"
    assert result["list"] == []
    assert result["row"]["attachments_revision"] == 3


def test_recovery_css_keeps_error_reason_wrapped_and_retry_touch_sized() -> None:
    from css_tokens import APP_CSS, declarations, parse_stylesheet, selector_text, top_level_rules

    styles = {}
    for rule in top_level_rules(parse_stylesheet(APP_CSS)):
        styles.setdefault(selector_text(rule), {}).update(declarations(rule))
    failed = styles[".stagedAttachmentChip[data-failed]"]
    reason = styles[".stagedAttachmentChip[data-failed] .stagedAttachmentPhase"]
    retry = styles[".stagedAttachmentRetry"]
    assert failed["height"] == "auto"
    assert failed["flex-wrap"] == "wrap"
    assert reason["white-space"] == "normal"
    assert reason["overflow-wrap"] == "anywhere"
    assert retry["min-height"] == "44px"
    assert retry["min-width"] == "44px"
    cancel = styles[".stagedAttachmentChip[data-pending] .stagedAttachmentRemove"]
    assert cancel["width"] == "44px"
    assert cancel["min-height"] == "44px"
    assert styles[".stagedAttachmentChip[data-pending]"]["height"] == "auto"


def test_unknown_upload_remains_send_blocking_when_retry_hits_a_session_blocker() -> None:
    result = run_attachments(
        """
        (async () => {
          apiImpl = async () => { throw new Error("unreachable"); };
          await controller.stageFiles([{ name: "unknown.txt", size: 3 }]);
          sessionState.set("sending", true);
          await tray.children[0].children.find((c) => c.className === "stagedAttachmentRetry").onclick();
          process.stdout.write(JSON.stringify({ calls: apiCalls.length, pending: controller.hasPendingUploads(), text: tray.children[0].children.find((c) => c.className === "stagedAttachmentPhase").textContent }));
        })();
        """
    )
    assert result["calls"] == 2
    assert result["pending"] is True
    assert "Wait for the current send" in result["text"]


def test_delete_ack_updates_original_session_even_after_selection_switch() -> None:
    result = run_attachments(
        """
        (async () => {
          sessionCatalog.set("latestSessions", [
            { session_id: "sid", start_ts: 10, attachments_revision: 1, staged_attachments: [{ id: "a" }], pending_attachment: true },
            { session_id: "other", start_ts: 20, attachments_revision: 0, staged_attachments: [], pending_attachment: false },
          ]);
          controller.syncStagedAttachmentsFromSelectedSession();
          let complete;
          apiImpl = async () => new Promise((resolve) => { complete = resolve; });
          const removing = tray.children[0].children.find((c) => c.className === "stagedAttachmentRemove").onclick();
          sessionState.set("selected", "other");
          complete({ ok: true, attachments: [], pending_attachment: false, attachments_revision: 2 });
          await removing;
          process.stdout.write(JSON.stringify({ old: sessionCatalog.get("sessionIndex").get("sid"), other: sessionCatalog.get("sessionIndex").get("other"), toasts }));
        })();
        """
    )
    assert result["old"]["attachments_revision"] == 2
    assert result["old"]["staged_attachments"] == []
    assert result["other"]["attachments_revision"] == 0
    assert result["toasts"] == []


def test_selection_switch_aborts_an_already_started_post_without_status_or_failure() -> None:
    result = run_attachments(
        """
        (async () => {
          let started = false, signal;
          apiImpl = async (path, options) => new Promise((resolve, reject) => {
            started = true; signal = options.signal;
            signal.listeners.push(() => reject(Object.assign(new Error("aborted"), { name: "AbortError" })));
          });
          const staging = controller.stageFiles([{ name: "switch.txt", size: 3 }]);
          while (!started) await new Promise((resolve) => setImmediate(resolve));
          sessionState.set("selected", "other");
          await staging;
          process.stdout.write(JSON.stringify({ aborted: signal.aborted, calls: apiCalls.length, chips: tray.children.length, pending: controller.hasPendingUploads(), toasts }));
        })();
        """
    )
    assert result == {"aborted": True, "calls": 1, "chips": 0, "pending": False, "toasts": []}


def test_successful_http_pending_response_reconciles_until_commit_instead_of_finishing() -> None:
    result = run_attachments(
        """
        (async () => {
          apiImpl = async (path) => path.endsWith("inject_file")
            ? { ok: true, state: "pending", attachments: [], pending_attachment: false, attachments_revision: 1 }
            : { ok: true, state: "committed", active: true, attachment: { id: "a" }, attachments: [{ id: "a" }], pending_attachment: true, attachments_revision: 2 };
          const staged = await controller.stageFiles([{ name: "pending.txt", size: 3 }]);
          process.stdout.write(JSON.stringify({ staged, paths: apiCalls.map((call) => call.path), list: controller.getStagedAttachments(), pending: controller.hasPendingUploads() }));
        })();
        """
    )
    assert result["staged"] is True
    assert len(result["paths"]) == 2
    assert result["paths"][1].split("?")[0].endswith("upload-status")
    assert [item["id"] for item in result["list"]] == ["a"]
    assert result["pending"] is False


def test_repeated_local_decode_failure_never_becomes_unknown_or_blocks_send() -> None:
    result = run_attachments(
        """
        (async () => {
          ctx.Image = class Image { async decode() { throw new Error("image decode failed"); } };
          const staged = await controller.stageFiles([{ name: "broken.png", type: "image/png", size: 2048 }]);
          const first = {
            pending: controller.hasPendingUploads(), calls: apiCalls.length,
            phase: tray.children[0].children.find((c) => c.className === "stagedAttachmentPhase").textContent,
          };
          const retry = await tray.children[0].children.find((c) => c.className === "stagedAttachmentRetry").onclick();
          process.stdout.write(JSON.stringify({
            staged, retry, first, calls: apiCalls.length, pending: controller.hasPendingUploads(), chips: tray.children.length,
            phase: tray.children[0].children.find((c) => c.className === "stagedAttachmentPhase").textContent,
            badge: attachBtn.children[0].textContent, badgeDisplay: attachBtn.children[0].style.display,
          }));
        })();
        """
    )
    assert result["staged"] is False
    assert result["retry"] is False
    assert result["first"] == {"pending": False, "calls": 0, "phase": "compressing: image decode failed"}
    assert result["calls"] == 0
    assert result["pending"] is False
    assert result["chips"] == 1
    assert result["phase"] == "compressing: image decode failed"
    assert result["badge"] == ""
    assert result["badgeDisplay"] == "none"


def test_badge_counts_unknown_upload_but_not_definite_rejection() -> None:
    result = run_attachments(
        """
        (async () => {
          apiImpl = async () => { throw Object.assign(new Error("rejected"), { status: 413 }); };
          await controller.stageFiles([{ name: "rejected.txt", size: 3 }]);
          const rejected = { badge: attachBtn.children[0].textContent, pending: controller.hasPendingUploads(), chips: tray.children.length };
          apiImpl = async () => { throw new Error("unreachable"); };
          await controller.stageFiles([{ name: "unknown.txt", size: 3 }]);
          process.stdout.write(JSON.stringify({ rejected, badge: attachBtn.children[0].textContent, pending: controller.hasPendingUploads(), chips: tray.children.length }));
        })();
        """
    )
    assert result["rejected"] == {"badge": "", "pending": False, "chips": 1}
    assert result["badge"] == "1"
    assert result["pending"] is True
    assert result["chips"] == 2


def test_retry_status_failure_preserves_real_prior_commit_uncertainty() -> None:
    result = run_attachments(
        """
        (async () => {
          apiImpl = async () => { throw new Error("connection interrupted"); };
          await controller.stageFiles([{ name: "uncertain.txt", size: 3 }]);
          await tray.children[0].children.find((c) => c.className === "stagedAttachmentRetry").onclick();
          process.stdout.write(JSON.stringify({
            paths: apiCalls.map((call) => call.path), pending: controller.hasPendingUploads(), badge: attachBtn.children[0].textContent,
            phase: tray.children[0].children.find((c) => c.className === "stagedAttachmentPhase").textContent,
          }));
        })();
        """
    )
    assert len(result["paths"]) == 3
    assert result["paths"][0].endswith("inject_file")
    assert result["paths"][1] == result["paths"][2]
    assert result["pending"] is True
    assert result["badge"] == "1"
    assert result["phase"] == "Upload outcome unknown: connection interrupted"
