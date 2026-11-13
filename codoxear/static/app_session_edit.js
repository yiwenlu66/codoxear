import * as CodoxearSnooze from "./app_snooze.js";

const global = window;


  function createSessionEditController(options = {}) {
    const {
      documentTarget = global.document,
      ElementCtor = global.HTMLElement,
      el,
      editCloseBtn,
      editStatus,
      editNameInput,
      editPriorityRange,
      editPriorityValue,
      editPriorityResetBtn,
      editSnoozeModeButtons,
      editSnoozeCustomDate,
      editSnoozeCustomTime,
      editSnoozeCustomRow,
      editDependencyBtn,
      editDependencyMenu,
      editSaveBtn,
      editCancelBtn,
      editViewer,
      getSessionInfo,
      getSessions,
      sessionState,
      sessionDisplayName,
      baseName,
      formatPriorityOffset,
      setPickerButtonContent,
      api,
      refreshSessions,
      setToast,
      prepareModalOpen,
      afterModalVisibilityChanged,
      positionDialogMenu,
      addAppEvent = (target, type, handler, eventOptions) => target.addEventListener(type, handler, eventOptions),
      now = () => Date.now(),
      HTMLElementCtor = ElementCtor,
    } = options;

    if (!editViewer || !editSaveBtn || !editDependencyBtn || !editDependencyMenu) {
      throw new Error("Codoxear session edit controller missing edit dialog dependencies");
    }
    let editSessionId = null;
    let dependencyMenuOpen = false;
    let snoozeMode = "none";
    function namePropagationNotice(propagation) {
      // The alias store is authoritative: the rename already succeeded locally,
      // and this only explains what happened to the backend's own title.
      if (!propagation || typeof propagation !== "object" || propagation.ok === true) return "";
      const backend = String(propagation.backend || "").toLowerCase();
      const label = backend === "pi" ? "Pi" : backend === "codex" ? "Codex" : backend === "cc" ? "Claude Code" : "the backend";
      const reason = String(propagation.reason || "");
      if (reason === "unsupported") {
        return backend === "pi"
          ? "renamed — Pi needs /reload before it can adopt the title"
          : `renamed — ${label} cannot update its title here`;
      }
      if (reason === "unsupported-clear") return `renamed — ${label} keeps its previous title`;
      if (reason === "pending" || reason === "no-thread") return "";
      return `renamed — ${label} title not updated`;
    }

    function applyMenus() {
      editDependencyMenu.classList.toggle("open", dependencyMenuOpen);
      editDependencyBtn.setAttribute("aria-expanded", dependencyMenuOpen ? "true" : "false");
      if (dependencyMenuOpen) positionDialogMenu(editDependencyMenu, editDependencyBtn);
    }

    function closeDependencyMenu() {
      if (!dependencyMenuOpen) return;
      dependencyMenuOpen = false;
      applyMenus();
    }

    function hideEditSession() {
      editSessionId = null;
      editStatus.textContent = "";
      editSaveBtn.disabled = false;
      closeDependencyMenu();
      if (editViewer.open && typeof editViewer.close === "function") editViewer.close();
      else editViewer.style.display = "none";
      afterModalVisibilityChanged();
    }

    function syncEditPriorityLabel() {
      editPriorityValue.textContent = formatPriorityOffset(editPriorityRange.value);
    }

    function setEditSnoozeMode(mode) {
      snoozeMode = ["none", "evening", "tomorrow", "weekend", "nextWeek", "indefinitely", "custom"].includes(mode) ? mode : "none";
      for (const [value, button] of editSnoozeModeButtons.entries()) button.classList.toggle("active", value === snoozeMode);
      editSnoozeCustomRow.style.display = snoozeMode === "custom" ? "grid" : "none";
    }

    function renderSnoozeChips(presets) {
      for (const preset of presets) {
        const button = editSnoozeModeButtons.get(preset.mode);
        if (!button || !button.children || button.children.length < 2) continue;
        const [label, time] = button.children;
        if (label) label.textContent = preset.label;
        if (time) time.textContent = preset.timeText ? `· ${preset.timeText}` : "";
      }
    }

    function fillCustomSnoozeInputs(tsSeconds) {
      const ts = Number(tsSeconds);
      const date = Number.isFinite(ts) && ts > 0 ? new Date(ts * 1000) : new Date(now() + 24 * 3600 * 1000);
      const yyyy = String(date.getFullYear()).padStart(4, "0");
      const mm = String(date.getMonth() + 1).padStart(2, "0");
      const dd = String(date.getDate()).padStart(2, "0");
      const hh = String(date.getHours()).padStart(2, "0");
      const mi = String(date.getMinutes()).padStart(2, "0");
      editSnoozeCustomDate.value = `${yyyy}-${mm}-${dd}`;
      editSnoozeCustomTime.value = `${hh}:${mi}`;
    }

    function setDependencyButtonContent() {
      const value = String(editDependencyBtn.dataset.value || "");
      let label = "No dependency";
      if (value) {
        const session = getSessionInfo(value);
        if (session) label = `${sessionDisplayName(session)}${session.cwd ? ` | ${baseName(session.cwd)}` : ""}`;
      }
      setPickerButtonContent(editDependencyBtn, label);
    }

    function fillDependencyOptions(currentSid, currentDependencySid) {
      editDependencyMenu.innerHTML = "";
      const addItem = (value, label, active) => {
        const button = el("button", { class: "fileMenuItem" + (active ? " active" : ""), type: "button", title: label });
        button.appendChild(el("span", { class: "fileMenuPath", text: label }));
        button.onclick = () => {
          editDependencyBtn.dataset.value = value || "";
          setDependencyButtonContent();
          closeDependencyMenu();
        };
        editDependencyMenu.appendChild(button);
      };
      addItem("", "No dependency", !currentDependencySid);
      for (const session of getSessions()) {
        if (!session || session.session_id === currentSid) continue;
        const label = `${sessionDisplayName(session)}${session.cwd ? ` | ${baseName(session.cwd)}` : ""}`;
        addItem(session.session_id, label, currentDependencySid === session.session_id);
      }
      editDependencyBtn.dataset.value = currentDependencySid || "";
      setDependencyButtonContent();
    }

    function openEditSession(sid) {
      if (!sid) return;
      const session = getSessionInfo(sid);
      if (!session) return;
      editSessionId = sid;
      editStatus.textContent = "";
      editSaveBtn.disabled = false;
      editNameInput.value = typeof session.alias === "string" ? session.alias : "";
      editNameInput.placeholder = sessionDisplayName(session) || "Conversation title";
      editPriorityRange.value = String(Number(session.priority_offset || 0));
      syncEditPriorityLabel();
      const currentNow = now();
      const presets = CodoxearSnooze.resolveSnoozePresets(currentNow);
      renderSnoozeChips(presets);
      const snoozeUntil = Number(session.snooze_until);
      const matchingPreset = Number.isFinite(snoozeUntil) && snoozeUntil > 0
        ? presets.find((preset) => Math.floor(snoozeUntil) === preset.epoch)
        : null;
      if (session.snoozed_indefinitely === true) {
        setEditSnoozeMode("indefinitely");
        fillCustomSnoozeInputs(CodoxearSnooze.snoozePresetEpoch("tomorrow", currentNow));
      } else if (matchingPreset) {
        setEditSnoozeMode(matchingPreset.mode);
        fillCustomSnoozeInputs(matchingPreset.epoch);
      } else if (snoozeUntil > currentNow / 1000) {
        setEditSnoozeMode("custom");
        fillCustomSnoozeInputs(snoozeUntil);
      } else {
        setEditSnoozeMode("none");
        fillCustomSnoozeInputs(CodoxearSnooze.snoozePresetEpoch("tomorrow", currentNow));
      }
      fillDependencyOptions(sid, session.dependency_session_id || "");
      prepareModalOpen();
      if (!editViewer.open && typeof editViewer.showModal === "function") editViewer.showModal();
      else editViewer.style.display = "flex";
      afterModalVisibilityChanged();
    }

    async function saveEditSession() {
      const sid = editSessionId;
      if (!sid || editSaveBtn.disabled) return;
      let snoozeUntil = null;
      if (snoozeMode === "custom") {
        const dateRaw = String(editSnoozeCustomDate.value || "").trim();
        const timeRaw = String(editSnoozeCustomTime.value || "").trim();
        if (!dateRaw || !timeRaw) { editStatus.textContent = "Choose both a custom date and time."; return; }
        const parsed = Date.parse(`${dateRaw}T${timeRaw}`);
        if (!Number.isFinite(parsed)) { editStatus.textContent = "Invalid snooze time."; return; }
        snoozeUntil = Math.floor(parsed / 1000);
      } else if (snoozeMode !== "none" && snoozeMode !== "indefinitely") {
        snoozeUntil = CodoxearSnooze.snoozePresetEpoch(snoozeMode, now());
      }
      try {
        editSaveBtn.disabled = true;
        editStatus.textContent = "Saving...";
        const editResponse = await api(`/api/sessions/${sid}/edit`, { method: "POST", body: {
          name: String(editNameInput.value || ""),
          priority_offset: Number(editPriorityRange.value || 0),
          snooze_until: snoozeUntil,
          snoozed_indefinitely: snoozeMode === "indefinitely",
          dependency_session_id: String(editDependencyBtn.dataset.value || "") || null,
        }});
        await refreshSessions();
        if (editSessionId !== sid) return;
        hideEditSession();
        setToast(namePropagationNotice(editResponse && editResponse.name_propagation) || "conversation updated");
      } catch (error) {
        if (editSessionId === sid) editStatus.textContent = error && error.message ? error.message : "Save failed";
      } finally {
        if (editSessionId === sid) editSaveBtn.disabled = false;
      }
    }


    editPriorityRange.oninput = syncEditPriorityLabel;
    editPriorityResetBtn.onclick = () => { editPriorityRange.value = "0"; syncEditPriorityLabel(); };
    for (const [mode, button] of editSnoozeModeButtons.entries()) button.onclick = () => {
      setEditSnoozeMode(mode);
      const presetEpoch = CodoxearSnooze.snoozePresetEpoch(mode, now());
      if (presetEpoch !== null) fillCustomSnoozeInputs(presetEpoch);
    };
    editDependencyBtn.onclick = (event) => { event.preventDefault(); event.stopPropagation(); dependencyMenuOpen = !dependencyMenuOpen; applyMenus(); };
    editCloseBtn.onclick = () => hideEditSession();
    editCancelBtn.onclick = () => hideEditSession();
    editViewer.addEventListener("cancel", (event) => { event.preventDefault(); hideEditSession(); });
    editViewer.onclick = (event) => { if (event.target === editViewer) hideEditSession(); };
    editSaveBtn.onclick = () => { void saveEditSession(); };
    addAppEvent(documentTarget, "click", (event) => {
      const target = event.target instanceof ElementCtor ? event.target : null;
      if (target && dependencyMenuOpen && !target.closest("#editDependencyBtn") && !target.closest("#editDependencyMenu")) closeDependencyMenu();
    });
    return {
      viewer: editViewer,
      openEditSession,
      hideEditSession,
      applyMenus,
      closeDependencyMenu,
    };
  }

export { createSessionEditController };
