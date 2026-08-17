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
      snoozeMode = ["none", "4h", "tomorrow", "custom"].includes(mode) ? mode : "none";
      for (const [value, button] of editSnoozeModeButtons.entries()) button.classList.toggle("active", value === snoozeMode);
      editSnoozeCustomRow.style.display = snoozeMode === "custom" ? "grid" : "none";
    }

    function tomorrowSnoozeSeconds() {
      const date = new Date(now());
      date.setDate(date.getDate() + 1);
      date.setHours(9, 0, 0, 0);
      return Math.floor(date.getTime() / 1000);
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
      const snoozeUntil = Number(session.snooze_until || 0);
      if (snoozeUntil > now() / 1000) {
        setEditSnoozeMode("custom");
        fillCustomSnoozeInputs(snoozeUntil);
      } else {
        setEditSnoozeMode("none");
        fillCustomSnoozeInputs(tomorrowSnoozeSeconds());
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
      if (snoozeMode === "4h") snoozeUntil = Math.floor(now() / 1000) + 4 * 3600;
      else if (snoozeMode === "tomorrow") snoozeUntil = tomorrowSnoozeSeconds();
      else if (snoozeMode === "custom") {
        const dateRaw = String(editSnoozeCustomDate.value || "").trim();
        const timeRaw = String(editSnoozeCustomTime.value || "").trim();
        if (!dateRaw || !timeRaw) { editStatus.textContent = "Choose both a custom date and time."; return; }
        const parsed = Date.parse(`${dateRaw}T${timeRaw}`);
        if (!Number.isFinite(parsed)) { editStatus.textContent = "Invalid snooze time."; return; }
        snoozeUntil = Math.floor(parsed / 1000);
      }
      try {
        editSaveBtn.disabled = true;
        editStatus.textContent = "Saving...";
        await api(`/api/sessions/${sid}/edit`, { method: "POST", body: {
          name: String(editNameInput.value || ""),
          priority_offset: Number(editPriorityRange.value || 0),
          snooze_until: snoozeUntil,
          dependency_session_id: String(editDependencyBtn.dataset.value || "") || null,
        }});
        await refreshSessions();
        if (editSessionId !== sid) return;
        hideEditSession();
        setToast("conversation updated");
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
      if (mode === "tomorrow") fillCustomSnoozeInputs(tomorrowSnoozeSeconds());
      else if (mode === "4h") fillCustomSnoozeInputs(Math.floor(now() / 1000) + 4 * 3600);
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
