/* File viewer vim keybindings: motions, edit sub-modes, verbs, hints, chip.

   State model (one owner per axis):
   - viewerMode "view" | "edit" is owned by the file viewer controller
     (`currentFileEditMode`); entering edit mode always starts the vim layer
     in the "insert" sub-mode, and exiting edit mode forgets the sub-mode.
   - vimSubMode "insert" | "normal" is owned here and is only meaningful
     while the viewer is in edit mode. In normal mode the editor is not
     writable (`activeFileEditorWritable()` is false through the vim gate),
     so no native path can mutate the buffer; verbs temporarily lift
     readOnly, mutate, and restore it.

   The controller registers one document capture-phase keydown listener
   before `bindFileEditorInteractions()` so it sees keys before every other
   document listener. Guard chain (first match returns): viewer not open,
   nested blocking dialog, hint mode active, touch-selection active, target
   is a text-entry element other than the active Monaco input area. Consumed
   keys call preventDefault() + stopImmediatePropagation().

   `f` belongs to hint mode everywhere outside insert mode: it calls the
   hint controller's enter() directly, bypassing the text-entry activation
   guard that would otherwise swallow `f` while Monaco's hidden textarea is
   focused, so vim char-find stays permanently displaced. While hint mode is
   active this layer defers entirely (guard 3), so hint labels are never
   consumed here.
*/

  function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`file vim dependency missing: ${name}`);
    return value;
  }

  function requireNode(value, name) {
    if (!value || typeof value !== "object") throw new TypeError(`file vim dependency missing: ${name}`);
    return value;
  }

  const MOTION_COMMANDS = Object.freeze({
    h: "cursorLeft",
    l: "cursorRight",
    j: "cursorDown",
    k: "cursorUp",
    w: "cursorWordStartRight",
    b: "cursorWordStartLeft",
    e: "cursorWordEndRight",
    0: "cursorHome",
    $: "cursorEnd",
  });

  function createFileVimController(options = {}) {
    const addAppEvent = requireFunction(options.addAppEvent, "addAppEvent");
    const documentTarget = requireNode(options.document, "document");
    const fileVimModeChip = requireNode(options.fileVimModeChip, "fileVimModeChip");
    const fileDiff = requireNode(options.fileDiff, "fileDiff");
    const isFileViewerOpen = requireFunction(options.isFileViewerOpen, "isFileViewerOpen");
    const hasBlockingFileEditorModal = requireFunction(options.hasBlockingFileEditorModal, "hasBlockingFileEditorModal");
    const hintModeActive = requireFunction(options.hintModeActive, "hintModeActive");
    const touchSelectActive = requireFunction(options.touchSelectActive, "touchSelectActive");
    const fileEditorShortcutBlocked = requireFunction(options.fileEditorShortcutBlocked, "fileEditorShortcutBlocked");
    const currentFileEditMode = requireFunction(options.currentFileEditMode, "currentFileEditMode");
    const setFileEditMode = requireFunction(options.setFileEditMode, "setFileEditMode");
    const currentFileDirty = requireFunction(options.currentFileDirty, "currentFileDirty");
    const currentFileEditorKind = requireFunction(options.currentFileEditorKind, "currentFileEditorKind");
    const activeFileEditor = requireFunction(options.activeFileEditor, "activeFileEditor");
    const focusActiveFileEditor = requireFunction(options.focusActiveFileEditor, "focusActiveFileEditor");
    const activeFileEditorInsertWritable = requireFunction(options.activeFileEditorInsertWritable, "activeFileEditorInsertWritable");
    const syncFileEditorReadOnly = requireFunction(options.syncFileEditorReadOnly, "syncFileEditorReadOnly");
    const getFileEditorText = requireFunction(options.getFileEditorText, "getFileEditorText");
    const currentActiveFileText = requireFunction(options.currentActiveFileText, "currentActiveFileText");
    const setFileDirty = requireFunction(options.setFileDirty, "setFileDirty");
    const setToast = requireFunction(options.setToast, "setToast");
    const enterHintMode = requireFunction(options.enterHintMode, "enterHintMode");

    let vimSubMode = "insert";
    let pendingPrefix = "";

    // --- mode chip: one writer, invoked at construction and on every change.

    function renderModeChip() {
      const inEdit = Boolean(currentFileEditMode());
      fileVimModeChip.hidden = !inEdit;
      fileVimModeChip.textContent = inEdit && vimSubMode === "normal" ? "NORMAL" : "INSERT";
    }

    function syncEditMode() {
      // Entering edit mode always starts in insert; exiting forgets the
      // sub-mode entirely.
      vimSubMode = "insert";
      pendingPrefix = "";
      renderModeChip();
    }

    // --- editor access helpers.

    function monacoSurface() {
      const kind = currentFileEditorKind();
      if (kind !== "file" && kind !== "diff") return null;
      return activeFileEditor();
    }

    function editorModel(editor) {
      return editor && typeof editor.getModel === "function" ? editor.getModel() : null;
    }

    function goToLine(editor, position) {
      const model = editorModel(editor);
      if (!model || typeof model.getLineCount !== "function") return false;
      const lineCount = Math.max(1, Number(model.getLineCount()) || 1);
      const target = { lineNumber: position === "first" ? 1 : lineCount, column: 1 };
      if (typeof editor.setPosition === "function") editor.setPosition(target);
      if (typeof editor.revealPositionInCenter === "function") editor.revealPositionInCenter(target);
      return true;
    }

    function runCursorCommand(editor, command) {
      if (!editor || typeof editor.trigger !== "function") return false;
      editor.trigger("file-vim", command, null);
      return true;
    }

    function runMotion(editor, key) {
      focusActiveFileEditor();
      if (key === "G") return goToLine(editor, "last");
      const command = MOTION_COMMANDS[key];
      return command ? runCursorCommand(editor, command) : false;
    }

    function runHalfPage(editor, direction) {
      focusActiveFileEditor();
      if (!editor || typeof editor.trigger !== "function") return false;
      editor.trigger("file-vim", "cursorMove", { to: direction, by: "halfPage", value: 1, select: false });
      return true;
    }

    // Scroll fallback for cursor-less surfaces (markdown preview, PDF,
    // plain-text fallback): j/k/d/u/gg/G scroll the active scroll element;
    // word and line-edge motions have no meaning there and stay no-ops.

    function scrollSurface() {
      if (typeof fileDiff.querySelector !== "function") return fileDiff;
      const preview = fileDiff.querySelector(".fileMarkdownPreview");
      return preview || fileDiff;
    }

    function scrollByHalfPage(surface, direction) {
      const viewport = Math.max(1, Number(surface.clientHeight) || 0);
      const delta = direction === "down" ? viewport / 2 : -viewport / 2;
      surface.scrollTop = Math.max(0, (Number(surface.scrollTop) || 0) + delta);
      return true;
    }

    function scrollLines(surface, direction) {
      surface.scrollTop = Math.max(0, (Number(surface.scrollTop) || 0) + (direction === "down" ? 24 : -24));
      return true;
    }

    function scrollToEdge(surface, position) {
      if (position === "first") surface.scrollTop = 0;
      else surface.scrollTop = Number(surface.scrollHeight) || 0;
      return true;
    }

    function dispatchMotionKey(event, key) {
      const editor = monacoSurface();
      if (!editor) {
        // No cursor exists on this surface: scroll it instead.
        const surface = scrollSurface();
        if (key === "d" && (event.ctrlKey || event.metaKey)) return scrollByHalfPage(surface, "down");
        if (key === "u" && (event.ctrlKey || event.metaKey)) return scrollByHalfPage(surface, "up");
        if (event.ctrlKey || event.metaKey || event.altKey) return false;
        if (key === "j") return scrollLines(surface, "down");
        if (key === "k") return scrollLines(surface, "up");
        if (key === "G") return scrollToEdge(surface, "last");
        return false;
      }
      if (key === "d" && (event.ctrlKey || event.metaKey)) return runHalfPage(editor, "down");
      if (key === "u" && (event.ctrlKey || event.metaKey)) return runHalfPage(editor, "up");
      if (event.ctrlKey || event.metaKey || event.altKey) return false;
      if (key in MOTION_COMMANDS || key === "G") return runMotion(editor, key);
      return false;
    }

    // --- edit sub-mode transitions. readOnly stays derived: the vim normal
    // --- gate makes activeFileEditorWritable() false, so syncFileEditorReadOnly
    // --- applies the right option on every transition.

    function enterInsert() {
      vimSubMode = "insert";
      pendingPrefix = "";
      renderModeChip();
      syncFileEditorReadOnly();
      focusActiveFileEditor();
      return true;
    }

    function exitToNormal() {
      vimSubMode = "normal";
      pendingPrefix = "";
      renderModeChip();
      syncFileEditorReadOnly();
      return true;
    }

    function syncDirtyAfterEdit() {
      setFileDirty(String(getFileEditorText() || "") !== String(currentActiveFileText() || ""));
    }

    // Verbs temporarily lift readOnly (Monaco rejects edits while readOnly),
    // mutate, then restore. The finally clause re-derives readOnly from the
    // sub-mode, which is still "normal" here.
    function withWritableEditor(run) {
      const editor = monacoSurface();
      if (!editor || typeof editor.updateOptions !== "function") return false;
      if (!activeFileEditorInsertWritable()) return false;
      editor.updateOptions({ readOnly: false });
      try {
        return run(editor) !== false;
      } finally {
        editor.updateOptions({ readOnly: true });
        syncDirtyAfterEdit();
        syncFileEditorReadOnly();
      }
    }

    function deleteChar() {
      return withWritableEditor((editor) => runCursorCommand(editor, "deleteRight"));
    }

    function deleteLine() {
      return withWritableEditor((editor) => {
        const model = editorModel(editor);
        if (!model || typeof model.getLineMaxColumn !== "function" || typeof editor.getPosition !== "function") return false;
        const position = editor.getPosition() || { lineNumber: 1, column: 1 };
        const lineCount = Math.max(1, Number(model.getLineCount && model.getLineCount()) || 1);
        const lineNumber = Math.max(1, Math.min(lineCount, Number(position.lineNumber) || 1));
        const isLast = lineNumber >= lineCount;
        const endLine = isLast ? lineNumber : lineNumber + 1;
        const endColumn = isLast ? Math.max(1, Number(model.getLineMaxColumn(lineNumber)) || 1) : 1;
        if (typeof editor.pushUndoStop === "function") editor.pushUndoStop();
        editor.executeEdits("file-vim", [{
          range: { startLineNumber: lineNumber, startColumn: 1, endLineNumber: endLine, endColumn },
          text: "",
          forceMoveMarkers: true,
        }]);
        if (typeof editor.pushUndoStop === "function") editor.pushUndoStop();
        return true;
      });
    }

    function openLine(below) {
      let opened = false;
      opened = withWritableEditor((editor) => {
        const model = editorModel(editor);
        if (!model || typeof model.getLineMaxColumn !== "function" || typeof editor.getPosition !== "function") return false;
        const position = editor.getPosition() || { lineNumber: 1, column: 1 };
        const lineCount = Math.max(1, Number(model.getLineCount && model.getLineCount()) || 1);
        const lineNumber = Math.max(1, Math.min(lineCount, Number(position.lineNumber) || 1));
        const insertLine = below ? lineNumber : lineNumber - 1;
        const column = below ? Math.max(1, Number(model.getLineMaxColumn(lineNumber)) || 1) : 1;
        if (typeof editor.pushUndoStop === "function") editor.pushUndoStop();
        editor.executeEdits("file-vim", [{
          range: { startLineNumber: insertLine, startColumn: column, endLineNumber: insertLine, endColumn: column },
          text: "\n",
          forceMoveMarkers: true,
        }]);
        if (typeof editor.pushUndoStop === "function") editor.pushUndoStop();
        const cursorLine = below ? lineNumber + 1 : lineNumber;
        const clamped = Math.max(1, Math.min(lineCount + 1, cursorLine));
        if (typeof editor.setPosition === "function") editor.setPosition({ lineNumber: clamped, column: 1 });
        return true;
      });
      if (opened) enterInsert();
      return opened;
    }

    function editHistory(direction) {
      return withWritableEditor((editor) => runCursorCommand(editor, direction === "redo" ? "redo" : "undo"));
    }

    // --- escape chain: insert -> normal -> (clean buffer only) view mode.

    function handleEscape() {
      if (!currentFileEditMode()) return false;
      if (vimSubMode === "insert") return exitToNormal();
      if (currentFileDirty()) {
        setToast("unsaved changes");
        return true;
      }
      setFileEditMode(false);
      return true;
    }

    // --- key dispatch.

    function consume(event) {
      if (typeof event.preventDefault === "function") event.preventDefault();
      if (typeof event.stopImmediatePropagation === "function") event.stopImmediatePropagation();
    }

    function isPlainPrintable(event, key) {
      return key.length === 1 && !event.ctrlKey && !event.metaKey && !event.altKey;
    }

    function dispatchNormalKey(event, key) {
      if (key === "i" && !event.ctrlKey && !event.metaKey && !event.altKey) return enterInsert();
      if (key === "f" && !event.ctrlKey && !event.metaKey && !event.altKey) {
        pendingPrefix = "";
        enterHintMode();
        return true;
      }
      if (pendingPrefix === "g") {
        pendingPrefix = "";
        if (key === "g") {
          const editor = monacoSurface();
          if (editor) {
            focusActiveFileEditor();
            goToLine(editor, "first");
          } else {
            scrollToEdge(scrollSurface(), "first");
          }
        }
        return true;
      }
      if (pendingPrefix === "d") {
        pendingPrefix = "";
        if (key === "d") deleteLine();
        return true;
      }
      if (key === "d" && !event.ctrlKey && !event.metaKey && !event.altKey) {
        pendingPrefix = "d";
        return true;
      }
      if (key === "g" && !event.ctrlKey && !event.metaKey && !event.altKey) {
        pendingPrefix = "g";
        return true;
      }
      if (key === "x") return deleteChar();
      if (key === "u" && !event.ctrlKey && !event.metaKey && !event.altKey) return editHistory("undo");
      if (key === "r" && (event.ctrlKey || event.metaKey)) return editHistory("redo");
      if (key === "a" && !event.ctrlKey && !event.metaKey && !event.altKey && !event.shiftKey) {
        const editor = monacoSurface();
        if (editor) runCursorCommand(editor, "cursorRight");
        return enterInsert();
      }
      if (key === "A" && !event.ctrlKey && !event.metaKey && !event.altKey) {
        const editor = monacoSurface();
        if (editor) runCursorCommand(editor, "cursorEnd");
        return enterInsert();
      }
      if (key === "o" && !event.ctrlKey && !event.metaKey && !event.altKey) return openLine(true);
      if (key === "O" && !event.ctrlKey && !event.metaKey && !event.altKey) return openLine(false);
      return dispatchMotionKey(event, key);
    }

    function dispatchViewKey(event, key) {
      if (key === "f" && !event.ctrlKey && !event.metaKey && !event.altKey) {
        pendingPrefix = "";
        enterHintMode();
        return true;
      }
      if (pendingPrefix === "g") {
        pendingPrefix = "";
        if (key === "g") {
          const editor = monacoSurface();
          if (editor) {
            focusActiveFileEditor();
            goToLine(editor, "first");
          } else {
            scrollToEdge(scrollSurface(), "first");
          }
        }
        return true;
      }
      if (key === "g" && !event.ctrlKey && !event.metaKey && !event.altKey) {
        pendingPrefix = "g";
        return true;
      }
      return dispatchMotionKey(event, key);
    }

    function handleKeydown(event) {
      const e = event || {};
      if (e.defaultPrevented) return;
      if (!isFileViewerOpen()) return;
      // Opening a nested dialog or entering another key mode abandons any
      // pending prefix.
      if (hasBlockingFileEditorModal() || hintModeActive() || touchSelectActive()) {
        pendingPrefix = "";
        return;
      }
      if (e.isComposing) return;
      const target = e.target && typeof e.target.closest === "function" ? e.target : null;
      if (fileEditorShortcutBlocked(target)) return;
      const key = String(e.key || "");
      if (key === "Escape") {
        if (handleEscape()) consume(e);
        return;
      }
      if (currentFileEditMode()) {
        if (vimSubMode === "insert") return;
        if (dispatchNormalKey(e, key)) {
          consume(e);
          return;
        }
        // Swallow remaining printable keys: in normal mode keystrokes never
        // reach the editor.
        if (isPlainPrintable(e, key)) consume(e);
        return;
      }
      if (dispatchViewKey(e, key)) {
        consume(e);
        return;
      }
      // View mode replaces direct letter activation of viewer buttons with
      // f-hints, so unconsumed printable keys are swallowed here.
      if (isPlainPrintable(e, key)) consume(e);
    }

    addAppEvent(documentTarget, "keydown", handleKeydown, true);

    renderModeChip();

    return Object.freeze({
      handleKeydown,
      handleEscape,
      syncEditMode,
      isNormalMode: () => Boolean(currentFileEditMode()) && vimSubMode === "normal",
      currentSubMode: () => vimSubMode,
      renderModeChip,
      dispose() {
        pendingPrefix = "";
        vimSubMode = "insert";
        renderModeChip();
      },
    });
  }

export { createFileVimController };
