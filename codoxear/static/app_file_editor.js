(function () {
  "use strict";

  function safeDispose(value) {
    if (!value || typeof value.dispose !== "function") return false;
    try {
      value.dispose();
      return true;
    } catch (_) {
      return false;
    }
  }

  function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`file editor dependency missing: ${name}`);
    return value;
  }

  function createFileEditorRuntime() {
    let editor = null;
    let models = [];
    let changeDisposable = null;

    function currentEditor() {
      return editor;
    }

    function setEditor(nextEditor) {
      editor = nextEditor || null;
      return editor;
    }

    function currentModels() {
      return models.slice();
    }

    function setModels(nextModels) {
      models = Array.isArray(nextModels) ? nextModels.filter(Boolean) : [];
      return currentModels();
    }

    function setChangeDisposable(nextDisposable) {
      changeDisposable = nextDisposable || null;
      return changeDisposable;
    }

    function activeCodeEditor(kind) {
      const editorKind = String(kind || "");
      if (editorKind === "diff" && editor && typeof editor.getModifiedEditor === "function") return editor.getModifiedEditor();
      if (editorKind === "file" && editor) return editor;
      return null;
    }

    function updateEditorOptions(kind, options) {
      if (String(kind || "") !== "diff" || !editor || typeof editor.updateOptions !== "function") return false;
      editor.updateOptions(options || {});
      return true;
    }

    function dispose(options = {}) {
      const clearHost = typeof options.clearHost === "function" ? options.clearHost : null;
      const afterDispose = typeof options.afterDispose === "function" ? options.afterDispose : null;
      safeDispose(changeDisposable);
      changeDisposable = null;
      if (clearHost) clearHost();
      for (const model of models) safeDispose(model);
      models = [];
      safeDispose(editor);
      editor = null;
      if (afterDispose) afterDispose();
      return true;
    }

    function withCurrentEditor(callback) {
      const fn = requireFunction(callback, "withCurrentEditor");
      return fn(editor);
    }

    return Object.freeze({
      activeCodeEditor,
      currentEditor,
      currentModels,
      dispose,
      setChangeDisposable,
      setEditor,
      setModels,
      updateEditorOptions,
      withCurrentEditor,
    });
  }

  window.CodoxearFileEditor = Object.freeze({
    createFileEditorRuntime,
  });
})();
