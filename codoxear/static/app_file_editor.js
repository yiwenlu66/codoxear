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

  const MONACO_THEME_NAME = "codoxear-github-light";

  function defineCodoxearMonacoTheme(monaco) {
    if (!monaco || !monaco.editor || typeof monaco.editor.defineTheme !== "function") throw new Error("monaco failed to initialize");
    monaco.editor.defineTheme(MONACO_THEME_NAME, {
      base: "vs",
      inherit: true,
      rules: [],
      colors: {
        "editor.background": "#ffffff",
        "editor.lineHighlightBackground": "#f6f8fa",
        "editorGutter.background": "#ffffff",
        "editorLineNumber.foreground": "#8c959f",
        "editorLineNumber.activeForeground": "#57606a",
        "diffEditor.insertedTextBackground": "#dafbe1",
        "diffEditor.removedTextBackground": "#ffebe9",
        "diffEditor.insertedLineBackground": "#f0fff4",
        "diffEditor.removedLineBackground": "#fff5f5",
      },
    });
  }

  function createMonacoLoader(options = {}) {
    const resolveAppUrl = requireFunction(options.resolveAppUrl, "resolveAppUrl");
    const globalObject = options.globalObject || window;
    const timeoutMs = Math.max(1, Number(options.timeoutMs || 4000));
    const pollMs = Math.max(1, Number(options.pollMs || 25));
    const timerSet = typeof options.setTimeout === "function" ? options.setTimeout : globalObject.setTimeout.bind(globalObject);
    let readyPromise = null;
    let monacoNs = null;
    let themeReady = false;

    function currentMonaco() {
      return monacoNs;
    }

    function selectionCtor() {
      return monacoNs && monacoNs.Selection ? monacoNs.Selection : null;
    }

    function editSupportAvailable() {
      return Boolean(monacoNs);
    }

    function ensure() {
      if (readyPromise) return readyPromise;
      readyPromise = new Promise((resolve, reject) => {
        let done = false;
        const startedAt = Date.now();
        const fail = (error) => {
          if (done) return;
          done = true;
          reject(error instanceof Error ? error : new Error(String(error || "monaco failed")));
        };
        const succeed = (value) => {
          if (done) return;
          done = true;
          resolve(value);
        };
        const finish = () => {
          if (done) return;
          if (!(globalObject.require && globalObject.require.config)) {
            fail(new Error("monaco loader unavailable"));
            return;
          }
          const base = resolveAppUrl("monaco/vs");
          globalObject.MonacoEnvironment = {
            getWorkerUrl(_moduleId, _label) {
              const src = `\nself.MonacoEnvironment={baseUrl:${JSON.stringify(base + "/")}};\nimportScripts(${JSON.stringify(base + "/base/worker/workerMain.js")});\n`;
              return `data:text/javascript;charset=utf-8,${encodeURIComponent(src)}`;
            },
          };
          globalObject.require.config({ paths: { vs: base } });
          globalObject.require(["vs/editor/editor.main"], () => {
            monacoNs = globalObject.monaco;
            if (!monacoNs) {
              fail(new Error("monaco failed to initialize"));
              return;
            }
            if (!themeReady) {
              defineCodoxearMonacoTheme(monacoNs);
              themeReady = true;
            }
            succeed(monacoNs);
          }, fail);
        };
        if (globalObject.monaco && globalObject.monaco.editor) {
          monacoNs = globalObject.monaco;
          finish();
          return;
        }
        if (globalObject.require && globalObject.require.config) {
          finish();
          return;
        }
        const waitForLoader = () => {
          if (done) return;
          if (globalObject.require && globalObject.require.config) {
            finish();
            return;
          }
          if (Date.now() - startedAt >= timeoutMs) {
            fail(new Error("monaco loader timed out"));
            return;
          }
          timerSet(waitForLoader, pollMs);
        };
        waitForLoader();
      });
      readyPromise.catch(() => {
        readyPromise = null;
      });
      return readyPromise;
    }

    return Object.freeze({
      currentMonaco,
      editSupportAvailable,
      ensure,
      selectionCtor,
    });
  }

  function editorLanguageForPath(path) {
    const ext = String(path || "").split(".").pop().toLowerCase();
    if (ext === "js") return "javascript";
    if (ext === "ts") return "typescript";
    if (ext === "json") return "json";
    if (ext === "py") return "python";
    if (ext === "sh" || ext === "bash" || ext === "zsh") return "bash";
    if (ext === "md") return "markdown";
    if (ext === "html" || ext === "htm") return "markup";
    if (ext === "css") return "css";
    if (ext === "yml" || ext === "yaml") return "yaml";
    if (ext === "toml") return "toml";
    if (ext === "rs") return "rust";
    if (ext === "go") return "go";
    if (ext === "java") return "java";
    if (ext === "c" || ext === "h") return "c";
    if (ext === "cpp" || ext === "cc" || ext === "hpp") return "cpp";
    return "";
  }

  function fileEditorCreateOptions({ language = "", value = "", readOnly = false } = {}) {
    return {
      language: language || "plaintext",
      value: String(value || ""),
      readOnly: Boolean(readOnly),
      theme: MONACO_THEME_NAME,
      lineNumbers: "on",
      minimap: { enabled: false },
      scrollBeyondLastLine: false,
      wordWrap: "on",
      folding: false,
      renderLineHighlight: "none",
      glyphMargin: false,
      overviewRulerBorder: false,
      stickyScroll: { enabled: false },
      automaticLayout: true,
      accessibilitySupport: "off",
      quickSuggestions: false,
      suggestOnTriggerCharacters: false,
      acceptSuggestionOnEnter: "off",
      inlineSuggest: { enabled: false },
      parameterHints: { enabled: false },
      snippetSuggestions: "none",
      tabCompletion: "off",
      wordBasedSuggestions: "off",
    };
  }

  function diffEditorCreateOptions() {
    return {
      readOnly: true,
      theme: MONACO_THEME_NAME,
      renderSideBySide: false,
      useInlineViewWhenSpaceIsLimited: true,
      lineNumbers: "on",
      minimap: { enabled: false },
      scrollBeyondLastLine: false,
      wordWrap: "on",
      diffWordWrap: "on",
      folding: false,
      renderLineHighlight: "none",
      glyphMargin: false,
      overviewRulerBorder: false,
      stickyScroll: { enabled: false },
      automaticLayout: true,
      hideUnchangedRegions: {
        enabled: true,
        contextLineCount: 4,
        minimumLineCount: 1,
        revealLineCount: 2,
      },
    };
  }

  function diffSideEditorOptions(modified = false) {
    return {
      wordWrap: "on",
      lineNumbers: modified ? "on" : "off",
      glyphMargin: false,
      lineDecorationsWidth: 0,
      lineNumbersMinChars: modified ? 3 : 0,
    };
  }

  function requireMonacoEditor(monaco, method) {
    if (!monaco || !monaco.editor || typeof monaco.editor[method] !== "function") throw new Error("monaco editor unavailable");
    return monaco.editor;
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

    function isActiveInput(kind, target, ElementCtor = null) {
      const Ctor = typeof ElementCtor === "function" ? ElementCtor : null;
      if (!target || (Ctor && !(target instanceof Ctor))) return false;
      if (!target.classList || typeof target.classList.contains !== "function" || !target.classList.contains("inputarea")) return false;
      const targetEditor = activeCodeEditor(kind);
      const node = targetEditor && typeof targetEditor.getDomNode === "function" ? targetEditor.getDomNode() : null;
      return Boolean(node && typeof node.contains === "function" && node.contains(target));
    }

    function updateEditorOptions(kind, options) {
      if (String(kind || "") !== "diff" || !editor || typeof editor.updateOptions !== "function") return false;
      editor.updateOptions(options || {});
      return true;
    }

    function createFileEditor(monaco, host, options = {}) {
      const editorApi = requireMonacoEditor(monaco, "create");
      const language = options.languageOverride || editorLanguageForPath(options.path);
      const nextEditor = editorApi.create(host, fileEditorCreateOptions({
        language,
        value: options.text,
        readOnly: Boolean(options.readOnly),
      }));
      setEditor(nextEditor);
      setModels([typeof nextEditor.getModel === "function" ? nextEditor.getModel() : null].filter(Boolean));
      const onDidChangeModelContent = requireFunction(options.onDidChangeModelContent, "onDidChangeModelContent");
      if (typeof nextEditor.onDidChangeModelContent !== "function") throw new Error("monaco file editor change listener unavailable");
      setChangeDisposable(nextEditor.onDidChangeModelContent(onDidChangeModelContent));
      return nextEditor;
    }

    function updateFileEditorText(monaco, options = {}) {
      const editorApi = requireMonacoEditor(monaco, "setModelLanguage");
      const targetEditor = currentEditor();
      if (!targetEditor || typeof targetEditor.getModel !== "function") throw new Error("file editor unavailable");
      const model = targetEditor.getModel();
      if (!model || typeof model.setValue !== "function") throw new Error("file editor model unavailable");
      const runProgrammaticChange = requireFunction(options.runProgrammaticChange, "runProgrammaticChange");
      const language = options.languageOverride || editorLanguageForPath(options.path) || "plaintext";
      runProgrammaticChange(() => {
        editorApi.setModelLanguage(model, language);
        model.setValue(String(options.text || ""));
      });
      return true;
    }

    function createDiffEditor(monaco, host, options = {}) {
      const editorApi = requireMonacoEditor(monaco, "createDiffEditor");
      if (typeof editorApi.createModel !== "function") throw new Error("monaco editor model creation unavailable");
      const language = editorLanguageForPath(options.path) || "plaintext";
      const originalModel = editorApi.createModel(String(options.originalText || ""), language);
      const modifiedModel = editorApi.createModel(String(options.modifiedText || ""), language);
      const diffEditor = editorApi.createDiffEditor(host, diffEditorCreateOptions());
      if (!diffEditor || typeof diffEditor.setModel !== "function") throw new Error("monaco diff editor unavailable");
      diffEditor.setModel({ original: originalModel, modified: modifiedModel });
      setEditor(diffEditor);
      setModels([originalModel, modifiedModel]);
      const originalEditor = typeof diffEditor.getOriginalEditor === "function" ? diffEditor.getOriginalEditor() : null;
      const modifiedEditor = typeof diffEditor.getModifiedEditor === "function" ? diffEditor.getModifiedEditor() : null;
      if (originalEditor && typeof originalEditor.updateOptions === "function") originalEditor.updateOptions(diffSideEditorOptions(false));
      if (modifiedEditor && typeof modifiedEditor.updateOptions === "function") modifiedEditor.updateOptions(diffSideEditorOptions(true));
      return { diffEditor, originalEditor, modifiedEditor };
    }

    function positionCurrentEditorAtLine(kind, lineNumber, normalizeLineNumber) {
      const normalize = requireFunction(normalizeLineNumber, "normalizeLineNumber");
      const requestedLine = normalize(lineNumber);
      const targetLine = requestedLine || 1;
      const editorKind = String(kind || "");
      if (editorKind === "diff") {
        if (!editor || typeof editor.getOriginalEditor !== "function" || typeof editor.getModifiedEditor !== "function") return null;
        const originalEditor = editor.getOriginalEditor();
        const modifiedEditor = editor.getModifiedEditor();
        if (!originalEditor || !modifiedEditor) return null;
        if (typeof originalEditor.setScrollPosition === "function") originalEditor.setScrollPosition({ scrollTop: 0, scrollLeft: 0 });
        if (typeof modifiedEditor.setScrollPosition === "function") modifiedEditor.setScrollPosition({ scrollTop: 0, scrollLeft: 0 });
        if (typeof originalEditor.setPosition === "function") originalEditor.setPosition({ lineNumber: targetLine, column: 1 });
        if (typeof modifiedEditor.setPosition === "function") modifiedEditor.setPosition({ lineNumber: targetLine, column: 1 });
        if (typeof modifiedEditor.revealPositionInCenter === "function") modifiedEditor.revealPositionInCenter({ lineNumber: targetLine, column: 1 });
        if (typeof editor.layout === "function") editor.layout();
        return Object.freeze({ requestedLine, targetLine });
      }
      if (!editor) return null;
      if (typeof editor.setScrollPosition === "function") editor.setScrollPosition({ scrollTop: 0, scrollLeft: 0 });
      if (typeof editor.setPosition === "function") editor.setPosition({ lineNumber: targetLine, column: 1 });
      if (typeof editor.revealPositionInCenter === "function") editor.revealPositionInCenter({ lineNumber: targetLine, column: 1 });
      if (typeof editor.layout === "function") editor.layout();
      return Object.freeze({ requestedLine, targetLine });
    }

    function focusActiveCodeEditor(kind) {
      const target = activeCodeEditor(kind);
      if (target && typeof target.focus === "function") target.focus();
      return target || null;
    }

    function normalizePosition(targetEditor, position) {
      if (!targetEditor || !position) return null;
      const model = typeof targetEditor.getModel === "function" ? targetEditor.getModel() : null;
      if (!model) return null;
      const lineCount = Math.max(1, Number(model.getLineCount && model.getLineCount()) || 1);
      const lineNumber = Math.max(1, Math.min(lineCount, Number(position.lineNumber) || 1));
      const lineMaxColumn = Math.max(1, Number(model.getLineMaxColumn && model.getLineMaxColumn(lineNumber)) || 1);
      const column = Math.max(1, Math.min(lineMaxColumn, Number(position.column) || 1));
      return { lineNumber, column };
    }

    function isCollapsedSelection(selection) {
      return !selection || (
        selection.startLineNumber === selection.endLineNumber &&
        selection.startColumn === selection.endColumn
      );
    }

    function applySelection(targetEditor, cursor, anchor = null, selectionCtor = null) {
      const Selection = typeof selectionCtor === "function" ? selectionCtor : null;
      if (!targetEditor || !Selection) return false;
      const nextCursor = normalizePosition(targetEditor, cursor);
      if (!nextCursor) return false;
      const nextAnchor = anchor ? normalizePosition(targetEditor, anchor) : null;
      const selection = nextAnchor
        ? new Selection(nextAnchor.lineNumber, nextAnchor.column, nextCursor.lineNumber, nextCursor.column)
        : new Selection(nextCursor.lineNumber, nextCursor.column, nextCursor.lineNumber, nextCursor.column);
      if (typeof targetEditor.setSelection === "function") targetEditor.setSelection(selection);
      if (!nextAnchor && typeof targetEditor.setPosition === "function") targetEditor.setPosition(nextCursor);
      if (typeof targetEditor.revealPositionInCenterIfOutsideViewport === "function") targetEditor.revealPositionInCenterIfOutsideViewport(nextCursor);
      else if (typeof targetEditor.revealPositionInCenter === "function") targetEditor.revealPositionInCenter(nextCursor);
      return true;
    }

    function selectionText(targetEditor) {
      if (!targetEditor || typeof targetEditor.getSelection !== "function" || typeof targetEditor.getModel !== "function") return "";
      const selection = targetEditor.getSelection();
      if (isCollapsedSelection(selection)) return "";
      const model = targetEditor.getModel();
      if (!model || typeof model.getValueInRange !== "function") return "";
      return String(model.getValueInRange(selection) || "");
    }

    function activeSelectionText(kind) {
      return selectionText(activeCodeEditor(kind));
    }

    function layoutCurrent() {
      if (!editor || typeof editor.layout !== "function") return false;
      editor.layout();
      return true;
    }

    function focusLine(kind, lineNumber, normalizeLineNumber) {
      const normalize = requireFunction(normalizeLineNumber, "normalizeLineNumber");
      const line = normalize(lineNumber);
      const target = activeCodeEditor(kind) || editor;
      if (!target || !line || typeof target.setPosition !== "function") return false;
      target.setPosition({ lineNumber: line, column: 1 });
      if (typeof target.revealLineInCenter === "function") target.revealLineInCenter(line);
      if (typeof target.focus === "function") target.focus();
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
      activeSelectionText,
      applySelection,
      createDiffEditor,
      createFileEditor,
      currentEditor,
      currentModels,
      dispose,
      focusActiveCodeEditor,
      focusLine,
      isActiveInput,
      isCollapsedSelection,
      layoutCurrent,
      normalizePosition,
      positionCurrentEditorAtLine,
      selectionText,
      setChangeDisposable,
      setEditor,
      setModels,
      updateEditorOptions,
      updateFileEditorText,
      withCurrentEditor,
    });
  }

  window.CodoxearFileEditor = Object.freeze({
    createFileEditorRuntime,
    createMonacoLoader,
  });
})();
