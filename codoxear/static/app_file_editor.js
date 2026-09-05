
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

  // One Monaco theme per UI theme family x resolved mode, named
  // codoxear-<family>-<mode>. paper-light is the original GitHub-light code
  // surface and is a regression surface: do not restyle it. The other five
  // follow their family's palette in themes/<family>.css. Dark diff tints are
  // translucent so they read on every dark paper.
  const DARK_DIFF_TINTS = Object.freeze({
    insertedText: "#4ade8030", insertedLine: "#4ade8014",
    removedText: "#f0755a30", removedLine: "#f0755a14",
  });
  const MONACO_THEMES = Object.freeze({
    "codoxear-paper-light": Object.freeze({
      base: "vs", background: "#ffffff", lineHighlight: "#f6f8fa",
      lineNumber: "#8c959f", activeLineNumber: "#57606a",
      insertedText: "#dafbe1", insertedLine: "#f0fff4",
      removedText: "#ffebe9", removedLine: "#fff5f5",
    }),
    "codoxear-paper-dark": Object.freeze({
      base: "vs-dark", background: "#201d17", lineHighlight: "#26231d",
      lineNumber: "#6e6a60", activeLineNumber: "#9d978a",
      ...DARK_DIFF_TINTS,
    }),
    "codoxear-clay-light": Object.freeze({
      base: "vs", background: "#f2ede2", lineHighlight: "#ece5d8",
      lineNumber: "#9a8f7f", activeLineNumber: "#7d7365",
      insertedText: "#15803d22", insertedLine: "#15803d12",
      removedText: "#bf3a2422", removedLine: "#bf3a2412",
    }),
    "codoxear-clay-dark": Object.freeze({
      base: "vs-dark", background: "#1f1b16", lineHighlight: "#2c2620",
      lineNumber: "#6e675b", activeLineNumber: "#a29785",
      ...DARK_DIFF_TINTS,
    }),
    "codoxear-slate-light": Object.freeze({
      base: "vs", background: "#f7f7f7", lineHighlight: "#f1f1f1",
      lineNumber: "#8a8a8a", activeLineNumber: "#5c5c5c",
      insertedText: "#15803d22", insertedLine: "#15803d12",
      removedText: "#d92d2022", removedLine: "#d92d2012",
    }),
    "codoxear-slate-dark": Object.freeze({
      base: "vs-dark", background: "#161616", lineHighlight: "#262626",
      lineNumber: "#6e6e6e", activeLineNumber: "#a0a0a0",
      ...DARK_DIFF_TINTS,
    }),
  });

  function monacoThemeName(family, mode) {
    const name = `codoxear-${String(family || "")}-${String(mode || "")}`;
    if (!MONACO_THEMES[name]) throw new Error(`unknown monaco theme: ${family}/${mode}`);
    return name;
  }

  const MONACO_DEFAULT_THEME_NAME = monacoThemeName("paper", "light");

  function monacoThemeData(palette) {
    return {
      base: palette.base,
      inherit: true,
      rules: [],
      colors: {
        "editor.background": palette.background,
        "editor.lineHighlightBackground": palette.lineHighlight,
        "editorGutter.background": palette.background,
        "editorLineNumber.foreground": palette.lineNumber,
        "editorLineNumber.activeForeground": palette.activeLineNumber,
        "diffEditor.insertedTextBackground": palette.insertedText,
        "diffEditor.removedTextBackground": palette.removedText,
        "diffEditor.insertedLineBackground": palette.insertedLine,
        "diffEditor.removedLineBackground": palette.removedLine,
      },
    };
  }

  function defineCodoxearMonacoThemes(monaco) {
    if (!monaco || !monaco.editor || typeof monaco.editor.defineTheme !== "function" || typeof monaco.editor.setTheme !== "function") {
      throw new Error("monaco failed to initialize");
    }
    for (const [name, palette] of Object.entries(MONACO_THEMES)) monaco.editor.defineTheme(name, monacoThemeData(palette));
  }

  function createMonacoLoader(options = {}) {
    const resolveAppUrl = requireFunction(options.resolveAppUrl, "resolveAppUrl");
    const subscribeTheme = requireFunction(options.subscribeTheme, "subscribeTheme");
    const globalObject = options.globalObject || window;
    const timeoutMs = Math.max(1, Number(options.timeoutMs || 4000));
    const pollMs = Math.max(1, Number(options.pollMs || 25));
    const timerSet = typeof options.setTimeout === "function" ? options.setTimeout : globalObject.setTimeout.bind(globalObject);
    let readyPromise = null;
    let monacoNs = null;
    let themeReady = false;
    // The UI theme store is the sole writer of this name. Snapshots that
    // arrive before Monaco loads are remembered and applied at init;
    // afterwards each snapshot switches the global Monaco theme directly.
    let themeName = MONACO_DEFAULT_THEME_NAME;

    function currentMonaco() {
      return monacoNs;
    }

    function currentThemeName() {
      return themeName;
    }

    subscribeTheme((snapshot) => {
      themeName = monacoThemeName(snapshot && snapshot.family, snapshot && snapshot.resolvedMode);
      if (themeReady) monacoNs.editor.setTheme(themeName);
    });

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
          // Monaco 0.55's vendored min build carries same-origin module worker
          // asset URLs (for example vs/assets/editor.worker-*.js) and uses them
          // when no legacy global worker URL override is installed. Do not
          // install the old encoded worker wrapper: the page CSP permits 'self'
          // and blob: workers, not data-scheme workers, and a global override
          // would bypass Monaco's per-language same-origin worker URLs.
          globalObject.MonacoEnvironment = globalObject.MonacoEnvironment || {};
          globalObject.require.config({ paths: { vs: base } });
          globalObject.require(["vs/editor/editor.main"], () => {
            monacoNs = globalObject.monaco;
            if (!monacoNs) {
              fail(new Error("monaco failed to initialize"));
              return;
            }
            if (!themeReady) {
              defineCodoxearMonacoThemes(monacoNs);
              themeReady = true;
            }
            monacoNs.editor.setTheme(themeName);
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
      currentThemeName,
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

  function requireThemeName(value) {
    if (!MONACO_THEMES[value]) throw new TypeError("file editor dependency missing: theme");
    return value;
  }

  function fileEditorCreateOptions({ language = "", value = "", readOnly = false, theme } = {}) {
    return {
      language: language || "plaintext",
      value: String(value || ""),
      readOnly: Boolean(readOnly),
      theme: requireThemeName(theme),
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

  function diffEditorCreateOptions(theme) {
    return {
      readOnly: true,
      theme: requireThemeName(theme),
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
      const editorKind = String(kind || "");
      const targetEditor = activeCodeEditor(editorKind);
      const node = targetEditor && typeof targetEditor.getDomNode === "function" ? targetEditor.getDomNode() : null;
      if (!target.classList || typeof target.classList.contains !== "function" || !target.classList.contains("inputarea")) return false;
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
        theme: options.theme,
      }));
      setEditor(nextEditor);
      setModels([typeof nextEditor.getModel === "function" ? nextEditor.getModel() : null].filter(Boolean));
      const onDidChangeModelContent = requireFunction(options.onDidChangeModelContent, "onDidChangeModelContent");
      if (typeof nextEditor.onDidChangeModelContent !== "function") throw new Error("monaco file editor change listener unavailable");
      setChangeDisposable(nextEditor.onDidChangeModelContent(onDidChangeModelContent));
      return nextEditor;
    }

    function currentFileText(kind, fallbackText = "") {
      const editorKind = String(kind || "");
      const targetEditor = currentEditor();
      if (editorKind !== "file") return String(fallbackText || "");
      if (!targetEditor || typeof targetEditor.getModel !== "function") return String(fallbackText || "");
      const model = targetEditor.getModel();
      if (!model || typeof model.getValue !== "function") return String(fallbackText || "");
      return String(model.getValue());
    }

    function restoreFileText(kind, text, runProgrammaticChange) {
      const editorKind = String(kind || "");
      const targetEditor = currentEditor();
      const run = requireFunction(runProgrammaticChange, "runProgrammaticChange");
      if (editorKind !== "file") return false;
      if (!targetEditor || typeof targetEditor.getModel !== "function") return false;
      const model = targetEditor.getModel();
      if (!model || typeof model.setValue !== "function") return false;
      run(() => {
        model.setValue(String(text || ""));
      });
      return true;
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
      const diffEditor = editorApi.createDiffEditor(host, diffEditorCreateOptions(options.theme));
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
      const editorKind = String(kind || "");
      const target = activeCodeEditor(editorKind);
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
      if (!targetEditor) return "";
      if (typeof targetEditor.selectedText === "function") return String(targetEditor.selectedText() || "");
      if (typeof targetEditor.getSelection !== "function" || typeof targetEditor.getModel !== "function") return "";
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

    function focusResolvedLine(kind, lineNumber) {
      const line = Math.max(1, Number(lineNumber) || 0);
      const target = activeCodeEditor(kind) || editor;
      if (!target || !line || typeof target.setPosition !== "function") return false;
      target.setPosition({ lineNumber: line, column: 1 });
      if (typeof target.revealLineInCenter === "function") target.revealLineInCenter(line);
      if (typeof target.focus === "function") target.focus();
      return true;
    }

    function focusLine(kind, lineNumber, normalizeLineNumber) {
      const normalize = requireFunction(normalizeLineNumber, "normalizeLineNumber");
      const line = normalize(lineNumber);
      return focusResolvedLine(kind, line);
    }

    function scheduleLineFocus(kind, requestedLine, options = {}) {
      const line = Math.max(1, Number(requestedLine) || 0);
      if (!line) return false;
      const requestFrame = requireFunction(options.requestAnimationFrame, "requestAnimationFrame");
      const setTimer = requireFunction(options.setTimeout, "setTimeout");
      const isCurrent = typeof options.isCurrent === "function" ? options.isCurrent : () => true;
      const delayMs = Math.max(0, Number(options.delayMs == null ? 60 : options.delayMs) || 0);
      const runFocus = () => {
        if (!isCurrent()) return false;
        if (!layoutCurrent()) return false;
        return focusResolvedLine(kind, line);
      };
      requestFrame(runFocus);
      setTimer(runFocus, delayMs);
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

    function disposeCurrentFile(options = {}) {
      const finishProgrammaticChange = requireFunction(options.finishProgrammaticChange, "finishProgrammaticChange");
      const clearHost = requireFunction(options.clearHost, "clearHost");
      const setFileEditorKind = requireFunction(options.setFileEditorKind, "setFileEditorKind");
      const clearFileTouchSelectionState = requireFunction(options.clearFileTouchSelectionState, "clearFileTouchSelectionState");
      finishProgrammaticChange();
      return dispose({
        clearHost,
        afterDispose: () => {
          setFileEditorKind("");
          clearFileTouchSelectionState();
        },
      });
    }

    function restoreCurrentFileText(text, options = {}) {
      const prepare = requireFunction(options.prepareFileEditorTextRestore, "prepareFileEditorTextRestore");
      const currentKind = requireFunction(options.currentFileEditorKind, "currentFileEditorKind");
      const runProgrammaticChange = requireFunction(options.runFileEditorProgrammaticChange, "runFileEditorProgrammaticChange");
      const finish = requireFunction(options.finishFileEditorTextRestore, "finishFileEditorTextRestore");
      const restorePlan = prepare(text);
      if (!restorePlan || restorePlan.kind !== "restore") return false;
      restoreFileText(currentKind(), restorePlan.text, runProgrammaticChange);
      finish();
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
      currentFileText,
      currentModels,
      dispose,
      disposeCurrentFile,
      focusActiveCodeEditor,
      focusLine,
      isActiveInput,
      isCollapsedSelection,
      layoutCurrent,
      normalizePosition,
      positionCurrentEditorAtLine,
      restoreCurrentFileText,
      restoreFileText,
      scheduleLineFocus,
      selectionText,
      setChangeDisposable,
      setEditor,
      setModels,
      updateEditorOptions,
      updateFileEditorText,
      withCurrentEditor,
    });
  }

  function requireObject(value, name) {
    if (!value || typeof value !== "object") throw new TypeError(`file editor dependency missing: ${name}`);
    return value;
  }

  function requireMethod(owner, method, label) {
    const object = requireObject(owner, label);
    if (typeof object[method] !== "function") throw new TypeError(`file editor dependency missing: ${label}.${method}`);
    return object[method].bind(object);
  }

  function createFileEditorRenderer(options = {}) {
    const runtime = requireObject(options.runtime, "runtime");
    const ensureMonaco = requireMethod(options.monacoLoader, "ensure", "monacoLoader");
    const currentThemeName = requireMethod(options.monacoLoader, "currentThemeName", "monacoLoader");
    const host = options.host;
    if (!host) throw new TypeError("file editor dependency missing: host");
    const normalizeLineNumber = requireFunction(options.normalizeLineNumber, "normalizeLineNumber");
    const requestFrame = requireFunction(options.requestAnimationFrame, "requestAnimationFrame");
    const setTimer = requireFunction(options.setTimeout, "setTimeout");
    const isCurrentFileOpenRequest = requireFunction(options.isCurrentFileOpenRequest, "isCurrentFileOpenRequest");
    const renderPlainTextFallback = requireFunction(options.renderPlainTextFallback, "renderPlainTextFallback");
    const disposeFileEditor = requireFunction(options.disposeFileEditor, "disposeFileEditor");
    const currentEditorKind = requireFunction(options.currentEditorKind, "currentEditorKind");
    const setEditorKind = requireFunction(options.setEditorKind, "setEditorKind");
    const currentFileEditMode = requireFunction(options.currentFileEditMode, "currentFileEditMode");
    const currentActiveFileEditable = requireFunction(options.currentActiveFileEditable, "currentActiveFileEditable");
    const isUnavailable = requireFunction(options.isUnavailable, "isUnavailable");
    const isProgrammaticChange = requireFunction(options.isProgrammaticChange, "isProgrammaticChange");
    const currentTouchSelectMode = requireFunction(options.currentTouchSelectMode, "currentTouchSelectMode");
    const resetTouchSelectionState = requireFunction(options.resetTouchSelectionState, "resetTouchSelectionState");
    const currentActiveFileText = requireFunction(options.currentActiveFileText, "currentActiveFileText");
    const setDirty = requireFunction(options.setDirty, "setDirty");
    const runProgrammaticChange = requireFunction(options.runProgrammaticChange, "runProgrammaticChange");
    const syncReadOnly = requireFunction(options.syncReadOnly, "syncReadOnly");
    const updateTouchToolbar = requireFunction(options.updateTouchToolbar, "updateTouchToolbar");
    const createFileEditor = requireMethod(runtime, "createFileEditor", "runtime");
    const updateFileEditorText = requireMethod(runtime, "updateFileEditorText", "runtime");
    const createDiffEditor = requireMethod(runtime, "createDiffEditor", "runtime");
    const positionCurrentEditorAtLine = requireMethod(runtime, "positionCurrentEditorAtLine", "runtime");
    const scheduleLineFocus = requireMethod(runtime, "scheduleLineFocus", "runtime");
    const currentFileText = requireMethod(runtime, "currentFileText", "runtime");

    function requestIsCurrent(request) {
      return !(request && !isCurrentFileOpenRequest(request));
    }

    function richEditorUnavailableReason(error, prefix) {
      const message = error && error.message ? String(error.message) : "";
      if (!prefix) return message || "Rich file viewer unavailable";
      return message ? `${prefix}: ${message}` : prefix;
    }

    function activeFileReadOnly() {
      return !(currentFileEditMode() && currentActiveFileEditable() && !isUnavailable());
    }

    function handleFileEditorContentChange() {
      if (isProgrammaticChange()) return;
      if (currentTouchSelectMode()) resetTouchSelectionState();
      const baselineText = String(currentActiveFileText() || "");
      setDirty(currentFileText("file", baselineText) !== baselineText);
    }

    function schedulePositionFocus(kind, lineNumber, request) {
      const positionState = positionCurrentEditorAtLine(kind, lineNumber, normalizeLineNumber);
      const requestedLine = positionState && positionState.requestedLine;
      if (!requestedLine) return positionState;
      scheduleLineFocus(kind, requestedLine, {
        requestAnimationFrame: requestFrame,
        setTimeout: setTimer,
        isCurrent: () => requestIsCurrent(request),
      });
      return positionState;
    }

    async function renderFile(rel, text, lineNumber = null, langOverride = "", request = null) {
      let monaco;
      try {
        monaco = await ensureMonaco();
      } catch (error) {
        if (!requestIsCurrent(request)) return false;
        const prefix = currentActiveFileEditable()
          ? "Code editor unavailable. Editing disabled"
          : "Code editor unavailable";
        const reason = richEditorUnavailableReason(error, prefix);
        renderPlainTextFallback(rel, text, lineNumber, reason);
        updateTouchToolbar();
        return Object.freeze({ ok: true, monacoUnavailable: true, status: `${rel} - editor unavailable`, reason });
      }
      if (!requestIsCurrent(request)) return false;
      if (currentEditorKind() !== "file") {
        disposeFileEditor();
        createFileEditor(monaco, host, {
          path: rel,
          text,
          languageOverride: langOverride,
          readOnly: activeFileReadOnly(),
          theme: currentThemeName(),
          onDidChangeModelContent: handleFileEditorContentChange,
        });
        setEditorKind("file");
      } else {
        updateFileEditorText(monaco, {
          path: rel,
          text,
          languageOverride: langOverride,
          runProgrammaticChange,
        });
      }
      syncReadOnly();
      schedulePositionFocus("file", lineNumber, request);
      updateTouchToolbar();
      return true;
    }

    async function renderDiff(rel, originalText, modifiedText, lineNumber = null, request = null) {
      let monaco;
      try {
        monaco = await ensureMonaco();
      } catch (error) {
        if (!requestIsCurrent(request)) return false;
        const reason = richEditorUnavailableReason(error, "Diff editor unavailable because Monaco failed to load");
        renderPlainTextFallback(rel, "", lineNumber, reason);
        updateTouchToolbar();
        return Object.freeze({ ok: true, monacoUnavailable: true, status: `${rel} - diff editor unavailable`, reason });
      }
      if (!requestIsCurrent(request)) return false;
      disposeFileEditor();
      createDiffEditor(monaco, host, { path: rel, originalText, modifiedText, theme: currentThemeName() });
      setEditorKind("diff");
      schedulePositionFocus("diff", lineNumber, request);
      updateTouchToolbar();
      return true;
    }

    return Object.freeze({
      ensureMonaco,
      renderDiff,
      renderFile,
    });
  }

export { MONACO_DEFAULT_THEME_NAME, createFileEditorRenderer, createFileEditorRuntime, createMonacoLoader, monacoThemeName };
