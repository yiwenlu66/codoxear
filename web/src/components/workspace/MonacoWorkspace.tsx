import type { ComponentChildren } from "preact";
import { useEffect, useRef, useState } from "preact/hooks";
import type * as Monaco from "monaco-editor";

import { loadMonaco } from "../../lib/monaco";

type MonacoWorkspaceMode = "diff" | "file";

type MonacoModule = Awaited<ReturnType<typeof loadMonaco>>;
type MonacoEditorInstance = Monaco.editor.IStandaloneCodeEditor | Monaco.editor.IStandaloneDiffEditor;
type MonacoModel = Monaco.editor.ITextModel;

interface MonacoWorkspaceProps {
  mode: MonacoWorkspaceMode;
  path: string;
  line?: number | null;
  originalText?: string;
  modifiedText: string;
  fallback: ComponentChildren;
}

function inferLanguage(path: string) {
  const extension = path.split(".").pop()?.toLowerCase() || "";
  switch (extension) {
    case "js":
    case "cjs":
    case "mjs":
      return "javascript";
    case "jsx":
      return "javascript";
    case "ts":
    case "cts":
    case "mts":
      return "typescript";
    case "tsx":
      return "typescript";
    case "json":
      return "json";
    case "md":
    case "markdown":
      return "markdown";
    case "py":
      return "python";
    case "sh":
    case "bash":
    case "zsh":
      return "shell";
    case "html":
    case "htm":
      return "html";
    case "css":
      return "css";
    case "scss":
      return "scss";
    case "less":
      return "less";
    case "yml":
    case "yaml":
      return "yaml";
    case "xml":
      return "xml";
    case "toml":
      return "ini";
    case "rs":
      return "rust";
    case "go":
      return "go";
    case "java":
      return "java";
    case "c":
    case "h":
      return "c";
    case "cpp":
    case "cc":
    case "cxx":
    case "hpp":
      return "cpp";
    case "sql":
      return "sql";
    default:
      return "plaintext";
  }
}

function normalizeLine(line?: number | null) {
  return typeof line === "number" && Number.isFinite(line) && line > 0 ? Math.floor(line) : null;
}

function shouldUseMonaco() {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
    return true;
  }
  return !window.matchMedia("(max-width: 767px), (pointer: coarse)").matches;
}

function commonEditorOptions() {
  return {
    automaticLayout: true,
    domReadOnly: true,
    glyphMargin: true,
    lineNumbersMinChars: 3,
    minimap: { enabled: false },
    readOnly: true,
    renderLineHighlight: "all" as const,
    scrollBeyondLastLine: false,
    smoothScrolling: true,
    stickyScroll: { enabled: false },
    quickSuggestions: false,
    suggestOnTriggerCharacters: false,
    acceptSuggestionOnEnter: "off" as const,
    tabCompletion: "off" as const,
    wordBasedSuggestions: "off" as const,
  };
}

function revealLine(monaco: MonacoModule, editor: Monaco.editor.IStandaloneCodeEditor | null, line?: number | null) {
  const targetLine = normalizeLine(line);
  if (!editor || !targetLine) {
    return;
  }

  const model = editor.getModel();
  if (!model) {
    return;
  }

  const boundedLine = Math.max(1, Math.min(targetLine, model.getLineCount()));
  const maxColumn = Math.max(1, model.getLineMaxColumn(boundedLine));
  editor.setSelection(new monaco.Selection(boundedLine, 1, boundedLine, maxColumn));
  editor.revealLineInCenter(boundedLine);
}

export function MonacoWorkspace({ mode, path, line = null, originalText = "", modifiedText, fallback }: MonacoWorkspaceProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const editorRef = useRef<MonacoEditorInstance | null>(null);
  const modelsRef = useRef<MonacoModel[]>([]);
  const [ready, setReady] = useState(false);
  const [useFallback, setUseFallback] = useState(false);

  useEffect(() => {
    if (!shouldUseMonaco()) {
      setReady(false);
      setUseFallback(true);
      return;
    }

    let cancelled = false;

    const disposeEditor = () => {
      const editor = editorRef.current;
      editorRef.current = null;
      if (editor) {
        editor.dispose();
      }
      const models = modelsRef.current;
      modelsRef.current = [];
      for (const model of models) {
        model.dispose();
      }
    };

    const mount = async () => {
      const container = containerRef.current;
      if (!container) {
        return;
      }

      setReady(false);
      setUseFallback(false);
      disposeEditor();

      try {
        const monaco = await loadMonaco();
        if (cancelled || !containerRef.current) {
          return;
        }

        const language = inferLanguage(path);
        if (mode === "diff") {
          const originalModel = monaco.editor.createModel(originalText, language);
          const modifiedModel = monaco.editor.createModel(modifiedText, language);
          const editor = monaco.editor.createDiffEditor(containerRef.current, {
            ...commonEditorOptions(),
            hideUnchangedRegions: {
              enabled: true,
              contextLineCount: 4,
              minimumLineCount: 1,
              revealLineCount: 2,
            },
            originalEditable: false,
            renderSideBySide: true,
          });
          editor.setModel({ original: originalModel, modified: modifiedModel });
          editorRef.current = editor;
          modelsRef.current = [originalModel, modifiedModel];
          revealLine(monaco, editor.getModifiedEditor(), line);
        } else {
          const model = monaco.editor.createModel(modifiedText, language);
          const editor = monaco.editor.create(containerRef.current, {
            ...commonEditorOptions(),
            model,
            wordWrap: "on",
          });
          editorRef.current = editor;
          modelsRef.current = [model];
          revealLine(monaco, editor, line);
        }

        setReady(true);
      } catch {
        if (cancelled) {
          return;
        }
        disposeEditor();
        setUseFallback(true);
      }
    };

    void mount();

    return () => {
      cancelled = true;
      setReady(false);
      disposeEditor();
    };
  }, [line, mode, modifiedText, originalText, path]);

  return (
    <div className="relative h-[58vh] overflow-hidden rounded-2xl border border-border/60 bg-slate-950">
      <div ref={containerRef} className={`h-full w-full ${ready && !useFallback ? "" : "invisible"}`} />
      {!ready || useFallback ? <div className="absolute inset-0">{fallback}</div> : null}
    </div>
  );
}
