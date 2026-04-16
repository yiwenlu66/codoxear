declare global {
  interface Window {
    MonacoEnvironment?: {
      getWorker?: (_workerId: string, label: string) => Worker;
    };
  }
}

import type * as Monaco from "monaco-editor";

type MonacoModule = typeof Monaco;
type WorkerFactory = { default: new () => Worker };
type WorkerModuleKey = "css" | "editor" | "html" | "json" | "typescript";

let configured = false;
let monacoPromise: Promise<MonacoModule> | null = null;
let workerFactoriesPromise: Promise<Record<WorkerModuleKey, WorkerFactory["default"]>> | null = null;

export function workerModuleKeyForLabel(label: string): WorkerModuleKey {
  switch (label) {
    case "json":
      return "json";
    case "css":
    case "scss":
    case "less":
      return "css";
    case "html":
    case "handlebars":
    case "razor":
      return "html";
    case "typescript":
    case "javascript":
      return "typescript";
    default:
      return "editor";
  }
}

async function loadWorkerFactories() {
  if (!workerFactoriesPromise) {
    workerFactoriesPromise = Promise.all([
      import("monaco-editor/esm/vs/editor/editor.worker?worker"),
      import("monaco-editor/esm/vs/language/css/css.worker?worker"),
      import("monaco-editor/esm/vs/language/html/html.worker?worker"),
      import("monaco-editor/esm/vs/language/json/json.worker?worker"),
      import("monaco-editor/esm/vs/language/typescript/ts.worker?worker"),
    ]).then(([
      editorWorker,
      cssWorker,
      htmlWorker,
      jsonWorker,
      tsWorker,
    ]) => ({
      editor: (editorWorker as WorkerFactory).default,
      css: (cssWorker as WorkerFactory).default,
      html: (htmlWorker as WorkerFactory).default,
      json: (jsonWorker as WorkerFactory).default,
      typescript: (tsWorker as WorkerFactory).default,
    }));
  }
  return workerFactoriesPromise;
}

async function ensureMonacoEnvironment() {
  if (configured || typeof window === "undefined") {
    return;
  }

  const workerFactories = await loadWorkerFactories();
  window.MonacoEnvironment = {
    getWorker(_workerId, label) {
      const WorkerCtor = workerFactories[workerModuleKeyForLabel(label)];
      return new WorkerCtor();
    },
  };
  configured = true;
}

export async function loadMonaco() {
  if (!monacoPromise) {
    monacoPromise = Promise.all([
      import("monaco-editor/min/vs/editor/editor.main.css"),
      import("monaco-editor"),
      ensureMonacoEnvironment(),
    ]).then(([, monaco]) => monaco as MonacoModule);
  }
  return monacoPromise;
}
