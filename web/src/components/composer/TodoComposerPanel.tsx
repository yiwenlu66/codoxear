import type { TodoSnapshot, TodoSnapshotItem } from "../../lib/types";

interface TodoComposerPanelProps {
  snapshot: unknown;
  expanded: boolean;
  onToggle: () => void;
}

function normalizeText(value: unknown): string | undefined {
  if (typeof value !== "string") {
    return undefined;
  }

  const trimmed = value.trim();

  return trimmed.length > 0 ? trimmed : undefined;
}

function normalizeTodoItem(value: unknown): TodoSnapshotItem | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }

  const item = value as Record<string, unknown>;
  const normalized: TodoSnapshotItem = {
    id: typeof item.id === "number" || typeof item.id === "string" ? item.id : undefined,
    title: normalizeText(item.title),
    status: normalizeText(item.status),
    description: normalizeText(item.description),
  };

  return [normalized.title, normalized.status, normalized.description].some(Boolean) ? normalized : null;
}

function normalizeSnapshot(snapshot: unknown): TodoSnapshot {
  if (!snapshot || typeof snapshot !== "object") {
    return { available: false, error: false, items: [] };
  }

  const raw = snapshot as Record<string, unknown>;

  return {
    available: raw.available === true,
    error: raw.error === true,
    progress_text: normalizeText(raw.progress_text),
    items: Array.isArray(raw.items)
      ? raw.items.map(normalizeTodoItem).filter((item): item is TodoSnapshotItem => Boolean(item))
      : [],
  };
}

export function getDisplayableTodoSnapshot(snapshot: unknown): TodoSnapshot | null {
  const todo = normalizeSnapshot(snapshot);

  return todo.available && todo.items.length > 0 ? todo : null;
}

function statusClassName(status: string | undefined) {
  return status ? status.replace(/[^a-z0-9_-]+/gi, "-") : "unknown";
}

export function TodoComposerPanel({ snapshot, expanded, onToggle }: TodoComposerPanelProps) {
  const todo = getDisplayableTodoSnapshot(snapshot);

  if (!todo) {
    return null;
  }

  const summary = todo.progress_text || "Todo";

  return (
    <div className="composerTodoBar">
      <button
        type="button"
        className={`composerTodoBarButton${expanded ? " isExpanded" : ""}`}
        aria-expanded={expanded ? "true" : "false"}
        onClick={onToggle}
      >
        <span className="composerTodoSummary">{summary}</span>
        <span className="composerTodoToggleHint">{expanded ? "Hide" : "Show"}</span>
      </button>
      {expanded ? (
        <div className="composerTodoPanel">
          <div className="composerTodoList">
            {todo.items.map((item, index) => (
              <article key={`${item.title || "todo"}-${index}`} className="composerTodoItem">
                <div className="composerTodoItemHead">
                  <strong>{item.title || "Untitled todo"}</strong>
                  <span className={`composerTodoStatus ${statusClassName(item.status)}`}>{item.status || "unknown"}</span>
                </div>
                {item.description ? <p>{item.description}</p> : null}
              </article>
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
}
