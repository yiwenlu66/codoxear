import type { TodoSnapshot, TodoSnapshotItem } from "../../lib/types";

interface TodoPopoverProps {
  snapshot: unknown;
}

function normalizeTodoItem(value: unknown): TodoSnapshotItem | null {
  if (!value || typeof value !== "object") {
    return null;
  }
  const item = value as Record<string, unknown>;
  return {
    id: typeof item.id === "number" || typeof item.id === "string" ? item.id : undefined,
    title: typeof item.title === "string" ? item.title : undefined,
    status: typeof item.status === "string" ? item.status : undefined,
    description: typeof item.description === "string" ? item.description : undefined,
  };
}

function normalizeSnapshot(snapshot: unknown): TodoSnapshot {
  if (!snapshot || typeof snapshot !== "object") {
    return { available: false, error: false, items: [] };
  }
  const raw = snapshot as Record<string, unknown>;
  const items = Array.isArray(raw.items)
    ? raw.items.map(normalizeTodoItem).filter((item): item is TodoSnapshotItem => Boolean(item))
    : [];
  return {
    available: raw.available === true,
    error: raw.error === true,
    progress_text: typeof raw.progress_text === "string" ? raw.progress_text : undefined,
    items,
  };
}

function statusClassName(status: string | undefined) {
  return status ? status.replace(/[^a-z0-9_-]+/gi, "-") : "unknown";
}

export function TodoPopover({ snapshot }: TodoPopoverProps) {
  const todo = normalizeSnapshot(snapshot);

  return (
    <section className="todoPopover" role="dialog" aria-label="Todo">
      <header className="todoPopoverHeader">
        <h3>Todo</h3>
        {todo.available && todo.progress_text ? <p className="todoPopoverSummary">{todo.progress_text}</p> : null}
      </header>
      {!todo.available ? (
        <p className="todoPopoverEmpty">{todo.error ? "Todo list unavailable" : "No todo list yet"}</p>
      ) : (
        <div className="todoPopoverList">
          {todo.items.map((item, index) => (
            <article key={`${item.title || "todo"}-${index}`} className="todoPopoverItem">
              <div className="todoPopoverItemHead">
                <strong>{item.title || "Untitled todo"}</strong>
                <span className={`todoPopoverStatus ${statusClassName(item.status)}`}>{item.status || "unknown"}</span>
              </div>
              {item.description ? <p>{item.description}</p> : null}
            </article>
          ))}
        </div>
      )}
    </section>
  );
}
