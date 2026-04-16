import type { ComponentChildren } from "preact";
import { useEffect, useRef, useState } from "preact/hooks";

interface SessionGroupProps {
  title: string;
  subtitle: string;
  collapsed?: boolean;
  canRename?: boolean;
  isSaving?: boolean;
  errorMessage?: string;
  onRename?: (value: string) => Promise<boolean> | boolean;
  onToggle?: () => void;
  children: ComponentChildren;
}

function ChevronIcon({ collapsed }: { collapsed: boolean }) {
  return (
    <svg
      viewBox="0 0 12 12"
      aria-hidden="true"
      className={`sessionGroupChevron${collapsed ? " isCollapsed" : ""}`}
    >
      <path d="M4 2.5 7.5 6 4 9.5" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function SessionGroup({
  title,
  subtitle,
  collapsed = false,
  canRename = false,
  isSaving = false,
  errorMessage = "",
  onRename,
  onToggle,
  children,
}: SessionGroupProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [draftTitle, setDraftTitle] = useState(title);
  const savingRef = useRef(false);
  const titleTooltip = subtitle?.trim() ? subtitle : title;

  useEffect(() => {
    if (!isEditing) {
      setDraftTitle(title);
    }
  }, [title, isEditing]);

  async function commitRename() {
    if (!onRename || savingRef.current) {
      return;
    }

    savingRef.current = true;
    const saved = await onRename(draftTitle);
    savingRef.current = false;
    if (saved) {
      setIsEditing(false);
    }
  }

  return (
    <section className="sessionGroup">
      <div className="sessionGroupShell">
        <div className="sessionGroupHeader" aria-expanded={!collapsed}>
          {isEditing ? (
            <span className="sessionGroupHeading sessionGroupHeadingEditing">
              <input
                type="text"
                className="sessionGroupRenameInput"
                value={draftTitle}
                onInput={(event) => setDraftTitle(event.currentTarget.value)}
                onBlur={() => {
                  void commitRename();
                }}
                onKeyDown={(event) => {
                  if (event.key === "Enter") {
                    event.preventDefault();
                    void commitRename();
                  }
                  if (event.key === "Escape") {
                    event.preventDefault();
                    setDraftTitle(title);
                    setIsEditing(false);
                  }
                }}
                disabled={isSaving}
                autoFocus
              />
            </span>
          ) : (
            <span className="sessionGroupHeading">
              {onToggle ? (
                <button
                  type="button"
                  className="sessionGroupTitleButton"
                  aria-expanded={!collapsed}
                  onClick={onToggle}
                  disabled={isSaving}
                  title={titleTooltip}
                >
                  <span className="sessionGroupToggle" aria-hidden="true">
                    <ChevronIcon collapsed={collapsed} />
                  </span>
                  <span className="sessionGroupTitle">{title}</span>
                </button>
              ) : (
                <span className="sessionGroupTitleButton isStatic" title={titleTooltip}>
                  <span className="sessionGroupTitle">{title}</span>
                </span>
              )}
            </span>
          )}
          <span className="sessionGroupActions">
            {canRename ? (
              <button
                type="button"
                className="sessionGroupRenameButton"
                onClick={() => {
                  setDraftTitle(title);
                  setIsEditing(true);
                }}
                disabled={isSaving}
              >
                Rename
              </button>
            ) : null}
          </span>
        </div>
        {errorMessage ? <p className="sessionGroupError">{errorMessage}</p> : null}
        {collapsed ? null : <div className="sessionGroupList">{children}</div>}
      </div>
    </section>
  );
}
