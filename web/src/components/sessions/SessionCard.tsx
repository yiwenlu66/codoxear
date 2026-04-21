import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { useEffect, useRef, useState } from "preact/hooks";

import { getSessionDisplayName } from "../../lib/session-display";
import type { SessionSummary } from "../../lib/types";

interface SessionCardProps {
  session: SessionSummary;
  active: boolean;
  onSelect: () => void;
  onToggleFocus?: () => void;
  onEdit?: () => void;
  onDuplicate?: () => void;
  onRestart?: () => void;
  onHandoff?: () => void;
  onDelete?: () => void;
}

function ActionIcon({ kind }: { kind: "edit" | "duplicate" | "delete" | "focus" | "handoff" | "restart" | "menu" }) {
  if (kind === "edit") {
    return (
      <svg viewBox="0 0 16 16" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.4">
        <path d="M11.9 2.3a1.5 1.5 0 0 1 2.1 2.1l-7.4 7.4-3 .8.8-3 7.5-7.3Z" />
        <path d="m10.7 3.5 1.8 1.8" />
      </svg>
    );
  }

  if (kind === "duplicate") {
    return (
      <svg viewBox="0 0 16 16" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.4">
        <rect x="5" y="3" width="8" height="9" rx="1.5" />
        <path d="M3.5 6.5V12A1.5 1.5 0 0 0 5 13.5h5.5" />
      </svg>
    );
  }

  if (kind === "focus") {
    return (
      <svg viewBox="0 0 16 16" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.4">
        <path d="m8 1.7 1.8 3.6 4 .6-2.9 2.8.7 4-3.6-1.9-3.6 1.9.7-4L2.2 5.9l4-.6Z" />
      </svg>
    );
  }

  if (kind === "handoff") {
    return (
      <svg viewBox="0 0 16 16" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.4">
        <path d="M3 8h7" />
        <path d="m8 4 4 4-4 4" />
        <path d="M3 3.5h4" />
        <path d="M3 12.5h4" />
      </svg>
    );
  }

  if (kind === "restart") {
    return (
      <svg viewBox="0 0 16 16" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.4">
        <path d="M12.8 6.5A4.8 4.8 0 1 0 13 8" />
        <path d="M10.5 3.2h2.8V6" />
      </svg>
    );
  }

  if (kind === "menu") {
    return (
      <svg viewBox="0 0 16 16" aria-hidden="true" fill="currentColor">
        <circle cx="3" cy="8" r="1.2" />
        <circle cx="8" cy="8" r="1.2" />
        <circle cx="13" cy="8" r="1.2" />
      </svg>
    );
  }

  return (
    <svg viewBox="0 0 16 16" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.4">
      <path d="M3.5 4.5h9" />
      <path d="M6 4.5V3.4c0-.5.4-.9.9-.9h2.2c.5 0 .9.4.9.9v1.1" />
      <path d="m5 6.2.5 6c.1.7.6 1.3 1.4 1.3h2.2c.7 0 1.3-.6 1.4-1.3l.5-6" />
    </svg>
  );
}

export function useDesktopSessionActions() {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
    return false;
  }
  return Boolean(window.matchMedia("(hover: hover) and (pointer: fine) and (min-width: 881px)").matches);
}

export function SessionCard({ session, active, onSelect, onToggleFocus, onEdit, onDuplicate, onRestart, onHandoff, onDelete }: SessionCardProps) {
  const title = getSessionDisplayName(session);
  const isHistorical = session.historical === true;
  const desktopActions = useDesktopSessionActions();
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement | null>(null);
  const hasInlineActions = Boolean(onToggleFocus || onEdit);
  const hasMenuActions = Boolean(onDuplicate || onRestart || onHandoff || onDelete);
  const hasActions = hasInlineActions || hasMenuActions;
  const showActions = hasActions && (active || desktopActions);
  const idBase = `session-${session.session_id.replace(/[^a-z0-9_-]/gi, "-")}`;
  const titleId = `${idBase}-title`;
  const accessibilityParts = [
    title,
    session.agent_backend || "codex",
    isHistorical ? "historical" : session.busy ? "busy" : "idle",
    !isHistorical && session.focused ? "focused" : null,
    !isHistorical && session.queue_len ? `${session.queue_len} queued` : null,
  ].filter(Boolean);
  const accessibilityLabel = accessibilityParts.join(", ");
  const stopActionClick = (event: MouseEvent) => {
    event.preventDefault();
    event.stopPropagation();
  };

  useEffect(() => {
    if (!menuOpen) {
      return undefined;
    }

    const handlePointerDown = (event: MouseEvent) => {
      const target = event.target instanceof Node ? event.target : null;
      if (target && menuRef.current?.contains(target)) {
        return;
      }
      setMenuOpen(false);
    };

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setMenuOpen(false);
      }
    };

    document.addEventListener("mousedown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("mousedown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [menuOpen]);

  const runMenuAction = (action?: () => void) => {
    setMenuOpen(false);
    action?.();
  };

  return (
    <div
      data-testid="session-card"
      className={cn("sessionCard", active && "active")}
      aria-current={active ? "true" : undefined}
    >
      <Card className={cn("sessionCardSurface h-full border-border/60 bg-card/90 shadow-sm", active && "ring-1 ring-primary/30 shadow-md")}>
        <CardContent className="sessionCardContent p-2">
          <button
            type="button"
            className="sessionCardButton compactSessionButton absolute inset-0 rounded-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/50"
            aria-current={active ? "true" : undefined}
            aria-label={accessibilityLabel}
            onClick={onSelect}
          />
          <div className="sessionCardLayout pointer-events-none relative z-10 px-2 py-1.5">
            <div className="sessionCardMainRow">
              <div className="sessionTitleWrap">
                <div id={titleId} className="sessionTitle">{title}</div>
              </div>
            </div>
            <div className="sessionCardFooterRow">
              <div className="sessionCardHeaderAside">
                <div className="sessionMetaBadges flex items-center justify-end gap-1">
                  {!isHistorical ? <span className={cn("stateDot", session.busy && "busy")} /> : null}
                  <Badge variant="secondary" className="backendBadge">{session.agent_backend || "codex"}</Badge>
                  {isHistorical ? <Badge variant="outline" className="ownerBadge">history</Badge> : null}
                  {!isHistorical && session.focused ? <Badge variant="outline" className="ownerBadge">Focus</Badge> : null}
                  {!isHistorical && session.queue_len ? <Badge className="queueBadge">{session.queue_len}</Badge> : null}
                </div>
              </div>
              <div className="sessionCardFooterAside">
                {showActions ? (
                  <div className="sessionActionRowInline flex items-center justify-end gap-1">
                    {onToggleFocus ? (
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        className={cn("sessionActionIconButton h-8 w-8 rounded-md text-muted-foreground hover:text-foreground", session.focused && "text-foreground")}
                        aria-label={session.focused ? "Remove from Focus" : "Add to Focus"}
                        onClick={(event) => {
                          stopActionClick(event);
                          setMenuOpen(false);
                          onToggleFocus();
                        }}
                      >
                        <ActionIcon kind="focus" />
                      </Button>
                    ) : null}
                    {onEdit ? (
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        className="sessionActionIconButton h-8 w-8 rounded-md text-muted-foreground hover:text-foreground"
                        aria-label="Edit session"
                        onClick={(event) => {
                          stopActionClick(event);
                          setMenuOpen(false);
                          onEdit();
                        }}
                      >
                        <ActionIcon kind="edit" />
                      </Button>
                    ) : null}
                    {hasMenuActions ? (
                      <div ref={menuRef} className="sessionActionMenuWrap relative">
                        <Button
                          type="button"
                          variant="ghost"
                          size="icon"
                          className="sessionActionIconButton h-8 w-8 rounded-md text-muted-foreground hover:text-foreground"
                          aria-label="More session actions"
                          aria-haspopup="menu"
                          aria-expanded={menuOpen ? "true" : "false"}
                          onClick={(event) => {
                            stopActionClick(event);
                            setMenuOpen((current) => !current);
                          }}
                        >
                          <ActionIcon kind="menu" />
                        </Button>
                        {menuOpen ? (
                          <div
                            role="menu"
                            aria-label="Session actions"
                            className="sessionActionMenu absolute right-0 top-[calc(100%+0.3rem)] z-30 min-w-[12rem] rounded-xl border border-border bg-card p-1 shadow-2xl"
                            onClick={(event) => stopActionClick(event as MouseEvent)}
                          >
                            {onDuplicate ? (
                              <button
                                type="button"
                                role="menuitem"
                                className="sessionActionMenuItem"
                                onClick={() => runMenuAction(onDuplicate)}
                              >
                                <ActionIcon kind="duplicate" />
                                <span>Duplicate</span>
                              </button>
                            ) : null}
                            {onRestart ? (
                              <button
                                type="button"
                                role="menuitem"
                                className="sessionActionMenuItem"
                                onClick={() => runMenuAction(onRestart)}
                              >
                                <ActionIcon kind="restart" />
                                <span>Restart Pi...</span>
                              </button>
                            ) : null}
                            {onHandoff ? (
                              <button
                                type="button"
                                role="menuitem"
                                className="sessionActionMenuItem"
                                onClick={() => runMenuAction(onHandoff)}
                              >
                                <ActionIcon kind="handoff" />
                                <span>Handoff...</span>
                              </button>
                            ) : null}
                            {onDelete ? (
                              <button
                                type="button"
                                role="menuitem"
                                className="sessionActionMenuItem danger"
                                onClick={() => runMenuAction(onDelete)}
                              >
                                <ActionIcon kind="delete" />
                                <span>Delete...</span>
                              </button>
                            ) : null}
                          </div>
                        ) : null}
                      </div>
                    ) : null}
                  </div>
                ) : null}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
