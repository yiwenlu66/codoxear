import { useEffect, useState } from "preact/hooks";
import { Button } from "@/components/ui/button";
import { SessionsPane } from "@/components/sessions/SessionsPane";
import { Composer } from "@/components/composer/Composer";
import { ConversationPane } from "@/components/conversation/ConversationPane";
import { FileIcon, HarnessIcon, StopIcon, TodoListIcon, WorkspaceIcon } from "./icons";

export type MobileShellTab = "sessions" | "read" | "chat" | "tools";

interface MobileShellProps {
  activeSessionId: string | null;
  activeTitle: string;
  announcementEnabled: boolean;
  announcementLabel: string;
  canInterrupt: boolean;
  notificationLabel: string;
  notificationsEnabled: boolean;
  onInterrupt(): void;
  onLogout(): void;
  onNewSession(): void;
  onOpenFilePath(path: string, line?: number | null): void;
  onOpenFiles(): void;
  onOpenHarness(): void;
  onOpenSettings(): void;
  onOpenTodo(): void;
  onOpenWorkspace(): void;
  onToggleAnnouncements(): void;
  onToggleNotifications(): void;
}

function MobileToolsSection({
  announcementEnabled,
  announcementLabel,
  notificationsEnabled,
  notificationLabel,
  onLogout,
  onNewSession,
  onOpenFiles,
  onOpenHarness,
  onOpenSettings,
  onOpenTodo,
  onOpenWorkspace,
  onToggleAnnouncements,
  onToggleNotifications,
}: Omit<MobileShellProps, "activeSessionId" | "activeTitle" | "canInterrupt" | "onInterrupt" | "onOpenFilePath">) {
  return (
    <section className="mobileToolsPage" aria-label="Tools">
      <header className="mobileSectionHeader">
        <div>
          <p className="mobileSectionEyebrow">Secondary actions</p>
          <h2 className="mobileSectionTitle">Tools</h2>
        </div>
      </header>
      <div className="mobileToolsGrid">
        <Button type="button" variant="outline" className="mobileToolCard" onClick={onNewSession}>
          <span className="mobileToolCardIcon" aria-hidden="true">+</span>
          <span className="mobileToolCardText">
            <strong>New session</strong>
            <span>Start a new browser-owned session.</span>
          </span>
        </Button>
        <Button type="button" variant="outline" className="mobileToolCard" onClick={onOpenFiles}>
          <FileIcon />
          <span className="mobileToolCardText">
            <strong>Files</strong>
            <span>Inspect tracked files and open diffs.</span>
          </span>
        </Button>
        <Button type="button" variant="outline" className="mobileToolCard" onClick={onOpenTodo}>
          <TodoListIcon />
          <span className="mobileToolCardText">
            <strong>Todo list</strong>
            <span>Review the current task snapshot.</span>
          </span>
        </Button>
        <Button type="button" variant="outline" className="mobileToolCard" onClick={onOpenWorkspace}>
          <WorkspaceIcon />
          <span className="mobileToolCardText">
            <strong>Workspace</strong>
            <span>Open diagnostics, queue state, and files.</span>
          </span>
        </Button>
        <Button type="button" variant="outline" className="mobileToolCard" onClick={onOpenHarness}>
          <HarnessIcon />
          <span className="mobileToolCardText">
            <strong>Harness</strong>
            <span>Inspect or adjust automation controls.</span>
          </span>
        </Button>
        <Button type="button" variant="outline" className="mobileToolCard" onClick={onOpenSettings}>
          <span className="mobileToolCardIcon" aria-hidden="true">S</span>
          <span className="mobileToolCardText">
            <strong>Settings</strong>
            <span>Voice, theme, and composer preferences.</span>
          </span>
        </Button>
      </div>
      <div className="mobileToggleStack">
        <Button type="button" variant="outline" className="mobileToggleButton" onClick={onToggleNotifications}>
          <span className="mobileToggleLabel">Notifications</span>
          <span className="mobileToggleValue">{notificationsEnabled ? "On" : "Off"}</span>
          <span className="visuallyHidden">{notificationLabel}</span>
        </Button>
        <Button type="button" variant="outline" className="mobileToggleButton" onClick={onToggleAnnouncements}>
          <span className="mobileToggleLabel">Announcements</span>
          <span className="mobileToggleValue">{announcementEnabled ? "On" : "Off"}</span>
          <span className="visuallyHidden">{announcementLabel}</span>
        </Button>
        <Button type="button" variant="outline" className="mobileToggleButton" onClick={onLogout}>
          <span className="mobileToggleLabel">Log out</span>
          <span className="mobileToggleValue">Session</span>
        </Button>
      </div>
    </section>
  );
}

function blurActiveInteractiveElement() {
  if (typeof document === "undefined") {
    return;
  }
  const active = document.activeElement;
  if (active instanceof HTMLElement) {
    active.blur();
  }
}

export function MobileShell({
  activeSessionId,
  activeTitle,
  announcementEnabled,
  announcementLabel,
  canInterrupt,
  notificationLabel,
  notificationsEnabled,
  onInterrupt,
  onLogout,
  onNewSession,
  onOpenFilePath,
  onOpenFiles,
  onOpenHarness,
  onOpenSettings,
  onOpenTodo,
  onOpenWorkspace,
  onToggleAnnouncements,
  onToggleNotifications,
}: MobileShellProps) {
  const [tab, setTab] = useState<MobileShellTab>(activeSessionId ? "read" : "sessions");

  useEffect(() => {
    if (activeSessionId) {
      setTab("read");
    } else {
      setTab("sessions");
    }
    blurActiveInteractiveElement();
  }, [activeSessionId]);

  useEffect(() => {
    if (tab !== "chat") {
      blurActiveInteractiveElement();
    }
  }, [tab]);

  return (
    <div className="mobileShell" data-testid="mobile-shell">
      <section className="mobileShellBody">
        {tab === "sessions" ? (
          <div className="mobilePane mobileSessionsPane">
            <SessionsPane onNewSession={onNewSession} />
          </div>
        ) : null}
        {tab === "read" ? (
          <section className="mobilePane mobileReadPane">
            <header className="mobileReadHeader">
              <div className="mobileChatHeading">
                <p className="mobileSectionEyebrow">Read</p>
                <h1 className="mobileChatTitle">{activeSessionId ? activeTitle : "No session selected"}</h1>
              </div>
              <div className="mobileReadHeaderActions">
                {canInterrupt ? (
                  <Button type="button" variant="outline" size="sm" className="mobileInterruptButton" onClick={onInterrupt}>
                    <StopIcon />
                    <span>Interrupt</span>
                  </Button>
                ) : null}
                <Button type="button" variant="outline" size="sm" className="mobileReplyButton" onClick={() => setTab("chat")}>Reply</Button>
              </div>
            </header>
            <ConversationPane
              key={activeSessionId || "no-session"}
              onOpenFilePath={(path, line) => onOpenFilePath(path, line ?? null)}
            />
          </section>
        ) : null}
        {tab === "chat" ? (
          <section className="mobilePane mobileChatPane">
            <header className="mobileChatHeader">
              <div className="mobileChatHeading">
                <p className="mobileSectionEyebrow">Chat</p>
                <h1 className="mobileChatTitle">{activeSessionId ? activeTitle : "No session selected"}</h1>
              </div>
              {canInterrupt ? (
                <Button type="button" variant="outline" size="sm" className="mobileInterruptButton" onClick={onInterrupt}>
                  <StopIcon />
                  <span>Interrupt</span>
                </Button>
              ) : null}
            </header>
            <ConversationPane
              key={activeSessionId || "no-session"}
              onOpenFilePath={(path, line) => onOpenFilePath(path, line ?? null)}
            />
            <Composer compactMobile />
          </section>
        ) : null}
        {tab === "tools" ? (
          <div className="mobilePane mobileToolsPane">
            <MobileToolsSection
              announcementEnabled={announcementEnabled}
              announcementLabel={announcementLabel}
              notificationLabel={notificationLabel}
              notificationsEnabled={notificationsEnabled}
              onLogout={onLogout}
              onNewSession={onNewSession}
              onOpenFiles={onOpenFiles}
              onOpenHarness={onOpenHarness}
              onOpenSettings={onOpenSettings}
              onOpenTodo={onOpenTodo}
              onOpenWorkspace={onOpenWorkspace}
              onToggleAnnouncements={onToggleAnnouncements}
              onToggleNotifications={onToggleNotifications}
            />
          </div>
        ) : null}
      </section>
      <nav className="mobileBottomNav" aria-label="Primary">
        <Button
          type="button"
          variant={tab === "sessions" ? "default" : "outline"}
          className="mobileBottomNavButton"
          onClick={() => setTab("sessions")}
        >
          Sessions
        </Button>
        <Button
          type="button"
          variant={tab === "read" ? "default" : "outline"}
          className="mobileBottomNavButton"
          onClick={() => setTab("read")}
        >
          Read
        </Button>
        <Button
          type="button"
          variant={tab === "chat" ? "default" : "outline"}
          className="mobileBottomNavButton"
          onClick={() => setTab("chat")}
        >
          Chat
        </Button>
        <Button
          type="button"
          variant={tab === "tools" ? "default" : "outline"}
          className="mobileBottomNavButton"
          onClick={() => setTab("tools")}
        >
          Tools
        </Button>
      </nav>
    </div>
  );
}
