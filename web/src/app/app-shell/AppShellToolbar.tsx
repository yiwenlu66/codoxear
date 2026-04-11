import { useEffect, useRef, useState } from "preact/hooks";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { FileIcon, HarnessIcon, MenuIcon, SessionsIcon, StopIcon, WorkspaceIcon } from "./icons";

interface AppShellToolbarProps {
  activeSessionId: string | null;
  activeTitle: string;
  canInterrupt: boolean;
  showInterruptAction: boolean;
  showMobileSessionsTrigger: boolean;
  showMobileToolbarMenu: boolean;
  onInterrupt(): void;
  onOpenFiles(): void;
  onOpenHarness(): void;
  onOpenSessions(): void;
  onOpenWorkspace(): void;
}

export function AppShellToolbar({
  activeSessionId,
  activeTitle,
  canInterrupt,
  showInterruptAction,
  showMobileSessionsTrigger,
  showMobileToolbarMenu,
  onInterrupt,
  onOpenFiles,
  onOpenHarness,
  onOpenSessions,
  onOpenWorkspace,
}: AppShellToolbarProps) {
  const [mobileToolsOpen, setMobileToolsOpen] = useState(false);
  const mobileToolsMenuRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    setMobileToolsOpen(false);
  }, [activeSessionId, showMobileToolbarMenu]);

  useEffect(() => {
    if (!mobileToolsOpen) {
      return undefined;
    }

    const handleDocumentClick = (event: MouseEvent) => {
      const target = event.target;
      if (!(target instanceof Node)) {
        return;
      }
      if (mobileToolsMenuRef.current?.contains(target)) {
        return;
      }
      setMobileToolsOpen(false);
    };

    document.addEventListener("click", handleDocumentClick);
    return () => {
      document.removeEventListener("click", handleDocumentClick);
    };
  }, [mobileToolsOpen]);

  const closeMobileToolsMenu = () => {
    setMobileToolsOpen(false);
  };

  const renderConversationActionButtons = (mobileMenu = false) => (
    <>
      {mobileMenu && showMobileSessionsTrigger ? (
        <Button
          type="button"
          variant="ghost"
          size="sm"
          className="conversationMenuItem"
          aria-label="Sessions"
          title="Sessions"
          onClick={() => {
            closeMobileToolsMenu();
            onOpenSessions();
          }}
        >
          <SessionsIcon />
          <span>Sessions</span>
        </Button>
      ) : null}
      <Button
        type="button"
        variant={mobileMenu ? "ghost" : "outline"}
        size={mobileMenu ? "sm" : "icon"}
        className={mobileMenu ? "conversationMenuItem" : "toolbarButton conversationToolButton"}
        aria-label="Files"
        title="Files"
        disabled={!activeSessionId}
        onClick={() => {
          closeMobileToolsMenu();
          onOpenFiles();
        }}
      >
        <FileIcon />
        {mobileMenu ? <span>Files</span> : null}
      </Button>
      <Button
        type="button"
        variant={mobileMenu ? "ghost" : "outline"}
        size={mobileMenu ? "sm" : "icon"}
        className={mobileMenu ? "conversationMenuItem" : "toolbarButton conversationToolButton"}
        aria-label="Workspace"
        title="Workspace"
        disabled={!activeSessionId}
        onClick={() => {
          closeMobileToolsMenu();
          onOpenWorkspace();
        }}
      >
        <WorkspaceIcon />
        {mobileMenu ? <span>Workspace</span> : null}
      </Button>
      {showInterruptAction ? (
        <Button
          type="button"
          variant={mobileMenu ? "ghost" : "outline"}
          size={mobileMenu ? "sm" : "icon"}
          className={cn(
            mobileMenu ? "conversationMenuItem conversationMenuItemDanger" : "toolbarButton conversationToolButton conversationToolButtonDanger",
          )}
          aria-label="Interrupt (Esc)"
          title="Interrupt (Esc)"
          disabled={!canInterrupt}
          onClick={() => {
            closeMobileToolsMenu();
            onInterrupt();
          }}
        >
          <StopIcon />
          {mobileMenu ? <span>Interrupt</span> : null}
        </Button>
      ) : null}
      <Button
        type="button"
        variant={mobileMenu ? "ghost" : "outline"}
        size={mobileMenu ? "sm" : "icon"}
        className={mobileMenu ? "conversationMenuItem" : "toolbarButton conversationToolButton"}
        aria-label="Harness mode"
        title="Harness mode"
        disabled={!activeSessionId}
        onClick={() => {
          closeMobileToolsMenu();
          onOpenHarness();
        }}
      >
        <HarnessIcon />
        {mobileMenu ? <span>Harness mode</span> : null}
      </Button>
    </>
  );

  return (
    <div className="conversationToolbar">
      <div className="conversationToolbarGroup conversationToolbarGroupPrimary">
        {showMobileToolbarMenu ? (
          <div ref={mobileToolsMenuRef} className="conversationMenuWrap">
            <Button
              type="button"
              variant="outline"
              size="icon"
              className="toolbarButton mobileToolsTrigger conversationMenuButton"
              aria-label="Conversation tools"
              title="Conversation tools"
              aria-expanded={mobileToolsOpen ? "true" : "false"}
              onClick={() => setMobileToolsOpen((current) => !current)}
            >
              <MenuIcon />
            </Button>
            {mobileToolsOpen ? <div className="conversationMenuPanel">{renderConversationActionButtons(true)}</div> : null}
          </div>
        ) : null}
        <div className="conversationTitle">{activeSessionId ? activeTitle : "No session selected"}</div>
      </div>
      <div className="conversationToolbarGroup conversationToolbarGroupActions">
        {showMobileToolbarMenu ? null : renderConversationActionButtons()}
      </div>
    </div>
  );
}
