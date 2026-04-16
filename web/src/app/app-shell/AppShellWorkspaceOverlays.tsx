import type { ComponentChildren } from "preact";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Sheet, SheetContent } from "@/components/ui/sheet";
import { NewSessionDialog } from "../../components/new-session/NewSessionDialog";
import { FileViewerDialog } from "../../components/workspace/FileViewerDialog";
import type { FileViewMode } from "../../components/workspace/FileViewerDialog";
import { HarnessDialog } from "../../components/workspace/HarnessDialog";

interface AppShellWorkspaceOverlaysProps {
  activeSessionId: string | null;
  detailsOpen: boolean;
  fileViewerLine: number | null;
  fileViewerMode: FileViewMode | null;
  fileViewerOpen: boolean;
  fileViewerPath: string;
  fileViewerRequestKey: number;
  harnessOpen: boolean;
  newSessionOpen: boolean;
  sessionsRail: ComponentChildren;
  sidebarOpen: boolean;
  voiceSettingsDialog: ComponentChildren;
  workspaceDetails: ComponentChildren;
  workspaceOpen: boolean;
  onCloseDetails(): void;
  onCloseFileViewer(): void;
  onCloseHarness(): void;
  onCloseNewSession(): void;
  onCloseSidebar(): void;
  onCloseWorkspace(): void;
}

export function AppShellWorkspaceOverlays({
  activeSessionId,
  detailsOpen,
  fileViewerLine,
  fileViewerMode,
  fileViewerOpen,
  fileViewerPath,
  fileViewerRequestKey,
  harnessOpen,
  newSessionOpen,
  sessionsRail,
  sidebarOpen,
  voiceSettingsDialog,
  workspaceDetails,
  workspaceOpen,
  onCloseDetails,
  onCloseFileViewer,
  onCloseHarness,
  onCloseNewSession,
  onCloseSidebar,
  onCloseWorkspace,
}: AppShellWorkspaceOverlaysProps) {
  return (
    <>
      <div data-testid="mobile-sessions-sheet">
        <Sheet open={sidebarOpen}>
          <button type="button" className="sheetBackdropButton" aria-label="Close sessions panel" onClick={onCloseSidebar} />
          <SheetContent side="left" className="mobileSheetContent" titleId="mobile-sessions-title">
            <div className="mobileSheetRail">
              <header className="mobileSheetHeader">
                <h2 id="mobile-sessions-title">Sessions</h2>
                <Button type="button" variant="ghost" size="sm" onClick={onCloseSidebar}>Close</Button>
              </header>
              {sessionsRail}
            </div>
          </SheetContent>
        </Sheet>
      </div>
      <div data-testid="mobile-workspace-sheet">
        <Sheet open={detailsOpen}>
          <button type="button" className="sheetBackdropButton" aria-label="Close workspace panel" onClick={onCloseDetails} />
          <SheetContent side="right" className="mobileSheetContent" titleId="mobile-workspace-title">
            <div className="mobileWorkspaceSheet">
              <header className="mobileSheetHeader">
                <h2 id="mobile-workspace-title">Workspace</h2>
                <Button type="button" variant="ghost" size="sm" onClick={onCloseDetails}>Close</Button>
              </header>
              {workspaceDetails}
            </div>
          </SheetContent>
        </Sheet>
      </div>
      <Dialog open={workspaceOpen} onOpenChange={(open) => {
        if (!open) {
          onCloseWorkspace();
        }
      }}>
        <DialogContent className="workspaceDialog max-w-none" titleId="workspace-dialog-title">
          <div data-testid="workspace-dialog" className="workspaceDialogBody">
            <DialogHeader className="workspaceDialogHeader">
              <div className="flex items-center justify-between gap-3">
                <div className="space-y-1">
                  <DialogTitle id="workspace-dialog-title">Workspace</DialogTitle>
                  <p className="text-sm text-muted-foreground">Inspect session details, queue state, tracked files, and UI requests.</p>
                </div>
                <Button type="button" variant="ghost" size="sm" onClick={onCloseWorkspace}>Close</Button>
              </div>
            </DialogHeader>
            <div className="min-h-0 flex-1 overflow-y-auto p-5">
              {workspaceDetails}
            </div>
          </div>
        </DialogContent>
      </Dialog>
      {voiceSettingsDialog}
      <FileViewerDialog
        open={fileViewerOpen}
        sessionId={activeSessionId}
        initialPath={fileViewerPath}
        initialLine={fileViewerLine}
        initialMode={fileViewerMode}
        openRequestKey={fileViewerRequestKey}
        onClose={onCloseFileViewer}
      />
      <HarnessDialog open={harnessOpen} sessionId={activeSessionId} onClose={onCloseHarness} />
      <NewSessionDialog open={newSessionOpen} onClose={onCloseNewSession} />
    </>
  );
}
