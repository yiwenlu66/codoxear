import { useEffect, useMemo, useRef, useState } from "preact/hooks";
import { api } from "../lib/api";
import { ConversationPane } from "../components/conversation/ConversationPane";
import { Composer } from "../components/composer/Composer";
import { SessionWorkspace } from "../components/workspace/SessionWorkspace";
import type { FileViewMode } from "../components/workspace/FileViewerDialog";
import { AppShellSidebar } from "./app-shell/AppShellSidebar";
import { AppShellToolbar } from "./app-shell/AppShellToolbar";
import { AppShellWorkspaceOverlays } from "./app-shell/AppShellWorkspaceOverlays";
import { VoiceSettingsDialog } from "./app-shell/VoiceSettingsDialog";
import { useAppShellAudio } from "./app-shell/useAppShellAudio";
import { useAppShellNotifications } from "./app-shell/useAppShellNotifications";
import { useAppShellSessionEffects } from "./app-shell/useAppShellSessionEffects";
import { useLiveSessionStoreApi, useMessagesStore, useSessionUiStore, useSessionUiStoreApi, useSessionsStore, useSessionsStoreApi } from "./providers";
import {
  shouldUseMobileWorkspaceSheet,
  shortSessionId,
} from "./app-shell/utils";

function EmptyDetailsWorkspace() {
  return (
    <aside className="workspacePane">
      <section className="workspaceSection">
        <h3>Diagnostics</h3>
        <p>No diagnostics available.</p>
      </section>
      <section className="workspaceSection">
        <h3>Queue</h3>
        <ul className="workspaceList">
          <li>No queued items</li>
        </ul>
      </section>
      <section className="workspaceSection">
        <h3>Files</h3>
        <ul className="workspaceList">
          <li>No tracked files</li>
        </ul>
      </section>
      <section className="workspaceSection">
        <h3>UI Requests</h3>
        <p>No pending requests</p>
      </section>
    </aside>
  );
}

export function AppShell() {
  const { bySessionId } = useMessagesStore();
  const { activeSessionId, items } = useSessionsStore();
  const { sessionId: sessionUiSessionId } = useSessionUiStore();
  const sessionsStoreApi = useSessionsStoreApi();
  const liveSessionStoreApi = useLiveSessionStoreApi();
  const sessionUiStoreApi = useSessionUiStoreApi();
  const [newSessionOpen, setNewSessionOpen] = useState(false);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [workspaceOpen, setWorkspaceOpen] = useState(false);
  const [fileViewerOpen, setFileViewerOpen] = useState(false);
  const [harnessOpen, setHarnessOpen] = useState(false);
  const [fileViewerPath, setFileViewerPath] = useState("");
  const [fileViewerLine, setFileViewerLine] = useState<number | null>(null);
  const [fileViewerMode, setFileViewerMode] = useState<FileViewMode | null>(null);
  const [fileViewerRequestKey, setFileViewerRequestKey] = useState(0);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const {
    announcementEnabled,
    announcementLabel,
    closeVoiceSettings,
    enterToSendDraft,
    liveAudioRef,
    narrationEnabledDraft,
    openVoiceSettings,
    saveVoiceSettings,
    setEnterToSendDraft,
    setNarrationEnabledDraft,
    setVoiceSettingsStatus,
    setVoiceApiKeyDraft,
    setVoiceBaseUrlDraft,
    startAnnouncementPlayback,
    toggleAnnouncements,
    voiceApiKeyDraft,
    voiceBaseUrlDraft,
    voiceSettings,
    voiceSettingsOpen,
    voiceSettingsStatus,
  } = useAppShellAudio();
  const backgroundReplySoundPrimedSessionIdsRef = useRef(new Set<string>());
  const suppressedReplySoundSessionIdsRef = useRef(new Set<string>());
  const activeSessionReplySoundPrimingRef = useRef<string | null>(null);

  useEffect(() => {
    if (!activeSessionId) return;
    suppressedReplySoundSessionIdsRef.current.add(activeSessionId);
    activeSessionReplySoundPrimingRef.current = activeSessionId;
  }, [activeSessionId]);

  useEffect(() => {
    sessionsStoreApi.refreshBootstrap().catch(() => undefined);
  }, [sessionsStoreApi]);

  const activeSession = items.find((session) => session.session_id === activeSessionId) ?? null;
  const activeTitle = activeSession
    ? activeSession.alias || activeSession.first_user_message || activeSession.title || shortSessionId(activeSession.session_id)
    : "No session selected";

  const playReplyBeep = () => {
    try {
      const AudioContextCtor = (window as any).AudioContext
        || (window as any).webkitAudioContext
        || (globalThis as any).AudioContext
        || (globalThis as any).webkitAudioContext;
      if (!AudioContextCtor) {
        return;
      }
      const ctx = new AudioContextCtor();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.type = "triangle";
      osc.frequency.setValueAtTime(987.77, ctx.currentTime);
      gain.gain.setValueAtTime(0.0001, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.08, ctx.currentTime + 0.01);
      gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.18);
      osc.start(ctx.currentTime);
      osc.stop(ctx.currentTime + 0.18);
      osc.onended = () => {
        void ctx.close().catch(() => undefined);
      };
    } catch {
      // Best-effort local cue only.
    }
  };

  const {
    notificationLabel,
    notificationsEnabled,
    replySoundEnabled,
    setReplySoundEnabled,
    toggleNotifications,
  } = useAppShellNotifications({
    activeSessionId,
    activeTitle,
    bySessionId,
    playReplyBeep,
    suppressedReplySoundSessionIdsRef,
    voiceSettings,
  });

  useAppShellSessionEffects({
    activeSessionBackend: activeSession?.agent_backend,
    activeSessionId,
    backgroundReplySoundPrimedSessionIdsRef,
    items,
    liveSessionStoreApi,
    replySoundEnabled,
    sessionUiStoreApi,
    sessionsStoreApi,
    workspaceOpen: workspaceOpen || detailsOpen,
    activeSessionReplySoundPrimingRef,
    suppressedReplySoundSessionIdsRef,
  });

  const sessionUiMatchesActiveSession = !!activeSessionId && sessionUiSessionId === activeSessionId;
  const showInterruptAction = !activeSessionId || Boolean(activeSession?.busy);
  const showMobileSessionsTrigger = shouldUseMobileWorkspaceSheet();
  const showMobileToolbarMenu = showMobileSessionsTrigger;

  const openFileViewer = (path = "", line: number | null = null, mode: FileViewMode | null = null) => {
    setFileViewerPath(path);
    setFileViewerLine(line);
    setFileViewerMode(mode);
    setFileViewerRequestKey((current) => current + 1);
    setFileViewerOpen(true);
  };

  const closeFileViewer = () => {
    setFileViewerOpen(false);
    setFileViewerPath("");
    setFileViewerLine(null);
    setFileViewerMode(null);
  };

  const logout = async () => {
    try {
      await api.logout();
      if (typeof window !== "undefined") {
        window.location.reload();
      }
    } catch {
      // allow retry from the UI
    }
  };

  useEffect(() => {
    if (activeSessionId) {
      setSidebarOpen(false);
    }
  }, [activeSessionId]);

  useEffect(() => {
    setFileViewerOpen(false);
    setHarnessOpen(false);
  }, [activeSessionId]);

  const shellClassName = useMemo(() => ["appShell", "editorialShell"].join(" "), []);

  const renderWorkspaceDetails = () => (
    sessionUiMatchesActiveSession ? <SessionWorkspace mode="details" /> : <EmptyDetailsWorkspace />
  );

  const openWorkspace = () => {
    if (shouldUseMobileWorkspaceSheet()) {
      setDetailsOpen(true);
      return;
    }
    setWorkspaceOpen(true);
  };

  const interruptActiveSession = async () => {
    if (!activeSessionId || !activeSession?.busy) return;
    await api.interruptSession(activeSessionId);
    await Promise.allSettled([
      sessionsStoreApi.refresh(),
      liveSessionStoreApi.loadInitial(activeSessionId),
      sessionUiStoreApi.refresh(activeSessionId, { agentBackend: activeSession.agent_backend }),
    ]);
  };

  const triggerTestPushNotification = async () => {
    setVoiceSettingsStatus("Sending test push...");
    try {
      const response = await api.triggerTestPushNotification() as { sent_count?: number; failed_count?: number; target_count?: number };
      const sent = Number(response.sent_count || 0);
      const failed = Number(response.failed_count || 0);
      const target = Number(response.target_count || sent + failed);
      if (sent > 0 && failed <= 0) {
        setVoiceSettingsStatus(`Test push sent to ${sent} device${sent === 1 ? "" : "s"}.`);
        return;
      }
      setVoiceSettingsStatus(`Test push sent to ${sent}/${target} devices${failed > 0 ? ` (${failed} failed)` : ""}.`);
    } catch (error) {
      setVoiceSettingsStatus(error instanceof Error ? `test push error: ${error.message}` : "test push error: unknown error");
    }
  };

  const renderSessionsRail = () => (
    <AppShellSidebar
      announcementEnabled={announcementEnabled}
      announcementLabel={announcementLabel}
      notificationLabel={notificationLabel}
      notificationsEnabled={notificationsEnabled}
      onBrandClick={() => {
        setSidebarOpen(false);
        if (announcementEnabled) {
          void startAnnouncementPlayback(voiceSettings, { resetSource: true, force: true });
        }
      }}
      onNewSession={() => setNewSessionOpen(true)}
      onOpenSettings={() => openVoiceSettings()}
      onLogout={() => {
        void logout();
      }}
      onToggleAnnouncements={() => {
        void toggleAnnouncements();
        if (!announcementEnabled) {
          void startAnnouncementPlayback(voiceSettings, { resetSource: true, force: true });
        }
      }}
      onToggleNotifications={() => {
        void toggleNotifications();
      }}
    />
  );


  return (
    <>
      <div className={shellClassName} data-testid="app-shell">
        <audio ref={liveAudioRef} className="liveAudioElement" preload="none" />
        <aside className="sidebarColumn desktopSessionsRail">{renderSessionsRail()}</aside>
        <section className="conversationColumn">
          <AppShellToolbar
            activeSessionId={activeSessionId}
            activeTitle={activeTitle}
            canInterrupt={Boolean(activeSessionId && activeSession?.busy)}
            showInterruptAction={showInterruptAction}
            showMobileSessionsTrigger={showMobileSessionsTrigger}
            showMobileToolbarMenu={showMobileToolbarMenu}
            onInterrupt={() => {
              void interruptActiveSession();
            }}
            onOpenFiles={() => openFileViewer()}
            onOpenHarness={() => setHarnessOpen(true)}
            onOpenSessions={() => setSidebarOpen(true)}
            onOpenWorkspace={openWorkspace}
          />
          <ConversationPane onOpenFilePath={(path, line) => openFileViewer(path, line ?? null, "file")} />
          <Composer />
        </section>
      </div>
      <AppShellWorkspaceOverlays
        activeSessionId={activeSessionId}
        detailsOpen={detailsOpen}
        fileViewerLine={fileViewerLine}
        fileViewerMode={fileViewerMode}
        fileViewerOpen={fileViewerOpen}
        fileViewerPath={fileViewerPath}
        fileViewerRequestKey={fileViewerRequestKey}
        harnessOpen={harnessOpen}
        newSessionOpen={newSessionOpen}
        sessionsRail={renderSessionsRail()}
        sidebarOpen={sidebarOpen}
        voiceSettingsDialog={(
          <VoiceSettingsDialog
            audioMeta={{
              enabledDevices: voiceSettings.notifications?.enabled_devices ?? 0,
              lastError: String(voiceSettings.audio?.last_error || ""),
              listeners: voiceSettings.audio?.active_listener_count ?? 0,
              queue: voiceSettings.audio?.queue_depth ?? 0,
              segments: voiceSettings.audio?.segment_count ?? 0,
              totalDevices: voiceSettings.notifications?.total_devices ?? 0,
            }}
            enterToSendDraft={enterToSendDraft}
            narrationEnabledDraft={narrationEnabledDraft}
            open={voiceSettingsOpen}
            replySoundEnabled={replySoundEnabled}
            status={voiceSettingsStatus}
            voiceApiKeyDraft={voiceApiKeyDraft}
            voiceBaseUrlDraft={voiceBaseUrlDraft}
            onChangeEnterToSend={setEnterToSendDraft}
            onChangeNarrationEnabled={setNarrationEnabledDraft}
            onChangeReplySoundEnabled={setReplySoundEnabled}
            onChangeVoiceApiKey={setVoiceApiKeyDraft}
            onChangeVoiceBaseUrl={setVoiceBaseUrlDraft}
            onClose={closeVoiceSettings}
            onSave={() => {
              void saveVoiceSettings();
            }}
            onTriggerTestPush={() => {
              void triggerTestPushNotification();
            }}
          />
        )}
        workspaceDetails={renderWorkspaceDetails()}
        workspaceOpen={workspaceOpen}
        onCloseDetails={() => setDetailsOpen(false)}
        onCloseFileViewer={closeFileViewer}
        onCloseHarness={() => setHarnessOpen(false)}
        onCloseNewSession={() => setNewSessionOpen(false)}
        onCloseSidebar={() => setSidebarOpen(false)}
        onCloseWorkspace={() => setWorkspaceOpen(false)}
      />
    </>
  );
}
