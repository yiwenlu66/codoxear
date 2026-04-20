import { useEffect, useMemo, useRef, useState } from "preact/hooks";
import { api } from "../lib/api";
import { ConversationPane } from "../components/conversation/ConversationPane";
import { Composer } from "../components/composer/Composer";
import { SessionWorkspace } from "../components/workspace/SessionWorkspace";
import type { FileViewMode } from "../components/workspace/FileViewerDialog";
import { TodoPopover } from "../components/workspace/TodoPopover";
import { AppShellSidebar } from "./app-shell/AppShellSidebar";
import { AppShellToolbar } from "./app-shell/AppShellToolbar";
import { AppShellWorkspaceOverlays } from "./app-shell/AppShellWorkspaceOverlays";
import { MobileShell } from "./app-shell/MobileShell";
import { VoiceSettingsDialog } from "./app-shell/VoiceSettingsDialog";
import { useAppShellAudio } from "./app-shell/useAppShellAudio";
import { useAppShellEvents } from "./app-shell/useAppShellEvents";
import { useAppShellNotifications } from "./app-shell/useAppShellNotifications";
import { useAppShellSessionEffects } from "./app-shell/useAppShellSessionEffects";
import { useLiveSessionStore, useLiveSessionStoreApi, useMessagesStore, useSessionUiStore, useSessionUiStoreApi, useSessionsStore, useSessionsStoreApi } from "./providers";
import {
  applyThemeMode,
  readThemeMode,
  shouldUseMobileWorkspaceSheet,
  shortSessionId,
  writeThemeMode,
} from "./app-shell/utils";
import { getSessionRuntimeId } from "../lib/session-identity";
import { getSessionDisplayName } from "../lib/session-display";

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
  const { busyBySessionId } = useLiveSessionStore();
  const { sessionId: sessionUiSessionId, diagnostics } = useSessionUiStore();
  const sessionsStoreApi = useSessionsStoreApi();
  const liveSessionStoreApi = useLiveSessionStoreApi();
  const sessionUiStoreApi = useSessionUiStoreApi();
  const [newSessionOpen, setNewSessionOpen] = useState(false);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [workspaceOpen, setWorkspaceOpen] = useState(false);
  const [fileViewerOpen, setFileViewerOpen] = useState(false);
  const [harnessOpen, setHarnessOpen] = useState(false);
  const [todoViewerOpen, setTodoViewerOpen] = useState(false);
  const [fileViewerPath, setFileViewerPath] = useState("");
  const [fileViewerLine, setFileViewerLine] = useState<number | null>(null);
  const [fileViewerMode, setFileViewerMode] = useState<FileViewMode | null>(null);
  const [fileViewerRequestKey, setFileViewerRequestKey] = useState(0);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [themeMode, setThemeMode] = useState(() => readThemeMode());
  const [sseConnected, setSseConnected] = useState(false);
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
  const activeSessionRuntimeId = getSessionRuntimeId(activeSession);
  const activeSessionPending = activeSession?.pending_startup === true;
  const activeSessionBusy = Boolean(
    (activeSessionId && busyBySessionId[activeSessionId] === true)
    || activeSession?.busy === true,
  );
  const activeTodoSnapshot = sessionUiSessionId === activeSessionId && diagnostics && typeof diagnostics === "object"
    ? (diagnostics as { todo_snapshot?: unknown }).todo_snapshot ?? null
    : null;
  const activeTitle = activeSession
    ? getSessionDisplayName(activeSession, shortSessionId(activeSession.session_id))
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
    refreshNotificationFeed,
    replySoundEnabled,
    setReplySoundEnabled,
    toggleNotifications,
  } = useAppShellNotifications({
    activeSessionId,
    activeTitle,
    bySessionId,
    playReplyBeep,
    realtimeConnected: sseConnected,
    suppressedReplySoundSessionIdsRef,
    voiceSettings,
  });

  useAppShellEvents({
    activeSessionBackend: activeSession?.agent_backend,
    activeSessionHistorical: activeSession?.historical === true,
    activeSessionId,
    activeSessionPending,
    activeSessionRuntimeId,
    items,
    liveSessionStoreApi,
    onConnectionChange: setSseConnected,
    refreshNotificationsFeed: refreshNotificationFeed,
    sessionUiStoreApi,
    sessionsStoreApi,
    workspaceOpen: workspaceOpen || detailsOpen,
  });

  useAppShellSessionEffects({
    activeSessionBackend: activeSession?.agent_backend,
    activeSessionHistorical: activeSession?.historical === true,
    activeSessionPending,
    activeSessionId,
    activeSessionRuntimeId,
    activeSessionLiveBusy: activeSessionId ? busyBySessionId[activeSessionId] === true : false,
    backgroundReplySoundPrimedSessionIdsRef,
    items,
    liveSessionStoreApi,
    realtimeConnected: sseConnected,
    replySoundEnabled,
    sessionUiStoreApi,
    sessionsStoreApi,
    workspaceOpen: workspaceOpen || detailsOpen,
    activeSessionReplySoundPrimingRef,
    suppressedReplySoundSessionIdsRef,
  });

  const sessionUiMatchesActiveSession = !!activeSessionId && sessionUiSessionId === activeSessionId;
  const showInterruptAction = !activeSessionId || activeSessionBusy;
  const [mobileLayout, setMobileLayout] = useState(() => shouldUseMobileWorkspaceSheet());

  useEffect(() => {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
      return;
    }

    const mediaQuery = window.matchMedia("(max-width: 880px)");
    const update = () => {
      setMobileLayout(mediaQuery.matches);
    };

    update();
    if (typeof mediaQuery.addEventListener === "function") {
      mediaQuery.addEventListener("change", update);
    } else {
      mediaQuery.addListener?.(update);
    }

    return () => {
      if (typeof mediaQuery.removeEventListener === "function") {
        mediaQuery.removeEventListener("change", update);
      } else {
        mediaQuery.removeListener?.(update);
      }
    };
  }, []);

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
    applyThemeMode(themeMode);
    writeThemeMode(themeMode);
  }, [themeMode]);

  useEffect(() => {
    if (activeSessionId) {
      setSidebarOpen(false);
    }
  }, [activeSessionId]);

  useEffect(() => {
    setFileViewerOpen(false);
    setHarnessOpen(false);
    setTodoViewerOpen(false);
  }, [activeSessionId]);

  const shellClassName = useMemo(() => ["appShell", "editorialShell"].join(" "), []);

  const renderWorkspaceDetails = () => (
    sessionUiMatchesActiveSession ? <SessionWorkspace mode="details" /> : <EmptyDetailsWorkspace />
  );

  const openWorkspace = () => {
    setWorkspaceOpen(true);
  };

  const interruptActiveSession = async () => {
    if (!activeSessionId || !activeSessionBusy) return;
    if (activeSessionRuntimeId) {
      await api.interruptSession(activeSessionId, activeSessionRuntimeId);
    } else {
      await api.interruptSession(activeSessionId);
    }
    await Promise.allSettled([
      sessionsStoreApi.refresh(),
      activeSessionRuntimeId
        ? liveSessionStoreApi.loadInitial(activeSessionId, activeSessionRuntimeId)
        : liveSessionStoreApi.loadInitial(activeSessionId),
      activeSessionRuntimeId
        ? sessionUiStoreApi.refresh(activeSessionId, { agentBackend: activeSession?.agent_backend, runtimeId: activeSessionRuntimeId })
        : sessionUiStoreApi.refresh(activeSessionId, { agentBackend: activeSession?.agent_backend }),
    ]);
  };

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.defaultPrevented || event.key !== "Escape") {
        return;
      }
      if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) {
        return;
      }
      if (!activeSessionId || !activeSessionBusy) {
        return;
      }
      event.preventDefault();
      void interruptActiveSession();
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [activeSessionBusy, activeSessionId, interruptActiveSession]);

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
        {mobileLayout ? (
          <MobileShell
            activeSessionId={activeSessionId}
            activeTitle={activeTitle}
            announcementEnabled={announcementEnabled}
            announcementLabel={announcementLabel}
            canInterrupt={Boolean(activeSessionId && activeSessionBusy)}
            notificationLabel={notificationLabel}
            notificationsEnabled={notificationsEnabled}
            onInterrupt={() => {
              void interruptActiveSession();
            }}
            onLogout={() => {
              void logout();
            }}
            onNewSession={() => setNewSessionOpen(true)}
            onOpenFilePath={(path, line) => openFileViewer(path, line ?? null, "file")}
            onOpenFiles={() => openFileViewer()}
            onOpenHarness={() => setHarnessOpen(true)}
            onOpenSettings={() => openVoiceSettings()}
            onOpenTodo={() => setTodoViewerOpen(true)}
            onOpenWorkspace={openWorkspace}
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
        ) : (
          <>
            <aside className="sidebarColumn desktopSessionsRail">{renderSessionsRail()}</aside>
            <section className="conversationColumn">
              <AppShellToolbar
                activeSessionId={activeSessionId}
                activeTitle={activeTitle}
                canInterrupt={Boolean(activeSessionId && activeSessionBusy)}
                showInterruptAction={showInterruptAction}
                showMobileSessionsTrigger={false}
                showMobileToolbarMenu={false}
                onInterrupt={() => {
                  void interruptActiveSession();
                }}
                onOpenFiles={() => openFileViewer()}
                onOpenHarness={() => setHarnessOpen(true)}
                onOpenSessions={() => setSidebarOpen(true)}
                onOpenTodo={() => setTodoViewerOpen(true)}
                onOpenWorkspace={openWorkspace}
              />
              <ConversationPane
                key={activeSessionId || "no-session"}
                onOpenFilePath={(path, line) => openFileViewer(path, line ?? null, "file")}
              />
              <Composer />
            </section>
          </>
        )}
      </div>
      <AppShellWorkspaceOverlays
        activeSessionId={activeSessionId}
        activeSessionRuntimeId={activeSessionRuntimeId}
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
            themeMode={themeMode}
            voiceApiKeyDraft={voiceApiKeyDraft}
            voiceBaseUrlDraft={voiceBaseUrlDraft}
            onChangeEnterToSend={setEnterToSendDraft}
            onChangeNarrationEnabled={setNarrationEnabledDraft}
            onChangeReplySoundEnabled={setReplySoundEnabled}
            onChangeThemeMode={setThemeMode}
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
        todoViewer={(
          <div data-testid="todo-viewer-dialog" className="todoViewerDialogBody">
            <TodoPopover snapshot={activeTodoSnapshot} />
          </div>
        )}
        todoViewerOpen={todoViewerOpen}
        workspaceDetails={renderWorkspaceDetails()}
        workspaceOpen={workspaceOpen}
        onCloseDetails={() => setDetailsOpen(false)}
        onCloseFileViewer={closeFileViewer}
        onCloseHarness={() => setHarnessOpen(false)}
        onCloseNewSession={() => setNewSessionOpen(false)}
        onCloseSidebar={() => setSidebarOpen(false)}
        onCloseTodoViewer={() => setTodoViewerOpen(false)}
        onCloseWorkspace={() => setWorkspaceOpen(false)}
      />
    </>
  );
}
