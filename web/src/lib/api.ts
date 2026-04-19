import { getJson, postJson } from "./http";
import { getSessionRouteId } from "./session-identity";
import type {
  AttachmentInjectResponse,
  AudioListenerResponse,
  CreateSessionResponse,
  DeleteSessionResponse,
  EditSessionResponse,
  FocusSessionResponse,
  GitFileVersionsResponse,
  HarnessConfigResponse,
  EditCwdGroupResponse,
  LiveSessionResponse,
  LoginResponse,
  MessagesResponse,
  NotificationFeedResponse,
  NotificationMessageResponse,
  NotificationSubscriptionStateResponse,
  RenameSessionResponse,
  LogoutResponse,
  SessionFileListResponse,
  SessionFileReadResponse,
  SessionResumeCandidatesResponse,
  SessionBootstrapResponse,
  SessionCommandsResponse,
  SessionDetailsResponse,
  SessionUiStateResponse,
  SessionsResponse,
  VoiceSettingsResponse,
  WorkspaceResponse,
} from "./types";

export const api = {
  me(signal?: AbortSignal) {
    return getJson<{ ok?: boolean }>("/api/me", signal);
  },
  login(password: string, signal?: AbortSignal) {
    return postJson<LoginResponse>("/api/login", { password }, signal);
  },
  listSessions(options?: { groupKey?: string; offset?: number; limit?: number; groupOffset?: number; groupLimit?: number }, signal?: AbortSignal) {
    const query = new URLSearchParams();
    if (options?.groupKey) {
      query.set("group_key", options.groupKey);
    }
    if (typeof options?.offset === "number" && Number.isFinite(options.offset) && options.offset > 0) {
      query.set("offset", String(options.offset));
    }
    if (typeof options?.limit === "number" && Number.isFinite(options.limit) && options.limit > 0) {
      query.set("limit", String(options.limit));
    }
    if (typeof options?.groupOffset === "number" && Number.isFinite(options.groupOffset) && options.groupOffset > 0) {
      query.set("group_offset", String(options.groupOffset));
    }
    if (typeof options?.groupLimit === "number" && Number.isFinite(options.groupLimit) && options.groupLimit > 0) {
      query.set("group_limit", String(options.groupLimit));
    }
    const suffix = query.size ? `?${query.toString()}` : "";
    return getJson<SessionsResponse>(`/api/sessions${suffix}`, signal);
  },
  getSessionsBootstrap(signal?: AbortSignal) {
    return getJson<SessionBootstrapResponse>("/api/sessions/bootstrap", signal);
  },
  getSessionDetails(sessionId: string, signal?: AbortSignal, runtimeId?: string | null) {
    const routeId = getSessionRouteId(sessionId, runtimeId);
    return getJson<SessionDetailsResponse>(`/api/sessions/${routeId}/details`, signal);
  },
  listMessages(sessionId: string, init = false, signal?: AbortSignal, offset?: number, before?: number, limit?: number, runtimeId?: string | null) {
    const query = new URLSearchParams();
    if (init) {
      query.set("init", "1");
    }
    if (typeof offset === "number" && Number.isFinite(offset) && offset > 0) {
      query.set("offset", String(offset));
    }
    if (typeof before === "number" && Number.isFinite(before) && before > 0) {
      query.set("before", String(before));
    }
    if (typeof limit === "number" && Number.isFinite(limit) && limit > 0) {
      query.set("limit", String(limit));
    }
    const suffix = query.size ? `?${query.toString()}` : "";
    const routeId = getSessionRouteId(sessionId, runtimeId);
    return getJson<MessagesResponse>(`/api/sessions/${routeId}/messages${suffix}`, signal);
  },
  getSessionUiState(sessionId: string, signal?: AbortSignal, runtimeId?: string | null) {
    const routeId = getSessionRouteId(sessionId, runtimeId);
    return getJson<SessionUiStateResponse>(`/api/sessions/${routeId}/ui_state`, signal);
  },
  getLiveSession(sessionId: string, offset?: number, requestsVersion?: string, signal?: AbortSignal, liveOffset?: number, runtimeId?: string | null, bridgeOffset?: number) {
    const query = new URLSearchParams();
    if (typeof offset === "number" && Number.isFinite(offset) && offset > 0) {
      query.set("offset", String(offset));
    }
    if (typeof requestsVersion === "string" && requestsVersion.length > 0) {
      query.set("requests_version", requestsVersion);
    }
    if (typeof liveOffset === "number" && Number.isFinite(liveOffset) && liveOffset > 0) {
      query.set("live_offset", String(liveOffset));
    }
    if (typeof bridgeOffset === "number" && Number.isFinite(bridgeOffset) && bridgeOffset > 0) {
      query.set("bridge_offset", String(bridgeOffset));
    }
    const suffix = query.size ? `?${query.toString()}` : "";
    const routeId = getSessionRouteId(sessionId, runtimeId);
    return getJson<LiveSessionResponse>(`/api/sessions/${routeId}/live${suffix}`, signal);
  },
  getWorkspace(sessionId: string, signal?: AbortSignal, runtimeId?: string | null) {
    const routeId = getSessionRouteId(sessionId, runtimeId);
    return getJson<WorkspaceResponse>(`/api/sessions/${routeId}/workspace`, signal);
  },
  getSessionCommands(sessionId: string, signal?: AbortSignal, runtimeId?: string | null) {
    const routeId = getSessionRouteId(sessionId, runtimeId);
    return getJson<SessionCommandsResponse>(`/api/sessions/${routeId}/commands`, signal);
  },
  attachSessionFile(sessionId: string, payload: { filename: string; data_b64: string; attachment_index: number }, runtimeId?: string | null) {
    const routeId = getSessionRouteId(sessionId, runtimeId);
    return postJson<AttachmentInjectResponse>(`/api/sessions/${routeId}/inject_image`, payload);
  },
  async sendMessage(sessionId: string, text: string, runtimeId?: string | null) {
    const routeId = getSessionRouteId(sessionId, runtimeId);
    return postJson(`/api/sessions/${routeId}/send`, { text });
  },
  async enqueueMessage(sessionId: string, text: string, runtimeId?: string | null) {
    const routeId = getSessionRouteId(sessionId, runtimeId);
    return postJson(`/api/sessions/${routeId}/enqueue`, { text });
  },
  deleteSession(sessionId: string, runtimeId?: string | null) {
    const routeId = getSessionRouteId(sessionId, runtimeId);
    return postJson<DeleteSessionResponse>(`/api/sessions/${routeId}/delete`, {});
  },
  async createSession(payload: Record<string, unknown>) {
    return postJson<CreateSessionResponse>(`/api/sessions`, payload);
  },
  getSessionResumeCandidates(cwd: string, agentBackend: string, options?: { offset?: number; limit?: number }) {
    const query = new URLSearchParams();
    query.set("cwd", cwd);
    query.set("backend", agentBackend);
    query.set("agent_backend", agentBackend);
    if (typeof options?.offset === "number") {
      query.set("offset", String(options.offset));
    }
    if (typeof options?.limit === "number") {
      query.set("limit", String(options.limit));
    }
    return getJson<SessionResumeCandidatesResponse>(`/api/session_resume_candidates?${query.toString()}`);
  },
  renameSession(sessionId: string, name: string) {
    return postJson<RenameSessionResponse>(`/api/sessions/${sessionId}/rename`, { name });
  },
  setSessionFocus(sessionId: string, focused: boolean, runtimeId?: string | null) {
    const routeId = getSessionRouteId(sessionId, runtimeId);
    return postJson<FocusSessionResponse>(`/api/sessions/${routeId}/focus`, { focused });
  },
  editSession(sessionId: string, payload: Record<string, unknown>) {
    return postJson<EditSessionResponse>(`/api/sessions/${sessionId}/edit`, payload);
  },
  logout() {
    return postJson<LogoutResponse>(`/api/logout`, {});
  },
  editCwdGroup(payload: { cwd: string; label?: string; collapsed?: boolean }) {
    return postJson<EditCwdGroupResponse>(`/api/cwd_groups/edit`, payload);
  },
  getVoiceSettings() {
    return getJson<VoiceSettingsResponse>(`/api/settings/voice`);
  },
  saveVoiceSettings(payload: Record<string, unknown>) {
    return postJson<VoiceSettingsResponse>(`/api/settings/voice`, payload);
  },
  setAudioListener(clientId: string, enabled: boolean) {
    return postJson<AudioListenerResponse>(`/api/audio/listener`, { client_id: clientId, enabled });
  },
  triggerTestAnnouncement() {
    return postJson(`/api/audio/test_announcement`, {});
  },
  triggerTestPushNotification() {
    return postJson(`/api/notifications/test_push`, {});
  },
  getNotificationsFeed(since: number) {
    const query = new URLSearchParams();
    query.set("since", String(since));
    return getJson<NotificationFeedResponse>(`/api/notifications/feed?${query.toString()}`);
  },
  getNotificationMessage(messageId: string) {
    const query = new URLSearchParams();
    query.set("message_id", messageId);
    return getJson<NotificationMessageResponse>(`/api/notifications/message?${query.toString()}`);
  },
  getNotificationSubscriptionState() {
    return getJson<NotificationSubscriptionStateResponse>(`/api/notifications/subscription`);
  },
  upsertNotificationSubscription(payload: Record<string, unknown>) {
    return postJson<NotificationSubscriptionStateResponse>(`/api/notifications/subscription`, payload);
  },
  toggleNotificationSubscription(endpoint: string, enabled: boolean) {
    return postJson<NotificationSubscriptionStateResponse>(`/api/notifications/subscription/toggle`, { endpoint, enabled });
  },
  getDiagnostics(sessionId: string, runtimeId?: string | null) {
    const routeId = getSessionRouteId(sessionId, runtimeId);
    return getJson(`/api/sessions/${routeId}/diagnostics`);
  },
  getQueue(sessionId: string, runtimeId?: string | null) {
    const routeId = getSessionRouteId(sessionId, runtimeId);
    return getJson(`/api/sessions/${routeId}/queue`);
  },
  getFiles(sessionId: string, path?: string, signal?: AbortSignal, runtimeId?: string | null) {
    const query = new URLSearchParams();
    if (path) {
      query.set("path", path);
    }
    const suffix = query.size ? `?${query.toString()}` : "";
    const routeId = getSessionRouteId(sessionId, runtimeId);
    return getJson<SessionFileListResponse>(`/api/sessions/${routeId}/file/list${suffix}`, signal);
  },
  getFileRead(sessionId: string, path: string, signal?: AbortSignal, runtimeId?: string | null) {
    const query = new URLSearchParams();
    query.set("path", path);
    const routeId = getSessionRouteId(sessionId, runtimeId);
    return getJson<SessionFileReadResponse>(`/api/sessions/${routeId}/file/read?${query.toString()}`, signal);
  },
  getGitFileVersions(sessionId: string, path: string, signal?: AbortSignal, runtimeId?: string | null) {
    const query = new URLSearchParams();
    query.set("path", path);
    const routeId = getSessionRouteId(sessionId, runtimeId);
    return getJson<GitFileVersionsResponse>(`/api/sessions/${routeId}/git/file_versions?${query.toString()}`, signal);
  },
  getHarness(sessionId: string, runtimeId?: string | null) {
    const routeId = getSessionRouteId(sessionId, runtimeId);
    return getJson<HarnessConfigResponse>(`/api/sessions/${routeId}/harness`);
  },
  saveHarness(sessionId: string, payload: Record<string, unknown>, runtimeId?: string | null) {
    const routeId = getSessionRouteId(sessionId, runtimeId);
    return postJson<HarnessConfigResponse>(`/api/sessions/${routeId}/harness`, payload);
  },
  interruptSession(sessionId: string, runtimeId?: string | null) {
    const routeId = getSessionRouteId(sessionId, runtimeId);
    return postJson(`/api/sessions/${routeId}/interrupt`, {});
  },
  submitUiResponse(sessionId: string, payload: Record<string, unknown>, runtimeId?: string | null) {
    const routeId = getSessionRouteId(sessionId, runtimeId);
    return postJson(`/api/sessions/${routeId}/ui_response`, payload);
  },
};
