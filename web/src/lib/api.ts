import { getJson, postJson } from "./http";
import type {
  AttachmentInjectResponse,
  AudioListenerResponse,
  CreateSessionResponse,
  DeleteSessionResponse,
  EditSessionResponse,
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
  getSessionDetails(sessionId: string, signal?: AbortSignal) {
    return getJson<SessionDetailsResponse>(`/api/sessions/${sessionId}/details`, signal);
  },
  listMessages(sessionId: string, init = false, signal?: AbortSignal, offset?: number, before?: number, limit?: number) {
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
    return getJson<MessagesResponse>(`/api/sessions/${sessionId}/messages${suffix}`, signal);
  },
  getSessionUiState(sessionId: string, signal?: AbortSignal) {
    return getJson<SessionUiStateResponse>(`/api/sessions/${sessionId}/ui_state`, signal);
  },
  getLiveSession(sessionId: string, offset?: number, requestsVersion?: string, signal?: AbortSignal, liveOffset?: number) {
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
    const suffix = query.size ? `?${query.toString()}` : "";
    return getJson<LiveSessionResponse>(`/api/sessions/${sessionId}/live${suffix}`, signal);
  },
  getWorkspace(sessionId: string, signal?: AbortSignal) {
    return getJson<WorkspaceResponse>(`/api/sessions/${sessionId}/workspace`, signal);
  },
  getSessionCommands(sessionId: string, signal?: AbortSignal) {
    return getJson<SessionCommandsResponse>(`/api/sessions/${sessionId}/commands`, signal);
  },
  attachSessionFile(sessionId: string, payload: { filename: string; data_b64: string; attachment_index: number }) {
    return postJson<AttachmentInjectResponse>(`/api/sessions/${sessionId}/inject_image`, payload);
  },
  async sendMessage(sessionId: string, text: string) {
    return postJson(`/api/sessions/${sessionId}/send`, { text });
  },
  async enqueueMessage(sessionId: string, text: string) {
    return postJson(`/api/sessions/${sessionId}/enqueue`, { text });
  },
  deleteSession(sessionId: string) {
    return postJson<DeleteSessionResponse>(`/api/sessions/${sessionId}/delete`, {});
  },
  async createSession(payload: Record<string, unknown>) {
    return postJson<CreateSessionResponse>(`/api/sessions`, payload);
  },
  getSessionResumeCandidates(cwd: string, agentBackend: string) {
    const query = new URLSearchParams();
    query.set("cwd", cwd);
    query.set("backend", agentBackend);
    query.set("agent_backend", agentBackend);
    return getJson<SessionResumeCandidatesResponse>(`/api/session_resume_candidates?${query.toString()}`);
  },
  renameSession(sessionId: string, name: string) {
    return postJson<RenameSessionResponse>(`/api/sessions/${sessionId}/rename`, { name });
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
  getDiagnostics(sessionId: string) {
    return getJson(`/api/sessions/${sessionId}/diagnostics`);
  },
  getQueue(sessionId: string) {
    return getJson(`/api/sessions/${sessionId}/queue`);
  },
  getFiles(sessionId: string, path?: string, signal?: AbortSignal) {
    const query = new URLSearchParams();
    if (path) {
      query.set("path", path);
    }
    const suffix = query.size ? `?${query.toString()}` : "";
    return getJson<SessionFileListResponse>(`/api/sessions/${sessionId}/file/list${suffix}`, signal);
  },
  getFileRead(sessionId: string, path: string, signal?: AbortSignal) {
    const query = new URLSearchParams();
    query.set("path", path);
    return getJson<SessionFileReadResponse>(`/api/sessions/${sessionId}/file/read?${query.toString()}`, signal);
  },
  getGitFileVersions(sessionId: string, path: string, signal?: AbortSignal) {
    const query = new URLSearchParams();
    query.set("path", path);
    return getJson<GitFileVersionsResponse>(`/api/sessions/${sessionId}/git/file_versions?${query.toString()}`, signal);
  },
  getHarness(sessionId: string) {
    return getJson<HarnessConfigResponse>(`/api/sessions/${sessionId}/harness`);
  },
  saveHarness(sessionId: string, payload: Record<string, unknown>) {
    return postJson<HarnessConfigResponse>(`/api/sessions/${sessionId}/harness`, payload);
  },
  interruptSession(sessionId: string) {
    return postJson(`/api/sessions/${sessionId}/interrupt`, {});
  },
  submitUiResponse(sessionId: string, payload: Record<string, unknown>) {
    return postJson(`/api/sessions/${sessionId}/ui_response`, payload);
  },
};
