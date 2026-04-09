import { getJson, postJson } from "./http";
import type {
  AudioListenerResponse,
  CreateSessionResponse,
  DeleteSessionResponse,
  EditSessionResponse,
  GitFileVersionsResponse,
  HarnessConfigResponse,
  EditCwdGroupResponse,
  MessagesResponse,
  NotificationFeedResponse,
  NotificationMessageResponse,
  NotificationSubscriptionStateResponse,
  RenameSessionResponse,
  LogoutResponse,
  SessionFileListResponse,
  SessionFileReadResponse,
  SessionResumeCandidatesResponse,
  SessionUiStateResponse,
  SessionsResponse,
  VoiceSettingsResponse,
} from "./types";

export const api = {
  listSessions(signal?: AbortSignal) {
    return getJson<SessionsResponse>("/api/sessions", signal);
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
  async sendMessage(sessionId: string, text: string) {
    return postJson(`/api/sessions/${sessionId}/send`, { text });
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
  getFiles(sessionId: string, signal?: AbortSignal) {
    return getJson<SessionFileListResponse>(`/api/sessions/${sessionId}/file/list`, signal);
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
  submitUiResponse(sessionId: string, payload: Record<string, unknown>) {
    return postJson(`/api/sessions/${sessionId}/ui_response`, payload);
  },
};
