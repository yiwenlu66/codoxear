export interface LaunchBackendDefaults {
  provider_choice?: string;
  provider_choices?: string[];
  model?: string | null;
  models?: string[];
  provider_models?: Record<string, string[]>;
  model_provider?: string | null;
  model_providers?: string[];
  preferred_auth_method?: string | null;
  reasoning_effort?: string | null;
  reasoning_efforts?: string[];
  service_tier?: string | null;
  supports_fast?: boolean;
}

export interface NewSessionDefaults {
  default_backend?: string;
  backends?: Record<string, LaunchBackendDefaults>;
}

export interface SessionSummary {
  session_id: string;
  thread_id?: string | null;
  resume_session_id?: string | null;
  title?: string;
  alias?: string;
  first_user_message?: string;
  cwd?: string;
  files?: string[];
  agent_backend?: "codex" | "pi" | string;
  broker_pid?: number;
  owned?: boolean;
  busy?: boolean;
  queue_len?: number;
  updated_ts?: number;
  git_branch?: string | null;
  model?: string | null;
  provider_choice?: string | null;
  reasoning_effort?: string | null;
  service_tier?: string | null;
  transport?: string | null;
  priority_offset?: number | null;
  snooze_until?: number | null;
  dependency_session_id?: string | null;
  blocked?: boolean;
  snoozed?: boolean;
  historical?: boolean;
}

export interface SessionsResponse {
  sessions: SessionSummary[];
  remaining_by_group?: Record<string, number>;
  omitted_group_count?: number;
}

export interface SessionBootstrapResponse {
  recent_cwds?: string[];
  cwd_groups?: Record<string, CwdGroupMeta>;
  new_session_defaults?: NewSessionDefaults;
  tmux_available?: boolean;
}

export interface SessionDetailsResponse {
  ok?: boolean;
  session: SessionSummary;
}

export interface CreateSessionResponse {
  ok?: boolean;
  session_id?: string;
  backend?: string;
  broker_pid?: number;
}

export interface DeleteSessionResponse {
  ok?: boolean;
}

export interface RenameSessionResponse {
  ok?: boolean;
  alias?: string;
}

export interface EditSessionResponse extends RenameSessionResponse {
  priority_offset?: number;
  snooze_until?: number | null;
  dependency_session_id?: string | null;
}

export interface CwdGroupMeta {
  label?: string;
  collapsed?: boolean;
}

export interface LoginResponse {
  ok?: boolean;
}

export interface EditCwdGroupResponse {
  ok?: boolean;
  cwd?: string;
  label?: string;
  collapsed?: boolean;
}

export interface LogoutResponse {
  ok?: boolean;
}

export interface SessionResumeCandidate {
  session_id: string;
  alias?: string;
  first_user_message?: string;
  updated_ts?: number;
  git_branch?: string | null;
}

export interface SessionResumeCandidatesResponse {
  ok?: boolean;
  exists?: boolean;
  will_create?: boolean;
  git_repo?: boolean;
  git_root?: string;
  git_branch?: string;
  sessions: SessionResumeCandidate[];
}

export interface VoiceSettingsResponse {
  ok?: boolean;
  tts_enabled_for_narration?: boolean;
  tts_enabled_for_final_response?: boolean;
  tts_base_url?: string;
  tts_api_key?: string;
  audio?: {
    queue_depth?: number;
    active_listener_count?: number;
    segment_count?: number;
    stream_url?: string;
    last_error?: string;
  };
  notifications?: {
    enabled_devices?: number;
    total_devices?: number;
    vapid_public_key?: string;
  };
}

export interface AudioListenerResponse {
  ok?: boolean;
  active_listener_count?: number;
}

export interface NotificationFeedItem {
  message_id?: string;
  session_display_name?: string;
  notification_text?: string;
  updated_ts?: number;
}

export interface NotificationFeedResponse {
  ok?: boolean;
  items: NotificationFeedItem[];
}

export interface NotificationSubscriptionRecord {
  endpoint?: string;
  notifications_enabled?: boolean;
  device_class?: string;
  device_label?: string;
  created_ts?: number;
  updated_ts?: number;
}

export interface NotificationSubscriptionStateResponse {
  ok?: boolean;
  vapid_public_key?: string;
  subscriptions: NotificationSubscriptionRecord[];
}

export interface NotificationMessageResponse {
  ok?: boolean;
  notification_text?: string;
  summary_status?: string;
}

export interface MessageEvent {
  type?: string;
  role?: string;
  ts?: number;
  text?: string;
  display?: boolean;
  name?: string;
  summary?: string;
  subject?: string;
  description?: string;
  details?: Record<string, unknown>;
  is_error?: boolean;
  answer?: string | string[];
  cancelled?: boolean;
  resolved?: boolean;
  tool_call_id?: string | null;
  was_custom?: boolean;
  agent?: string;
  task?: string;
  output?: string | null;
  progress_text?: string;
  operation?: string;
  custom_type?: string;
  task_id?: string;
  task_list_id?: string;
  owner?: string;
  assigned_by?: string;
  items?: TodoSnapshotItem[];
  options?: Array<{ label?: string; value?: string; title?: string; description?: string } | string>;
  allow_freeform?: boolean;
  allow_multiple?: boolean;
  timeout_ms?: number | null;
  message?: {
    role?: string;
    content?: Array<{ type?: string; text?: string }>;
  };
  question?: string;
  context?: string;
  questions?: Array<{ header?: string; question?: string; options?: Array<{ label?: string; description?: string; preview?: string }>; multiSelect?: boolean }>;
  answers_by_question?: Record<string, string | string[]>;
  toolName?: string;
  prompt_fallback_available?: boolean;
  streaming?: boolean;
  completed?: boolean;
  stream_id?: string;
  turn_id?: string;
  [key: string]: unknown;
}

export interface MessagesResponse {
  events: MessageEvent[];
  offset?: number;
  has_older?: boolean;
  next_before?: number;
  ui_version?: string;
}

export interface SessionUiRequest {
  id?: string;
  method?: "select" | "confirm" | "input" | "editor" | string;
  label?: string;
  title?: string;
  message?: string;
  question?: string;
  context?: string;
  prefill?: string;
  value?: string | string[];
  confirmed?: boolean;
  cancelled?: boolean;
  allow_freeform?: boolean;
  allow_multiple?: boolean;
  options?: Array<{ label?: string; value?: string; title?: string; description?: string } | string>;
  [key: string]: unknown;
}

export interface SessionUiStateResponse {
  requests: SessionUiRequest[];
}

export interface LiveSessionResponse {
  ok?: boolean;
  session_id?: string;
  offset?: number;
  live_offset?: number;
  busy?: boolean;
  requests_version?: string;
  events: MessageEvent[];
  requests?: SessionUiRequest[];
}

export interface WorkspaceResponse {
  ok?: boolean;
  session_id?: string;
  diagnostics?: Record<string, unknown> | null;
  queue?: {
    items?: Array<string | { text?: string }>;
  } | null;
}

export interface SessionCommand {
  name: string;
  description?: string;
  source?: string;
}

export interface SessionCommandsResponse {
  commands: SessionCommand[];
}

export interface AttachmentInjectResponse {
  ok?: boolean;
  path?: string;
  inject_text?: string;
  broker?: Record<string, unknown>;
}

export interface SessionFileListEntry {
  name: string;
  path: string;
  kind: "dir" | "file";
}

export interface SessionFileListResponse {
  ok?: boolean;
  cwd?: string;
  path?: string;
  entries: SessionFileListEntry[];
}

export interface SessionFileReadResponse {
  ok?: boolean;
  kind?: "text" | "image" | string;
  path?: string;
  rel?: string;
  size?: number;
  text?: string;
  editable?: boolean;
  version?: string;
  image_url?: string;
  content_type?: string;
}

export interface GitFileVersionsResponse {
  ok?: boolean;
  cwd?: string;
  path?: string;
  abs_path?: string;
  current_exists?: boolean;
  current_size?: number;
  current_text?: string;
  base_exists?: boolean;
  base_text?: string;
}

export interface HarnessConfigResponse {
  ok?: boolean;
  enabled?: boolean;
  request?: string;
  cooldown_minutes?: number;
  remaining_injections?: number;
}

export interface TodoSnapshotItem {
  id?: number | string;
  title?: string;
  status?: string;
  description?: string;
  owner?: string;
  assigned_by?: string;
  updated_at?: string;
  source?: string;
}

export interface TodoSnapshot {
  available?: boolean;
  error?: boolean;
  progress_text?: string;
  items: TodoSnapshotItem[];
  counts?: Record<string, number>;
}
