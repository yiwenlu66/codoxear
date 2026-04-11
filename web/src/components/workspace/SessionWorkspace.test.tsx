import { render } from "preact";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AppProviders } from "../../app/providers";
import { SessionWorkspace } from "./SessionWorkspace";

vi.mock("../../lib/api", () => ({
  api: {
    submitUiResponse: vi.fn().mockResolvedValue({ ok: true }),
  },
}));

async function flush() {
  await Promise.resolve();
  await Promise.resolve();
}

function createStaticStore<TState extends object, TActions extends Record<string, (...args: any[]) => any>>(
  state: TState,
  actions: TActions,
) {
  return {
    getState: () => state,
    subscribe: () => () => undefined,
    ...actions,
  };
}

describe("SessionWorkspace", () => {
  let root: HTMLDivElement | null = null;

  const ASK_USER_BRIDGE_PREFIX = "__codoxear_ask_user_bridge_v1__";

  afterEach(() => {
    vi.clearAllMocks();
    if (root) {
      render(null, root);
      root.remove();
      root = null;
    }
  });

  it("renders structured ask_user prompts and submits multi-select with freeform input", async () => {
    const { api } = await import("../../lib/api");
    const workspaceRefresh = vi.fn().mockResolvedValue(undefined);
    const liveRefresh = vi.fn().mockResolvedValue(undefined);
    const liveSessionStore = createStaticStore(
      {
        offsetsBySessionId: {},
        requestsBySessionId: {
          "sess-1": [
            {
              id: "req-1",
              method: "select",
              question: "Choose deployment targets",
              context: "You can pick more than one.",
              allow_multiple: true,
              allow_freeform: true,
              options: [
                { title: "Alpha", description: "Primary region" },
                { title: "Beta", description: "Backup region" },
              ],
            },
          ],
        },
        requestVersionsBySessionId: {},
        busyBySessionId: {},
        loadingBySessionId: {},
      },
      { loadInitial: liveRefresh, poll: vi.fn() },
    );
    const sessionUiStore = createStaticStore(
      {
        sessionId: "sess-1",
        diagnostics: null,
        queue: null,
        loading: false,
      },
      { refresh: workspaceRefresh },
    );

    root = document.createElement("div");
    document.body.appendChild(root);
    render(
      <AppProviders liveSessionStore={liveSessionStore as any} sessionUiStore={sessionUiStore as any}>
        <SessionWorkspace />
      </AppProviders>,
      root,
    );

    expect(root.textContent).toContain("Choose deployment targets");
    expect(root.textContent).toContain("You can pick more than one.");
    expect(root.textContent).toContain("Alpha");
    expect(root.textContent).toContain("Primary region");

    const checkboxes = root.querySelectorAll('input[type="checkbox"]');
    const alphaCheckbox = checkboxes[0] as HTMLInputElement;
    alphaCheckbox.click();
    await flush();

    const freeform = root.querySelector('textarea[placeholder="Other response"]') as HTMLTextAreaElement;
    freeform.value = "Gamma";
    freeform.dispatchEvent(new Event("input", { bubbles: true }));
    await flush();

    const confirm = Array.from(root.querySelectorAll("button")).find((button) => button.textContent === "Confirm") as HTMLButtonElement;
    confirm.click();
    await flush();

    expect(api.submitUiResponse).toHaveBeenCalledWith("sess-1", {
      id: "req-1",
      value: ["Alpha", "Gamma"],
    });
    expect(liveRefresh).toHaveBeenCalledWith("sess-1");
    expect(workspaceRefresh).toHaveBeenCalledWith("sess-1", { agentBackend: "pi" });
  });

  it("locks request actions while submitting and shows an error when submission fails", async () => {
    const { api } = await import("../../lib/api");
    const refresh = vi.fn().mockResolvedValue(undefined);
    let rejectRequest: (error?: unknown) => void = () => undefined;
    const pendingSubmission = new Promise((_resolve, reject) => {
      rejectRequest = reject;
    });
    vi.mocked(api.submitUiResponse).mockReturnValueOnce(pendingSubmission as any);
    const sessionUiStore = createStaticStore(
      {
        sessionId: "sess-error",
        diagnostics: null,
        queue: null,
        files: [],
        loading: false,
        requests: [
          {
            id: "req-error",
            method: "confirm",
            question: "Continue with deploy?",
          },
        ],
      },
      { refresh },
    );

    root = document.createElement("div");
    document.body.appendChild(root);
    render(
      <AppProviders sessionUiStore={sessionUiStore as any}>
        <SessionWorkspace />
      </AppProviders>,
      root,
    );

    const confirm = Array.from(root.querySelectorAll("button")).find((button) => button.textContent === "Confirm") as HTMLButtonElement;
    confirm.click();
    confirm.click();
    await flush();

    expect(api.submitUiResponse).toHaveBeenCalledTimes(1);
    expect(confirm.disabled).toBe(true);
    expect(root.textContent).toContain("Submitting...");

    rejectRequest(new Error("Broker unavailable"));
    await flush();

    expect(root.textContent).toContain("Broker unavailable");
    expect(refresh).not.toHaveBeenCalled();
    expect(confirm.disabled).toBe(false);
  });

  it("uses the first option as the default single-select answer", async () => {
    const { api } = await import("../../lib/api");
    const refresh = vi.fn().mockResolvedValue(undefined);
    const sessionUiStore = createStaticStore(
      {
        sessionId: "sess-2",
        diagnostics: null,
        queue: null,
        files: [],
        loading: false,
        requests: [
          {
            id: "req-2",
            method: "select",
            question: "Choose a model",
            options: [
              { title: "fast", description: "Quickest option" },
              { title: "balanced", description: "Default option" },
            ],
          },
        ],
      },
      { refresh },
    );

    root = document.createElement("div");
    document.body.appendChild(root);
    render(
      <AppProviders sessionUiStore={sessionUiStore as any}>
        <SessionWorkspace />
      </AppProviders>,
      root,
    );

    const confirm = Array.from(root.querySelectorAll("button")).find((button) => button.textContent === "Confirm") as HTMLButtonElement;
    confirm.click();
    await flush();

    expect(api.submitUiResponse).toHaveBeenCalledWith("sess-2", {
      id: "req-2",
      value: "fast",
    });
  });

  it("renders bridged AskUserQuestion editor requests as structured multi-question cards", async () => {
    const { api } = await import("../../lib/api");
    const refresh = vi.fn().mockResolvedValue(undefined);
    const sessionUiStore = createStaticStore(
      {
        sessionId: "sess-bridge",
        diagnostics: null,
        queue: null,
        files: [],
        loading: false,
        requests: [
          {
            id: "req-bridge-1",
            method: "editor",
            title: "AskUserQuestion",
            prefill: `${ASK_USER_BRIDGE_PREFIX}\n${JSON.stringify({
              questions: [
                {
                  header: "展示位置",
                  question: "这个 `claude-todo-v2-state` 你希望优先展示在哪一层？",
                  options: [
                    { label: "Composer 上方 (Recommended)", description: "最靠近输入区。" },
                    { label: "会话详情", description: "只放在右侧面板。" },
                  ],
                },
                {
                  header: "展示内容",
                  question: "你希望这个 state 怎么用？",
                  options: [
                    { label: "只作显隐控制", description: "只决定面板显示。" },
                    { label: "显示一个状态标签", description: "同时展示启用状态。" },
                  ],
                },
              ],
              metadata: { source: "brainstorming" },
            })}`,
          },
        ],
      },
      { refresh },
    );

    root = document.createElement("div");
    document.body.appendChild(root);
    render(
      <AppProviders sessionUiStore={sessionUiStore as any}>
        <SessionWorkspace />
      </AppProviders>,
      root,
    );

    expect(root.textContent).toContain("这个 `claude-todo-v2-state` 你希望优先展示在哪一层？");
    expect(root.textContent).toContain("你希望这个 state 怎么用？");
    expect(root.textContent).toContain("Composer 上方 (Recommended)");
    expect(root.textContent).toContain("显示一个状态标签");

    const optionButtons = Array.from(root.querySelectorAll("button"));
    const firstAnswer = optionButtons.find((button) => button.textContent?.includes("Composer 上方")) as HTMLButtonElement;
    const secondAnswer = optionButtons.find((button) => button.textContent?.includes("显示一个状态标签")) as HTMLButtonElement;
    expect(firstAnswer).toBeDefined();
    expect(secondAnswer).toBeDefined();

    firstAnswer.click();
    secondAnswer.click();
    await flush();

    const confirm = optionButtons.find((button) => button.textContent === "Confirm") as HTMLButtonElement;
    confirm.click();
    await flush();

    expect(api.submitUiResponse).toHaveBeenCalledWith("sess-bridge", {
      id: "req-bridge-1",
      value: `${ASK_USER_BRIDGE_PREFIX}\n${JSON.stringify({
        action: "answered",
        answers: {
          "这个 `claude-todo-v2-state` 你希望优先展示在哪一层？": "Composer 上方 (Recommended)",
          "你希望这个 state 怎么用？": "显示一个状态标签",
        },
      })}`,
    });
  });

  it("renders only user-facing request content instead of diagnostics-heavy panels", () => {
    const sessionUiStore = createStaticStore(
      {
        sessionId: "sess-1",
        diagnostics: { status: "ok" },
        queue: { items: [{ text: "next task" }] },
        files: ["src/main.tsx"],
        loading: false,
        requests: [
          {
            id: "req-3",
            method: "confirm",
            question: "Continue with deploy?",
            context: "Need explicit confirmation.",
          },
        ],
      },
      { refresh: vi.fn() },
    );

    root = document.createElement("div");
    document.body.appendChild(root);
    render(
      <AppProviders sessionUiStore={sessionUiStore as any}>
        <SessionWorkspace />
      </AppProviders>,
      root,
    );

    expect(root.querySelector('[data-testid="workspace-card"]')).not.toBeNull();
    expect(root.querySelectorAll('[data-testid="workspace-tab"]').length).toBeGreaterThanOrEqual(3);
    expect(root.textContent).toContain("UI Requests");
    expect(root.textContent).toContain("Continue with deploy?");
    expect(root.textContent).toContain("Diagnostics");
    expect(root.textContent).toContain("Queue");
  });

  it("can render diagnostics in details mode when explicitly requested", () => {
    const sessionUiStore = createStaticStore(
      {
        sessionId: "sess-9",
        diagnostics: { status: "ok", queue_len: 2 },
        queue: { items: [{ text: "next task" }] },
        files: ["src/main.tsx"],
        loading: false,
        requests: [],
      },
      { refresh: vi.fn() },
    );

    root = document.createElement("div");
    document.body.appendChild(root);
    render(
      <AppProviders sessionUiStore={sessionUiStore as any}>
        <SessionWorkspace mode="details" />
      </AppProviders>,
      root,
    );

    expect(root.textContent).toContain("Diagnostics");
    expect(root.textContent).toContain("Status");
    expect(root.textContent).toContain("Queue");
    expect(root.textContent).toContain("next task");
    expect(root.textContent).toContain("src/main.tsx");
  });

  it("renders Pi details with session-file rows and a formatted todo snapshot", () => {
    const sessionUiStore = createStaticStore(
      {
        sessionId: "pi-details",
        diagnostics: {
          log_path: "/tmp/pi-broker.log",
          session_file_path: "/tmp/pi-session.jsonl",
          updated_ts: 1_700_000_100,
          todo_snapshot: {
            available: true,
            error: false,
            progress_text: "1/2 completed",
            items: [
              {
                id: 1,
                title: "Explore project context",
                description: "Inspect the current web app",
                status: "completed",
              },
              {
                id: 2,
                title: "Restore history controls",
                status: "in-progress",
              },
            ],
          },
        },
        queue: null,
        files: [],
        loading: false,
        requests: [],
      },
      { refresh: vi.fn().mockResolvedValue(undefined) },
    );

    root = document.createElement("div");
    document.body.appendChild(root);
    render(
      <AppProviders sessionUiStore={sessionUiStore as any}>
        <SessionWorkspace mode="details" />
      </AppProviders>,
      root,
    );

    expect(root.textContent).toContain("Session file");
    expect(root.textContent).toContain("/tmp/pi-session.jsonl");
    expect(root.textContent).toContain("Log");
    expect(root.textContent).toContain("/tmp/pi-broker.log");
    expect(root.textContent).toContain("Todo list");
    expect(root.textContent).toContain("1/2 completed");
    expect(root.textContent).toContain("Explore project context");
    expect(root.textContent).toContain("Restore history controls");
    expect(root.textContent).not.toContain("session_file_path");
    expect(root.textContent).not.toContain("todo_snapshot");
  });
});
