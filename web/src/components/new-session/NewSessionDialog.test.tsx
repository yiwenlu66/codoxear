import { render } from "preact";
import { act } from "preact/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AppProviders } from "../../app/providers";
import { NewSessionDialog } from "./NewSessionDialog";

vi.mock("../../lib/api", () => ({
  api: {
    createSession: vi.fn(),
    getSessionResumeCandidates: vi.fn(),
    renameSession: vi.fn(),
  },
}));

async function flush() {
  await Promise.resolve();
  await Promise.resolve();
}

async function setInputValue(element: HTMLInputElement, value: string) {
  await act(async () => {
    element.value = value;
    element.dispatchEvent(new Event("input", { bubbles: true }));
    element.dispatchEvent(new Event("change", { bubbles: true }));
  });
}

async function setSelectValue(element: HTMLSelectElement, value: string) {
  await act(async () => {
    element.value = value;
    element.dispatchEvent(new Event("input", { bubbles: true }));
    element.dispatchEvent(new Event("change", { bubbles: true }));
  });
}

async function setCheckboxValue(element: HTMLInputElement, checked: boolean) {
  await act(async () => {
    element.checked = checked;
    element.dispatchEvent(new Event("change", { bubbles: true }));
  });
}

async function submitForm(element: HTMLFormElement) {
  await act(async () => {
    if (typeof element.requestSubmit === "function") {
      element.requestSubmit();
    } else {
      element.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
    }
  });
}

function createSessionsStore(initialState: any) {
  let state = initialState;
  const listeners = new Set<() => void>();
  return {
    getState: () => state,
    subscribe(listener: () => void) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    async refresh(options?: { preferNewest?: boolean }) {
      if (options?.preferNewest) {
        state = {
          ...state,
          activeSessionId: state.items[0]?.session_id ?? null,
        };
      }
      listeners.forEach((listener) => listener());
    },
    async refreshBootstrap() {
      state = { ...state, bootstrapLoaded: true };
      listeners.forEach((listener) => listener());
    },
    select: vi.fn((sessionId: string) => {
      state = { ...state, activeSessionId: sessionId };
      listeners.forEach((listener) => listener());
    }),
    setState(next: any) {
      state = next;
      listeners.forEach((listener) => listener());
    },
  };
}

describe("NewSessionDialog", () => {
  let root: HTMLDivElement | null = null;

  afterEach(() => {
    vi.clearAllMocks();
    if (root) {
      render(null, root);
      root.remove();
      root = null;
    }
  });

  it("creates a session and selects the returned session id", async () => {
    const { api } = await import("../../lib/api");
    vi.mocked(api.createSession).mockResolvedValue({ session_id: "new", broker_pid: 42, backend: "codex", ok: true } as any);
    vi.mocked(api.getSessionResumeCandidates).mockResolvedValue({
      exists: true,
      will_create: false,
      git_repo: false,
      sessions: [],
    } as any);
    vi.mocked(api.renameSession).mockResolvedValue({ ok: true } as any);
    const sessionsStore = createSessionsStore({
      items: [
        { session_id: "old" },
        { session_id: "new" },
      ],
      activeSessionId: "old",
      loading: false,
      bootstrapLoaded: true,
      recentCwds: ["/tmp/project"],
      tmuxAvailable: true,
      newSessionDefaults: {
        default_backend: "codex",
        backends: {
          pi: { provider_choice: "macaron", model: "gpt-5.4", reasoning_effort: "high" },
          codex: {
            provider_choice: "chatgpt",
            provider_choices: ["chatgpt", "openai-api"],
            model: "gpt-5",
            reasoning_effort: "medium",
            reasoning_efforts: ["medium", "high"],
            supports_fast: true,
          },
        },
      },
    });
    const onClose = vi.fn();

    root = document.createElement("div");
    document.body.appendChild(root);
    await act(async () => {
      render(
        <AppProviders sessionsStore={sessionsStore as any}>
          <NewSessionDialog open onClose={onClose} />
        </AppProviders>,
        root!,
      );
    });
    await flush();

    expect(root.querySelector('[data-testid="new-session-dialog"]')).not.toBeNull();
    expect(root.querySelector('[data-testid="backend-tab-codex"]')).not.toBeNull();
    expect(root.querySelector('input[name="cwd"]')).not.toBeNull();
    expect(root.querySelector('button[type="submit"]')).not.toBeNull();
    expect(root.textContent).toContain("Working directory");
    expect(root.textContent).toContain("Session name");
    expect(root.textContent).toContain("Model");
    expect(root.textContent).toContain("Provider");
    expect(root.textContent).toContain("Reasoning effort");
    expect(root.textContent).toContain("Resume conversation");
    expect(root.textContent).toContain("Launch mode");
    expect(root.textContent).toContain("Git worktree branch");
    expect(root.textContent).toContain("Speed");

    const cwdInput = root.querySelector('input[placeholder="/path/to/project"]') as HTMLInputElement;
    await setInputValue(cwdInput, "/tmp/project");
    await flush();

    const nameInput = root.querySelector('input[name="sessionName"]') as HTMLInputElement;
    await setInputValue(nameInput, "Inbox cleanup");
    await flush();

    const modelInput = root.querySelector('input[name="model"]') as HTMLInputElement;
    await setInputValue(modelInput, "gpt-5.4");
    await flush();

    const providerSelect = root.querySelector('select[name="providerChoice"]') as HTMLSelectElement;
    await setSelectValue(providerSelect, "openai-api");

    const reasoningSelect = root.querySelector('select[name="reasoningEffort"]') as HTMLSelectElement;
    await setSelectValue(reasoningSelect, "high");
    await flush();

    const fastCheckbox = root.querySelector('input[name="fastMode"]') as HTMLInputElement;
    const tmuxCheckbox = root.querySelector('input[name="createInTmux"]') as HTMLInputElement;
    const worktreeCheckbox = root.querySelector('input[name="useWorktree"]') as HTMLInputElement;
    await setCheckboxValue(fastCheckbox, true);
    await setCheckboxValue(tmuxCheckbox, true);
    await setCheckboxValue(worktreeCheckbox, true);
    await flush();

    const worktreeInput = root.querySelector('input[name="worktreeBranch"]') as HTMLInputElement;
    await setInputValue(worktreeInput, "feature/inbox-cleanup");
    await flush();

    const form = root.querySelector("form") as HTMLFormElement;
    await submitForm(form);
    await flush();

    expect(api.createSession).toHaveBeenCalledWith({
      cwd: "/tmp/project",
      backend: "codex",
      create_in_tmux: true,
      model: "gpt-5.4",
      reasoning_effort: "high",
      model_provider: "openai",
      preferred_auth_method: "apikey",
      resume_session_id: undefined,
      service_tier: "fast",
      worktree_branch: "feature/inbox-cleanup",
    });
    expect(api.renameSession).toHaveBeenCalledWith("new", "Inbox cleanup");
    expect(sessionsStore.select).toHaveBeenCalledWith("new");
    expect(onClose).toHaveBeenCalled();
  });

  it("prefills the working directory from the active session", async () => {
    const sessionsStore = createSessionsStore({
      items: [
        { session_id: "active", cwd: "/Users/demo/current-project" },
        { session_id: "other", cwd: "/Users/demo/other-project" },
      ],
      activeSessionId: "active",
      loading: false,
      bootstrapLoaded: true,
      recentCwds: ["/tmp/project"],
      tmuxAvailable: false,
      newSessionDefaults: {
        default_backend: "codex",
        backends: {
          codex: { provider_choice: "chatgpt" },
          pi: {},
        },
      },
    });

    root = document.createElement("div");
    document.body.appendChild(root);
    await act(async () => {
      render(
        <AppProviders sessionsStore={sessionsStore as any}>
          <NewSessionDialog open onClose={() => undefined} />
        </AppProviders>,
        root!,
      );
    });
    await flush();

    const cwdInput = root.querySelector('input[name="cwd"]') as HTMLInputElement;
    expect(cwdInput.value).toBe("/Users/demo/current-project");
  });

  it("still selects and closes when rename fails after launch", async () => {
    const { api } = await import("../../lib/api");
    vi.mocked(api.createSession).mockResolvedValue({ session_id: "new", broker_pid: 42, backend: "codex", ok: true } as any);
    vi.mocked(api.getSessionResumeCandidates).mockResolvedValue({
      exists: true,
      will_create: false,
      git_repo: false,
      sessions: [],
    } as any);
    vi.mocked(api.renameSession).mockRejectedValue(new Error("rename failed"));
    const sessionsStore = createSessionsStore({
      items: [
        { session_id: "old" },
        { session_id: "new" },
      ],
      activeSessionId: "old",
      loading: false,
      bootstrapLoaded: true,
      recentCwds: ["/tmp/project"],
      tmuxAvailable: false,
      newSessionDefaults: {
        default_backend: "codex",
        backends: {
          codex: { provider_choice: "chatgpt" },
          pi: {},
        },
      },
    });
    const onClose = vi.fn();

    root = document.createElement("div");
    document.body.appendChild(root);
    await act(async () => {
      render(
        <AppProviders sessionsStore={sessionsStore as any}>
          <NewSessionDialog open onClose={onClose} />
        </AppProviders>,
        root!,
      );
    });
    await flush();

    const input = root.querySelector('input[name="cwd"]') as HTMLInputElement;
    await setInputValue(input, "/tmp/project");
    const sessionNameInput = root.querySelector('input[name="sessionName"]') as HTMLInputElement;
    await setInputValue(sessionNameInput, "Inbox cleanup");
    await flush();

    const form = root.querySelector("form") as HTMLFormElement;
    await submitForm(form);
    await flush();

    expect(api.createSession).toHaveBeenCalled();
    expect(api.renameSession).toHaveBeenCalledWith("new", "Inbox cleanup");
    expect(sessionsStore.select).toHaveBeenCalledWith("new");
    expect(onClose).toHaveBeenCalled();
    expect(root.textContent).not.toContain("rename failed");
  });

  it("ignores duplicate submit events while launch is already in progress", async () => {
    const { api } = await import("../../lib/api");
    let resolveCreate: (value: unknown) => void = () => undefined;
    const pendingCreate = new Promise((resolve) => {
      resolveCreate = resolve;
    });
    vi.mocked(api.createSession).mockReturnValueOnce(pendingCreate as any);
    vi.mocked(api.getSessionResumeCandidates).mockResolvedValue({
      exists: true,
      will_create: false,
      git_repo: false,
      sessions: [],
    } as any);
    const sessionsStore = createSessionsStore({
      items: [{ session_id: "pending" }],
      activeSessionId: "pending",
      loading: false,
      bootstrapLoaded: true,
      recentCwds: ["/tmp/project"],
      tmuxAvailable: false,
      newSessionDefaults: {
        default_backend: "codex",
        backends: {
          codex: { provider_choice: "chatgpt" },
          pi: {},
        },
      },
    });

    root = document.createElement("div");
    document.body.appendChild(root);
    await act(async () => {
      render(
        <AppProviders sessionsStore={sessionsStore as any}>
          <NewSessionDialog open onClose={() => undefined} />
        </AppProviders>,
        root!,
      );
    });
    await flush();

    const input = root.querySelector('input[name="cwd"]') as HTMLInputElement;
    await setInputValue(input, "/tmp/project");
    await flush();

    const form = root.querySelector("form") as HTMLFormElement;
    await act(async () => {
      form.requestSubmit();
      form.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
      await Promise.resolve();
    });

    expect(api.createSession).toHaveBeenCalledTimes(1);

    resolveCreate({ session_id: "pending-new", broker_pid: 12, backend: "codex", ok: true });
    await flush();
  });

  it("hydrates late-arriving backend defaults for the selected backend without overwriting later user edits", async () => {
    const sessionsStore = createSessionsStore({
      items: [],
      activeSessionId: null,
      loading: false,
      recentCwds: [],
      tmuxAvailable: true,
      newSessionDefaults: null,
    });

    root = document.createElement("div");
    document.body.appendChild(root);
    await act(async () => {
      render(
        <AppProviders sessionsStore={sessionsStore as any}>
          <NewSessionDialog open onClose={() => undefined} />
        </AppProviders>,
        root!,
      );
    });
    await flush();

    const piTab = root.querySelector('[data-testid="backend-tab-pi"]') as HTMLButtonElement;
    await act(async () => {
      piTab.click();
      await Promise.resolve();
    });

    act(() => {
      sessionsStore.setState({
        ...sessionsStore.getState(),
        newSessionDefaults: {
          default_backend: "codex",
          backends: {
            pi: { provider_choice: "macaron", model: "gpt-5.4", reasoning_effort: "high" },
            codex: { provider_choice: "chatgpt", model: "gpt-5", reasoning_effort: "medium", supports_fast: true },
          },
        },
      });
    });
    await flush();

    expect((root.querySelector('input[name="model"]') as HTMLInputElement).value).toBe("gpt-5.4");
    expect((root.querySelector('select[name="providerChoice"]') as HTMLSelectElement).value).toBe("macaron");
    expect((root.querySelector('select[name="reasoningEffort"]') as HTMLSelectElement).value).toBe("high");

    const modelInput = root.querySelector('input[name="model"]') as HTMLInputElement;
    await setInputValue(modelInput, "custom-model");
    await flush();

    act(() => {
      sessionsStore.setState({
        ...sessionsStore.getState(),
        newSessionDefaults: {
          default_backend: "pi",
          backends: {
            pi: { provider_choice: "other", model: "replacement", reasoning_effort: "low" },
            codex: { provider_choice: "chatgpt" },
          },
        },
      });
    });
    await flush();

    expect((root.querySelector('input[name="model"]') as HTMLInputElement).value).toBe("custom-model");
  });

  it("falls back to preferNewest when no broker pid match is found", async () => {
    const { api } = await import("../../lib/api");
    vi.mocked(api.createSession).mockResolvedValue({ session_id: "newest", broker_pid: 99, backend: "codex", ok: true } as any);
    vi.mocked(api.getSessionResumeCandidates).mockResolvedValue({
      exists: true,
      will_create: false,
      git_repo: false,
      sessions: [],
    } as any);
    const sessionsStore = createSessionsStore({
      items: [
        { session_id: "newest" },
        { session_id: "older" },
      ],
      activeSessionId: "older",
      loading: false,
      recentCwds: [],
      tmuxAvailable: false,
      newSessionDefaults: {
        default_backend: "codex",
        backends: {
          codex: { provider_choice: "chatgpt" },
          pi: {},
        },
      },
    });

    root = document.createElement("div");
    document.body.appendChild(root);
    await act(async () => {
      render(
        <AppProviders sessionsStore={sessionsStore as any}>
          <NewSessionDialog open onClose={() => undefined} />
        </AppProviders>,
        root!,
      );
    });
    await flush();

    const input = root.querySelector('input[placeholder="/path/to/project"]') as HTMLInputElement;
    await setInputValue(input, "/tmp/project");
    await flush();
    await flush();
    await flush();

    const form = root.querySelector("form") as HTMLFormElement;
    await submitForm(form);
    await flush();
    await flush();

    expect(api.createSession).toHaveBeenCalled();
    expect(sessionsStore.getState().activeSessionId).toBe("newest");
  });

  it("updates Pi model suggestions when the provider changes", async () => {
    const sessionsStore = createSessionsStore({
      items: [],
      activeSessionId: null,
      loading: false,
      recentCwds: [],
      tmuxAvailable: false,
      newSessionDefaults: {
        default_backend: "pi",
        backends: {
          pi: {
            provider_choice: "macaron",
            provider_choices: ["macaron", "anthropic"],
            model: "gpt-5.4",
            models: ["gpt-5.4", "gpt-5.3-codex"],
            provider_models: {
              macaron: ["gpt-5.4", "gpt-5.3-codex"],
              anthropic: ["claude-sonnet-4-6", "claude-opus-4-6"],
            },
            reasoning_effort: "high",
            reasoning_efforts: ["medium", "high"],
          } as any,
          codex: { provider_choice: "chatgpt" },
        },
      },
    });

    root = document.createElement("div");
    document.body.appendChild(root);
    await act(async () => {
      render(
        <AppProviders sessionsStore={sessionsStore as any}>
          <NewSessionDialog open onClose={() => undefined} />
        </AppProviders>,
        root!,
      );
    });
    await flush();

    const providerSelect = root.querySelector('select[name="providerChoice"]') as HTMLSelectElement;
    expect(Array.from(root.querySelectorAll('#new-session-models option')).map((option) => option.getAttribute("value"))).toEqual([
      "gpt-5.4",
      "gpt-5.3-codex",
    ]);

    await setSelectValue(providerSelect, "anthropic");
    await flush();

    expect(Array.from(root.querySelectorAll('#new-session-models option')).map((option) => option.getAttribute("value"))).toEqual([
      "claude-sonnet-4-6",
      "claude-opus-4-6",
    ]);
  });

  it("only replaces the Pi model value on provider change when the model input is untouched", async () => {
    const sessionsStore = createSessionsStore({
      items: [],
      activeSessionId: null,
      loading: false,
      recentCwds: [],
      tmuxAvailable: false,
      newSessionDefaults: {
        default_backend: "pi",
        backends: {
          pi: {
            provider_choice: "macaron",
            provider_choices: ["macaron", "anthropic"],
            model: "gpt-5.4",
            models: ["gpt-5.4", "gpt-5.3-codex"],
            provider_models: {
              macaron: ["gpt-5.4", "gpt-5.3-codex"],
              anthropic: ["claude-sonnet-4-6", "claude-opus-4-6"],
            },
            reasoning_effort: "high",
            reasoning_efforts: ["medium", "high"],
          } as any,
          codex: { provider_choice: "chatgpt" },
        },
      },
    });

    root = document.createElement("div");
    document.body.appendChild(root);
    await act(async () => {
      render(
        <AppProviders sessionsStore={sessionsStore as any}>
          <NewSessionDialog open onClose={() => undefined} />
        </AppProviders>,
        root!,
      );
    });
    await flush();

    const modelInput = root.querySelector('input[name="model"]') as HTMLInputElement;
    const providerSelect = root.querySelector('select[name="providerChoice"]') as HTMLSelectElement;

    expect(modelInput.value).toBe("gpt-5.4");
    await setSelectValue(providerSelect, "anthropic");
    await flush();
    expect(modelInput.value).toBe("claude-sonnet-4-6");

    await setInputValue(modelInput, "custom-model");
    await setSelectValue(providerSelect, "macaron");
    await flush();
    expect(modelInput.value).toBe("custom-model");
  });

  it("keeps Pi provider changes auto-updating the model until the user edits it after switching backends", async () => {
    const sessionsStore = createSessionsStore({
      items: [],
      activeSessionId: null,
      loading: false,
      recentCwds: [],
      tmuxAvailable: false,
      newSessionDefaults: {
        default_backend: "codex",
        backends: {
          codex: { provider_choice: "chatgpt", model: "gpt-5" },
          pi: {
            provider_choice: "macaron",
            provider_choices: ["macaron", "anthropic"],
            model: "gpt-5.4",
            models: ["gpt-5.4", "gpt-5.3-codex"],
            provider_models: {
              macaron: ["gpt-5.4", "gpt-5.3-codex"],
              anthropic: ["claude-sonnet-4-6", "claude-opus-4-6"],
            },
            reasoning_effort: "high",
            reasoning_efforts: ["medium", "high"],
          } as any,
        },
      },
    });

    root = document.createElement("div");
    document.body.appendChild(root);
    await act(async () => {
      render(
        <AppProviders sessionsStore={sessionsStore as any}>
          <NewSessionDialog open onClose={() => undefined} />
        </AppProviders>,
        root!,
      );
    });
    await flush();

    const piTab = root.querySelector('[data-testid="backend-tab-pi"]') as HTMLButtonElement;
    await act(async () => {
      piTab.click();
      await Promise.resolve();
    });
    await flush();

    const modelInput = root.querySelector('input[name="model"]') as HTMLInputElement;
    const providerSelect = root.querySelector('select[name="providerChoice"]') as HTMLSelectElement;

    expect(modelInput.value).toBe("gpt-5.4");
    await setSelectValue(providerSelect, "anthropic");
    await flush();
    expect(modelInput.value).toBe("claude-sonnet-4-6");

    await setInputValue(modelInput, "custom-model");
    await setSelectValue(providerSelect, "macaron");
    await flush();
    expect(modelInput.value).toBe("custom-model");
  });

  it("keeps the working directory when session defaults refresh while the dialog is open", async () => {
    const sessionsStore = createSessionsStore({
      items: [],
      activeSessionId: null,
      loading: false,
      recentCwds: ["/tmp/project", "/tmp/other"],
      tmuxAvailable: true,
      newSessionDefaults: {
        default_backend: "codex",
        backends: {
          codex: {
            provider_choice: "chatgpt",
            provider_choices: ["chatgpt", "openai-api"],
            model: "gpt-5",
            reasoning_effort: "medium",
            reasoning_efforts: ["medium", "high"],
            supports_fast: true,
          },
          pi: { provider_choice: "macaron" },
        },
      },
    });

    root = document.createElement("div");
    document.body.appendChild(root);
    await act(async () => {
      render(
        <AppProviders sessionsStore={sessionsStore as any}>
          <NewSessionDialog open onClose={() => undefined} />
        </AppProviders>,
        root!,
      );
    });
    await flush();

    const cwdInput = root.querySelector('input[placeholder="/path/to/project"]') as HTMLInputElement;
    await setInputValue(cwdInput, "/tmp/project");
    await flush();

    await act(async () => {
      sessionsStore.setState({
        ...sessionsStore.getState(),
        recentCwds: ["/tmp/project", "/tmp/other", "/tmp/new"],
        newSessionDefaults: {
          default_backend: "codex",
          backends: {
            codex: {
              provider_choice: "chatgpt",
              provider_choices: ["chatgpt", "openai-api"],
              model: "gpt-5",
              reasoning_effort: "medium",
              reasoning_efforts: ["medium", "high"],
              supports_fast: true,
            },
            pi: { provider_choice: "macaron" },
          },
        },
      });
    });
    await flush();

    expect(cwdInput.value).toBe("/tmp/project");
  });

  it("lets Pi sessions launch in tmux and explains the pi-rpc host split", async () => {
    const { api } = await import("../../lib/api");
    vi.mocked(api.createSession).mockResolvedValue({ session_id: "pi-new", broker_pid: 84, backend: "pi", ok: true } as any);
    vi.mocked(api.getSessionResumeCandidates).mockResolvedValue({
      exists: true,
      will_create: false,
      git_repo: false,
      sessions: [],
    } as any);
    vi.mocked(api.renameSession).mockResolvedValue({ ok: true } as any);

    const sessionsStore = createSessionsStore({
      items: [{ session_id: "old" }, { session_id: "pi-new" }],
      activeSessionId: "old",
      loading: false,
      bootstrapLoaded: true,
      recentCwds: ["/tmp/pi-project"],
      tmuxAvailable: true,
      newSessionDefaults: {
        default_backend: "pi",
        backends: {
          pi: {
            provider_choice: "macaron",
            provider_choices: ["macaron", "anthropic"],
            model: "gpt-5.4",
            reasoning_effort: "high",
            reasoning_efforts: ["medium", "high"],
          },
          codex: { provider_choice: "chatgpt" },
        },
      },
    });

    root = document.createElement("div");
    document.body.appendChild(root);
    await act(async () => {
      render(
        <AppProviders sessionsStore={sessionsStore as any}>
          <NewSessionDialog open onClose={() => undefined} />
        </AppProviders>,
        root!,
      );
    });
    await flush();

    const tmuxCheckbox = root.querySelector('input[name="createInTmux"]') as HTMLInputElement;
    expect(tmuxCheckbox.disabled).toBe(false);
    expect(root.textContent).toContain("Host the new Pi session in tmux while pi-rpc handles web control.");

    await setCheckboxValue(tmuxCheckbox, true);
    await flush();

    const form = root.querySelector("form") as HTMLFormElement;
    await submitForm(form);
    await flush();

    expect(api.createSession).toHaveBeenCalledWith({
      cwd: "/tmp/pi-project",
      backend: "pi",
      create_in_tmux: true,
      model: "gpt-5.4",
      reasoning_effort: "high",
      model_provider: "macaron",
      resume_session_id: undefined,
      worktree_branch: undefined,
    });
  });
});
