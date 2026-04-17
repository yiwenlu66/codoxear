import { render } from "preact";
import { act } from "preact/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AppProviders } from "../../app/providers";
import { ConversationPane } from "./ConversationPane";

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

function createMutableStore<TState extends object, TActions extends Record<string, (...args: any[]) => any>>(
  initialState: TState,
  actions: (getState: () => TState, setState: (next: TState) => void) => TActions,
) {
  let state = initialState;
  const listeners = new Set<() => void>();

  const setState = (next: TState) => {
    state = next;
    listeners.forEach((listener) => listener());
  };

  return {
    getState: () => state,
    subscribe(listener: () => void) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    setState,
    ...actions(() => state, setState),
  };
}

describe("ConversationPane", () => {
  let root: HTMLDivElement | null = null;

  afterEach(() => {
    if (root) {
      render(null, root);
      root.remove();
      root = null;
    }
  });

  it("renders user, assistant, and ask_user events with card surfaces", () => {
    const sessionsStore = createStaticStore(
      { items: [], activeSessionId: "sess-1", loading: false, newSessionDefaults: null },
      { refresh: () => Promise.resolve(), select: () => undefined },
    );
    const messagesStore = createStaticStore(
      {
        bySessionId: {
          "sess-1": [
            { role: "user", text: "Hello" },
            { type: "ask_user", question: "Choose a provider", answer: "openai", resolved: true },
            { type: "ask_user", question: "Need anything else?", cancelled: true, resolved: true },
            { role: "assistant", text: "All done." },
          ],
        },
        offsetsBySessionId: { "sess-1": 3 },
        loading: false,
      },
      { loadInitial: () => Promise.resolve(), poll: () => Promise.resolve() },
    );

    root = document.createElement("div");
    document.body.appendChild(root);
    render(
      <AppProviders sessionsStore={sessionsStore as any} messagesStore={messagesStore as any}>
        <ConversationPane />
      </AppProviders>,
      root!,
    );

    const text = root.textContent || "";
    expect(root.querySelectorAll("[data-testid='message-surface']").length).toBeGreaterThanOrEqual(4);
    const userSurface = root.querySelector("[data-testid='message-surface'][data-kind='user']") as HTMLElement | null;
    expect(userSurface).not.toBeNull();
    expect(userSurface?.className).toContain("text-foreground");
    expect(userSurface?.className).not.toContain("text-primary-foreground");
    expect(root.querySelector("[data-testid='message-surface'][data-kind='assistant']")).not.toBeNull();
    expect(root.querySelector("[data-testid='message-surface'][data-kind='ask_user']")).not.toBeNull();
    expect(text).toContain("Hello");
    expect(text).toContain("Choose a provider");
    expect(text).toContain("Answer: openai");
    expect(text).toContain("Need anything else?");
    expect(text).toContain("Cancelled");
    expect(text).toContain("All done.");
  });

  it("renders legacy-visible activity events in the main conversation flow", () => {
    const sessionsStore = createStaticStore(
      { items: [], activeSessionId: "sess-2", loading: false, newSessionDefaults: null },
      { refresh: () => Promise.resolve(), select: () => undefined },
    );
    const messagesStore = createStaticStore(
      {
        bySessionId: {
          "sess-2": [
            { role: "user", text: "Please fix this" },
            { type: "reasoning", text: "thinking...", summary: "Inspecting current state" },
            { type: "tool", name: "read" },
            { type: "tool_result", name: "read", text: "{\"ok\":true}" },
            { type: "subagent", agent: "reviewer", task: "Check the patch", output: "Looks good" },
            {
              type: "todo_snapshot",
              progress_text: "2/3 completed",
              items: [
                { title: "Explore project context", status: "completed" },
                { title: "Render message types", status: "in-progress" },
              ],
            },
            { type: "pi_model_change", summary: "Switched to gpt-5.4" },
            { role: "assistant", text: "Fixed." },
          ],
        },
        offsetsBySessionId: { "sess-2": 7 },
        loading: false,
      },
      { loadInitial: () => Promise.resolve(), poll: () => Promise.resolve() },
    );

    root = document.createElement("div");
    document.body.appendChild(root);
    render(
      <AppProviders sessionsStore={sessionsStore as any} messagesStore={messagesStore as any}>
        <ConversationPane />
      </AppProviders>,
      root,
    );

    const text = root.textContent || "";
    expect(root.querySelectorAll(".messageRow")).toHaveLength(8);
    expect(root.querySelector("[data-testid='message-surface'][data-kind='reasoning']")).not.toBeNull();
    expect(root.querySelector("[data-testid='message-surface'][data-kind='tool']")).not.toBeNull();
    expect(root.querySelector("[data-testid='message-surface'][data-kind='tool_result']")).not.toBeNull();
    expect(root.querySelector("[data-testid='message-surface'][data-kind='subagent']")).not.toBeNull();
    expect(root.querySelector("[data-testid='message-surface'][data-kind='todo_snapshot']")).not.toBeNull();
    expect(root.querySelector("[data-testid='message-surface'][data-kind='pi_model_change']")).not.toBeNull();
    expect(text).toContain("Please fix this");
    expect(text).toContain("thinking...");
    expect(text).toContain("Inspecting current state");
    expect(text).toContain("read");
    expect(text).toContain('{"ok":true}');
    expect(text).toContain("reviewer");
    expect(text).toContain("Check the patch");
    expect(text).toContain("Looks good");
    expect(text).toContain("2/3 completed");
    expect(text).toContain("Switched to gpt-5.4");
    expect(text).toContain("Fixed.");
  });

  it("renders Claude Todo V2 task-assignment custom messages as dedicated timeline events", () => {
    const sessionsStore = createStaticStore(
      { items: [], activeSessionId: "sess-custom", loading: false, newSessionDefaults: null },
      { refresh: () => Promise.resolve(), select: () => undefined },
    );
    const messagesStore = createStaticStore(
      {
        bySessionId: {
          "sess-custom": [
            {
              type: "custom_message",
              custom_type: "claude-todo-v2-task-assignment",
              text: "Task #1 assigned to @Codex",
              subject: "Clarify Claude Todo V2 compatibility goal",
              description: "Ask the user a focused requirement question.",
              owner: "Codex",
              assigned_by: "team-lead",
              ts: 100,
            },
          ],
        },
        offsetsBySessionId: { "sess-custom": 1 },
        loading: false,
      },
      { loadInitial: () => Promise.resolve(), poll: () => Promise.resolve() },
    );

    root = document.createElement("div");
    document.body.appendChild(root);
    render(
      <AppProviders sessionsStore={sessionsStore as any} messagesStore={messagesStore as any}>
        <ConversationPane />
      </AppProviders>,
      root,
    );

    expect(root.querySelector("[data-testid='message-surface'][data-kind='custom_message']")).not.toBeNull();
    expect(root.textContent).toContain("Task #1 assigned to @Codex");
    expect(root.textContent).toContain("Clarify Claude Todo V2 compatibility goal");
    expect(root.textContent).toContain("Codex");
    expect(root.textContent).toContain("team-lead");
    expect(root.textContent).toContain("Ask the user a focused requirement question.");
  });

  it("renders unknown custom messages through a safe fallback card", () => {
    const sessionsStore = createStaticStore(
      { items: [], activeSessionId: "sess-custom-fallback", loading: false, newSessionDefaults: null },
      { refresh: () => Promise.resolve(), select: () => undefined },
    );
    const messagesStore = createStaticStore(
      {
        bySessionId: {
          "sess-custom-fallback": [
            {
              type: "custom_message",
              custom_type: "claude-todo-v2-task-note",
              text: "Task note added",
              ts: 200,
            },
          ],
        },
        offsetsBySessionId: { "sess-custom-fallback": 1 },
        loading: false,
      },
      { loadInitial: () => Promise.resolve(), poll: () => Promise.resolve() },
    );

    root = document.createElement("div");
    document.body.appendChild(root);
    render(
      <AppProviders sessionsStore={sessionsStore as any} messagesStore={messagesStore as any}>
        <ConversationPane />
      </AppProviders>,
      root,
    );

    expect(root.querySelector("[data-testid='message-surface'][data-kind='custom_message']")).not.toBeNull();
    expect(root.textContent).toContain("Task note added");
  });

  it("groups consecutive assistant messages and avoids showing role labels as body text", () => {
    const sessionsStore = createStaticStore(
      { items: [], activeSessionId: "sess-3", loading: false, newSessionDefaults: null },
      { refresh: () => Promise.resolve(), select: () => undefined },
    );
    const messagesStore = createStaticStore(
      {
        bySessionId: {
          "sess-3": [
            { role: "assistant", text: "First answer" },
            { role: "assistant", text: "Second answer" },
            { role: "user", text: "Thanks" },
          ],
        },
        offsetsBySessionId: { "sess-3": 3 },
        loading: false,
      },
      { loadInitial: () => Promise.resolve(), poll: () => Promise.resolve() },
    );

    root = document.createElement("div");
    document.body.appendChild(root);
    render(
      <AppProviders sessionsStore={sessionsStore as any} messagesStore={messagesStore as any}>
        <ConversationPane />
      </AppProviders>,
      root,
    );

    expect(root.querySelectorAll(".messageRow")).toHaveLength(3);
    expect(root.querySelectorAll(".messageRow.grouped")).toHaveLength(1);
    expect(root.textContent).not.toContain("assistantassistant");
  });

  it("renders markdown code blocks and inline code inside assistant messages", () => {
    const sessionsStore = createStaticStore(
      { items: [], activeSessionId: "sess-4", loading: false, newSessionDefaults: null },
      { refresh: () => Promise.resolve(), select: () => undefined },
    );
    const messagesStore = createStaticStore(
      {
        bySessionId: {
          "sess-4": [
            {
              role: "assistant",
              text: "Run `npm test` before `<div>` output.\n\n```ts\nconst answer = 42;\n```",
            },
          ],
        },
        offsetsBySessionId: { "sess-4": 1 },
        loading: false,
      },
      { loadInitial: () => Promise.resolve(), poll: () => Promise.resolve() },
    );

    root = document.createElement("div");
    document.body.appendChild(root);
    render(
      <AppProviders sessionsStore={sessionsStore as any} messagesStore={messagesStore as any}>
        <ConversationPane />
      </AppProviders>,
      root,
    );

    const inlineCode = Array.from(root.querySelectorAll(".messageBody p code")).map((node) => node.textContent || "");
    const blockCode = root.querySelector(".messageBody pre code");
    expect(inlineCode).toContain("npm test");
    expect(inlineCode).toContain("<div>");
    expect(blockCode?.textContent).toContain("const answer = 42;");
  });

  it("renders markdown unordered and ordered lists with explicit list styling", () => {
    const sessionsStore = createStaticStore(
      { items: [], activeSessionId: "sess-4-list", loading: false, newSessionDefaults: null },
      { refresh: () => Promise.resolve(), select: () => undefined },
    );
    const messagesStore = createStaticStore(
      {
        bySessionId: {
          "sess-4-list": [
            {
              role: "assistant",
              text: "- first\n- second\n\n1. alpha\n2. beta",
            },
          ],
        },
        offsetsBySessionId: { "sess-4-list": 1 },
        loading: false,
      },
      { loadInitial: () => Promise.resolve(), poll: () => Promise.resolve() },
    );

    root = document.createElement("div");
    document.body.appendChild(root);
    render(
      <AppProviders sessionsStore={sessionsStore as any} messagesStore={messagesStore as any}>
        <ConversationPane />
      </AppProviders>,
      root,
    );

    const unordered = root.querySelector(".messageBody ul") as HTMLUListElement | null;
    const ordered = root.querySelector(".messageBody ol") as HTMLOListElement | null;
    expect(unordered).not.toBeNull();
    expect(ordered).not.toBeNull();
    expect(unordered?.className).toContain("list-disc");
    expect(unordered?.textContent).toContain("first");
    expect(unordered?.textContent).toContain("second");
    expect(ordered?.className).toContain("list-decimal");
    expect(ordered?.textContent).toContain("alpha");
    expect(ordered?.textContent).toContain("beta");
  });

  it("copies plain chat bubble text from the copy button", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(globalThis.navigator, "clipboard", {
      configurable: true,
      value: { writeText },
    });

    const sessionsStore = createStaticStore(
      { items: [], activeSessionId: "sess-copy", loading: false, newSessionDefaults: null },
      { refresh: () => Promise.resolve(), select: () => undefined },
    );
    const messagesStore = createStaticStore(
      {
        bySessionId: {
          "sess-copy": [
            { role: "assistant", text: "Copy me please" },
          ],
        },
        offsetsBySessionId: { "sess-copy": 1 },
        loading: false,
      },
      { loadInitial: () => Promise.resolve(), poll: () => Promise.resolve() },
    );

    root = document.createElement("div");
    document.body.appendChild(root);
    render(
      <AppProviders sessionsStore={sessionsStore as any} messagesStore={messagesStore as any}>
        <ConversationPane />
      </AppProviders>,
      root,
    );

    const copyButton = root.querySelector("button[aria-label='Copy assistant message']") as HTMLButtonElement | null;
    expect(copyButton).not.toBeNull();

    await act(async () => {
      copyButton?.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
      await Promise.resolve();
    });

    expect(writeText).toHaveBeenCalledWith("Copy me please");
    expect(copyButton?.textContent?.trim()).toBe("");
    expect(copyButton?.className).toContain("isCopied");
  });

  it("only binds the newest duplicate ask_user card to a matching live request", () => {
    const sessionsStore = createStaticStore(
      { items: [], activeSessionId: "sess-dup", loading: false, newSessionDefaults: null },
      { refresh: () => Promise.resolve(), select: () => undefined },
    );
    const messagesStore = createStaticStore(
      {
        bySessionId: {
          "sess-dup": [
            {
              type: "ask_user",
              tool_call_id: "historic-old",
              question: "Choose a provider",
              context: "Pick one option.",
              options: ["OpenAI", "Anthropic"],
              allow_freeform: false,
              allow_multiple: false,
              resolved: false,
              ts: 100,
            },
            {
              type: "ask_user",
              tool_call_id: "historic-new",
              question: "Choose a provider",
              context: "Pick one option.",
              options: ["OpenAI", "Anthropic"],
              allow_freeform: false,
              allow_multiple: false,
              resolved: false,
              ts: 200,
            },
          ],
        },
        offsetsBySessionId: { "sess-dup": 2 },
        loading: false,
      },
      { loadInitial: () => Promise.resolve(), poll: () => Promise.resolve() },
    );
    const sessionUiStore = createStaticStore(
      {
        sessionId: "sess-dup",
        diagnostics: null,
        queue: null,
        files: [],
        loading: false,
        requests: [
          {
            id: "ui-live-1",
            method: "select",
            question: "Choose a provider",
            context: "Pick one option.",
            options: ["OpenAI", "Anthropic"],
            allow_freeform: false,
            allow_multiple: false,
          },
        ],
      },
      { refresh: () => Promise.resolve() },
    );

    root = document.createElement("div");
    document.body.appendChild(root);
    render(
      <AppProviders sessionsStore={sessionsStore as any} messagesStore={messagesStore as any} sessionUiStore={sessionUiStore as any}>
        <ConversationPane />
      </AppProviders>,
      root,
    );

    const openAiButtons = Array.from(root.querySelectorAll("button")).filter((button) => button.textContent?.includes("OpenAI")) as HTMLButtonElement[];
    expect(openAiButtons).toHaveLength(2);
    expect(openAiButtons[0].disabled).toBe(true);
    expect(openAiButtons[1].disabled).toBe(false);
  });

  it("enables unresolved historical ask_user cards for legacy pi sessions without live ui transport", () => {
    const sessionsStore = createStaticStore(
      {
        items: [{ session_id: "sess-legacy", alias: "Legacy Pi", agent_backend: "pi", transport: null }],
        activeSessionId: "sess-legacy",
        loading: false,
        newSessionDefaults: null,
      },
      { refresh: () => Promise.resolve(), select: () => undefined },
    );
    const messagesStore = createStaticStore(
      {
        bySessionId: {
          "sess-legacy": [
            {
              type: "ask_user",
              tool_call_id: "legacy-ask-1",
              question: "Choose a provider",
              context: "This prompt should still be answerable.",
              options: ["OpenAI", "Anthropic"],
              allow_freeform: true,
              allow_multiple: false,
              resolved: false,
            },
          ],
        },
        offsetsBySessionId: { "sess-legacy": 1 },
        loading: false,
      },
      { loadInitial: () => Promise.resolve(), poll: () => Promise.resolve() },
    );
    const sessionUiStore = createStaticStore(
      {
        sessionId: "sess-legacy",
        diagnostics: null,
        queue: null,
        files: [],
        loading: false,
        requests: [],
      },
      { refresh: () => Promise.resolve() },
    );

    root = document.createElement("div");
    document.body.appendChild(root);
    render(
      <AppProviders sessionsStore={sessionsStore as any} messagesStore={messagesStore as any} sessionUiStore={sessionUiStore as any}>
        <ConversationPane />
      </AppProviders>,
      root,
    );

    const optionButton = Array.from(root.querySelectorAll("button")).find((button) => button.textContent?.includes("OpenAI")) as
      | HTMLButtonElement
      | undefined;
    expect(optionButton).toBeDefined();
    expect(optionButton?.disabled).toBe(false);
  });

  it("resets expandable message state when switching sessions", async () => {
    const longReasoningA = Array.from({ length: 10 }, (_value, index) => `session A reasoning ${index + 1}`).join("\n");
    const longReasoningB = Array.from({ length: 10 }, (_value, index) => `session B reasoning ${index + 1}`).join("\n");
    const messagesStore = createStaticStore(
      {
        bySessionId: {
          "sess-expand-a": [{ type: "reasoning", text: longReasoningA }],
          "sess-expand-b": [{ type: "reasoning", text: longReasoningB }],
        },
        offsetsBySessionId: { "sess-expand-a": 1, "sess-expand-b": 1 },
        loading: false,
      },
      { loadInitial: () => Promise.resolve(), poll: () => Promise.resolve() },
    );

    root = document.createElement("div");
    document.body.appendChild(root);
    render(
      <AppProviders
        sessionsStore={createStaticStore(
          { items: [], activeSessionId: "sess-expand-a", loading: false, newSessionDefaults: null },
          { refresh: () => Promise.resolve(), select: () => undefined },
        ) as any}
        messagesStore={messagesStore as any}
      >
        <ConversationPane />
      </AppProviders>,
      root,
    );

    const firstButton = root.querySelector("[data-testid='message-surface'][data-kind='reasoning'] .messageExpandButton") as HTMLButtonElement | null;
    const firstContent = root.querySelector("[data-testid='message-surface'][data-kind='reasoning'] .messageExpandableContent") as HTMLDivElement | null;

    expect(firstButton?.getAttribute("aria-expanded")).toBe("false");
    await act(async () => {
      firstButton?.click();
      await Promise.resolve();
    });
    expect(firstButton?.getAttribute("aria-expanded")).toBe("true");
    expect(firstContent?.classList.contains("isCollapsed")).toBe(false);

    await act(async () => {
      render(
        <AppProviders
          sessionsStore={createStaticStore(
            { items: [], activeSessionId: "sess-expand-b", loading: false, newSessionDefaults: null },
            { refresh: () => Promise.resolve(), select: () => undefined },
          ) as any}
          messagesStore={messagesStore as any}
        >
          <ConversationPane />
        </AppProviders>,
        root!,
      );
      await Promise.resolve();
    });

    const nextButton = root.querySelector("[data-testid='message-surface'][data-kind='reasoning'] .messageExpandButton") as HTMLButtonElement | null;
    const nextContent = root.querySelector("[data-testid='message-surface'][data-kind='reasoning'] .messageExpandableContent") as HTMLDivElement | null;

    expect(nextButton?.getAttribute("aria-expanded")).toBe("false");
    expect(nextContent?.classList.contains("isCollapsed")).toBe(true);
    expect(root.textContent).toContain("session B reasoning 1");
  });

  it("collapses long reasoning and tool-result cards with expandable toggles", async () => {
    const sessionsStore = createStaticStore(
      { items: [], activeSessionId: "sess-6", loading: false, newSessionDefaults: null },
      { refresh: () => Promise.resolve(), select: () => undefined },
    );
    const longReasoning = Array.from({ length: 10 }, (_value, index) => `reasoning line ${index + 1}`).join("\n");
    const longToolResult = Array.from({ length: 12 }, (_value, index) => `result line ${index + 1}`).join("\n");
    const messagesStore = createStaticStore(
      {
        bySessionId: {
          "sess-6": [
            { type: "reasoning", text: longReasoning },
            { type: "tool_result", name: "bash", text: longToolResult },
          ],
        },
        offsetsBySessionId: { "sess-6": 2 },
        loading: false,
      },
      { loadInitial: () => Promise.resolve(), poll: () => Promise.resolve() },
    );

    root = document.createElement("div");
    document.body.appendChild(root);
    render(
      <AppProviders sessionsStore={sessionsStore as any} messagesStore={messagesStore as any}>
        <ConversationPane />
      </AppProviders>,
      root,
    );

    const reasoningButton = root.querySelector("[data-testid='message-surface'][data-kind='reasoning'] .messageExpandButton") as HTMLButtonElement | null;
    const reasoningContent = root.querySelector("[data-testid='message-surface'][data-kind='reasoning'] .messageExpandableContent") as HTMLDivElement | null;
    const toolButton = root.querySelector("[data-testid='message-surface'][data-kind='tool_result'] .messageToolToggle") as HTMLButtonElement | null;
    const toolContent = root.querySelector("[data-testid='message-surface'][data-kind='tool_result'] .messageToolDetails") as HTMLDivElement | null;

    expect(reasoningButton?.textContent).toBe("Show more");
    expect(reasoningButton?.getAttribute("aria-expanded")).toBe("false");
    expect(reasoningContent?.classList.contains("isCollapsed")).toBe(true);
    expect(toolButton?.textContent).toBe("Expand");
    expect(toolButton?.getAttribute("aria-expanded")).toBe("false");
    expect(toolContent).toBeNull();

    reasoningButton?.click();
    toolButton?.click();
    await Promise.resolve();

    expect(reasoningButton?.textContent).toBe("Show less");
    expect(reasoningButton?.getAttribute("aria-expanded")).toBe("true");
    expect(reasoningContent?.classList.contains("isCollapsed")).toBe(false);
    expect(toolButton?.textContent).toBe("Collapse");
    expect(toolButton?.getAttribute("aria-expanded")).toBe("true");
    expect(root.querySelector("[data-testid='message-surface'][data-kind='tool_result'] .messageToolDetails")).not.toBeNull();
  });

  it("shows tool and tool-result cards as one-line summaries until expanded", async () => {
    const sessionsStore = createStaticStore(
      { items: [], activeSessionId: "sess-tool-compact", loading: false, newSessionDefaults: null },
      { refresh: () => Promise.resolve(), select: () => undefined },
    );
    const messagesStore = createStaticStore(
      {
        bySessionId: {
          "sess-tool-compact": [
            { type: "tool", name: "read", text: "web/src/components/conversation/ConversationPane.tsx" },
            { type: "tool_result", name: "read", text: '{"ok":true,"path":"ConversationPane.tsx"}' },
          ],
        },
        offsetsBySessionId: { "sess-tool-compact": 2 },
        loading: false,
      },
      { loadInitial: () => Promise.resolve(), poll: () => Promise.resolve() },
    );

    root = document.createElement("div");
    document.body.appendChild(root);
    render(
      <AppProviders sessionsStore={sessionsStore as any} messagesStore={messagesStore as any}>
        <ConversationPane />
      </AppProviders>,
      root,
    );

    const toolSurface = root.querySelector("[data-testid='message-surface'][data-kind='tool']") as HTMLElement | null;
    const toolResultSurface = root.querySelector("[data-testid='message-surface'][data-kind='tool_result']") as HTMLElement | null;
    const toolToggle = toolSurface?.querySelector(".messageToolToggle") as HTMLButtonElement | null;
    const toolResultToggle = toolResultSurface?.querySelector(".messageToolToggle") as HTMLButtonElement | null;

    expect(toolSurface?.className).not.toContain("messageCard");
    expect(toolResultSurface?.className).not.toContain("messageCard");
    expect(toolSurface?.querySelector(".messageToolSummary")?.textContent).toContain("web/src/components/conversation/ConversationPane.tsx");
    expect(toolResultSurface?.querySelector(".messageToolSummary")?.textContent).toContain('{"ok":true,"path":"ConversationPane.tsx"}');
    expect(toolSurface?.querySelector(".messageToolRow")).not.toBeNull();
    expect(toolResultSurface?.querySelector(".messageToolRow")).not.toBeNull();
    expect(toolSurface?.querySelector(".messageToolDetails")).toBeNull();
    expect(toolResultSurface?.querySelector(".messageToolDetails")).toBeNull();
    expect(toolToggle?.getAttribute("aria-expanded")).toBe("false");
    expect(toolResultToggle?.getAttribute("aria-expanded")).toBe("false");

    await act(async () => {
      toolToggle?.click();
      toolResultToggle?.click();
      await Promise.resolve();
    });

    expect(toolSurface?.querySelector(".messageToolDetails .messageBody")?.textContent).toContain("web/src/components/conversation/ConversationPane.tsx");
    expect(toolResultSurface?.querySelector(".messageToolDetails .messageBody")?.textContent).toContain('{"ok":true,"path":"ConversationPane.tsx"}');
    expect(toolToggle?.getAttribute("aria-expanded")).toBe("true");
    expect(toolResultToggle?.getAttribute("aria-expanded")).toBe("true");
  });

  it("scrolls the conversation pane to the bottom on initial render when messages exist", async () => {
    const sessionsStore = createStaticStore(
      { items: [], activeSessionId: "sess-7", loading: false, newSessionDefaults: null },
      { refresh: () => Promise.resolve(), select: () => undefined },
    );
    const messagesStore = createStaticStore(
      {
        bySessionId: {
          "sess-7": [
            { role: "assistant", text: "One" },
            { role: "assistant", text: "Two" },
            { role: "assistant", text: "Three" },
          ],
        },
        offsetsBySessionId: { "sess-7": 3 },
        loading: false,
      },
      { loadInitial: () => Promise.resolve(), poll: () => Promise.resolve() },
    );

    const scrollTo = vi.fn();
    const originalScrollTo = HTMLElement.prototype.scrollTo;
    HTMLElement.prototype.scrollTo = scrollTo;
    const scrollHeightDescriptor = Object.getOwnPropertyDescriptor(HTMLElement.prototype, "scrollHeight");
    const clientHeightDescriptor = Object.getOwnPropertyDescriptor(HTMLElement.prototype, "clientHeight");
    Object.defineProperty(HTMLElement.prototype, "scrollHeight", {
      configurable: true,
      get() {
        return this.classList?.contains("conversationPane") ? 500 : (scrollHeightDescriptor?.get?.call(this) ?? 0);
      },
    });
    Object.defineProperty(HTMLElement.prototype, "clientHeight", {
      configurable: true,
      get() {
        return this.classList?.contains("conversationPane") ? 200 : (clientHeightDescriptor?.get?.call(this) ?? 0);
      },
    });

    root = document.createElement("div");
    document.body.appendChild(root);
    render(
      <AppProviders sessionsStore={sessionsStore as any} messagesStore={messagesStore as any}>
        <ConversationPane />
      </AppProviders>,
      root,
    );

    await Promise.resolve();
    await Promise.resolve();

    expect(scrollTo).toHaveBeenCalledWith({ top: 500 });
    HTMLElement.prototype.scrollTo = originalScrollTo;

    if (scrollHeightDescriptor) {
      Object.defineProperty(HTMLElement.prototype, "scrollHeight", scrollHeightDescriptor);
    }
    if (clientHeightDescriptor) {
      Object.defineProperty(HTMLElement.prototype, "clientHeight", clientHeightDescriptor);
    }
  });

  it("renders legacy timestamps plus markdown tables, local file refs, and local images", () => {
    const sessionsStore = createStaticStore(
      {
        items: [{ session_id: "sess-8", cwd: "/repo/docs", agent_backend: "pi" }],
        activeSessionId: "sess-8",
        loading: false,
        newSessionDefaults: null,
      },
      { refresh: () => Promise.resolve(), select: () => undefined },
    );
    const messagesStore = createStaticStore(
      {
        bySessionId: {
          "sess-8": [
            {
              role: "assistant",
              ts: 1_744_000_000,
              text: "[server.py](/repo/codoxear/server.py#L123)\n\n| Name | Urgency |\n|---|---:|\n| offline | 2 |\n\n![diagram](./flow.png)",
            },
            { role: "assistant", ts: 1_650_000, text: "synthetic timestamp should stay hidden" },
          ],
        },
        offsetsBySessionId: { "sess-8": 2 },
        hasOlderBySessionId: {},
        olderBeforeBySessionId: {},
        loadingOlderBySessionId: {},
        loading: false,
      },
      { loadInitial: () => Promise.resolve(), poll: () => Promise.resolve(), loadOlder: () => Promise.resolve() },
    );

    root = document.createElement("div");
    document.body.appendChild(root);
    render(
      <AppProviders sessionsStore={sessionsStore as any} messagesStore={messagesStore as any}>
        <ConversationPane />
      </AppProviders>,
      root,
    );

    const timestamps = root.querySelectorAll("time.messageTimestamp");
    expect(timestamps).toHaveLength(1);
    expect(timestamps[0]?.getAttribute("datetime")).toBe(new Date(1_744_000_000 * 1000).toISOString());

    const fileLink = root.querySelector(".messageBody a[data-file-path='/repo/codoxear/server.py']") as HTMLAnchorElement | null;
    expect(fileLink?.textContent).toBe("server.py#L123");
    expect(fileLink?.getAttribute("href")).toContain("api/sessions/sess-8/file/blob?path=%2Frepo%2Fcodoxear%2Fserver.py");

    const table = root.querySelector(".messageBody .mdTableWrap table");
    expect(table).not.toBeNull();
    expect(table?.textContent).toContain("Name");
    expect(table?.textContent).toContain("offline");

    const image = root.querySelector(".messageBody img") as HTMLImageElement | null;
    expect(image?.getAttribute("src")).toContain("api/sessions/sess-8/file/blob?path=%2Frepo%2Fdocs%2Fflow.png");
    expect(image?.getAttribute("alt")).toBe("diagram");
  });

  it("rewrites memory citation blocks into clickable memory links and can open local file refs", () => {
    const sessionsStore = createStaticStore(
      {
        items: [{ session_id: "sess-citation", cwd: "/repo", agent_backend: "pi" }],
        activeSessionId: "sess-citation",
        loading: false,
        newSessionDefaults: null,
      },
      { refresh: () => Promise.resolve(), select: () => undefined },
    );
    const messagesStore = createStaticStore(
      {
        bySessionId: {
          "sess-citation": [
            {
              role: "assistant",
              text: "<oai-mem-citation>\n<citation_entries>\nMEMORY.md:4773-4779|note=[used corrected guidance]\n</citation_entries>\n<rollout_ids>\n</rollout_ids>\n</oai-mem-citation>",
            },
          ],
        },
        offsetsBySessionId: { "sess-citation": 1 },
        loading: false,
      },
      { loadInitial: () => Promise.resolve(), poll: () => Promise.resolve(), loadOlder: () => Promise.resolve() },
    );

    root = document.createElement("div");
    document.body.appendChild(root);
    render(
      <AppProviders sessionsStore={sessionsStore as any} messagesStore={messagesStore as any}>
        <ConversationPane />
      </AppProviders>,
      root,
    );

    expect(root.textContent).toContain("Memory citations");
    const link = root.querySelector("a[data-file-path]") as HTMLAnchorElement | null;
    expect(link).not.toBeNull();
    expect(link?.getAttribute("data-file-path")).toBe("~/.codex/memories/MEMORY.md");
    expect(link?.getAttribute("data-file-line")).toBe("4773");
  });

  it("shows history controls and wires load-older plus jump-to-latest actions", async () => {
    const sessionsStore = createStaticStore(
      { items: [{ session_id: "sess-9", agent_backend: "pi" }], activeSessionId: "sess-9", loading: false, newSessionDefaults: null },
      { refresh: () => Promise.resolve(), select: () => undefined },
    );
    const loadInitial = vi.fn().mockResolvedValue(undefined);
    const liveLoadInitial = vi.fn().mockResolvedValue(undefined);
    const loadOlder = vi.fn().mockResolvedValue(undefined);
    const messagesStore = createStaticStore(
      {
        bySessionId: {
          "sess-9": [{ role: "assistant", text: "Newest reply" }],
        },
        offsetsBySessionId: { "sess-9": 1 },
        hasOlderBySessionId: { "sess-9": true },
        olderBeforeBySessionId: { "sess-9": 5 },
        loadingOlderBySessionId: { "sess-9": false },
        loading: false,
      },
      { loadInitial, poll: () => Promise.resolve(), loadOlder },
    );
    const liveSessionStore = createStaticStore(
      { offsetsBySessionId: {}, requestsBySessionId: {}, requestVersionsBySessionId: {}, busyBySessionId: {}, loadingBySessionId: {} },
      { loadInitial: liveLoadInitial, poll: () => Promise.resolve() },
    );

    root = document.createElement("div");
    document.body.appendChild(root);
    render(
      <AppProviders liveSessionStore={liveSessionStore as any} sessionsStore={sessionsStore as any} messagesStore={messagesStore as any}>
        <ConversationPane />
      </AppProviders>,
      root,
    );

    const loadOlderButton = Array.from(root.querySelectorAll("button")).find((button) => button.textContent === "Load older") as HTMLButtonElement | undefined;
    const jumpButton = Array.from(root.querySelectorAll("button")).find((button) => button.textContent === "Jump to latest") as HTMLButtonElement | undefined;

    expect(loadOlderButton).toBeDefined();
    expect(jumpButton).toBeDefined();

    const initialLegacyLoadCalls = loadInitial.mock.calls.length;

    loadOlderButton?.click();
    jumpButton?.click();
    await Promise.resolve();

    expect(loadOlder).toHaveBeenCalledWith("sess-9");
    expect(loadInitial).toHaveBeenCalledTimes(initialLegacyLoadCalls);
    expect(liveLoadInitial).toHaveBeenCalledWith("sess-9");
  });

  it("renders one assistant bubble for a streamed pi reply", () => {
    const sessionsStore = createStaticStore(
      { items: [{ session_id: "sess-stream", agent_backend: "pi" }], activeSessionId: "sess-stream", loading: false, newSessionDefaults: null },
      { refresh: () => Promise.resolve(), select: () => undefined },
    );
    const messagesStore = createStaticStore(
      {
        bySessionId: {
          "sess-stream": [
            { role: "assistant", text: "hello", streaming: true, stream_id: "pi-stream:turn-001", turn_id: "turn-001" },
          ],
        },
        offsetsBySessionId: { "sess-stream": 1 },
        loading: false,
      },
      { loadInitial: () => Promise.resolve(), poll: () => Promise.resolve(), loadOlder: () => Promise.resolve() },
    );

    root = document.createElement("div");
    document.body.appendChild(root);
    render(
      <AppProviders sessionsStore={sessionsStore as any} messagesStore={messagesStore as any}>
        <ConversationPane />
      </AppProviders>,
      root,
    );

    const assistantRows = Array.from(root.querySelectorAll(".messageRow")).filter((row) => row.textContent?.includes("hello"));
    expect(assistantRows).toHaveLength(1);
  });

  it("keeps the working indicator visible when live session state is still busy", () => {
    const sessionsStore = createStaticStore(
      { items: [{ session_id: "sess-live-busy", agent_backend: "pi", busy: false }], activeSessionId: "sess-live-busy", loading: false, newSessionDefaults: null },
      { refresh: () => Promise.resolve(), select: () => undefined },
    );
    const messagesStore = createStaticStore(
      {
        bySessionId: {
          "sess-live-busy": [{ role: "assistant", text: "Still thinking" }],
        },
        offsetsBySessionId: { "sess-live-busy": 1 },
        loading: false,
      },
      { loadInitial: () => Promise.resolve(), poll: () => Promise.resolve(), loadOlder: () => Promise.resolve() },
    );
    const liveSessionStore = createStaticStore(
      {
        offsetsBySessionId: { "sess-live-busy": 1 },
        liveOffsetsBySessionId: { "sess-live-busy": 1 },
        requestsBySessionId: {},
        requestVersionsBySessionId: {},
        busyBySessionId: { "sess-live-busy": true },
        loadingBySessionId: {},
      },
      { loadInitial: () => Promise.resolve(), poll: () => Promise.resolve() },
    );

    root = document.createElement("div");
    document.body.appendChild(root);
    render(
      <AppProviders
        sessionsStore={sessionsStore as any}
        messagesStore={messagesStore as any}
        liveSessionStore={liveSessionStore as any}
      >
        <ConversationPane />
      </AppProviders>,
      root,
    );

    expect(root.textContent).toContain("Working");
  });

  it("shows a floating previous-user button only after scrolling above an earlier user message", async () => {
    const sessionsStore = createStaticStore(
      { items: [{ session_id: "sess-jump", agent_backend: "pi" }], activeSessionId: "sess-jump", loading: false, newSessionDefaults: null },
      { refresh: () => Promise.resolve(), select: () => undefined },
    );
    const messagesStore = createStaticStore(
      {
        bySessionId: {
          "sess-jump": [
            { role: "user", text: "First question" },
            { role: "assistant", text: "First answer" },
            { role: "user", text: "Second question" },
            { role: "assistant", text: "Second answer" },
            { role: "assistant", text: "Newest answer" },
          ],
        },
        offsetsBySessionId: { "sess-jump": 5 },
        loading: false,
      },
      { loadInitial: () => Promise.resolve(), poll: () => Promise.resolve(), loadOlder: () => Promise.resolve() },
    );

    root = document.createElement("div");
    document.body.appendChild(root);
    render(
      <AppProviders sessionsStore={sessionsStore as any} messagesStore={messagesStore as any}>
        <ConversationPane />
      </AppProviders>,
      root,
    );

    const pane = root.querySelector(".conversationPane") as HTMLDivElement | null;
    const rows = Array.from(root.querySelectorAll<HTMLElement>(".messageRow"));

    expect(pane).not.toBeNull();
    expect(root.querySelector("[data-testid='jump-to-previous-user']")).toBeNull();

    rows.forEach((row, index) => {
      Object.defineProperty(row, "offsetTop", { configurable: true, value: index * 180 });
    });
    Object.defineProperty(pane!, "clientHeight", { configurable: true, value: 360 });
    Object.defineProperty(pane!, "scrollTop", { configurable: true, writable: true, value: 190 });

    await act(async () => {
      pane?.dispatchEvent(new Event("scroll"));
      await Promise.resolve();
      await Promise.resolve();
    });

    const jumpButton = root.querySelector("[data-testid='jump-to-previous-user']") as HTMLButtonElement | null;
    expect(jumpButton).not.toBeNull();
    expect(jumpButton?.textContent).toBe("");
    expect(jumpButton?.getAttribute("aria-label")).toBe("Jump to previous user message");
  });

  it("jumps to the nearest earlier rendered user message and can step backward again", async () => {
    const sessionsStore = createStaticStore(
      {
        items: [{ session_id: "sess-jump-target", agent_backend: "pi" }],
        activeSessionId: "sess-jump-target",
        loading: false,
        newSessionDefaults: null,
      },
      { refresh: () => Promise.resolve(), select: () => undefined },
    );
    const loadOlder = vi.fn().mockResolvedValue(undefined);
    const messagesStore = createStaticStore(
      {
        bySessionId: {
          "sess-jump-target": [
            { role: "user", text: "Question 1" },
            { role: "assistant", text: "Answer 1" },
            { role: "user", text: "Question 2" },
            { role: "assistant", text: "Answer 2" },
            { role: "assistant", text: "Answer 3" },
          ],
        },
        offsetsBySessionId: { "sess-jump-target": 5 },
        loading: false,
      },
      { loadInitial: () => Promise.resolve(), poll: () => Promise.resolve(), loadOlder },
    );

    const scrollTo = vi.fn();

    root = document.createElement("div");
    document.body.appendChild(root);
    render(
      <AppProviders sessionsStore={sessionsStore as any} messagesStore={messagesStore as any}>
        <ConversationPane />
      </AppProviders>,
      root,
    );

    const pane = root.querySelector(".conversationPane") as HTMLDivElement | null;
    Object.defineProperty(pane!, "scrollTo", { configurable: true, value: scrollTo });
    const rows = Array.from(root.querySelectorAll<HTMLElement>(".messageRow"));
    rows.forEach((row, index) => {
      Object.defineProperty(row, "offsetTop", { configurable: true, value: index * 240 });
    });
    Object.defineProperty(pane!, "clientHeight", { configurable: true, value: 360 });
    Object.defineProperty(pane!, "scrollTop", { configurable: true, writable: true, value: 700 });

    await act(async () => {
      pane?.dispatchEvent(new Event("scroll"));
      await Promise.resolve();
      await Promise.resolve();
    });

    const jumpButton = root.querySelector("[data-testid='jump-to-previous-user']") as HTMLButtonElement | null;
    expect(jumpButton).not.toBeNull();
    await act(async () => {
      jumpButton?.click();
      await Promise.resolve();
    });

    expect(scrollTo).toHaveBeenCalledWith({ top: 464, behavior: "smooth" });
    expect(loadOlder).not.toHaveBeenCalled();

    pane!.scrollTop = 464;
    await act(async () => {
      pane?.dispatchEvent(new Event("scroll"));
      await Promise.resolve();
      await Promise.resolve();
    });

    const secondJumpButton = root.querySelector("[data-testid='jump-to-previous-user']") as HTMLButtonElement | null;
    expect(secondJumpButton).not.toBeNull();
    await act(async () => {
      secondJumpButton?.click();
      await Promise.resolve();
    });

    expect(scrollTo).toHaveBeenCalledWith({ top: 0, behavior: "smooth" });
  });

  it("clears the previous-user jump button when switching to a session without earlier user rows", async () => {
    const messagesStore = createStaticStore(
      {
        bySessionId: {
          "sess-a": [
            { role: "user", text: "Question A" },
            { role: "assistant", text: "Answer A" },
            { role: "user", text: "Question A2" },
            { role: "assistant", text: "Answer A2" },
            { role: "assistant", text: "Newest answer" },
          ],
          "sess-b": [{ role: "assistant", text: "Only answer" }],
        },
        offsetsBySessionId: { "sess-a": 5, "sess-b": 1 },
        loading: false,
      },
      { loadInitial: () => Promise.resolve(), poll: () => Promise.resolve(), loadOlder: () => Promise.resolve() },
    );

    root = document.createElement("div");
    document.body.appendChild(root);
    render(
      <AppProviders
        sessionsStore={createStaticStore(
          {
            items: [
              { session_id: "sess-a", agent_backend: "pi" },
              { session_id: "sess-b", agent_backend: "pi" },
            ],
            activeSessionId: "sess-a",
            loading: false,
            newSessionDefaults: null,
          },
          { refresh: () => Promise.resolve(), select: () => undefined },
        ) as any}
        messagesStore={messagesStore as any}
      >
        <ConversationPane />
      </AppProviders>,
      root,
    );

    const pane = root.querySelector(".conversationPane") as HTMLDivElement | null;
    const rows = Array.from(root.querySelectorAll<HTMLElement>(".messageRow"));
    rows.forEach((row, index) => {
      Object.defineProperty(row, "offsetTop", { configurable: true, value: index * 180 });
    });
    Object.defineProperty(pane!, "clientHeight", { configurable: true, value: 360 });
    Object.defineProperty(pane!, "scrollTop", { configurable: true, writable: true, value: 410 });

    await act(async () => {
      pane?.dispatchEvent(new Event("scroll"));
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(root.querySelector("[data-testid='jump-to-previous-user']")).not.toBeNull();

    await act(async () => {
      render(
        <AppProviders
          sessionsStore={createStaticStore(
            {
              items: [
                { session_id: "sess-a", agent_backend: "pi" },
                { session_id: "sess-b", agent_backend: "pi" },
              ],
              activeSessionId: "sess-b",
              loading: false,
              newSessionDefaults: null,
            },
            { refresh: () => Promise.resolve(), select: () => undefined },
          ) as any}
          messagesStore={messagesStore as any}
        >
          <ConversationPane />
        </AppProviders>,
        root!,
      );
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(root.querySelector("[data-testid='jump-to-previous-user']")).toBeNull();
  });

  it("shows a scroll-to-bottom button when the pane is away from the latest content", async () => {
    const sessionsStore = createStaticStore(
      { items: [{ session_id: "sess-bottom", agent_backend: "pi" }], activeSessionId: "sess-bottom", loading: false, newSessionDefaults: null },
      { refresh: () => Promise.resolve(), select: () => undefined },
    );
    const messagesStore = createStaticStore(
      {
        bySessionId: {
          "sess-bottom": [
            { role: "user", text: "Question 1" },
            { role: "assistant", text: "Answer 1" },
            { role: "assistant", text: "Answer 2" },
            { role: "assistant", text: "Answer 3" },
          ],
        },
        offsetsBySessionId: { "sess-bottom": 4 },
        loading: false,
      },
      { loadInitial: () => Promise.resolve(), poll: () => Promise.resolve(), loadOlder: () => Promise.resolve() },
    );

    root = document.createElement("div");
    document.body.appendChild(root);
    render(
      <AppProviders sessionsStore={sessionsStore as any} messagesStore={messagesStore as any}>
        <ConversationPane />
      </AppProviders>,
      root,
    );

    await act(async () => {
      await Promise.resolve();
    });

    const scrollHeightDescriptor = Object.getOwnPropertyDescriptor(HTMLElement.prototype, "scrollHeight");
    const clientHeightDescriptor = Object.getOwnPropertyDescriptor(HTMLElement.prototype, "clientHeight");
    Object.defineProperty(HTMLElement.prototype, "scrollHeight", {
      configurable: true,
      get() {
        return this.classList?.contains("conversationPane") ? 1500 : (scrollHeightDescriptor?.get?.call(this) ?? 0);
      },
    });
    Object.defineProperty(HTMLElement.prototype, "clientHeight", {
      configurable: true,
      get() {
        return this.classList?.contains("conversationPane") ? 360 : (clientHeightDescriptor?.get?.call(this) ?? 0);
      },
    });

    const pane = root.querySelector(".conversationPane") as HTMLDivElement | null;
    Object.defineProperty(pane!, "scrollTop", { configurable: true, writable: true, value: 620 });
    const rows = Array.from(root.querySelectorAll<HTMLElement>(".messageRow"));
    rows.forEach((row, index) => {
      Object.defineProperty(row, "offsetTop", { configurable: true, value: index * 400 });
    });

    await act(async () => {
      pane?.dispatchEvent(new Event("scroll"));
      await Promise.resolve();
      await Promise.resolve();
    });

    const bottomButton = root.querySelector("[data-testid='scroll-to-bottom']") as HTMLButtonElement | null;
    expect(bottomButton).not.toBeNull();
    expect(bottomButton?.textContent).toBe("");
    expect(bottomButton?.getAttribute("aria-label")).toBe("Scroll to conversation bottom");

    if (scrollHeightDescriptor) {
      Object.defineProperty(HTMLElement.prototype, "scrollHeight", scrollHeightDescriptor);
    }
    if (clientHeightDescriptor) {
      Object.defineProperty(HTMLElement.prototype, "clientHeight", clientHeightDescriptor);
    }
  });

  it("shows both floating buttons together and scrolls to the bottom when requested", async () => {
    const sessionsStore = createStaticStore(
      { items: [{ session_id: "sess-both", agent_backend: "pi" }], activeSessionId: "sess-both", loading: false, newSessionDefaults: null },
      { refresh: () => Promise.resolve(), select: () => undefined },
    );
    const messagesStore = createStaticStore(
      {
        bySessionId: {
          "sess-both": [
            { role: "user", text: "Question 1" },
            { role: "assistant", text: "Answer 1" },
            { role: "user", text: "Question 2" },
            { role: "assistant", text: "Answer 2" },
            { role: "assistant", text: "Answer 3" },
            { role: "assistant", text: "Newest answer" },
          ],
        },
        offsetsBySessionId: { "sess-both": 6 },
        loading: false,
      },
      { loadInitial: () => Promise.resolve(), poll: () => Promise.resolve(), loadOlder: () => Promise.resolve() },
    );

    const scrollTo = vi.fn();

    root = document.createElement("div");
    document.body.appendChild(root);
    render(
      <AppProviders sessionsStore={sessionsStore as any} messagesStore={messagesStore as any}>
        <ConversationPane />
      </AppProviders>,
      root,
    );

    await act(async () => {
      await Promise.resolve();
    });

    const scrollHeightDescriptor = Object.getOwnPropertyDescriptor(HTMLElement.prototype, "scrollHeight");
    const clientHeightDescriptor = Object.getOwnPropertyDescriptor(HTMLElement.prototype, "clientHeight");
    Object.defineProperty(HTMLElement.prototype, "scrollHeight", {
      configurable: true,
      get() {
        return this.classList?.contains("conversationPane") ? 1800 : (scrollHeightDescriptor?.get?.call(this) ?? 0);
      },
    });
    Object.defineProperty(HTMLElement.prototype, "clientHeight", {
      configurable: true,
      get() {
        return this.classList?.contains("conversationPane") ? 360 : (clientHeightDescriptor?.get?.call(this) ?? 0);
      },
    });

    const pane = root.querySelector(".conversationPane") as HTMLDivElement | null;
    Object.defineProperty(pane!, "scrollTo", { configurable: true, value: scrollTo });
    Object.defineProperty(pane!, "scrollTop", { configurable: true, writable: true, value: 700 });
    const rows = Array.from(root.querySelectorAll<HTMLElement>(".messageRow"));
    rows.forEach((row, index) => {
      Object.defineProperty(row, "offsetTop", { configurable: true, value: index * 350 });
    });

    await act(async () => {
      pane?.dispatchEvent(new Event("scroll"));
      await Promise.resolve();
      await Promise.resolve();
    });

    const previousButton = root.querySelector("[data-testid='jump-to-previous-user']") as HTMLButtonElement | null;
    const bottomButton = root.querySelector("[data-testid='scroll-to-bottom']") as HTMLButtonElement | null;

    expect(previousButton).not.toBeNull();
    expect(bottomButton).not.toBeNull();

    await act(async () => {
      bottomButton?.click();
      await Promise.resolve();
    });

    expect(scrollTo).toHaveBeenCalledWith({ top: 1800, behavior: "smooth" });

    if (scrollHeightDescriptor) {
      Object.defineProperty(HTMLElement.prototype, "scrollHeight", scrollHeightDescriptor);
    }
    if (clientHeightDescriptor) {
      Object.defineProperty(HTMLElement.prototype, "clientHeight", clientHeightDescriptor);
    }
  });

  it("keeps existing messages visible while a background refresh is in flight", () => {
    const sessionsStore = createStaticStore(
      { items: [{ session_id: "sess-10", agent_backend: "pi" }], activeSessionId: "sess-10", loading: false, newSessionDefaults: null },
      { refresh: () => Promise.resolve(), select: () => undefined },
    );
    const messagesStore = createStaticStore(
      {
        bySessionId: {
          "sess-10": [{ role: "assistant", text: "Keep this visible" }],
        },
        offsetsBySessionId: { "sess-10": 1 },
        hasOlderBySessionId: {},
        olderBeforeBySessionId: {},
        loadingOlderBySessionId: {},
        loading: true,
      },
      { loadInitial: () => Promise.resolve(), poll: () => Promise.resolve(), loadOlder: () => Promise.resolve() },
    );

    root = document.createElement("div");
    document.body.appendChild(root);
    render(
      <AppProviders sessionsStore={sessionsStore as any} messagesStore={messagesStore as any}>
        <ConversationPane />
      </AppProviders>,
      root,
    );

    expect(root.textContent).toContain("Keep this visible");
    expect(root.querySelector("[data-testid='message-surface'][data-kind='loading']")).toBeNull();
  });

  it("renders optimistic pending user messages from the composer store", () => {
    const sessionsStore = createStaticStore(
      { items: [{ session_id: "sess-pending", agent_backend: "pi" }], activeSessionId: "sess-pending", loading: false, newSessionDefaults: null },
      { refresh: () => Promise.resolve(), select: () => undefined },
    );
    const messagesStore = createStaticStore(
      {
        bySessionId: {
          "sess-pending": [{ role: "assistant", text: "Working on it" }],
        },
        offsetsBySessionId: { "sess-pending": 1 },
        loading: false,
      },
      { loadInitial: () => Promise.resolve(), poll: () => Promise.resolve(), loadOlder: () => Promise.resolve() },
    );
    const composerStore = createStaticStore(
      {
        draft: "",
        sending: true,
        pendingBySessionId: {
          "sess-pending": [{ role: "user", text: "Please continue", pending: true, localId: "local-1" }],
        },
      },
      { setDraft: () => undefined, submit: () => Promise.resolve() },
    );

    root = document.createElement("div");
    document.body.appendChild(root);
    render(
      <AppProviders sessionsStore={sessionsStore as any} messagesStore={messagesStore as any} composerStore={composerStore as any}>
        <ConversationPane />
      </AppProviders>,
      root,
    );

    const text = root.textContent || "";
    expect(text).toContain("Working on it");
    expect(text).toContain("Please continue");
    expect(root.querySelectorAll("[data-testid='message-surface'][data-kind='user']")).toHaveLength(1);
  });

  it("inserts day separators when consecutive messages cross calendar days", () => {
    const sessionsStore = createStaticStore(
      { items: [{ session_id: "sess-days", agent_backend: "pi" }], activeSessionId: "sess-days", loading: false, newSessionDefaults: null },
      { refresh: () => Promise.resolve(), select: () => undefined },
    );
    const messagesStore = createStaticStore(
      {
        bySessionId: {
          "sess-days": [
            { role: "assistant", ts: 1_744_000_000, text: "Day one" },
            { role: "assistant", ts: 1_744_086_500, text: "Day two" },
          ],
        },
        offsetsBySessionId: { "sess-days": 2 },
        loading: false,
      },
      { loadInitial: () => Promise.resolve(), poll: () => Promise.resolve(), loadOlder: () => Promise.resolve() },
    );

    root = document.createElement("div");
    document.body.appendChild(root);
    render(
      <AppProviders sessionsStore={sessionsStore as any} messagesStore={messagesStore as any}>
        <ConversationPane />
      </AppProviders>,
      root,
    );

    expect(root.querySelectorAll(".daySeparator").length).toBe(2);
    expect(root.textContent).toContain("Day one");
    expect(root.textContent).toContain("Day two");
  });

  it("does not keep showing the skeleton when a different session is still loading", () => {
    const sessionsStore = createStaticStore(
      { items: [{ session_id: "sess-1", agent_backend: "pi" }], activeSessionId: "sess-1", loading: false, newSessionDefaults: null },
      { refresh: () => Promise.resolve(), select: () => undefined },
    );
    const messagesStore = createStaticStore(
      {
        bySessionId: {
          "sess-1": [{ role: "assistant", text: "Loaded message" }],
        },
        offsetsBySessionId: { "sess-1": 1 },
        hasOlderBySessionId: {},
        olderBeforeBySessionId: {},
        loadingOlderBySessionId: {},
        loadingBySessionId: { "sess-2": true },
        loadedBySessionId: { "sess-1": true },
        loading: true,
      },
      { loadInitial: () => Promise.resolve(), poll: () => Promise.resolve(), loadOlder: () => Promise.resolve() },
    );

    root = document.createElement("div");
    document.body.appendChild(root);
    render(
      <AppProviders sessionsStore={sessionsStore as any} messagesStore={messagesStore as any}>
        <ConversationPane />
      </AppProviders>,
      root,
    );

    expect(root.textContent).toContain("Loaded message");
    expect(root.querySelector("[data-kind='loading']")).toBeNull();
  });

  it("loads historical pi messages when the synthetic history session becomes active", async () => {
    const sessionsStore = createStaticStore(
      {
        items: [{ session_id: "history:pi:resume-hist", agent_backend: "pi", historical: true }],
        activeSessionId: "history:pi:resume-hist",
        loading: false,
        newSessionDefaults: null,
      },
      { refresh: () => Promise.resolve(), select: () => undefined },
    );
    const loadInitial = vi.fn().mockResolvedValue(undefined);
    const messagesStore = createStaticStore(
      {
        bySessionId: {},
        offsetsBySessionId: {},
        hasOlderBySessionId: {},
        olderBeforeBySessionId: {},
        loadingOlderBySessionId: {},
        loadingBySessionId: {},
        loadedBySessionId: {},
        loading: false,
      },
      { loadInitial, poll: () => Promise.resolve(), loadOlder: () => Promise.resolve() },
    );

    root = document.createElement("div");
    document.body.appendChild(root);
    await act(async () => {
      render(
        <AppProviders sessionsStore={sessionsStore as any} messagesStore={messagesStore as any}>
          <ConversationPane />
        </AppProviders>,
        root!,
      );
      await Promise.resolve();
    });

    expect(loadInitial).toHaveBeenCalledWith("history:pi:resume-hist");
  });

  it("shows the loading skeleton immediately for an unloaded historical pi session", async () => {
    const sessionsStore = createStaticStore(
      {
        items: [{ session_id: "history:pi:resume-empty", agent_backend: "pi", historical: true }],
        activeSessionId: "history:pi:resume-empty",
        loading: false,
        newSessionDefaults: null,
      },
      { refresh: () => Promise.resolve(), select: () => undefined },
    );
    let resolveLoad: (() => void) | null = null;
    const messagesStore = createMutableStore(
      {
        bySessionId: {},
        offsetsBySessionId: {},
        hasOlderBySessionId: {},
        olderBeforeBySessionId: {},
        loadingOlderBySessionId: {},
        loadingBySessionId: {},
        loadedBySessionId: {},
        loading: false,
      },
      (getState, setState) => ({
        loadInitial: vi.fn(async (sessionId: string) => {
          setState({
            ...getState(),
            loadingBySessionId: { ...getState().loadingBySessionId, [sessionId]: true },
            loading: true,
          });
          await new Promise<void>((resolve) => {
            resolveLoad = resolve;
          });
          setState({
            ...getState(),
            loadingBySessionId: { ...getState().loadingBySessionId, [sessionId]: false },
            loadedBySessionId: { ...getState().loadedBySessionId, [sessionId]: true },
            bySessionId: { ...getState().bySessionId, [sessionId]: [] },
            loading: false,
          });
        }),
        poll: () => Promise.resolve(),
        loadOlder: () => Promise.resolve(),
      }),
    );

    root = document.createElement("div");
    document.body.appendChild(root);
    render(
      <AppProviders sessionsStore={sessionsStore as any} messagesStore={messagesStore as any}>
        <ConversationPane />
      </AppProviders>,
      root,
    );

    expect(root.querySelector("[data-kind='loading']")).not.toBeNull();
    expect(root.textContent).not.toContain("No conversation events yet.");

    await act(async () => {
      resolveLoad?.();
      await Promise.resolve();
    });
  });
});
