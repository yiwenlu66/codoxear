import type { ExtensionAPI, ExtensionContext } from "@mariozechner/pi-coding-agent";

type AskUserOption = {
  label: string;
  description: string;
  preview?: string;
};

type AskUserQuestion = {
  question: string;
  header: string;
  options: AskUserOption[];
  multiSelect?: boolean;
};

type AskUserParams = {
  questions: AskUserQuestion[];
  answers?: Record<string, string>;
  annotations?: Record<string, { preview?: string; notes?: string }>;
  metadata?: { source?: string };
};

type AskUserOutcome = {
  action: "answered" | "clarify_with_user" | "finish_plan_interview" | "cancelled";
  answers: Record<string, string>;
  annotations?: Record<string, { preview?: string; notes?: string }>;
};

const ASK_USER_BRIDGE_PREFIX = "__codoxear_ask_user_bridge_v1__";

function encodeAskUserBridgeRequest(params: AskUserParams) {
  return `${ASK_USER_BRIDGE_PREFIX}\n${JSON.stringify({
    questions: params.questions,
    ...(params.metadata ? { metadata: params.metadata } : {}),
  })}`;
}

function parseAskUserBridgeResponse(value: string | undefined): AskUserOutcome {
  if (typeof value !== "string" || !value.startsWith(`${ASK_USER_BRIDGE_PREFIX}\n`)) {
    return { action: "cancelled", answers: {} };
  }

  try {
    const parsed = JSON.parse(value.slice(ASK_USER_BRIDGE_PREFIX.length + 1)) as Partial<AskUserOutcome>;
    const action = parsed.action === "answered" || parsed.action === "clarify_with_user" || parsed.action === "finish_plan_interview"
      ? parsed.action
      : "cancelled";
    const answers = parsed.answers && typeof parsed.answers === "object" ? parsed.answers : {};
    const annotations = parsed.annotations && typeof parsed.annotations === "object" ? parsed.annotations : undefined;
    return {
      action,
      answers: answers as Record<string, string>,
      ...(annotations ? { annotations } : {}),
    };
  } catch {
    return { action: "cancelled", answers: {} };
  }
}

async function askViaEditor(ctx: ExtensionContext, params: AskUserParams): Promise<AskUserOutcome> {
  if (params.answers) {
    return {
      action: "answered",
      answers: params.answers,
      ...(params.annotations ? { annotations: params.annotations } : {}),
    };
  }

  if (!ctx.hasUI) {
    throw new Error("AskUserQuestion requires an interactive UI");
  }

  const response = await ctx.ui.editor("AskUserQuestion", encodeAskUserBridgeRequest(params));
  return parseAskUserBridgeResponse(response);
}

export default function codoxearAskUserBridge(pi: ExtensionAPI): void {
  let resetBridge: null | (() => void) = null;

  void import("pi-claude-runtime-core/runtime-bridge")
    .then(({ setSharedAskUserQuestionBridge }) => {
      setSharedAskUserQuestionBridge({
        ask: (ctx: ExtensionContext, params: AskUserParams) => askViaEditor(ctx, params),
      });
      resetBridge = () => setSharedAskUserQuestionBridge(null);
    })
    .catch(() => {
      // The AskUserQuestion package is optional. If it is not installed, leave Pi startup unaffected.
    });

  pi.on("session_shutdown", async () => {
    resetBridge?.();
  });
}
