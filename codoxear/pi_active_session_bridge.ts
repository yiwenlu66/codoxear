const fs = require("node:fs");
const path = require("node:path");

type SessionManager = {
	getSessionFile(): string | undefined;
	getSessionId(): string;
	getCwd(): string;
};

type ThinkingLevel = "off" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max";

type ExtensionUI = {
	notify(message: string, type?: "info" | "warning" | "error"): void;
};

type ExtensionContext = {
	sessionManager: SessionManager;
	ui: ExtensionUI;
};

type ExtensionCommandContext = ExtensionContext;

type ExtensionAPI = {
	on(event: "session_start", handler: (event: { type: "session_start" }, ctx: ExtensionContext) => void): void;
	on(
		event: "session_switch",
		handler: (event: { type: "session_switch"; reason: "new" | "resume" }, ctx: ExtensionContext) => void,
	): void;
	on(event: "session_fork", handler: (event: { type: "session_fork" }, ctx: ExtensionContext) => void): void;
	registerCommand(
		name: string,
		options: {
			description?: string;
			handler: (args: string, ctx: ExtensionCommandContext) => void | Promise<void>;
		},
	): void;
	getThinkingLevel(): ThinkingLevel;
	setThinkingLevel(level: ThinkingLevel): void;
};

const THINKING_LEVELS: readonly ThinkingLevel[] = ["off", "minimal", "low", "medium", "high", "xhigh", "max"];

function writeActiveSession(ctx: ExtensionContext, reason: string): void {
	const markerPath = process.env.CODEX_WEB_PI_ACTIVE_SESSION_FILE;
	if (!markerPath) return;
	const sessionFile = ctx.sessionManager.getSessionFile();
	if (!sessionFile) return;
	const payload = {
		version: 1,
		reason,
		sessionFile,
		sessionId: ctx.sessionManager.getSessionId(),
		cwd: ctx.sessionManager.getCwd(),
		pid: process.pid,
		updatedAt: new Date().toISOString(),
	};
	try {
		fs.mkdirSync(path.dirname(markerPath), { recursive: true });
		const tmp = `${markerPath}.${process.pid}.tmp`;
		fs.writeFileSync(tmp, `${JSON.stringify(payload)}\n`, { mode: 0o600 });
		fs.renameSync(tmp, markerPath);
		try {
			fs.chmodSync(markerPath, 0o600);
		} catch {
			// Best effort; write mode already restricts newly created files.
		}
	} catch {
		// Do not let Codoxear bookkeeping affect Pi session operation.
	}
}

export default function (pi: ExtensionAPI): void {
	pi.on("session_start", (_event, ctx) => writeActiveSession(ctx, "session_start"));
	pi.on("session_switch", (event, ctx) => writeActiveSession(ctx, event.reason));
	pi.on("session_fork", (_event, ctx) => writeActiveSession(ctx, "fork"));
	pi.registerCommand("thinking", {
		description: "Set the thinking level for the current model",
		handler: (args, ctx) => {
			const requested = args.trim().toLowerCase();
			if (!THINKING_LEVELS.includes(requested as ThinkingLevel)) {
				ctx.ui.notify(
					`Thinking level: ${pi.getThinkingLevel()}. Choose one of: ${THINKING_LEVELS.join(", ")}.`,
					"warning",
				);
				return;
			}
			pi.setThinkingLevel(requested as ThinkingLevel);
			const effective = pi.getThinkingLevel();
			const message = effective === requested
				? `Thinking level: ${effective}`
				: `Thinking level: ${effective} (requested ${requested}; adjusted for the current model)`;
			ctx.ui.notify(message, "info");
		},
	});
}
