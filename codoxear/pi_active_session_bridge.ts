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
	getCommands(): Array<{ name: string; description?: string }>;
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
		bridgeVersion: 2,
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

let activePi;

function readLiveRunSettings(pi) {
	if (!pi) return {};
	const out = {};
	try {
		if (typeof pi.getModel === "function") {
			const m = pi.getModel();
			if (m && typeof m === "object") {
				if (typeof m.provider === "string" && m.provider) out.model_provider = m.provider;
				if (typeof m.id === "string" && m.id) out.model = m.id;
			}
		}
	} catch {}
	try {
		if (typeof pi.getThinkingLevel === "function") {
			const level = pi.getThinkingLevel();
			if (typeof level === "string" && level) out.reasoning_effort = level;
		}
	} catch {}
	return out;
}

function writeThinkingCapabilities(commands) {
	const markerPath = process.env.CODEX_WEB_PI_ACTIVE_SESSION_FILE;
	if (!markerPath) return;
	const capsPath = `${markerPath}.caps`;
	const payload = {
		bridgeVersion: 2,
		features: ["effort", "thinking"],
		commands: Array.isArray(commands) ? commands.map((command) => ({ name: command.name, description: command.description || "" })) : undefined,
		pid: process.pid,
		updatedAt: new Date().toISOString(),
		...readLiveRunSettings(activePi),
	};
	try {
		fs.mkdirSync(path.dirname(capsPath), { recursive: true });
		const tmp = `${capsPath}.${process.pid}.tmp`;
		fs.writeFileSync(tmp, `${JSON.stringify(payload)}\n`, { mode: 0o600 });
		fs.renameSync(tmp, capsPath);
		try {
			fs.chmodSync(capsPath, 0o600);
		} catch {
			// Best effort; write mode already restricts newly created files.
		}
	} catch {
		// Do not let Codoxear bookkeeping affect Pi session operation.
	}
}

export default function (pi: ExtensionAPI): void {
	activePi = pi;
	writeThinkingCapabilities();
	let commandsRegistered = false;
	function registerEffortCommands() {
		if (commandsRegistered) return;
		try {
			pi.registerCommand("effort", {
				description: "Set the reasoning effort (thinking level) for the current model",
				handler: effortHandler,
			});
			// Backwards-compatible alias; /effort is the primary spelling.
			pi.registerCommand("thinking", {
				description: "Alias for /effort",
				handler: effortHandler,
			});
			commandsRegistered = true;
			try {
				const live = typeof pi.getCommands === "function" ? pi.getCommands() : undefined;
				writeThinkingCapabilities(
					Array.isArray(live) ? live.map((c) => ({ name: c.name, description: c.description || "" })) : undefined,
				);
			} catch {
				// Caps without the command list is fine; the next lifecycle event retries.
			}
		} catch {
			// Runtime not yet bound (extension loading) or already registered;
			// retried on subsequent lifecycle events until it succeeds.
		}
	}
	pi.on("session_start", (event, ctx) => {
		registerEffortCommands();
		writeActiveSession(ctx, "session_start");
	});
	pi.on("session_switch", (event, ctx) => {
		registerEffortCommands();
		writeActiveSession(ctx, event.reason);
	});
	pi.on("session_fork", (_event, ctx) => writeActiveSession(ctx, "fork"));
	const effortHandler = (args, ctx) => {
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
	};
}
