const fs = require("node:fs");
const path = require("node:path");

type SessionManager = {
	getSessionFile(): string | undefined;
	getSessionId(): string;
	getCwd(): string;
	fileEntries?: Array<{ type?: unknown }>;
	flushed?: boolean;
};

type ThinkingLevel = "off" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max";

type ExtensionUI = {
	notify(message: string, type?: "info" | "warning" | "error"): void;
};

type ExtensionContext = {
	sessionManager: SessionManager;
	ui: ExtensionUI;
	// ExtensionContextActions.getModel — the live model is only reachable via
	// ctx in event handlers; ExtensionAPI (pi.*) has no getModel.
	getModel?: () => { provider?: unknown; id?: unknown } | undefined;
};

type ExtensionCommandContext = ExtensionContext;

type SessionStartReason = "startup" | "reload" | "new" | "resume" | "fork";

type ExtensionAPI = {
	on(event: "session_start", handler: (event: { type: "session_start"; reason?: SessionStartReason }, ctx: ExtensionContext) => void): void;
	on(
		event: "session_switch",
		handler: (event: { type: "session_switch"; reason: "new" | "resume" | "fork" }, ctx: ExtensionContext) => void,
	): void;
	on(event: "session_fork", handler: (event: { type: "session_fork" }, ctx: ExtensionContext) => void): void;
	on(event: "turn_end", handler: (event: { type: "turn_end"; turnIndex: number; message?: unknown; toolResults?: unknown[] }, ctx: ExtensionContext) => void): void;
	on(event: "model_select", handler: (event: { type: "model_select" }, ctx: ExtensionContext) => void): void;
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

function materializeNewSession(ctx: ExtensionContext): void {
	try {
		const mgr = ctx.sessionManager;
		const sessionFile = mgr.getSessionFile();
		const header = mgr.fileEntries?.[0];
		if (!header || header.type !== "session" || typeof sessionFile !== "string" || fs.existsSync(sessionFile)) return;
		fs.writeFileSync(sessionFile, `${JSON.stringify(header)}\n`, { mode: 0o600, flag: "wx" });
		mgr.flushed = true;
	} catch {
		// The bridge must not interfere with Pi if session internals change.
	}
}

let activePi;

function readLiveRunSettings(pi, ctx?: ExtensionContext) {
	const out = {};
	try {
		// ctx.getModel() is the current Pi extension API; pi.getModel() is kept
		// as a fallback for older runtimes that exposed it on the API object.
		const getModel = (ctx && typeof ctx.getModel === "function" && ctx.getModel.bind(ctx))
			|| (pi && typeof pi.getModel === "function" && pi.getModel.bind(pi))
			|| null;
		if (getModel) {
			const m = getModel();
			if (m && typeof m === "object") {
				if (typeof m.provider === "string" && m.provider) out.model_provider = m.provider;
				if (typeof m.id === "string" && m.id) out.model = m.id;
			}
		}
	} catch {}
	try {
		if (pi && typeof pi.getThinkingLevel === "function") {
			const level = pi.getThinkingLevel();
			if (typeof level === "string" && level) out.reasoning_effort = level;
		}
	} catch {}
	return out;
}

let lastCapsModel;
let lastCapsProvider;
let lastCapsEffort;
let lastCommandsSnapshot;

function sameCommands(left, right) {
	return Array.isArray(left)
		&& Array.isArray(right)
		&& left.length === right.length
		&& left.every((command, index) => command.name === right[index].name && command.description === right[index].description);
}

function captureCommands() {
	try {
		const snapshot = activePi.getCommands().map((command) => ({
			name: command.name,
			description: command.description || "",
		}));
		lastCommandsSnapshot = sameCommands(lastCommandsSnapshot, snapshot) ? lastCommandsSnapshot : snapshot;
		return lastCommandsSnapshot;
	} catch {
		return undefined;
	}
}

function refreshCaps(ctx?: ExtensionContext) {
	// Re-read Pi's current settings and registry from a live lifecycle event.
	// Retaining an unchanged registry snapshot avoids rewriting caps when the
	// browser-visible state has not changed.
	const previousCommandsSnapshot = lastCommandsSnapshot;
	const commands = captureCommands();
	const live = readLiveRunSettings(activePi, ctx);
	const model = typeof live.model === "string" ? live.model : undefined;
	const provider = typeof live.model_provider === "string" ? live.model_provider : undefined;
	const effort = typeof live.reasoning_effort === "string" ? live.reasoning_effort : undefined;
	const commandsChanged = commands !== undefined && commands !== previousCommandsSnapshot;
	if (model === lastCapsModel && provider === lastCapsProvider && effort === lastCapsEffort && !commandsChanged) {
		return; // nothing changed
	}
	lastCapsModel = model;
	lastCapsProvider = provider;
	lastCapsEffort = effort;
	writeThinkingCapabilities(commands, live);
}

function writeThinkingCapabilities(commands, liveSettings) {
	const markerPath = process.env.CODEX_WEB_PI_ACTIVE_SESSION_FILE;
	if (!markerPath) return;
	const capsPath = `${markerPath}.caps`;
	const commandsToWrite = commands !== undefined ? commands : lastCommandsSnapshot;
	const payload = {
		bridgeVersion: 2,
		features: ["effort", "thinking"],
		commands: Array.isArray(commandsToWrite) ? commandsToWrite.map((command) => ({ name: command.name, description: command.description || "" })) : undefined,
		pid: process.pid,
		updatedAt: new Date().toISOString(),
		...(liveSettings || readLiveRunSettings(activePi)),
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
	function registerEffortCommands(ctx?: ExtensionContext) {
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
			// Read settings with the same ctx as refreshCaps: a write missing the
			// live model would clobber a richer caps file that refreshCaps then
			// wrongly considers current via its in-memory dedupe.
			writeThinkingCapabilities(captureCommands(), readLiveRunSettings(pi, ctx));
		} catch {
			// Runtime not yet bound (extension loading) or already registered;
			// retried on subsequent lifecycle events until it succeeds.
		}
	}
	pi.on("session_start", (event, ctx) => {
		registerEffortCommands(ctx);
		writeActiveSession(ctx, event.reason || "session_start");
		if (event.reason === "new") materializeNewSession(ctx);
		refreshCaps(ctx);
	});
	pi.on("session_switch", (event, ctx) => {
		registerEffortCommands(ctx);
		writeActiveSession(ctx, event.reason);
		if (event.reason === "new") materializeNewSession(ctx);
		refreshCaps(ctx);
	});
	pi.on("session_fork", (_event, ctx) => writeActiveSession(ctx, "fork"));
	// A native /model switch is only recorded in the session log; model_select
	// lets the caps file track it immediately instead of at the next turn_end.
	pi.on("model_select", (_event, ctx) => refreshCaps(ctx));
	pi.on("turn_end", (_event, ctx) => {
		registerEffortCommands(ctx);
		refreshCaps(ctx);
	});
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
