const fs = require("node:fs");
const path = require("node:path");

type SessionManager = {
	getSessionFile(): string | undefined;
	getSessionId(): string;
	getCwd(): string;
};

type ExtensionContext = {
	sessionManager: SessionManager;
};

type ExtensionAPI = {
	on(event: "session_start", handler: (event: { type: "session_start" }, ctx: ExtensionContext) => void): void;
	on(
		event: "session_switch",
		handler: (event: { type: "session_switch"; reason: "new" | "resume" }, ctx: ExtensionContext) => void,
	): void;
	on(event: "session_fork", handler: (event: { type: "session_fork" }, ctx: ExtensionContext) => void): void;
};

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
}
