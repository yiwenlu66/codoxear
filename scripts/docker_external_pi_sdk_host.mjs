// Container-only fixture: two real Pi SDK sessions in one Node process, using
// normal agentDir discovery. Deliberately do not call bindExtensions(): this is
// the headless embedding path that never delivers session_start to extensions.
import { createServer } from "node:http";
import { mkdirSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { createAgentSession } from "/usr/local/lib/node_modules/@earendil-works/pi-coding-agent/dist/index.js";

const home = homedir();
const sessions = new Map();
for (const key of ["a", "b"]) {
  const cwd = join(home, `sdk-${key}`);
  mkdirSync(cwd, { recursive: true });
  const { session, extensionsResult } = await createAgentSession({
    cwd,
    agentDir: join(home, ".pi", "agent"),
    noTools: true,
  });
  if (extensionsResult.errors.length) throw Error(JSON.stringify(extensionsResult.errors));
  sessions.set(key, { session, errors: [] });
}

function info() {
  return {
    pid: process.pid,
    sessions: [...sessions].map(([key, { session, errors }]) => ({
      key,
      sessionId: session.sessionManager.getSessionId(),
      sessionFile: session.sessionManager.getSessionFile(),
      cwd: session.sessionManager.getCwd(),
      errors,
    })),
  };
}

const server = createServer(async (request, response) => {
  try {
    const url = new URL(request.url, "http://127.0.0.1:19723");
    if (request.method === "GET") {
      response.setHeader("Content-Type", "application/json");
      response.end(JSON.stringify(info()));
      return;
    }
    let body = "";
    for await (const chunk of request) body += chunk;
    const data = JSON.parse(body || "{}");
    const record = sessions.get(data.key);
    if (!record) throw Error("unknown SDK session");
    if (url.pathname === "/prompt") {
      await record.session.prompt(data.text);
    } else if (url.pathname === "/dispose") {
      await record.session.abort();
      record.session.dispose();
      sessions.delete(data.key);
    } else {
      throw Error("unknown fixture operation");
    }
    response.setHeader("Content-Type", "application/json");
    response.end(JSON.stringify(info()));
  } catch (error) {
    response.writeHead(500, { "Content-Type": "application/json" });
    response.end(JSON.stringify({ error: String(error) }));
  }
});
server.listen(19723, "127.0.0.1", () => console.log(JSON.stringify(info())));
