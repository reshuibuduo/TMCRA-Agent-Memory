import { appendFile } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import {
  handleSessionStart,
  handleStop,
  handleUserPrompt,
  restrictOwnerAccess,
  restrictOwnerDirectory,
} from "../lib/local_memory.mjs";

async function input() {
  let value = "";
  for await (const chunk of process.stdin) value += chunk;
  return value.trim() ? JSON.parse(value) : {};
}

async function diagnostic(event, error) {
  try {
    const path = join(homedir(), ".tmcra", "integrations", "logs", "hooks.jsonl");
    await restrictOwnerDirectory(dirname(path));
    await appendFile(path, `${JSON.stringify({
      at: new Date().toISOString(),
      event,
      error: error?.name || "Error",
      message: String(error?.message || "hook failed").slice(0, 300),
    })}\n`, { encoding: "utf8", mode: 0o600 });
    await restrictOwnerAccess(path);
  } catch {
    // A logging failure must not block the host agent.
  }
}

const event = String(process.argv[2] || "");
const platform = String(process.argv[3] || "codex").trim().toLowerCase();
const allowedEvents = new Set(["session-start", "user-prompt", "stop", "stop-failure"]);
const allowedPlatforms = new Set(["codex", "claude-code", "zcode"]);

try {
  if (!allowedEvents.has(event) || !allowedPlatforms.has(platform)) {
    throw new Error("invalid TMCRA local hook target");
  }
  const value = await input();
  const result = event === "session-start"
    ? await handleSessionStart(value, platform)
    : event === "user-prompt"
      ? await handleUserPrompt(value, platform)
      : await handleStop(value, platform, event === "stop-failure");
  process.stdout.write(`${JSON.stringify(result)}\n`);
} catch (error) {
  await diagnostic(event || "unknown", error);
  process.stdout.write('{"continue":true}\n');
}
