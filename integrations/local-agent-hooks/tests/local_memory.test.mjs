import assert from "node:assert/strict";
import { createServer } from "node:http";
import { mkdtemp, mkdir, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import {
  handleStop,
  handleUserPrompt,
  validateLoopbackBaseUrl,
  writeIntegrationConfig,
} from "../lib/local_memory.mjs";

test("local integration rejects every non-loopback API URL", () => {
  assert.throws(
    () => validateLoopbackBaseUrl("https://example.invalid"),
    /non-loopback/u,
  );
  assert.throws(
    () => validateLoopbackBaseUrl("http://name:secret@127.0.0.1:2009"), // public-audit: allow-test-fixture
    /credentials/u,
  );
  assert.equal(validateLoopbackBaseUrl("http://127.0.0.1:2009"), "http://127.0.0.1:2009");
});

test("configuration stores only a token-file reference", async () => {
  const root = await mkdtemp(join(tmpdir(), "tmcra-local-hooks-config-"));
  const configRoot = join(root, "runtime-config-root");
  const tokenPath = join(configRoot, "runtime", "secrets", "local-api.token");
  await mkdir(join(configRoot, "runtime", "secrets"), { recursive: true });
  const secret = "test-local-token-value-never-print"; // public-audit: allow-test-fixture
  await writeFile(tokenPath, secret, "utf8");
  const runtimeConfig = join(root, "local-runtime.json");
  await writeFile(runtimeConfig, JSON.stringify({ installation: { config_root: configRoot } }), "utf8");
  const output = join(root, "integration.json");
  const result = await writeIntegrationConfig({ runtimeConfigPath: runtimeConfig, outputPath: output });
  const raw = await readFile(output, "utf8");
  assert.equal(result.secretPrinted, false);
  assert.doesNotMatch(raw, new RegExp(secret, "u"));
  assert.equal(JSON.parse(raw).tokenFile, tokenPath);
});

test("two host tools share one project while preserving sessions and roles", async (context) => {
  const root = await mkdtemp(join(tmpdir(), "tmcra-local-hooks-contract-"));
  const tokenFile = join(root, "local-api.token");
  const integrationConfig = join(root, "integration.json");
  const stateDir = join(root, "state");
  const workspace = join(root, "workspace");
  const localToken = "test-loopback-bearer-token-value"; // public-audit: allow-test-fixture
  await mkdir(workspace, { recursive: true });
  await writeFile(tokenFile, localToken, "utf8");

  const messages = [];
  const recalls = [];
  const server = createServer(async (request, response) => {
    const chunks = [];
    for await (const chunk of request) chunks.push(chunk);
    const body = chunks.length ? JSON.parse(Buffer.concat(chunks).toString("utf8")) : null;
    assert.equal(request.headers.authorization, `Bearer ${localToken}`);
    response.setHeader("Content-Type", "application/json");
    if (request.method === "POST" && request.url === "/v1/messages") {
      messages.push(body);
      response.end(JSON.stringify({ message_id: `message-${messages.length}`, scopes: [{}] }));
      return;
    }
    if (request.method === "POST" && request.url === "/v1/recall") {
      recalls.push(body);
      response.end(JSON.stringify({
        prompt_evidence: { content: "Previous work completed the local storage adapter." },
      }));
      return;
    }
    if (request.method === "GET" && request.url === "/v1/projects") {
      response.end(JSON.stringify({ projects: [] }));
      return;
    }
    response.statusCode = 404;
    response.end(JSON.stringify({ detail: "not found" }));
  });
  await new Promise((accept) => server.listen(0, "127.0.0.1", accept));
  context.after(() => new Promise((accept) => server.close(accept)));
  const address = server.address();
  assert.ok(address && typeof address === "object");
  await writeFile(integrationConfig, JSON.stringify({
    schemaVersion: 1,
    baseUrl: `http://127.0.0.1:${address.port}`,
    tokenFile,
    stateDir,
    topK: 8,
    userVisibility: "both",
    timeoutMs: 5_000,
  }), "utf8");
  const environment = { TMCRA_LOCAL_INTEGRATION_CONFIG: integrationConfig };

  const codexRecall = await handleUserPrompt({
    session_id: "codex-native-thread",
    turn_id: "codex-turn-1",
    cwd: workspace,
    prompt: "Continue the storage adapter. API key = private-example-value", // public-audit: allow-test-fixture
  }, "codex", environment);
  assert.match(codexRecall.hookSpecificOutput.additionalContext, /Previous work/u);
  assert.match(codexRecall.hookSpecificOutput.additionalContext, /trust="untrusted"/u);
  await handleStop({
    session_id: "codex-native-thread",
    turn_id: "codex-turn-1",
    cwd: workspace,
    last_assistant_message: "Implemented the storage adapter and verified it.",
  }, "codex", false, environment);

  await handleUserPrompt({
    session_id: "claude-native-thread",
    turn_id: "claude-turn-1",
    cwd: workspace,
    prompt: "What did the other tool finish?",
  }, "claude-code", environment);

  assert.equal(messages.length, 3);
  assert.deepEqual(messages.map((item) => item.role), ["user", "assistant", "user"]);
  assert.deepEqual(messages.map((item) => item.visibility), ["both", "project", "both"]);
  assert.equal(messages[0].project_id, messages[2].project_id);
  assert.notEqual(messages[0].session_id, messages[2].session_id);
  assert.equal(messages[0].actor.actor_role, "user");
  assert.equal(messages[1].actor.actor_role, "assistant");
  assert.doesNotMatch(messages[0].content, /private-example-value/u);
  assert.match(messages[0].content, /\[REDACTED\]/u);
  assert.equal(recalls[0].project_id, recalls[1].project_id);
});
