import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { Context } from "@deepseek-ai/cordis";
import AgentLoop from "@deepseek-ai/dsh-agent-loop";
import { mountAgentLoopTestDependencies } from "@deepseek-ai/dsh-agent-loop-testkit";
import {
  createUserMessage,
  LlmAdapter,
  type GenerateOptions,
  type StreamChunk,
} from "@deepseek-ai/dsh-llm";
import { SessionId } from "@deepseek-ai/dsh-session";
import { apply, type Config } from "../src/index.ts";

interface Call {
  method: string;
  path: string;
  authorization: string;
  body: any;
}

const temporaryDirectories: string[] = [];

afterEach(async () => {
  for (const directory of temporaryDirectories.splice(0)) {
    await rm(directory, { recursive: true, force: true });
  }
});

function answerChunks(text: string): StreamChunk[] {
  return [
    { type: "block-start", index: 0, blockType: "text" },
    { type: "text-delta", index: 0, text },
    { type: "block-end", index: 0, block: { type: "text", text } },
    { type: "usage", usage: { inputTokens: 10, outputTokens: text.length } },
    { type: "finish", reason: { kind: "stop" } },
  ];
}

class RecordingAdapter extends LlmAdapter {
  readonly requests: GenerateOptions[] = [];

  constructor(private readonly answers: string[]) {
    super();
  }

  async *stream(options: GenerateOptions): AsyncIterable<StreamChunk> {
    this.requests.push(options);
    const answer = this.answers.shift();
    if (answer === undefined) throw new Error("test adapter script exhausted");
    for (const chunk of answerChunks(answer)) yield chunk;
  }
}

function json(response: ServerResponse, status: number, value: unknown): void {
  response.writeHead(status, { "content-type": "application/json" });
  response.end(JSON.stringify(value));
}

async function body(request: IncomingMessage): Promise<any> {
  const chunks: Buffer[] = [];
  for await (const chunk of request) chunks.push(Buffer.from(chunk));
  const raw = Buffer.concat(chunks).toString("utf8");
  return raw ? JSON.parse(raw) : undefined;
}

async function startMemoryServer(token: string) {
  const calls: Call[] = [];
  const projectMessages = new Map<string, any[]>();
  const server = createServer(async (request, response) => {
    const value = await body(request);
    calls.push({
      method: request.method || "GET",
      path: request.url || "/",
      authorization: String(request.headers.authorization || ""),
      body: value,
    });
    if (request.headers.authorization !== `Bearer ${token}`) {
      return json(response, 401, { detail: "unauthorized" });
    }
    if (request.method === "POST" && request.url === "/v1/messages") {
      const items = projectMessages.get(value.project_id) || [];
      if (!items.some((item) => item.native_message_id === value.native_message_id)) items.push(value);
      projectMessages.set(value.project_id, items);
      return json(response, 200, { message_id: `message-${items.length}`, scopes: [{}] });
    }
    if (request.method === "POST" && request.url === "/v1/recall") {
      const items = projectMessages.get(value.project_id) || [];
      const content = items
        .filter((item) => item.role === "assistant")
        .map((item) => item.content)
        .join("\n");
      return json(response, 200, { prompt_evidence: { content } });
    }
    if (request.method === "GET" && request.url === "/v1/projects") {
      return json(response, 200, { projects: [] });
    }
    return json(response, 404, { detail: "not found" });
  });
  await new Promise<void>((accept, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", accept);
  });
  const address = server.address();
  if (!address || typeof address === "string") throw new Error("mock server did not bind");
  return {
    baseUrl: `http://127.0.0.1:${address.port}`,
    calls,
    close: () => new Promise<void>((accept, reject) =>
      server.close((error) => error ? reject(error) : accept())),
  };
}

async function testContext(adapter: RecordingAdapter, config: Config) {
  const context = new Context();
  await mountAgentLoopTestDependencies(context);
  await context.plugin(AgentLoop, { agents: [] });
  context.llm.registerAdapter(["mock"], adapter);
  await context.plugin({ name: "tmcra-local-test", inject: ["agents"], apply }, config);
  return context;
}

function send(agent: ReturnType<Context["agentLoop"]["create"]>, text: string): void {
  agent.followup(createUserMessage({
    content: [{ type: "text", text }],
    source: { kind: "user" },
  }));
}

function requestText(options: GenerateOptions): string {
  return options.messages.flatMap((message) => message.content)
    .flatMap((block) => block.type === "text" ? [block.text] : [])
    .join("\n");
}

describe("TMCRA owner-local DeepSeek Harness plugin", () => {
  it("recalls across Harness sessions and keeps user/agent provenance separate", async () => {
    const root = await mkdtemp(join(tmpdir(), "tmcra-dsh-local-"));
    temporaryDirectories.push(root);
    const workspace = join(root, "workspace");
    const tokenFile = join(root, "local-api.token");
    const configPath = join(root, "local-integration.json");
    const token = "dsh-local-test-token-value-12345"; // public-audit: allow-test-fixture
    await mkdir(workspace, { recursive: true });
    await writeFile(tokenFile, token, "utf8");
    const server = await startMemoryServer(token);
    await writeFile(configPath, JSON.stringify({
      schemaVersion: 1,
      baseUrl: server.baseUrl,
      tokenFile,
      stateDir: join(root, "state"),
      topK: 8,
      userVisibility: "both",
      timeoutMs: 5_000,
    }), "utf8");
    const adapter = new RecordingAdapter([
      "The parser migration is complete.",
      "Continuing from the parser migration.",
    ]);
    const context = await testContext(adapter, { configPath });
    try {
      const first = context.agentLoop.create(
        SessionId("dsh-local-session-a"),
        { provider: "mock", model: "mock" },
        { cwd: workspace },
      );
      send(first, "Finish the parser migration");
      await first.whenIdle();

      const second = context.agentLoop.create(
        SessionId("dsh-local-session-b"),
        { provider: "mock", model: "mock" },
        { cwd: workspace },
      );
      send(second, "Continue the work from the other conversation");
      await second.whenIdle();
      await context.fiber.dispose();

      expect(adapter.requests).toHaveLength(2);
      expect(requestText(adapter.requests[1]!)).toContain("The parser migration is complete.");
      expect(requestText(adapter.requests[1]!)).toContain('trust="untrusted"');
      const writes = server.calls.filter((call) => call.path === "/v1/messages");
      expect(writes.map((call) => call.body.role)).toEqual([
        "user",
        "assistant",
        "user",
        "assistant",
      ]);
      expect(writes[0]!.body.visibility).toBe("both");
      expect(writes[1]!.body.visibility).toBe("project");
      expect(writes[0]!.body.actor.actor_role).toBe("user");
      expect(writes[1]!.body.actor.actor_role).toBe("assistant");
      expect(writes[0]!.body.project_id).toBe(writes[2]!.body.project_id);
      expect(writes[0]!.body.session_id).not.toBe(writes[2]!.body.session_id);
      expect(server.calls.every((call) => call.authorization === `Bearer ${token}`)).toBe(true);
    } finally {
      await context.fiber.dispose();
      await server.close();
    }
  });
});
