// Live free-model aggregator.
//
// Listens on an ephemeral port, registers the OpenRouter catalog, and answers
// OpenAI and Anthropic completion routes with a real free model dialled through
// the grader's gateway. Each prompt goes to the highest-rated free model that
// meets its flags. A 4xx/5xx fails over to the next model; a 429 retires that
// model for the rest of the trial. `--list` prints the catalog and exits before
// the server starts. The process leaves when GET /work reports the trial is done.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import { freeModels, isRecord, readBrief, type Brief, type Json } from "./lib/brief.ts";
import { anthropicModelList, buildCatalog, listedModels, openAIModelList, type Catalog } from "./lib/catalog.ts";
import { needFor, rankModels } from "./lib/rank.ts";
import {
  anthropicJson,
  anthropicSse,
  anthropicToChat,
  chatBody,
  answerCarries,
  declaredTools,
  failureOf,
  openAIJson,
  openAISse,
  parseUpstream,
  present,
  tokensOf,
  type Completion,
  type UpstreamFailure,
} from "./lib/translate.ts";

const PROTOCOL_VERSION = 1;
const POLL_MS = 100;
const MAX_BODY_BYTES = 8 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? "").replace(/\/+$/, "");
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? "";
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? "").replace(/\/+$/, "");
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? "";

const brief: Brief = readBrief(PROMPT_FILE);
const catalog: Catalog = buildCatalog(brief);
/** Prompts arrive in brief order. The cursor advances once per completion body. */
let promptCursor = 0;
/** Models that answered 429 or 404. Later prompts skip them; nothing waits on retry-after. */
const cooled = new Set<string>();
/** A 401 is the account, not one model. Further dials would only burn the call cap. */
let authBlocked = false;
/** Per prompt, so a run of refusals cannot walk into the 60-call cap. */
const MAX_ATTEMPTS = 6;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Chat Completions on the gateway. A base that already ends in `/v1` is not doubled. */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === "") return "";
  return GATEWAY_URL.endsWith("/v1") ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

function catalogBytes(): string {
  return JSON.stringify(catalog);
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, {
    "content-type": "application/json",
    "content-length": Buffer.byteLength(text),
  });
  res.end(text);
}

function sendError(res: ServerResponse, wire: "openai" | "anthropic", status: number, type: string, message: string): void {
  if (wire === "anthropic") sendJson(res, status, { type: "error", error: { type, message } });
  else sendJson(res, status, { error: { message, type, code: type } });
}

function wireOf(req: IncomingMessage, pathname: string): "openai" | "anthropic" {
  if (pathname === "/v1/messages") return "anthropic";
  if (typeof req.headers["x-api-key"] === "string") return "anthropic";
  return "openai";
}

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString("utf8"));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** GET /v1/models. Bearer is the OpenAI list; x-api-key is the Anthropic list. */
function serveModels(req: IncomingMessage, res: ServerResponse): void {
  const models = listedModels(brief);
  if (typeof req.headers["x-api-key"] === "string") sendJson(res, 200, anthropicModelList(models));
  else sendJson(res, 200, openAIModelList(models));
}

async function dial(modelId: string, body: Json): Promise<Response> {
  const headers: Record<string, string> = {
    "content-type": "application/json",
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = "text/event-stream";
  return fetch(chatCompletionsUrl(), { method: "POST", headers, body: JSON.stringify(body) });
}

function sendAnswer(
  res: ServerResponse,
  wire: "openai" | "anthropic",
  modelId: string,
  shown: Completion,
  stream: boolean,
  failures: readonly UpstreamFailure[],
): void {
  const extra: Record<string, string> = {};
  if (failures.length > 0) extra["x-makefun-upstream-failures"] = JSON.stringify(failures.slice(0, 8));
  if (stream) {
    const payload = wire === "anthropic" ? anthropicSse(modelId, shown) : openAISse(modelId, shown);
    res.writeHead(200, {
      "content-type": "text/event-stream",
      "cache-control": "no-cache",
      connection: "keep-alive",
      ...extra,
    });
    res.end(payload);
    return;
  }
  const text = JSON.stringify(wire === "anthropic" ? anthropicJson(modelId, shown) : openAIJson(modelId, shown));
  res.writeHead(200, {
    "content-type": "application/json",
    "content-length": Buffer.byteLength(text),
    ...extra,
  });
  res.end(text);
}

/**
 * Dial the best eligible free model. A 4xx/5xx fails over to the next one and
 * is reported on `x-makefun-upstream-failures` when a later model serves the
 * prompt. A 429 or 404 retires that model for the rest of the trial.
 */
async function serveCompletion(res: ServerResponse, wire: "openai" | "anthropic", body: Json): Promise<void> {
  const row = brief.prompts[promptCursor];
  promptCursor += 1;
  const need = needFor(row, body);
  const ranked = rankModels(freeModels(brief), need, cooled);
  if (ranked.length === 0 || authBlocked) {
    sendError(res, wire, 503, wire === "anthropic" ? "overloaded_error" : "server_error", "no free model meets this request");
    return;
  }
  const tokens = tokensOf(body);
  const declared = declaredTools(body, wire);
  const failures: UpstreamFailure[] = [];
  const label = row?.id ?? `request-${promptCursor}`;
  /** The best real answer that did not carry the token. One backup model is enough. */
  let fallback: { id: string; shown: Completion } | null = null;
  let weakBudget = 1;
  let attempts = 0;
  for (const candidate of ranked) {
    if (authBlocked || attempts >= MAX_ATTEMPTS) break;
    attempts += 1;
    const upstreamBody = wire === "anthropic" ? anthropicToChat(body, candidate.id) : chatBody(body, candidate.id);
    let upstream: Response;
    try {
      upstream = await dial(candidate.id, upstreamBody);
    } catch (err) {
      log(`${label} ${candidate.id} unreachable: ${(err as Error).message}`);
      continue;
    }
    const text = await upstream.text();
    if (!upstream.ok) {
      const failure = failureOf(upstream.status, text, candidate.id);
      if (failures.length < 8) failures.push(failure);
      if (upstream.status === 429 || upstream.status === 404 || failure.code === "model_not_allowed") cooled.add(candidate.id);
      if (upstream.status === 401) authBlocked = true;
      log(`${label} ${candidate.id} refused HTTP ${upstream.status} ${failure.code}`);
      continue;
    }
    const parsed = parseUpstream(text, body.stream === true);
    if (parsed === null) {
      log(`${label} ${candidate.id} returned no completion`);
      continue;
    }
    const shown = present(parsed, tokens, need.json, declared);
    if (answerCarries(shown, tokens)) {
      log(`${label} served by ${candidate.id}`);
      sendAnswer(res, wire, candidate.id, shown, body.stream === true, failures);
      return;
    }
    if (fallback === null) fallback = { id: candidate.id, shown };
    if (weakBudget > 0) {
      weakBudget -= 1;
      log(`${label} ${candidate.id} omitted the prompt token`);
      continue;
    }
    break;
  }
  if (fallback !== null) {
    log(`${label} served by ${fallback.id} without a confirmed token`);
    sendAnswer(res, wire, fallback.id, fallback.shown, body.stream === true, []);
    return;
  }
  sendError(
    res,
    wire,
    502,
    wire === "anthropic" ? "api_error" : "upstream_error",
    failures.length === 0 ? "the gateway did not answer" : `every eligible model refused the prompt (${failures.length})`,
  );
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? "/", "http://aggregator.local");
  const pathname = url.pathname;
  if (req.method === "GET" && pathname === "/v1/models") {
    serveModels(req, res);
    return;
  }
  if (req.method === "POST" && (pathname === "/v1/chat/completions" || pathname === "/v1/messages")) {
    const body = await readJson(req);
    const wire = wireOf(req, pathname);
    if (body === null) {
      sendError(res, wire, 400, "invalid_request_error", "the request body is not one JSON object");
      return;
    }
    await serveCompletion(res, wire, body);
    return;
  }
  sendError(res, wireOf(req, pathname), 404, "not_found", `no route for ${req.method ?? "GET"} ${pathname}`);
}

function listen(server: Server): Promise<number> {
  return new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "0.0.0.0", () => {
      const address = server.address();
      if (address === null || typeof address === "string") {
        reject(new Error("the listener has no TCP port"));
        return;
      }
      resolve(address.port);
    });
  });
}

async function register(port: number): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) {
    throw new Error(`POST /register answered HTTP ${res.status}: ${(await res.text()).slice(0, 300)}`);
  }
  const reply: unknown = await res.json();
  const requests = isRecord(reply) && typeof reply.requests === "number" ? reply.requests : 0;
  log(`registered on ${port} for ${requests} prompts, ${listedModels(brief).length} free models`);
}

function pollUntilDone(server: Server): void {
  let busy = false;
  const timer = setInterval(() => {
    if (busy) return;
    busy = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is briefly unreachable between polls. The next tick tries again.
      } finally {
        busy = false;
      }
    })();
  }, POLL_MS);
}

async function writeStdout(text: string): Promise<void> {
  await new Promise<void>((resolve, reject) => {
    process.stdout.write(text, (err) => (err ? reject(err) : resolve()));
  });
}

async function main(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes("--list")) {
    await writeStdout(`${catalogBytes()}\n`);
    process.exit(0);
  }
  if (ENV_URL === "") throw new Error("MAKEFUN_ENV_URL is not set");
  if (chatCompletionsUrl() === "") throw new Error("OPENAI_BASE_URL is not set");
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendError(res, wireOf(req, req.url ?? ""), 500, "internal_error", "the server failed before it could answer");
      }
    });
  });
  const port = await listen(server);
  if (port === 8080) throw new Error("refusing port 8080: the env listens there");
  log(`listening on 0.0.0.0:${port}`);
  await register(port);
  process.on("SIGTERM", () => process.exit(0));
  process.on("SIGINT", () => process.exit(0));
  pollUntilDone(server);
}

try {
  await main();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
