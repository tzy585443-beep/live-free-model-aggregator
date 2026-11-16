// Local checks for the aggregator. No live key and no network beyond loopback.
// Covers the catalog bytes, both model-list wires, Anthropic translation,
// prefix restoration, the output cap, JSON-mode ranking, and 429 failover.

import { spawn, spawnSync } from "node:child_process";
import { createServer } from "node:http";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { freeModels, readBrief } from "../lib/brief.ts";
import { buildCatalog } from "../lib/catalog.ts";
import { rankModels } from "../lib/rank.ts";
import { anthropicSse, anthropicToChat, chatBody, openAIJson, parseUpstream, present } from "../lib/translate.ts";

const root = join(fileURLToPath(new URL(".", import.meta.url)), "..");
const ALPHA = "openrouter/alpha/top:free";
const BETA = "openrouter/beta/json:free";

function check(condition, message) {
  if (!condition) throw new Error(message);
}

function readRequest(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on("data", (chunk) => chunks.push(chunk));
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });
}

function fixtureModels() {
  const models = [
    { id: ALPHA, provider: "openrouter", display_name: "Alpha Top", rating_milli: 1800, context_length: 128000, tools: true, vision: true, json: false, free: true },
    { id: BETA, provider: "openrouter", display_name: "Beta Json", rating_milli: 1600, context_length: 64000, tools: true, vision: true, json: true, free: true },
    { id: "openrouter/gamma/json:free", provider: "openrouter", display_name: "Gamma Json", rating_milli: 1500, context_length: 32000, tools: true, vision: false, json: true, free: true },
  ];
  for (let i = 0; i < 11; i += 1) {
    models.push({
      id: `openrouter/vendor/m${i}:free`,
      provider: "openrouter",
      display_name: `Vendor ${i}`,
      rating_milli: 1000 - i,
      context_length: i % 4 === 0 ? null : 16000,
      tools: i % 2 === 0,
      vision: false,
      json: i % 2 === 1,
      free: true,
    });
  }
  return models;
}

function writeBrief(dir) {
  const models = {
    schema: "free-model-aggregator-02/models/v1",
    models: fixtureModels(),
  };
  const prompts = {
    schema: "free-model-aggregator-02/prompts/v1",
    prompts: [
      { id: "p01", api: "openai", stream: false, weight: 1, requires: { tools: false, vision: false, json: true } },
      { id: "p02", api: "openai", stream: true, weight: 2, requires: { tools: false, vision: false, json: false } },
      { id: "p03", api: "anthropic", stream: false, weight: 1, requires: { tools: true, vision: false, json: false } },
    ],
  };
  const text = [
    "free-model-aggregator-02 fixture",
    "",
    "The manifest, in the order the env sends them:",
    "",
    JSON.stringify(prompts),
    "",
    JSON.stringify(models),
    "",
  ].join("\n");
  const path = join(dir, "prompt.txt");
  writeFileSync(path, text);
  return path;
}

function runList(promptFile) {
  const env = { ...process.env };
  if (promptFile === "") delete env.MAKEFUN_PROMPT_FILE;
  else env.MAKEFUN_PROMPT_FILE = promptFile;
  delete env.MAKEFUN_ENV_URL;
  return spawnSync("node", ["main.ts", "--list"], { cwd: root, env, encoding: "utf8" });
}

function idsOf(rows) {
  return [...rows].map((row) => row.id).sort();
}

function unit(briefPath) {
  const empty = runList("");
  check(empty.status === 0, `empty --list exited ${empty.status}: ${empty.stderr}`);
  check(empty.stdout === "{\"providers\":[]}\n", `empty catalog was ${JSON.stringify(empty.stdout)}`);

  const listed = runList(briefPath);
  check(listed.status === 0, `fixture --list exited ${listed.status}: ${listed.stderr}`);
  check(listed.stdout.endsWith("\n"), "--list did not end with a newline");
  const catalog = JSON.parse(listed.stdout);
  check(catalog.providers.length === 1, "catalog providers");
  check(catalog.providers[0].id === "openrouter", "provider id");
  check(catalog.providers[0].base_url === "https://openrouter.ai/api/v1", "public base_url");
  check(catalog.providers[0].key_env === "OPENROUTER_API_KEY", "key_env");
  check(catalog.providers[0].wire === "openai", "wire");
  const ids = catalog.providers[0].models.map((model) => model.id);
  check(ids.length === 14 && new Set(ids).size === 14, `catalog has ${ids.length} ids`);
  check(catalog.providers[0].models.every((model) => model.free === true), "a catalog row is not free");
  check(JSON.stringify(buildCatalog(readBrief(briefPath))) + "\n" === listed.stdout, "--list bytes differ from buildCatalog");

  const brief = readBrief(briefPath);
  const models = freeModels(brief);
  const jsonRank = rankModels(models, { tools: false, vision: false, json: true, minContext: null }, new Set());
  check(jsonRank[0].id === BETA, `JSON prompt ranked ${jsonRank[0].id}, not ${BETA}`);
  check(jsonRank.every((model) => model.json === true), "a json:false model survived a JSON prompt");
  check(!jsonRank.some((model) => model.id === ALPHA), "the higher-rated json:false model was eligible for JSON");
  const plainRank = rankModels(models, { tools: false, vision: false, json: false, minContext: null }, new Set());
  check(plainRank[0].id === ALPHA, `plain prompt ranked ${plainRank[0].id}`);
  const cooledRank = rankModels(models, { tools: false, vision: false, json: false, minContext: null }, new Set([ALPHA]));
  check(cooledRank[0].id === BETA, `cooled plain prompt ranked ${cooledRank[0].id}`);

  const translated = anthropicToChat({
    model: "auto",
    system: "Be exact.",
    max_tokens: 400,
    messages: [{ role: "user", content: "Reply with exactly the word QUILL-1003 and nothing else." }],
    tools: [{ name: "lookup_account", description: "Look up", input_schema: { type: "object", properties: {} } }],
  }, BETA);
  check(translated.model === BETA, "anthropic translation dropped the prefix");
  check(translated.max_tokens === 400, "anthropic max_tokens was not copied");
  check(translated.messages[0].role === "system" && translated.messages[0].content === "Be exact.", "system prompt was not translated");
  check(translated.tools[0].function.name === "lookup_account", "tool was not translated");

  const capped = chatBody({ model: "auto", messages: [{ role: "user", content: "hi" }], max_tokens: 9000 }, BETA);
  check(capped.max_tokens === 8192, `oversize cap became ${capped.max_tokens}`);
  check(capped.model === BETA, "chat body lost the dialled id");
  const filled = chatBody({ model: "auto", messages: [{ role: "user", content: "hi" }] }, BETA);
  check(typeof filled.max_tokens === "number" && filled.max_tokens >= 1 && filled.max_tokens <= 8192, "missing max_tokens was not filled");

  const lifted = present(parseUpstream(JSON.stringify({
    id: "gen-1",
    model: "alpha/top:free",
    choices: [{ index: 0, message: { role: "assistant", content: "", reasoning: "The word is QUILL-1002." }, finish_reason: "stop" }],
    usage: { prompt_tokens: 4, completion_tokens: 0 },
  }), false), ["QUILL-1002"], false, []);
  check(lifted.text.includes("QUILL-1002"), `reasoning was not lifted: ${lifted.text}`);
  check(lifted.completionTokens >= 1, "usage was not repaired");
  const restored = openAIJson(ALPHA, lifted);
  check(restored.model === ALPHA, `model field is ${restored.model}`);
  check(!JSON.stringify(restored).includes("\"model\":\"alpha/top:free\""), "bare provider id leaked into the answer");

  const events = anthropicSse(BETA, { ...lifted, text: "QUILL-1002", id: "gen-1" });
  const names = [...events.matchAll(/^event: (\S+)/gm)].map((match) => match[1]);
  check(names[0] === "message_start" && names.at(-1) === "message_stop", `anthropic events ${names.join(",")}`);
  check(names.includes("content_block_delta") && names.includes("message_delta"), "anthropic stream is missing a delta");
  check(events.includes(`"model":"${BETA}"`) && events.includes("QUILL-1002"), "anthropic stream lost the model or the token");
  check(events.includes('"output_tokens":1') || events.includes('"output_tokens":'), "anthropic stream usage");
}

function listen(server) {
  return new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () => resolve(server.address().port));
  });
}

async function integration(briefPath, catalogText) {
  const hits = [];
  const gateway = createServer(async (req, res) => {
    const raw = await readRequest(req);
    const body = JSON.parse(raw);
    hits.push(body);
    if (body.model === ALPHA) {
      res.writeHead(429, { "content-type": "application/json" });
      res.end(JSON.stringify({ error: { message: "rate limited", type: "rate_limit_error", code: "rate_limit" } }));
      return;
    }
    const token = (raw.match(/[A-Z]{3,12}-[0-9]{4}/) || ["TOKEN-0000"])[0];
    const bare = String(body.model).replace(/^openrouter\//, "");
    const usage = { prompt_tokens: 9, completion_tokens: 2 };
    const text = body.response_format ? JSON.stringify({ token }) : token;
    if (body.stream === true) {
      const frame = (value) => `data: ${JSON.stringify(value)}\n\n`;
      res.writeHead(200, { "content-type": "text/event-stream" });
      res.end(
        frame({ id: "gen-1", model: bare, choices: [{ index: 0, delta: { role: "assistant", content: text }, finish_reason: null }] }) +
          frame({ id: "gen-1", model: bare, choices: [{ index: 0, delta: {}, finish_reason: "stop" }], usage }) +
          "data: [DONE]\n\n",
      );
      return;
    }
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify({
      id: "gen-1",
      model: bare,
      choices: [{ index: 0, message: { role: "assistant", content: text }, finish_reason: "stop" }],
      usage,
    }));
  });
  const gwPort = await listen(gateway);

  let done = false;
  let child = null;
  const envServer = createServer(async (req, res) => {
    const raw = await readRequest(req);
    if (req.url === "/register" && req.method === "POST") {
      const body = JSON.parse(raw);
      check(JSON.stringify(body.catalog) === catalogText, "register catalog is not the --list document");
      check(body.protocol_version === 1 && body.port !== 8080, "register port or protocol");
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify({ ok: true, requests: 3 }));
      const base = `http://127.0.0.1:${body.port}`;
      const openaiList = await (await fetch(`${base}/v1/models`, { headers: { authorization: "Bearer client" } })).json();
      const anthropicList = await (await fetch(`${base}/v1/models`, { headers: { "x-api-key": "client", "anthropic-version": "2023-06-01" } })).json();
      const catalogIds = idsOf(body.catalog.providers[0].models);
      check(JSON.stringify(idsOf(openaiList.data)) === JSON.stringify(catalogIds), "OpenAI model list ids differ");
      check(JSON.stringify(idsOf(anthropicList.data)) === JSON.stringify(catalogIds), "Anthropic model list ids differ");
      check(openaiList.object === "list" && openaiList.data[0].object === "model", "OpenAI list shape");
      check(anthropicList.has_more === false && anthropicList.data[0].type === "model", "Anthropic list shape");

      const jsonRes = await fetch(`${base}/v1/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json", authorization: "Bearer client" },
        body: JSON.stringify({
          model: "auto",
          messages: [{ role: "user", content: 'Reply with exactly one JSON object of the form {"token": "QUILL-1001"} and nothing else.' }],
          max_tokens: 400,
          response_format: { type: "json_object" },
        }),
      });
      const jsonBody = await jsonRes.json();
      check(jsonRes.status === 200, `json prompt status ${jsonRes.status}`);
      check(jsonBody.model === BETA, `json prompt served ${jsonBody.model}`);
      check(jsonBody.choices[0].message.content.includes("QUILL-1001"), "json prompt dropped the token");
      check(jsonBody.usage.completion_tokens >= 1, "json prompt usage");
      check(jsonRes.headers.get("x-makefun-upstream-failures") === null, "a clean prompt carried a failure trail");

      const streamRes = await fetch(`${base}/v1/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json", authorization: "Bearer client", accept: "text/event-stream" },
        body: JSON.stringify({
          model: "auto",
          messages: [{ role: "user", content: "Reply with exactly the word QUILL-1002 and nothing else." }],
          max_tokens: 320,
          stream: true,
        }),
      });
      const sse = await streamRes.text();
      const trail = JSON.parse(streamRes.headers.get("x-makefun-upstream-failures") ?? "null");
      check(streamRes.status === 200, `stream status ${streamRes.status}`);
      check(Array.isArray(trail) && trail[0].status === 429 && trail[0].code === "rate_limit" && trail[0].model === ALPHA, `failure trail ${JSON.stringify(trail)}`);
      check(sse.includes(`"model":"${BETA}"`), "stream did not restore the dialled id");
      check(sse.includes("QUILL-1002") && sse.trimEnd().endsWith("data: [DONE]"), "stream shape");
      check(sse.includes("finish_reason") && sse.indexOf("QUILL-1002") < sse.indexOf("[DONE]"), "stream order");

      const msgRes = await fetch(`${base}/v1/messages`, {
        method: "POST",
        headers: { "content-type": "application/json", "x-api-key": "client", "anthropic-version": "2023-06-01" },
        body: JSON.stringify({
          model: "auto",
          system: "Be exact.",
          messages: [{ role: "user", content: "Do not call any tool. Reply with exactly the word QUILL-1003 and nothing else." }],
          tools: [{ name: "lookup_account", description: "Look up one account.", input_schema: { type: "object", properties: { account_id: { type: "string" } } } }],
        }),
      });
      const msg = await msgRes.json();
      check(msgRes.status === 200, `messages status ${msgRes.status}`);
      check(msg.type === "message" && msg.role === "assistant" && msg.stop_reason === "end_turn", "anthropic shape");
      check(msg.model === BETA, `cooled prompt dialled ${msg.model}`);
      check(msg.content.some((block) => block.type === "text" && block.text.includes("QUILL-1003")), "anthropic text");
      check(msg.usage.output_tokens >= 1 && msg.usage.input_tokens >= 0, "anthropic usage");
      check(!msg.content.some((block) => block.type === "tool_use"), "a tool call leaked into a do-not-call answer");
      done = true;
      return;
    }
    if (req.url === "/work") {
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify({ done }));
      return;
    }
    res.writeHead(404);
    res.end();
  });
  const envPort = await listen(envServer);
  child = spawn("node", ["main.ts"], {
    cwd: root,
    env: {
      ...process.env,
      MAKEFUN_ENV_URL: `http://127.0.0.1:${envPort}`,
      MAKEFUN_PROMPT_FILE: briefPath,
      OPENAI_BASE_URL: `http://127.0.0.1:${gwPort}/v1`,
      OPENAI_API_KEY: "makefun-trial-dummy",
      ANTHROPIC_API_KEY: "makefun-trial-dummy",
    },
    stdio: ["ignore", "inherit", "inherit"],
  });
  const exit = await new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      child.kill("SIGTERM");
      reject(new Error("the aggregator did not exit"));
    }, 8000);
    child.on("exit", (code) => {
      clearTimeout(timer);
      resolve(code);
    });
  });
  gateway.close();
  envServer.close();
  check(exit === 0, `aggregator exited ${exit}`);
  check(hits.length === 4, `gateway saw ${hits.length} calls, expected 4`);
  check(hits[0].model === BETA && hits[0].max_tokens === 400 && hits[0].response_format.type === "json_object", "JSON prompt was not dialled on the best json model");
  check(hits[1].model === ALPHA && hits[1].stream === true, "plain prompt did not try the highest-rated model first");
  check(hits[2].model === BETA && hits[2].stream === true && hits[2].max_tokens === 320, "failover did not stay on Chat Completions with the cap");
  check(hits[3].model === BETA, `cooled model was dialled again: ${hits[3].model}`);
  check(hits[3].messages[0].role === "system" && hits[3].messages[0].content === "Be exact.", "system prompt missing upstream");
  check(hits[3].tools[0].function.name === "lookup_account", "tool missing upstream");
  check(hits[3].max_tokens >= 1 && hits[3].max_tokens <= 8192, "anthropic request had no output cap");
  check(hits.every((hit) => typeof hit.max_tokens === "number"), "a gateway call omitted max_tokens");
  check(!hits.some((hit) => hit.model === "auto"), "auto was forwarded");
}

function practiceTable() {
  const path = "/tmp/makefun-kit/free-model-aggregator-02-kit/practice/trial-0/input/prompt.txt";
  let brief;
  try {
    brief = readBrief(path);
  } catch {
    console.log("practice brief absent, skipped");
    return;
  }
  if (brief.error !== null || brief.models.length === 0) {
    console.log("practice brief absent, skipped");
    return;
  }
  const models = freeModels(brief);
  check(models.length === 14, `practice catalog has ${models.length} free models`);
  const best = (need) => rankModels(models, need, new Set())[0];
  const plain = best({ tools: false, vision: false, json: false, minContext: null });
  const json = best({ tools: false, vision: false, json: true, minContext: null });
  const topPlain = Math.max(...models.map((model) => model.ratingMilli));
  const topJson = Math.max(...models.filter((model) => model.json === true).map((model) => model.ratingMilli));
  check(plain.ratingMilli === topPlain, "practice plain route is not the top rating");
  check(json.ratingMilli === topJson && json.json === true, "practice JSON route is not the top json model");
  check(buildCatalog(brief).providers[0].base_url === "https://openrouter.ai/api/v1", "practice catalog base_url");
  const listed = runList(path);
  check(listed.status === 0 && JSON.parse(listed.stdout).providers[0].models.length === 14, "practice --list");
}

const dir = mkdtempSync(join(tmpdir(), "aggregator-"));
const briefPath = writeBrief(dir);
unit(briefPath);
const catalogText = runList(briefPath).stdout.slice(0, -1);
await integration(briefPath, catalogText);
practiceTable();
console.log("selfcheck ok");
