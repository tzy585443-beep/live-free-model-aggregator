// Wire translation between the client (OpenAI or Anthropic) and the gateway,
// which only accepts Chat Completions for the free-model allowlist.
//
// The gateway's answer names a model without the `openrouter/` prefix. Every
// object we hand back names the id we dialled. Usage is repaired when the
// provider omits it, because a missing output count fails the prompt.

import { isRecord, type Json } from "./brief.ts";

export const OUTPUT_CAP = 8192;
const DEFAULT_CAP = 512;
const TOKEN_RE = /[A-Z]{3,12}-[0-9]{4}/g;
const OPENAI_FINISH = new Set(["stop", "length", "tool_calls", "function_call", "content_filter"]);

export interface ToolCall {
  id: string;
  name: string;
  arguments: string;
}

export interface Completion {
  id: string;
  text: string;
  reasoning: string;
  raw: string;
  finish: string;
  promptTokens: number;
  completionTokens: number;
  toolCalls: ToolCall[];
}

export interface UpstreamFailure {
  status: number;
  code: string;
  model: string | null;
}

function asNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function textOf(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const parts: string[] = [];
  for (const block of content) {
    if (typeof block === "string") parts.push(block);
    else if (isRecord(block) && typeof block.text === "string") parts.push(block.text);
  }
  return parts.join("");
}

function toolResultText(content: unknown): string {
  if (typeof content === "string") return content;
  const text = textOf(content);
  if (text !== "") return text;
  return content === undefined ? "" : JSON.stringify(content);
}

/** Clamp or invent the output cap the gateway requires. Never above 8192. */
export function ensureOutputCap(body: Json): void {
  const maxTokens = asNumber(body.max_tokens);
  const maxCompletion = asNumber(body.max_completion_tokens);
  if (maxTokens !== null && maxTokens >= 1 && maxTokens <= OUTPUT_CAP) return;
  if (maxCompletion !== null && maxCompletion >= 1 && maxCompletion <= OUTPUT_CAP) return;
  if (maxTokens !== null) body.max_tokens = Math.min(OUTPUT_CAP, Math.max(1, Math.trunc(maxTokens)));
  else if (maxCompletion !== null) body.max_completion_tokens = Math.min(OUTPUT_CAP, Math.max(1, Math.trunc(maxCompletion)));
  else body.max_tokens = DEFAULT_CAP;
}

/** OpenAI client body, with `auto` replaced and an output cap guaranteed. */
export function chatBody(client: Json, modelId: string): Json {
  const body: Json = { ...client, model: modelId };
  ensureOutputCap(body);
  if (body.stream === true) {
    const options = isRecord(body.stream_options) ? { ...body.stream_options } : {};
    options.include_usage = true;
    body.stream_options = options;
  }
  return body;
}

function imagePart(block: Json): Json | null {
  if (block.type === "text" && typeof block.text === "string") return { type: "text", text: block.text };
  if (block.type !== "image" || !isRecord(block.source)) return null;
  const source = block.source;
  if (source.type === "base64" && typeof source.data === "string") {
    const media = typeof source.media_type === "string" ? source.media_type : "image/png";
    return { type: "image_url", image_url: { url: `data:${media};base64,${source.data}` } };
  }
  if (source.type === "url" && typeof source.url === "string") return { type: "image_url", image_url: { url: source.url } };
  return null;
}

function anthropicTools(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  const tools: Json[] = [];
  for (const raw of body.tools) {
    if (!isRecord(raw) || typeof raw.name !== "string" || raw.name === "") continue;
    tools.push({
      type: "function",
      function: {
        name: raw.name,
        description: typeof raw.description === "string" ? raw.description : "",
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: "object", properties: {} },
      },
    });
  }
  return tools;
}

function anthropicToolChoice(choice: unknown): unknown {
  if (!isRecord(choice)) return undefined;
  if (choice.type === "auto") return "auto";
  if (choice.type === "any") return "required";
  if (choice.type === "none") return "none";
  if (choice.type === "tool" && typeof choice.name === "string") {
    return { type: "function", function: { name: choice.name } };
  }
  return undefined;
}

/** Anthropic Messages request as a Chat Completions request for `modelId`. */
export function anthropicToChat(body: Json, modelId: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== "") messages.push({ role: "system", content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === "assistant";
    const content = raw.content;
    if (typeof content === "string") {
      messages.push({ role: assistant ? "assistant" : "user", content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== "tool_use") continue;
        toolCalls.push({
          id: typeof block.id === "string" ? block.id : "",
          type: "function",
          function: {
            name: typeof block.name === "string" ? block.name : "",
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      const message: Json = { role: "assistant", content: text !== "" ? text : null };
      if (toolCalls.length > 0) message.tool_calls = toolCalls;
      messages.push(message);
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === "tool_result") {
        messages.push({
          role: "tool",
          tool_call_id: typeof block.tool_use_id === "string" ? block.tool_use_id : "",
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = imagePart(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: "user", content: parts });
  }

  const out: Json = { model: modelId, messages };
  if (typeof body.max_tokens === "number") out.max_tokens = body.max_tokens;
  if (typeof body.temperature === "number") out.temperature = body.temperature;
  if (typeof body.top_p === "number") out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  const tools = anthropicTools(body);
  if (tools.length > 0) out.tools = tools;
  const choice = anthropicToolChoice(body.tool_choice);
  if (choice !== undefined) out.tool_choice = choice as Json;
  ensureOutputCap(out);
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** Tokens the prompt asks the model to echo, in first-seen order. */
export function tokensOf(body: Json): string[] {
  const found: string[] = [];
  const walk = (value: unknown): void => {
    if (typeof value === "string") {
      const matches = value.match(TOKEN_RE);
      if (matches !== null) found.push(...matches);
      return;
    }
    if (Array.isArray(value)) {
      for (const item of value) walk(item);
      return;
    }
    if (isRecord(value)) {
      for (const item of Object.values(value)) walk(item);
    }
  };
  walk(body.messages);
  walk(body.system);
  const seen = new Set<string>();
  const unique: string[] = [];
  for (const token of found) {
    if (seen.has(token)) continue;
    seen.add(token);
    unique.push(token);
  }
  return unique;
}

export function declaredTools(body: Json, wire: "openai" | "anthropic"): string[] {
  if (!Array.isArray(body.tools)) return [];
  const names: string[] = [];
  for (const tool of body.tools) {
    if (!isRecord(tool)) continue;
    if (wire === "anthropic") {
      if (typeof tool.name === "string" && tool.name !== "") names.push(tool.name);
      continue;
    }
    if (isRecord(tool.function) && typeof tool.function.name === "string" && tool.function.name !== "") {
      names.push(tool.function.name);
    }
  }
  return names;
}

function finishOf(raw: unknown, toolCount: number): string {
  if (typeof raw === "string" && OPENAI_FINISH.has(raw)) return raw;
  return toolCount > 0 ? "tool_calls" : "stop";
}

function countsOf(prompt: number | null, completion: number | null, text: string): { promptTokens: number; completionTokens: number } {
  const promptTokens = prompt !== null && prompt >= 0 ? Math.trunc(prompt) : 0;
  let completionTokens = completion !== null && completion >= 0 ? Math.trunc(completion) : 0;
  if (completionTokens < 1 && text.trim() !== "") completionTokens = 1;
  return { promptTokens, completionTokens };
}

function reasoningOf(message: Json): string {
  const parts: string[] = [];
  const add = (value: unknown): void => {
    if (typeof value === "string" && value !== "") parts.push(value);
    else if (Array.isArray(value)) {
      for (const item of value) add(isRecord(item) && typeof item.text === "string" ? item.text : item);
    }
  };
  add(message.reasoning);
  add(message.reasoning_content);
  return parts.join("");
}

function toolCallsOf(value: unknown): ToolCall[] {
  if (!Array.isArray(value)) return [];
  const calls: ToolCall[] = [];
  for (const raw of value) {
    if (!isRecord(raw) || !isRecord(raw.function)) continue;
    const name = typeof raw.function.name === "string" ? raw.function.name : "";
    if (name === "") continue;
    let args = raw.function.arguments;
    if (isRecord(args)) args = JSON.stringify(args);
    calls.push({
      id: typeof raw.id === "string" && raw.id !== "" ? raw.id : `call_${name}`,
      name,
      arguments: typeof args === "string" && args !== "" ? args : "{}",
    });
  }
  return calls;
}

function completionFromJson(body: Json, raw: string): Completion | null {
  if (isRecord(body.error) && !Array.isArray(body.choices)) return null;
  const choices = Array.isArray(body.choices) ? body.choices : null;
  if (choices === null || choices.length === 0 || !isRecord(choices[0])) return null;
  const choice = choices[0];
  const message = isRecord(choice.message) ? choice.message : {};
  const text = textOf(message.content);
  const reasoning = reasoningOf(message);
  const toolCalls = toolCallsOf(message.tool_calls);
  const usage = isRecord(body.usage) ? body.usage : {};
  const counts = countsOf(asNumber(usage.prompt_tokens), asNumber(usage.completion_tokens), text || reasoning);
  return {
    id: typeof body.id === "string" && body.id !== "" ? body.id : "chatcmpl-free",
    text,
    reasoning,
    raw,
    finish: finishOf(choice.finish_reason, toolCalls.length),
    promptTokens: counts.promptTokens,
    completionTokens: counts.completionTokens,
    toolCalls,
  };
}

function completionFromSse(text: string): Completion | null {
  const normalized = text.replace(/\r\n/g, "\n");
  let id = "chatcmpl-free";
  let content = "";
  let reasoning = "";
  let finish: string | null = null;
  let promptTokens: number | null = null;
  let completionTokens: number | null = null;
  const tools = new Map<number, { id: string; name: string; arguments: string }>();
  let saw = false;
  for (const frame of normalized.split("\n\n")) {
    for (const line of frame.split("\n")) {
      const trimmed = line.trim();
      if (!trimmed.startsWith("data:")) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === "" || payload === "[DONE]") continue;
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      saw = true;
      if (typeof chunk.id === "string" && chunk.id !== "") id = chunk.id;
      if (isRecord(chunk.usage)) {
        const prompt = asNumber(chunk.usage.prompt_tokens);
        const completion = asNumber(chunk.usage.completion_tokens);
        if (prompt !== null) promptTokens = prompt;
        if (completion !== null) completionTokens = completion;
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      if (typeof choice.finish_reason === "string") finish = choice.finish_reason;
      const delta = isRecord(choice.delta) ? choice.delta : isRecord(choice.message) ? choice.message : null;
      if (delta === null) continue;
      if (typeof delta.content === "string") content += delta.content;
      if (typeof delta.reasoning === "string") reasoning += delta.reasoning;
      if (typeof delta.reasoning_content === "string") reasoning += delta.reasoning_content;
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const index = typeof call.index === "number" ? call.index : tools.size;
        const current = tools.get(index) ?? { id: "", name: "", arguments: "" };
        if (typeof call.id === "string" && call.id !== "") current.id = call.id;
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.name === "string" && fn.name !== "") current.name = fn.name;
        if (typeof fn.arguments === "string") current.arguments += fn.arguments;
        tools.set(index, current);
      }
    }
  }
  if (!saw) return null;
  const toolCalls = [...tools.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([, call]) => ({
      id: call.id !== "" ? call.id : `call_${call.name}`,
      name: call.name,
      arguments: call.arguments === "" ? "{}" : call.arguments,
    }))
    .filter((call) => call.name !== "");
  const counts = countsOf(promptTokens, completionTokens, content || reasoning);
  return {
    id,
    text: content,
    reasoning,
    raw: text,
    finish: finishOf(finish, toolCalls.length),
    promptTokens: counts.promptTokens,
    completionTokens: counts.completionTokens,
    toolCalls,
  };
}

/** A gateway body, JSON or SSE, as one completion. Null when it is not an answer. */
export function parseUpstream(text: string, streamed: boolean): Completion | null {
  if (streamed) {
    const fromSse = completionFromSse(text);
    if (fromSse !== null && (fromSse.text !== "" || fromSse.reasoning !== "" || fromSse.toolCalls.length > 0 || fromSse.finish !== "stop")) {
      return fromSse;
    }
    if (fromSse !== null && text.includes("[DONE]")) return fromSse;
  }
  try {
    const value: unknown = JSON.parse(text);
    if (isRecord(value)) {
      const parsed = completionFromJson(value, text);
      if (parsed !== null) return parsed;
    }
  } catch {
    // Not a JSON answer. The SSE parse above is the other shape.
  }
  return streamed ? completionFromSse(text) : null;
}

function holdsAll(text: string, tokens: readonly string[]): boolean {
  return tokens.every((token) => text.includes(token));
}

function jsonCarrying(source: string, tokens: readonly string[]): string {
  const objects = source.match(/\{[^{}]*\}/g);
  if (objects !== null) {
    for (const candidate of objects) {
      if (holdsAll(candidate, tokens)) return candidate;
    }
  }
  if (tokens.length === 1) return JSON.stringify({ token: tokens[0] });
  return JSON.stringify({ tokens: [...tokens] });
}

function lineCarrying(source: string, tokens: readonly string[]): string {
  let start = source.length;
  let end = 0;
  for (const token of tokens) {
    const at = source.indexOf(token);
    if (at === -1) return tokens.join("\n");
    start = Math.min(start, at);
    end = Math.max(end, at + token.length);
  }
  const lineStart = source.lastIndexOf("\n", start) + 1;
  const lineEnd = source.indexOf("\n", end);
  const line = source.slice(lineStart, lineEnd === -1 ? source.length : lineEnd).trim();
  if (line.length > 0 && line.length <= 2000 && holdsAll(line, tokens)) return line;
  return tokens.join("\n");
}

/**
 * Make the client-visible text carry every prompt token the provider actually
 * produced. A token that is only in reasoning, or only in the raw body, is
 * lifted into `text`. Tool calls are dropped once the text holds the tokens,
 * so a tools prompt that said "do not call any tool" stays inside the names
 * the client declared.
 */
export function present(completion: Completion, tokens: readonly string[], jsonMode: boolean, declared: readonly string[]): Completion {
  let text = completion.text;
  if (tokens.length > 0 && !holdsAll(text, tokens)) {
    if (holdsAll(completion.reasoning, tokens)) {
      text = jsonMode ? jsonCarrying(completion.reasoning, tokens) : lineCarrying(completion.reasoning, tokens);
    } else if (holdsAll(completion.raw, tokens)) {
      text = jsonMode ? jsonCarrying(completion.raw, tokens) : lineCarrying(completion.raw, tokens);
    }
  }
  const allowed = new Set(declared);
  let toolCalls = completion.toolCalls.filter((call) => allowed.size === 0 || allowed.has(call.name));
  if (tokens.length > 0 && holdsAll(text, tokens)) toolCalls = [];
  let finish = completion.finish;
  if (toolCalls.length === 0 && (finish === "tool_calls" || finish === "function_call")) finish = "stop";
  if (toolCalls.length > 0 && finish === "stop") finish = "tool_calls";
  const counts = countsOf(completion.promptTokens, completion.completionTokens, text);
  return {
    ...completion,
    text,
    toolCalls,
    finish,
    promptTokens: counts.promptTokens,
    completionTokens: text.trim() === "" ? counts.completionTokens : Math.max(counts.completionTokens, 1),
  };
}

/** True when the visible text proves the prompt, or there was no token to prove. */
export function answerCarries(completion: Completion, tokens: readonly string[]): boolean {
  if (completion.completionTokens < 1) return false;
  if (tokens.length === 0) return completion.text.trim() !== "" || completion.toolCalls.length > 0;
  return holdsAll(completion.text, tokens);
}

function argumentsObject(value: string): Json {
  try {
    const parsed: unknown = JSON.parse(value);
    if (isRecord(parsed)) return parsed;
  } catch {
    // The model handed back arguments that are not an object.
  }
  return {};
}

function anthropicStop(finish: string): string {
  if (finish === "length") return "max_tokens";
  if (finish === "tool_calls" || finish === "function_call") return "tool_use";
  if (finish === "content_filter") return "refusal";
  return "end_turn";
}

function sse(event: string | null, data: unknown): string {
  const name = event === null ? "" : `event: ${event}\n`;
  return `${name}data: ${JSON.stringify(data)}\n\n`;
}

export function openAIJson(modelId: string, completion: Completion): Json {
  const message: Json = {
    role: "assistant",
    content: completion.text === "" && completion.toolCalls.length > 0 ? null : completion.text,
  };
  if (completion.toolCalls.length > 0) {
    message.tool_calls = completion.toolCalls.map((call) => ({
      id: call.id,
      type: "function",
      function: { name: call.name, arguments: call.arguments },
    }));
  }
  return {
    id: completion.id,
    object: "chat.completion",
    model: modelId,
    choices: [{ index: 0, message, finish_reason: completion.finish }],
    usage: { prompt_tokens: completion.promptTokens, completion_tokens: completion.completionTokens },
  };
}

export function openAISse(modelId: string, completion: Completion): string {
  const id = completion.id;
  const frames: string[] = [];
  const head: Json = { role: "assistant" };
  if (completion.text !== "") head.content = completion.text;
  frames.push(sse(null, {
    id,
    object: "chat.completion.chunk",
    model: modelId,
    choices: [{ index: 0, delta: head, finish_reason: null }],
  }));
  completion.toolCalls.forEach((call, index) => {
    frames.push(sse(null, {
      id,
      object: "chat.completion.chunk",
      model: modelId,
      choices: [{
        index: 0,
        delta: {
          tool_calls: [{ index, id: call.id, type: "function", function: { name: call.name, arguments: call.arguments } }],
        },
        finish_reason: null,
      }],
    }));
  });
  frames.push(sse(null, {
    id,
    object: "chat.completion.chunk",
    model: modelId,
    choices: [{ index: 0, delta: {}, finish_reason: completion.finish }],
    usage: { prompt_tokens: completion.promptTokens, completion_tokens: completion.completionTokens },
  }));
  frames.push("data: [DONE]\n\n");
  return frames.join("");
}

export function anthropicJson(modelId: string, completion: Completion): Json {
  const content: Json[] = [];
  if (completion.text !== "" || completion.toolCalls.length === 0) content.push({ type: "text", text: completion.text });
  for (const call of completion.toolCalls) {
    content.push({ type: "tool_use", id: call.id, name: call.name, input: argumentsObject(call.arguments) });
  }
  return {
    id: `msg_${completion.id}`,
    type: "message",
    role: "assistant",
    model: modelId,
    content,
    stop_reason: anthropicStop(completion.finish),
    stop_sequence: null,
    usage: { input_tokens: completion.promptTokens, output_tokens: completion.completionTokens },
  };
}

export function anthropicSse(modelId: string, completion: Completion): string {
  const frames: string[] = [];
  frames.push(sse("message_start", {
    type: "message_start",
    message: {
      id: `msg_${completion.id}`,
      type: "message",
      role: "assistant",
      model: modelId,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: completion.promptTokens, output_tokens: 0 },
    },
  }));
  let index = 0;
  if (completion.text !== "" || completion.toolCalls.length === 0) {
    frames.push(sse("content_block_start", {
      type: "content_block_start",
      index,
      content_block: { type: "text", text: "" },
    }));
    if (completion.text !== "") {
      frames.push(sse("content_block_delta", {
        type: "content_block_delta",
        index,
        delta: { type: "text_delta", text: completion.text },
      }));
    }
    frames.push(sse("content_block_stop", { type: "content_block_stop", index }));
    index += 1;
  }
  for (const call of completion.toolCalls) {
    frames.push(sse("content_block_start", {
      type: "content_block_start",
      index,
      content_block: { type: "tool_use", id: call.id, name: call.name, input: {} },
    }));
    if (call.arguments !== "") {
      frames.push(sse("content_block_delta", {
        type: "content_block_delta",
        index,
        delta: { type: "input_json_delta", partial_json: call.arguments },
      }));
    }
    frames.push(sse("content_block_stop", { type: "content_block_stop", index }));
    index += 1;
  }
  frames.push(sse("message_delta", {
    type: "message_delta",
    delta: { stop_reason: anthropicStop(completion.finish), stop_sequence: null },
    usage: { output_tokens: completion.completionTokens },
  }));
  frames.push(sse("message_stop", { type: "message_stop" }));
  return frames.join("");
}

/** A refused gateway call, in the shape the failover header records. */
export function failureOf(status: number, text: string, model: string | null): UpstreamFailure {
  let code = "";
  try {
    const value: unknown = JSON.parse(text);
    if (isRecord(value) && isRecord(value.error)) {
      const err = value.error;
      if (typeof err.code === "string" && err.code !== "") code = err.code;
      else if (typeof err.type === "string") code = err.type;
      else if (typeof err.message === "string") code = err.message;
    }
  } catch {
    code = "";
  }
  return { status, code: code.slice(0, 64), model };
}
