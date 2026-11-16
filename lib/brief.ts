// The trial brief is prose wrapped around two JSON documents. This module finds those
// documents and turns them into the model table and the six-prompt manifest. Routing,
// the catalog, and `--list` all read this parse, so it stays a pure function of the file.

import { readFileSync } from "node:fs";

export const MODELS_SCHEMA = "free-model-aggregator-02/models/v1";
export const PROMPTS_SCHEMA = "free-model-aggregator-02/prompts/v1";

const SCHEMA_PREFIX = "free-model-aggregator-02/";

export type Json = Record<string, unknown>;

export function isRecord(value: unknown): value is Json {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

/** One free model the gateway will dial. `id` always carries the `openrouter/` prefix. */
export interface ModelRow {
  id: string;
  provider: string;
  displayName: string;
  ratingMilli: number;
  contextLength: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
  free: boolean;
}

/** Requirements a candidate must meet before it may serve a prompt. */
export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  minContext?: number;
}

export interface PromptRow {
  id: string;
  api: "openai" | "anthropic";
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

export interface Brief {
  models: ModelRow[];
  prompts: PromptRow[];
  error: string | null;
}

function endOfObject(text: string, open: number): number {
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let i = open; i < text.length; i += 1) {
    const ch = text[i];
    if (inString) {
      if (escaped) escaped = false;
      else if (ch === "\\") escaped = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') {
      inString = true;
      continue;
    }
    if (ch === "{") depth += 1;
    else if (ch === "}") {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** JSON objects in `text` that carry a string `schema`, in appearance order. */
export function schemaDocuments(text: string): Json[] {
  const found: Json[] = [];
  for (let start = text.indexOf("{"); start !== -1; start = text.indexOf("{", start + 1)) {
    const end = endOfObject(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === "string") found.push(value);
    } catch {
      // A brace inside prose, or a document cut off: keep scanning.
    }
  }
  return found;
}

function asString(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function asNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function asBool(value: unknown): boolean | null {
  return typeof value === "boolean" ? value : null;
}

function documentWith(docs: readonly Json[], keys: readonly string[]): { doc: Json; key: string } | null {
  for (const doc of docs) {
    if (!asString(doc.schema).includes(SCHEMA_PREFIX)) continue;
    for (const key of keys) {
      if (Array.isArray(doc[key])) return { doc, key };
    }
  }
  return null;
}

/**
 * Gateway id for a brief row. The brief usually spells `openrouter/<upstream>:free`.
 * A bare upstream id paired with provider `openrouter` is prefixed so the allowlist matches.
 */
export function dialableId(raw: string, provider: string): string {
  if (raw.toLowerCase().startsWith("openrouter/")) return raw;
  if (provider === "" || provider === "openrouter") return `openrouter/${raw}`;
  return raw;
}

function modelOf(value: unknown): ModelRow | null {
  if (!isRecord(value)) return null;
  const raw = asString(value.id) !== "" ? asString(value.id) : asString(value.model);
  if (raw === "") return null;
  const named = asString(value.provider) !== "" ? asString(value.provider) : asString(value.wire);
  const provider = named === "" ? "openrouter" : named;
  const id = dialableId(raw, provider);
  const label = asString(value.display_name);
  return {
    id,
    provider,
    displayName: label !== "" ? label : raw,
    ratingMilli: asNumber(value.rating_milli) ?? 0,
    contextLength: asNumber(value.context_length),
    tools: asBool(value.tools),
    vision: asBool(value.vision),
    json: asBool(value.json),
    free: value.free !== false,
  };
}

function requirementOf(value: unknown): Requirement {
  if (!isRecord(value)) return {};
  const requires: Requirement = {};
  if (value.tools === true) requires.tools = true;
  if (value.vision === true) requires.vision = true;
  if (value.json === true) requires.json = true;
  const minContext = asNumber(value.min_context);
  if (minContext !== null) requires.minContext = minContext;
  return requires;
}

function promptOf(value: unknown): PromptRow | null {
  if (!isRecord(value)) return null;
  const id = asString(value.id);
  if (id === "") return null;
  return {
    id,
    api: value.api === "anthropic" ? "anthropic" : "openai",
    stream: value.stream === true,
    weight: value.weight === 2 ? 2 : 1,
    requires: requirementOf(value.requires),
  };
}

function rowsOf<T>(list: unknown, map: (value: unknown) => T | null): T[] {
  if (!Array.isArray(list)) return [];
  const out: T[] = [];
  for (const item of list) {
    const row = map(item);
    if (row !== null) out.push(row);
  }
  return out;
}

/** Read the brief at `path`. An empty path yields an empty brief and no throw. */
export function readBrief(path: string): Brief {
  if (path === "") return { models: [], prompts: [], error: "MAKEFUN_PROMPT_FILE is not set" };
  let text: string;
  try {
    text = readFileSync(path, "utf8");
  } catch (err) {
    return { models: [], prompts: [], error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = schemaDocuments(text);
  const modelsDoc = documentWith(docs, ["models", "ratings"]);
  if (modelsDoc === null) {
    return { models: [], prompts: [], error: `the brief at ${path} holds no ${MODELS_SCHEMA} document` };
  }
  const models = rowsOf(modelsDoc.doc[modelsDoc.key], modelOf);
  const promptsDoc = documentWith(docs, ["prompts", "manifest"]);
  if (promptsDoc === null) {
    return { models, prompts: [], error: `the brief at ${path} holds no ${PROMPTS_SCHEMA} document` };
  }
  return { models, prompts: rowsOf(promptsDoc.doc[promptsDoc.key], promptOf), error: null };
}

/** Free rows, brief order, first occurrence of each id. */
export function freeModels(brief: Brief): ModelRow[] {
  const seen = new Set<string>();
  const out: ModelRow[] = [];
  for (const model of brief.models) {
    if (!model.free || seen.has(model.id)) continue;
    seen.add(model.id);
    out.push(model);
  }
  return out;
}

/**
 * A model meets a requirement only when the brief says it can. A null capability
 * (the brief did not say) does not satisfy a flag the prompt needs.
 */
export function meets(requires: Requirement, model: ModelRow): boolean {
  if (requires.tools === true && model.tools !== true) return false;
  if (requires.vision === true && model.vision !== true) return false;
  if (requires.json === true && model.json !== true) return false;
  if (requires.minContext !== undefined) {
    if (model.contextLength === null || model.contextLength < requires.minContext) return false;
  }
  return true;
}
