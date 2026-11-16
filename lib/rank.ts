// Candidate order for one prompt. Highest rating_milli that meets the prompt's
// flags wins. A JSON prompt never sees a model whose brief row says json: false,
// even when that model has the best overall rating. Ties keep brief order.

import { isRecord, meets, type Json, type ModelRow, type PromptRow, type Requirement } from "./brief.ts";

export interface Need {
  tools: boolean;
  vision: boolean;
  json: boolean;
  minContext: number | null;
}

function contentHasImage(content: unknown): boolean {
  if (!Array.isArray(content)) return false;
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === "image" || block.type === "image_url") return true;
  }
  return false;
}

/** Flags read off the request itself, used when the brief has no row and as a backstop. */
export function needFromBody(body: Json): Need {
  let vision = false;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (isRecord(message) && contentHasImage(message.content)) vision = true;
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === "json_object" || format === "json_schema",
    minContext: null,
  };
}

/**
 * The brief row is the scored requirement. A flag the body also demands is
 * added, so a JSON response_format is never sent to a model that cannot do it.
 */
export function needFor(row: PromptRow | undefined, body: Json): Need {
  const sniffed = needFromBody(body);
  if (row === undefined) return sniffed;
  const minContext = row.requires.minContext;
  return {
    tools: row.requires.tools === true || sniffed.tools,
    vision: row.requires.vision === true || sniffed.vision,
    json: row.requires.json === true || sniffed.json,
    minContext: minContext === undefined ? null : minContext,
  };
}

function requirementOf(need: Need): Requirement {
  const requires: Requirement = {};
  if (need.tools) requires.tools = true;
  if (need.vision) requires.vision = true;
  if (need.json) requires.json = true;
  if (need.minContext !== null) requires.minContext = need.minContext;
  return requires;
}

/**
 * Eligible free models, best rating first. `cooled` models (a 429 earlier in
 * this trial) are skipped: nothing retries them on a timer.
 */
export function rankModels(models: readonly ModelRow[], need: Need, cooled: ReadonlySet<string>): ModelRow[] {
  const requires = requirementOf(need);
  const ranked: { model: ModelRow; index: number }[] = [];
  models.forEach((model, index) => {
    if (!model.free || cooled.has(model.id)) return;
    if (!meets(requires, model)) return;
    ranked.push({ model, index });
  });
  ranked.sort((a, b) => b.model.ratingMilli - a.model.ratingMilli || a.index - b.index);
  return ranked.map((entry) => entry.model);
}
