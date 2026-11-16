// The catalog document registered with the env and printed by `--list`.
// Both callers stringify the same object, so the bytes match.
// `base_url` is OpenRouter's public origin: the gateway origin changes between
// the live pass and the replay, and the catalog is part of the exported state.

import { freeModels, type Brief, type Json, type ModelRow } from "./brief.ts";

export const OPENROUTER_PUBLIC_BASE = "https://openrouter.ai/api/v1";
export const OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY";

/** Fixed listing timestamps. A clock reading would change the bytes between passes. */
const OPENAI_CREATED = 1767225600;
const ANTHROPIC_CREATED_AT = "2026-01-01T00:00:00Z";

export interface CatalogModel {
  id: string;
  free: true;
  context_length: number | null;
}

export interface CatalogProvider {
  id: "openrouter";
  base_url: string;
  key_env: string;
  wire: "openai";
  models: CatalogModel[];
}

export interface Catalog {
  providers: CatalogProvider[];
}

export interface ListedModel {
  id: string;
  provider: string;
  displayName: string;
}

/** One provider, every free id once, brief order. No models yields an empty provider list. */
export function buildCatalog(brief: Brief): Catalog {
  const models: CatalogModel[] = freeModels(brief).map((model) => ({
    id: model.id,
    free: true,
    context_length: model.contextLength,
  }));
  if (models.length === 0) return { providers: [] };
  return {
    providers: [
      {
        id: "openrouter",
        base_url: OPENROUTER_PUBLIC_BASE,
        key_env: OPENROUTER_KEY_ENV,
        wire: "openai",
        models,
      },
    ],
  };
}

export function listedModels(brief: Brief): ListedModel[] {
  return freeModels(brief).map((model: ModelRow) => ({
    id: model.id,
    provider: model.provider,
    displayName: model.displayName,
  }));
}

/** OpenAI's GET /v1/models list. Ids are the dialable gateway ids. */
export function openAIModelList(models: readonly ListedModel[]): Json {
  return {
    object: "list",
    data: models.map((model) => ({
      id: model.id,
      object: "model",
      created: OPENAI_CREATED,
      owned_by: model.provider,
    })),
  };
}

/** Anthropic's GET /v1/models list, served when the client sends `x-api-key`. */
export function anthropicModelList(models: readonly ListedModel[]): Json {
  return {
    data: models.map((model) => ({
      type: "model",
      id: model.id,
      display_name: model.displayName,
      created_at: ANTHROPIC_CREATED_AT,
    })),
    has_more: false,
  };
}
