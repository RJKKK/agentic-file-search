/*
Reference: legacy/python/src/fs_explorer/model_config.py
*/

const DEFAULT_TEXT_MODEL = "gpt-4o-mini";
const DEFAULT_VISION_MODEL = "gpt-4o-mini";
const DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small";

export interface ModelEndpointConfig {
  modelName: string;
  apiKey: string | null;
  baseUrl: string | null;
}

export function isConfigured(config: ModelEndpointConfig): boolean {
  return Boolean(config.apiKey && config.modelName);
}

function firstNonEmpty(...values: Array<string | null | undefined>): string | null {
  for (const value of values) {
    const stripped = value?.trim();
    if (stripped) {
      return stripped;
    }
  }
  return null;
}

export function resolveTextConfig(input: Partial<ModelEndpointConfig> = {}): ModelEndpointConfig {
  return {
    modelName:
      firstNonEmpty(
        input.modelName,
        process.env.FS_EXPLORER_TEXT_MODEL,
        process.env.TEXT_MODEL_NAME,
        DEFAULT_TEXT_MODEL,
      ) ?? DEFAULT_TEXT_MODEL,
    apiKey: firstNonEmpty(
      input.apiKey,
      process.env.FS_EXPLORER_TEXT_API_KEY,
      process.env.TEXT_API_KEY,
      process.env.OPENAI_API_KEY,
    ),
    baseUrl: firstNonEmpty(
      input.baseUrl,
      process.env.FS_EXPLORER_TEXT_BASE_URL,
      process.env.TEXT_BASE_URL,
      process.env.OPENAI_BASE_URL,
    ),
  };
}

export function resolveVisionConfig(input: Partial<ModelEndpointConfig> = {}): ModelEndpointConfig {
  const textConfig = resolveTextConfig();
  return {
    modelName:
      firstNonEmpty(
        input.modelName,
        process.env.FS_EXPLORER_VISION_MODEL,
        process.env.VISION_MODEL_NAME,
        textConfig.modelName,
        DEFAULT_VISION_MODEL,
      ) ?? DEFAULT_VISION_MODEL,
    apiKey: firstNonEmpty(
      input.apiKey,
      process.env.FS_EXPLORER_VISION_API_KEY,
      process.env.VISION_API_KEY,
      textConfig.apiKey,
    ),
    baseUrl: firstNonEmpty(
      input.baseUrl,
      process.env.FS_EXPLORER_VISION_BASE_URL,
      process.env.VISION_BASE_URL,
      textConfig.baseUrl,
    ),
  };
}

export function resolveEmbeddingConfig(
  input: Partial<ModelEndpointConfig> = {},
): ModelEndpointConfig {
  const textConfig = resolveTextConfig();
  return {
    modelName:
      firstNonEmpty(
        input.modelName,
        process.env.FS_EXPLORER_EMBEDDING_MODEL,
        process.env.EMBEDDING_MODEL_NAME,
        DEFAULT_EMBEDDING_MODEL,
      ) ?? DEFAULT_EMBEDDING_MODEL,
    apiKey: firstNonEmpty(
      input.apiKey,
      process.env.FS_EXPLORER_EMBEDDING_API_KEY,
      process.env.EMBEDDING_API_KEY,
      textConfig.apiKey,
    ),
    baseUrl: firstNonEmpty(
      input.baseUrl,
      process.env.FS_EXPLORER_EMBEDDING_BASE_URL,
      process.env.EMBEDDING_BASE_URL,
      textConfig.baseUrl,
    ),
  };
}

export function resolveLangextractConfig(
  input: Partial<ModelEndpointConfig> = {},
): ModelEndpointConfig {
  const textConfig = resolveTextConfig();
  return {
    modelName:
      firstNonEmpty(
        input.modelName,
        process.env.FS_EXPLORER_LANGEXTRACT_MODEL,
        process.env.LANGEXTRACT_MODEL_NAME,
        textConfig.modelName,
        DEFAULT_TEXT_MODEL,
      ) ?? DEFAULT_TEXT_MODEL,
    apiKey: firstNonEmpty(
      input.apiKey,
      process.env.FS_EXPLORER_LANGEXTRACT_API_KEY,
      process.env.LANGEXTRACT_API_KEY,
      textConfig.apiKey,
    ),
    baseUrl: firstNonEmpty(
      input.baseUrl,
      process.env.FS_EXPLORER_LANGEXTRACT_BASE_URL,
      process.env.LANGEXTRACT_BASE_URL,
      textConfig.baseUrl,
    ),
  };
}

export function configuredTextCosts(): { inputCostPerMillion: number; outputCostPerMillion: number } {
  return {
    inputCostPerMillion: Number(process.env.TEXT_INPUT_COST_PER_MILLION || "0"),
    outputCostPerMillion: Number(process.env.TEXT_OUTPUT_COST_PER_MILLION || "0"),
  };
}
