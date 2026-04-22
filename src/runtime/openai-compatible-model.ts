/*
Reference: legacy/python/src/fs_explorer/openai_compat.py
Reference: legacy/python/src/fs_explorer/model_config.py
*/

import type { ActionModel, ActionModelRequest, FinalAnswerRequest } from "../agent/agent.js";
import {
  isConfigured,
  resolveTextConfig,
  type ModelEndpointConfig,
} from "./model-config.js";

interface ChatCompletionChoice {
  delta?: {
    content?: string | Array<{ type?: string; text?: string }>;
  };
  message?: {
    content?: string | Array<{ type?: string; text?: string }>;
  };
}

interface ChatCompletionResponse {
  choices?: ChatCompletionChoice[];
}

function normalizeBaseUrl(baseUrl: string | null): string {
  const resolved = baseUrl?.trim() || "https://api.openai.com/v1";
  return resolved.replace(/\/+$/, "");
}

function resolveRequestTimeoutMs(): number {
  const raw = Number.parseInt(process.env.FS_EXPLORER_TEXT_TIMEOUT_MS || "", 10);
  if (Number.isFinite(raw) && raw > 0) {
    return raw;
  }
  return 60_000;
}

async function fetchWithTimeout(
  input: string,
  init: RequestInit,
  timeoutMs: number,
): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(input, {
      ...init,
      signal: controller.signal,
    });
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") {
      throw new Error(`Text model request timed out after ${timeoutMs}ms.`);
    }
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

function extractContentText(
  content: string | Array<{ type?: string; text?: string }> | undefined,
  trim: boolean,
): string {
  if (typeof content === "string") {
    return trim ? content.trim() : content;
  }
  if (Array.isArray(content)) {
    const text = content
      .map((item) => (item.type === "text" || item.text ? String(item.text ?? "") : ""))
      .filter(Boolean)
      .join("\n");
    return trim ? text.trim() : text;
  }
  return "";
}

function extractChoiceText(response: ChatCompletionResponse): string {
  return extractContentText(
    response.choices?.[0]?.message?.content ?? response.choices?.[0]?.delta?.content,
    true,
  );
}

function extractDeltaText(response: ChatCompletionResponse): string {
  return extractContentText(
    response.choices?.[0]?.delta?.content ?? response.choices?.[0]?.message?.content,
    false,
  );
}

async function* iterSseDataLines(stream: ReadableStream<Uint8Array>): AsyncIterable<string> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        break;
      }
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
      let separatorIndex = buffer.indexOf("\n\n");
      while (separatorIndex >= 0) {
        const block = buffer.slice(0, separatorIndex);
        buffer = buffer.slice(separatorIndex + 2);
        const dataLines = block
          .split(/\r?\n/)
          .filter((line) => line.startsWith("data:"))
          .map((line) => line.slice(5).trimStart());
        if (dataLines.length > 0) {
          yield dataLines.join("\n");
        }
        separatorIndex = buffer.indexOf("\n\n");
      }
    }
  } finally {
    reader.releaseLock();
  }
}

export class OpenAICompatibleActionModel implements ActionModel {
  constructor(private readonly config: ModelEndpointConfig = resolveTextConfig()) {
    if (!isConfigured(config)) {
      throw new Error(
        "Text model is not configured. Set FS_EXPLORER_TEXT_API_KEY, TEXT_API_KEY, or OPENAI_API_KEY.",
      );
    }
  }

  async generateAction(request: ActionModelRequest): Promise<unknown> {
    const messages = [
      {
        role: "system",
        content: request.systemPrompt,
      },
      ...request.messages.map((message) => ({
        role: message.role === "assistant" ? "assistant" : "user",
        content: message.content,
      })),
    ];

    const payload = {
      model: this.config.modelName,
      messages,
      temperature: 0,
      response_format: { type: "json_object" },
    };
    const endpoint = `${normalizeBaseUrl(this.config.baseUrl)}/chat/completions`;
    const timeoutMs = resolveRequestTimeoutMs();

    let response = await fetchWithTimeout(endpoint, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${this.config.apiKey}`,
      },
      body: JSON.stringify(payload),
    }, timeoutMs);
    if (!response.ok && response.status >= 400) {
      // Reference: legacy/python/src/fs_explorer/openai_compat.py falls back when
      // response_format is unsupported by an OpenAI-compatible endpoint.
      const fallbackPayload = { ...payload };
      delete (fallbackPayload as Partial<typeof payload>).response_format;
      response = await fetchWithTimeout(endpoint, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          authorization: `Bearer ${this.config.apiKey}`,
        },
        body: JSON.stringify(fallbackPayload),
      }, timeoutMs);
    }
    if (!response.ok) {
      const body = await response.text().catch(() => "");
      throw new Error(`Text model request failed: HTTP ${response.status} ${body}`.trim());
    }
    const data = (await response.json()) as ChatCompletionResponse;
    return extractChoiceText(data);
  }

  async *streamFinalAnswer(request: FinalAnswerRequest): AsyncIterable<string> {
    const messages = [
      {
        role: "system",
        content: request.systemPrompt,
      },
      ...request.messages.map((message) => ({
        role: message.role === "assistant" ? "assistant" : "user",
        content: message.content,
      })),
    ];
    const endpoint = `${normalizeBaseUrl(this.config.baseUrl)}/chat/completions`;
    const timeoutMs = resolveRequestTimeoutMs();
    const payload = {
      model: this.config.modelName,
      messages,
      temperature: 0,
      stream: true,
    };

    let response = await fetchWithTimeout(
      endpoint,
      {
        method: "POST",
        headers: {
          "content-type": "application/json",
          authorization: `Bearer ${this.config.apiKey}`,
        },
        body: JSON.stringify(payload),
      },
      timeoutMs,
    );

    if (!response.ok || !response.body) {
      const fallback = await fetchWithTimeout(
        endpoint,
        {
          method: "POST",
          headers: {
            "content-type": "application/json",
            authorization: `Bearer ${this.config.apiKey}`,
          },
          body: JSON.stringify({
            model: this.config.modelName,
            messages,
            temperature: 0,
          }),
        },
        timeoutMs,
      );
      if (!fallback.ok) {
        const body = await fallback.text().catch(() => "");
        throw new Error(`Text model request failed: HTTP ${fallback.status} ${body}`.trim());
      }
      const data = (await fallback.json()) as ChatCompletionResponse;
      const fullText = extractChoiceText(data);
      if (fullText) {
        yield fullText;
      }
      return;
    }

    for await (const dataLine of iterSseDataLines(response.body)) {
      if (!dataLine || dataLine === "[DONE]") {
        if (dataLine === "[DONE]") {
          break;
        }
        continue;
      }
      const chunk = JSON.parse(dataLine) as ChatCompletionResponse;
      const text = extractDeltaText(chunk);
      if (text) {
        yield text;
      }
    }
  }
}

export function createDefaultActionModel(): ActionModel | null {
  const config = resolveTextConfig();
  if (!isConfigured(config)) {
    return null;
  }
  return new OpenAICompatibleActionModel(config);
}
