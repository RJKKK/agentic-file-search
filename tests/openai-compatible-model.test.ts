import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  createDefaultActionModel,
  loadEnvFile,
  OpenAICompatibleActionModel,
  resolveTextConfig,
} from "../src/index.js";

describe("openai-compatible action model", () => {
  it("resolves text config with legacy environment variable precedence", () => {
    const previous = {
      fsKey: process.env.FS_EXPLORER_TEXT_API_KEY,
      textKey: process.env.TEXT_API_KEY,
      openaiKey: process.env.OPENAI_API_KEY,
      fsModel: process.env.FS_EXPLORER_TEXT_MODEL,
      textModel: process.env.TEXT_MODEL_NAME,
      fsBase: process.env.FS_EXPLORER_TEXT_BASE_URL,
      textBase: process.env.TEXT_BASE_URL,
    };
    process.env.FS_EXPLORER_TEXT_API_KEY = "fs-key";
    process.env.TEXT_API_KEY = "text-key";
    process.env.OPENAI_API_KEY = "openai-key";
    process.env.FS_EXPLORER_TEXT_MODEL = "fs-model";
    process.env.TEXT_MODEL_NAME = "text-model";
    process.env.FS_EXPLORER_TEXT_BASE_URL = "https://fs.example/v1";
    process.env.TEXT_BASE_URL = "https://text.example/v1";

    const config = resolveTextConfig();
    assert.equal(config.apiKey, "fs-key");
    assert.equal(config.modelName, "fs-model");
    assert.equal(config.baseUrl, "https://fs.example/v1");

    for (const [key, value] of Object.entries(previous)) {
      const envKey = {
        fsKey: "FS_EXPLORER_TEXT_API_KEY",
        textKey: "TEXT_API_KEY",
        openaiKey: "OPENAI_API_KEY",
        fsModel: "FS_EXPLORER_TEXT_MODEL",
        textModel: "TEXT_MODEL_NAME",
        fsBase: "FS_EXPLORER_TEXT_BASE_URL",
        textBase: "TEXT_BASE_URL",
      }[key]!;
      if (value == null) {
        delete process.env[envKey];
      } else {
        process.env[envKey] = value;
      }
    }
  });

  it("calls OpenAI-compatible chat completions and extracts action text", async () => {
    const calls: unknown[] = [];
    const previousFetch = globalThis.fetch;
    globalThis.fetch = (async (_url: string | URL | Request, init?: RequestInit) => {
      calls.push(JSON.parse(String(init?.body)));
      return new Response(
        JSON.stringify({
          choices: [
            {
              message: {
                content: '{"action":{"final_result":"done"},"reason":"ok"}',
              },
            },
          ],
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      );
    }) as typeof fetch;
    try {
      const model = new OpenAICompatibleActionModel({
        modelName: "gpt-test",
        apiKey: "key",
        baseUrl: "https://example.test/v1",
      });
      const action = await model.generateAction({
        systemPrompt: "system",
        messages: [{ role: "user", content: "task" }],
        task: "task",
        availableTools: [],
      });
      assert.equal(action, '{"action":{"final_result":"done"},"reason":"ok"}');
      assert.equal((calls[0] as { model: string }).model, "gpt-test");
    } finally {
      globalThis.fetch = previousFetch;
    }
  });

  it("streams final-answer chunks from an OpenAI-compatible SSE response", async () => {
    const previousFetch = globalThis.fetch;
    globalThis.fetch = (async (_url: string | URL | Request, init?: RequestInit) => {
      const payload = JSON.parse(String(init?.body));
      if (payload.stream) {
        const encoder = new TextEncoder();
        return new Response(
          new ReadableStream({
            start(controller) {
              controller.enqueue(
                encoder.encode(
                  [
                    'data: {"choices":[{"delta":{"content":"Hello"}}]}',
                    "",
                    'data: {"choices":[{"delta":{"content":" world"}}]}',
                    "",
                    "data: [DONE]",
                    "",
                  ].join("\n"),
                ),
              );
              controller.close();
            },
          }),
          { status: 200, headers: { "content-type": "text/event-stream" } },
        );
      }
      return new Response(
        JSON.stringify({
          choices: [
            {
              message: {
                content: '{"action":{"final_result":"done"},"reason":"ok"}',
              },
            },
          ],
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      );
    }) as typeof fetch;
    try {
      const model = new OpenAICompatibleActionModel({
        modelName: "gpt-test",
        apiKey: "key",
        baseUrl: "https://example.test/v1",
      });
      const chunks: string[] = [];
      for await (const chunk of model.streamFinalAnswer!({
        systemPrompt: "system",
        messages: [{ role: "user", content: "task" }],
        task: "task",
        draftAnswer: "draft",
      })) {
        chunks.push(chunk);
      }
      assert.deepEqual(chunks, ["Hello", " world"]);
    } finally {
      globalThis.fetch = previousFetch;
    }
  });

  it("loads .env files without overriding existing variables by default", async () => {
    const previous = process.env.TEXT_API_KEY;
    process.env.TEXT_API_KEY = "existing";
    loadEnvFile({ path: "missing.env" });
    assert.equal(process.env.TEXT_API_KEY, "existing");
    assert.ok(createDefaultActionModel());
    if (previous == null) {
      delete process.env.TEXT_API_KEY;
    } else {
      process.env.TEXT_API_KEY = previous;
    }
  });

  it("fails fast when the text model request exceeds the configured timeout", async () => {
    const previousFetch = globalThis.fetch;
    const previousTimeout = process.env.FS_EXPLORER_TEXT_TIMEOUT_MS;
    globalThis.fetch = (async (_url: string | URL | Request, init?: RequestInit) =>
      new Promise<Response>((_resolve, reject) => {
        const signal = init?.signal;
        if (signal?.aborted) {
          reject(new DOMException("Aborted", "AbortError"));
          return;
        }
        signal?.addEventListener(
          "abort",
          () => reject(new DOMException("Aborted", "AbortError")),
          { once: true },
        );
      })) as typeof fetch;
    process.env.FS_EXPLORER_TEXT_TIMEOUT_MS = "25";
    try {
      const model = new OpenAICompatibleActionModel({
        modelName: "gpt-test",
        apiKey: "key",
        baseUrl: "https://example.test/v1",
      });
      await assert.rejects(
        model.generateAction({
          systemPrompt: "system",
          messages: [{ role: "user", content: "task" }],
          task: "task",
          availableTools: [],
        }),
        /timed out/i,
      );
    } finally {
      globalThis.fetch = previousFetch;
      if (previousTimeout == null) {
        delete process.env.FS_EXPLORER_TEXT_TIMEOUT_MS;
      } else {
        process.env.FS_EXPLORER_TEXT_TIMEOUT_MS = previousTimeout;
      }
    }
  });
});
