/*
Reference: legacy/python/src/fs_explorer/server.py
*/

import { createRequire } from "node:module";

const CJK_RUN_PATTERN = /[\u4e00-\u9fff]{1,}|[a-z0-9_]{2,}/gi;
const CHINESE_RUN_ONLY_PATTERN = /^[\u4e00-\u9fff]+$/;
const MIN_CHINESE_SUBWORD_LENGTH = 2;
const MAX_CHINESE_SUBWORD_LENGTH = 6;
const require = createRequire(import.meta.url);

interface JiebaLike {
  cutForSearch(sentence: string | Uint8Array, hmm?: boolean | undefined | null): string[];
}

let jiebaTokenizer: JiebaLike | null | undefined;
let jiebaLoadErrorLogged = false;

function normalizeWhitespace(value: string): string {
  return String(value || "").replace(/\r\n/g, "\n").replace(/\n{3,}/g, "\n\n").trim();
}

function uniqueOrdered(values: string[]): string[] {
  const seen = new Set<string>();
  const ordered: string[] = [];
  for (const value of values) {
    const normalized = String(value || "").trim();
    if (!normalized || seen.has(normalized)) {
      continue;
    }
    seen.add(normalized);
    ordered.push(normalized);
  }
  return ordered;
}

function extractBaseTokens(value: string): string[] {
  return uniqueOrdered(String(value || "").toLowerCase().match(CJK_RUN_PATTERN) ?? []);
}

function getJiebaTokenizer(): JiebaLike | null {
  if (jiebaTokenizer !== undefined) {
    return jiebaTokenizer;
  }
  try {
    const jiebaModule = require("@node-rs/jieba") as {
      Jieba: {
        withDict(dict: Uint8Array): JiebaLike;
      };
    };
    const dictModule = require("@node-rs/jieba/dict") as {
      dict: Uint8Array;
    };
    jiebaTokenizer = jiebaModule.Jieba.withDict(dictModule.dict);
    return jiebaTokenizer;
  } catch (error) {
    jiebaTokenizer = null;
    if (!jiebaLoadErrorLogged) {
      jiebaLoadErrorLogged = true;
      console.warn(
        `[chinese-fts] Failed to load @node-rs/jieba; falling back to n-gram tokenization: ${String(error)}`,
      );
    }
    return jiebaTokenizer;
  }
}

function segmentChineseRunWithJieba(token: string): string[] {
  const normalized = String(token || "").trim().toLowerCase();
  if (!CHINESE_RUN_ONLY_PATTERN.test(normalized)) {
    return normalized ? [normalized] : [];
  }
  const jieba = getJiebaTokenizer();
  if (!jieba) {
    return [];
  }
  try {
    return uniqueOrdered(
      jieba
        .cutForSearch(normalized, true)
        .map((item) => String(item || "").trim().toLowerCase())
        .filter((item) => item.length > 1),
    );
  } catch (error) {
    if (!jiebaLoadErrorLogged) {
      jiebaLoadErrorLogged = true;
      console.warn(
        `[chinese-fts] jieba tokenization failed; falling back to n-gram tokenization: ${String(error)}`,
      );
    }
    return [];
  }
}

function expandChineseRun(token: string): string[] {
  const normalized = String(token || "").trim().toLowerCase();
  if (!CHINESE_RUN_ONLY_PATTERN.test(normalized)) {
    return normalized ? [normalized] : [];
  }
  const expanded: string[] = [normalized, ...segmentChineseRunWithJieba(normalized)];
  for (let length = MIN_CHINESE_SUBWORD_LENGTH; length <= MAX_CHINESE_SUBWORD_LENGTH; length += 1) {
    if (normalized.length < length) {
      break;
    }
    for (let start = 0; start + length <= normalized.length; start += 1) {
      expanded.push(normalized.slice(start, start + length));
    }
  }
  return uniqueOrdered(expanded);
}

export function extractChineseFtsQueryTokens(value: string): string[] {
  return uniqueOrdered(extractBaseTokens(value).flatMap((token) => expandChineseRun(token)));
}

export function extractPlainQueryTokens(value: string): string[] {
  return extractBaseTokens(value);
}

export function buildChineseFtsIndexText(value: string): string {
  const normalized = normalizeWhitespace(value);
  if (!normalized) {
    return normalized;
  }
  const expanded = extractChineseFtsQueryTokens(normalized);
  if (expanded.length === 0) {
    return normalized;
  }
  return normalizeWhitespace([normalized, expanded.join(" ")].join("\n"));
}
