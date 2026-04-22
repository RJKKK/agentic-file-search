/*
Reference: legacy/python/src/fs_explorer/fs.py
Reference: legacy/python/src/fs_explorer/page_store.py
*/

import { readFile } from "node:fs/promises";
import os from "node:os";

import iconv from "iconv-lite";

export function normalizeText(text: string): string {
  return String(text).split(/\s+/).filter(Boolean).join(" ").trim();
}

export function snippet(text: string, limit = 220): string {
  const normalized = normalizeText(text);
  if (normalized.length <= limit) {
    return normalized;
  }
  return `${normalized.slice(0, Math.max(0, limit - 3)).trimEnd()}...`;
}

function candidateTextEncodings(): string[] {
  const preferred = (() => {
    const locale = Intl.DateTimeFormat().resolvedOptions().locale.toLowerCase();
    if (locale.includes("zh")) {
      return os.platform() === "win32" ? "cp936" : "gbk";
    }
    return "utf-8";
  })();

  const candidates = [
    "utf-8",
    "utf-8-sig",
    preferred,
    "cp936",
    "gbk",
    "gb18030",
    "utf-16le",
    "latin1",
  ];
  return [...new Set(candidates.map((item) => item.trim().toLowerCase()).filter(Boolean))];
}

export async function readTextFile(filePath: string): Promise<string> {
  const raw = await readFile(filePath);
  for (const encoding of candidateTextEncodings()) {
    try {
      const decoded = iconv.decode(raw, encoding);
      if (!decoded.includes("\uFFFD")) {
        return decoded;
      }
    } catch {
      continue;
    }
  }
  return iconv.decode(raw, candidateTextEncodings()[0] ?? "utf-8");
}

export function parsePageFrontMatter(markdown: string): {
  header: Record<string, string>;
  body: string;
} {
  const text = String(markdown ?? "");
  if (!text.startsWith("---\n")) {
    return { header: {}, body: text };
  }
  const marker = text.indexOf("\n---\n", 4);
  if (marker === -1) {
    return { header: {}, body: text };
  }
  const rawHeader = text.slice(4, marker);
  const body = text.slice(marker + 5).replace(/^\n+/, "");
  const header: Record<string, string> = {};
  for (const line of rawHeader.split(/\r?\n/)) {
    const idx = line.indexOf(":");
    if (idx === -1) {
      continue;
    }
    const key = line.slice(0, idx).trim();
    const value = line.slice(idx + 1).trim().replace(/^"(.*)"$/, "$1");
    if (key) {
      header[key] = value;
    }
  }
  return { header, body };
}
