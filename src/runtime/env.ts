/*
Reference: legacy/python/src/fs_explorer/model_config.py
*/

import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

export function loadEnvFile(input: { path?: string; override?: boolean } = {}): void {
  const envPath = resolve(input.path ?? ".env");
  if (!existsSync(envPath)) {
    return;
  }
  const content = readFileSync(envPath, "utf8");
  for (const rawLine of content.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) {
      continue;
    }
    const index = line.indexOf("=");
    if (index <= 0) {
      continue;
    }
    const key = line.slice(0, index).trim();
    let value = line.slice(index + 1).trim();
    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }
    if (input.override || process.env[key] == null) {
      process.env[key] = value;
    }
  }
}
