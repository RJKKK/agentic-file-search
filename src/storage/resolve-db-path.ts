/*
Reference: legacy/python/src/fs_explorer/index_config.py
*/

import { resolve } from "node:path";
import { dirname } from "node:path";
import { fileURLToPath } from "node:url";

export const DEFAULT_SQLITE_DB_PATH = "data/agentic-file-search.db";
export const ENV_SQLITE_DB_PATH = "FS_EXPLORER_SQLITE_PATH";
export const ENV_DB_PATH_LEGACY = "FS_EXPLORER_DB_PATH";

function repositoryRootFromModule(): string {
  return resolve(dirname(fileURLToPath(import.meta.url)), "../..");
}

export function resolveSqliteDbPath(overridePath?: string | null): string {
  const explicit = overridePath?.trim();
  if (explicit) {
    return resolve(repositoryRootFromModule(), explicit);
  }

  const envPath = process.env[ENV_SQLITE_DB_PATH]?.trim();
  if (envPath) {
    return resolve(repositoryRootFromModule(), envPath);
  }

  const legacyPath = process.env[ENV_DB_PATH_LEGACY]?.trim();
  if (legacyPath && !legacyPath.includes("://")) {
    return resolve(repositoryRootFromModule(), legacyPath);
  }

  return resolve(repositoryRootFromModule(), DEFAULT_SQLITE_DB_PATH);
}
