/*
Reference: legacy/python/src/fs_explorer/index_config.py
*/

import { resolve } from "node:path";

export const DEFAULT_SQLITE_DB_PATH = "agentic-file-search.sqlite";
export const ENV_SQLITE_DB_PATH = "FS_EXPLORER_SQLITE_PATH";
export const ENV_DB_PATH_LEGACY = "FS_EXPLORER_DB_PATH";

export function resolveSqliteDbPath(overridePath?: string | null): string {
  const explicit = overridePath?.trim();
  if (explicit) {
    return resolve(explicit);
  }

  const envPath = process.env[ENV_SQLITE_DB_PATH]?.trim();
  if (envPath) {
    return resolve(envPath);
  }

  const legacyPath = process.env[ENV_DB_PATH_LEGACY]?.trim();
  if (legacyPath && !legacyPath.includes("://")) {
    return resolve(legacyPath);
  }

  return resolve(DEFAULT_SQLITE_DB_PATH);
}
