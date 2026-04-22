/*
Reference: legacy/python/src/fs_explorer/blob_store.py
*/

import { mkdir, readdir, readFile, rm, stat, unlink, writeFile } from "node:fs/promises";
import { dirname, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import type { BlobHead, BlobStore } from "../types/library.js";

const DEFAULT_OBJECT_STORE_DIR = "data/object_store";
const FILENAME_SANITIZE_RE = /[^A-Za-z0-9._-]+/g;

function repositoryRootFromModule(): string {
  return resolve(dirname(fileURLToPath(import.meta.url)), "../..");
}

function normalizeObjectKey(objectKey: string): string {
  return String(objectKey).trim().replace(/\\/g, "/").replace(/^\/+/, "");
}

function assertWithinRoot(rootDir: string, targetPath: string): void {
  const rel = relative(rootDir, targetPath);
  if (rel.startsWith("..") || rel === "") {
    if (rel === "") {
      return;
    }
    throw new Error(`Object key escapes object-store root: ${targetPath}`);
  }
}

export function resolveObjectStoreDir(): string {
  const configured = process.env.FS_EXPLORER_OBJECT_STORE_DIR?.trim() || DEFAULT_OBJECT_STORE_DIR;
  return resolve(repositoryRootFromModule(), configured);
}

export function sanitizeFilename(filename: string): string {
  const base = filename.split(/[\\/]/).pop()?.trim() ?? "";
  if (!base) {
    return "document";
  }
  const sanitized = base.replace(FILENAME_SANITIZE_RE, "-").replace(/^[-.]+|[-.]+$/g, "");
  return sanitized || "document";
}

export class LocalBlobStore implements BlobStore {
  readonly rootDir: string;

  constructor(rootDir?: string) {
    this.rootDir = resolve(rootDir || resolveObjectStoreDir());
  }

  private async ensureRoot(): Promise<void> {
    await mkdir(this.rootDir, { recursive: true });
  }

  private resolveKeyPath(objectKey: string): string {
    const safeKey = normalizeObjectKey(objectKey);
    const target = resolve(this.rootDir, safeKey);
    assertWithinRoot(this.rootDir, target);
    return target;
  }

  private async headForPath(objectKey: string, targetPath: string): Promise<BlobHead> {
    const info = await stat(targetPath);
    return {
      objectKey,
      storageUri: `blob://library/default/${objectKey}`,
      size: Number(info.size),
      absolutePath: targetPath,
    };
  }

  private async pruneEmptyParents(startPath: string): Promise<void> {
    let current = startPath;
    while (current !== this.rootDir) {
      try {
        await rm(current, { recursive: false });
      } catch {
        break;
      }
      current = dirname(current);
    }
  }

  async put(input: { objectKey: string; data: Uint8Array }): Promise<BlobHead> {
    await this.ensureRoot();
    const objectKey = normalizeObjectKey(input.objectKey);
    const target = this.resolveKeyPath(objectKey);
    await mkdir(dirname(target), { recursive: true });
    await writeFile(target, input.data);
    return this.headForPath(objectKey, target);
  }

  async get(input: { objectKey: string }): Promise<Uint8Array> {
    const target = this.resolveKeyPath(input.objectKey);
    return readFile(target);
  }

  async materialize(input: { objectKey: string }): Promise<string> {
    const target = this.resolveKeyPath(input.objectKey);
    const info = await stat(target).catch(() => null);
    if (!info?.isFile()) {
      throw new Error(`Blob not found: ${input.objectKey}`);
    }
    return target;
  }

  async delete(input: { objectKey: string }): Promise<boolean> {
    const target = this.resolveKeyPath(input.objectKey);
    const info = await stat(target).catch(() => null);
    if (!info?.isFile()) {
      return false;
    }
    await unlink(target).catch(() => undefined);
    await this.pruneEmptyParents(dirname(target));
    return true;
  }

  async head(input: { objectKey: string }): Promise<BlobHead | null> {
    const objectKey = normalizeObjectKey(input.objectKey);
    const target = this.resolveKeyPath(objectKey);
    const info = await stat(target).catch(() => null);
    if (!info?.isFile()) {
      return null;
    }
    return this.headForPath(objectKey, target);
  }

  async deletePrefix(input: { prefix: string }): Promise<number> {
    const prefix = normalizeObjectKey(input.prefix).replace(/\/+$/, "");
    const target = this.resolveKeyPath(prefix);
    const info = await stat(target).catch(() => null);
    if (!info) {
      return 0;
    }
    if (info.isFile()) {
      await unlink(target).catch(() => undefined);
      await this.pruneEmptyParents(dirname(target));
      return 1;
    }
    const listed = await this.listPrefix({ prefix });
    await rm(target, { recursive: true, force: true });
    await this.pruneEmptyParents(dirname(target));
    return listed.length;
  }

  async listPrefix(input: { prefix: string }): Promise<BlobHead[]> {
    const prefix = normalizeObjectKey(input.prefix).replace(/\/+$/, "");
    const target = this.resolveKeyPath(prefix);
    const info = await stat(target).catch(() => null);
    if (!info) {
      return [];
    }
    if (info.isFile()) {
      return [await this.headForPath(prefix, target)];
    }
    const results: BlobHead[] = [];
    const walk = async (directory: string): Promise<void> => {
      const entries = await readdir(directory, { withFileTypes: true });
      for (const entry of entries) {
        const child = resolve(directory, entry.name);
        if (entry.isDirectory()) {
          await walk(child);
          continue;
        }
        if (!entry.isFile()) {
          continue;
        }
        const objectKey = relative(this.rootDir, child).replace(/\\/g, "/");
        results.push(await this.headForPath(objectKey, child));
      }
    };
    await walk(target);
    results.sort((left, right) => left.objectKey.localeCompare(right.objectKey));
    return results;
  }
}
