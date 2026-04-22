import assert from "node:assert/strict";
import { readdir, readFile } from "node:fs/promises";
import { join } from "node:path";
import { describe, it } from "node:test";

async function collectTsFiles(root: string): Promise<string[]> {
  const entries = await readdir(root, { withFileTypes: true });
  const files: string[] = [];
  for (const entry of entries) {
    const current = join(root, entry.name);
    if (entry.isDirectory()) {
      files.push(...(await collectTsFiles(current)));
      continue;
    }
    if (entry.isFile() && current.endsWith(".ts")) {
      files.push(current);
    }
  }
  return files;
}

describe("reference comments", () => {
  it("ensures every new node source file contains a Reference header", async () => {
    const files = await collectTsFiles(join(process.cwd(), "src"));
    for (const file of files) {
      const content = await readFile(file, "utf8");
      assert.match(content, /Reference:/);
      assert.match(content, /legacy\/python\//);
    }
  });
});
