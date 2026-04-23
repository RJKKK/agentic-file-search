import assert from "node:assert/strict";
import { mkdtemp, mkdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, it } from "node:test";

import { loadOutdatedSkillManifests, loadSkills } from "../src/runtime/load-skills.js";

describe("skill loader", () => {
  it("loads active skills and excludes outdated skills from the runtime registry", async () => {
    const loaded = await loadSkills(join(process.cwd(), "skills"));
    const names = loaded.flatMap((skill) => skill.manifest.tools.map((tool) => tool.name));

    assert.ok(names.includes("glob"));
    assert.ok(names.includes("read"));
    assert.ok(names.includes("page_boundary_context"));
    assert.ok(names.includes("get_document"));
    assert.ok(!names.includes("scan_folder"));
    assert.ok(!names.includes("semantic_search"));

    const outdated = await loadOutdatedSkillManifests(join(process.cwd(), "skills"));
    assert.ok(outdated.map((item) => item.id).includes("scan-folder-skill"));
    assert.ok(outdated.map((item) => item.id).includes("semantic-search-skill"));
  });

  it("fails fast for duplicate skill ids", async () => {
    const root = await mkdtemp(join(tmpdir(), "afs-duplicate-"));
    await mkdir(join(root, "active", "a"), { recursive: true });
    await mkdir(join(root, "active", "b"), { recursive: true });
    await writeFile(
      join(root, "active", "a", "manifest.json"),
      JSON.stringify({
        id: "dup",
        name: "A",
        status: "active",
        description: "A",
        entry: "./index.mjs",
        tools: [{ name: "one", description: "one", parameters: [] }],
        enabledByDefault: true,
      }),
    );
    await writeFile(
      join(root, "active", "a", "index.mjs"),
      "export const skillModule = { tools: { one: async () => ({ output: 'ok' }) } };",
    );
    await writeFile(
      join(root, "active", "b", "manifest.json"),
      JSON.stringify({
        id: "dup",
        name: "B",
        status: "active",
        description: "B",
        entry: "./index.mjs",
        tools: [{ name: "two", description: "two", parameters: [] }],
        enabledByDefault: true,
      }),
    );
    await writeFile(
      join(root, "active", "b", "index.mjs"),
      "export const skillModule = { tools: { two: async () => ({ output: 'ok' }) } };",
    );

    await assert.rejects(loadSkills(root), /Duplicate skill id detected/);
  });

  it("fails when one manifest tool has no handler export", async () => {
    const root = await mkdtemp(join(tmpdir(), "afs-missing-handler-"));
    await mkdir(join(root, "active", "bad"), { recursive: true });
    await writeFile(
      join(root, "active", "bad", "manifest.json"),
      JSON.stringify({
        id: "bad",
        name: "Bad",
        status: "active",
        description: "Bad",
        entry: "./index.mjs",
        tools: [{ name: "missing", description: "missing", parameters: [] }],
        enabledByDefault: true,
      }),
    );
    await writeFile(
      join(root, "active", "bad", "index.mjs"),
      "export const skillModule = { tools: {} };",
    );

    await assert.rejects(loadSkills(root), /missing a handler/);
  });
});
