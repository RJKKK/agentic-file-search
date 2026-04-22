/*
Reference: legacy/python/src/fs_explorer/agent.py
Reference: legacy/python/src/fs_explorer/models.py
*/

import { readdir, readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

import {
  SkillManifestSchema,
  type LoadedSkill,
  type SkillManifest,
  type SkillModule,
} from "../types/skills.js";

async function loadManifest(manifestPath: string): Promise<SkillManifest> {
  const raw = await readFile(manifestPath, "utf8");
  return SkillManifestSchema.parse(JSON.parse(raw));
}

export async function loadSkills(skillsRoot: string): Promise<LoadedSkill[]> {
  const activeRoot = resolve(skillsRoot, "active");
  let entries;
  try {
    entries = await readdir(activeRoot, { withFileTypes: true });
  } catch {
    return [];
  }

  entries = [...entries].sort((left, right) => left.name.localeCompare(right.name));

  const loaded: LoadedSkill[] = [];
  const seenIds = new Set<string>();

  for (const entry of entries) {
    if (!entry.isDirectory()) {
      continue;
    }
    const directory = resolve(activeRoot, entry.name);
    const manifestPath = resolve(directory, "manifest.json");
    const manifest = await loadManifest(manifestPath);
    if (manifest.status !== "active") {
      throw new Error(`Active skill manifest ${manifest.id} must declare status=active.`);
    }
    if (seenIds.has(manifest.id)) {
      throw new Error(`Duplicate skill id detected: ${manifest.id}`);
    }
    seenIds.add(manifest.id);

    const entryPath = resolve(directory, manifest.entry);
    const imported = (await import(pathToFileURL(entryPath).href)) as {
      skillModule?: SkillModule;
    };
    if (!imported.skillModule || typeof imported.skillModule !== "object") {
      throw new Error(`Skill ${manifest.id} must export skillModule.`);
    }
    for (const tool of manifest.tools) {
      if (typeof imported.skillModule.tools[tool.name] !== "function") {
        throw new Error(
          `Skill ${manifest.id} is missing a handler for manifest tool ${tool.name}.`,
        );
      }
    }
    loaded.push({
      manifest,
      directory,
      module: imported.skillModule,
    });
  }

  return loaded;
}

export async function loadOutdatedSkillManifests(skillsRoot: string): Promise<SkillManifest[]> {
  const outdatedRoot = resolve(skillsRoot, "outdated");
  let entries;
  try {
    entries = await readdir(outdatedRoot, { withFileTypes: true });
  } catch {
    return [];
  }

  entries = [...entries].sort((left, right) => left.name.localeCompare(right.name));

  const manifests: SkillManifest[] = [];
  for (const entry of entries) {
    if (!entry.isDirectory()) {
      continue;
    }
    const manifest = await loadManifest(resolve(outdatedRoot, entry.name, "manifest.json"));
    manifests.push(manifest);
  }
  return manifests;
}
