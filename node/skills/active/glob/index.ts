/*
Reference: legacy/python/src/fs_explorer/fs.py
Reference: legacy/python/src/fs_explorer/agent.py
*/

import { stat } from "node:fs/promises";
import fg from "fast-glob";

import type { SkillModule } from "../../../src/types/skills.js";
import { resolve } from "node:path";

export const skillModule: SkillModule = {
  tools: {
    async glob(input) {
      const directory = String(input.directory ?? "");
      const pattern = String(input.pattern ?? "");
      if (!directory || !pattern) {
        return { output: "glob requires directory and pattern." };
      }

      try {
        const info = await stat(directory);
        if (!info.isDirectory()) {
          return { output: `No such directory: ${directory}` };
        }
      } catch {
        return { output: `No such directory: ${directory}` };
      }

      const matches = (
        await fg(pattern, {
        cwd: directory,
        absolute: true,
        onlyFiles: false,
        })
      ).sort((left, right) => left.localeCompare(right));
      if (matches.length === 0) {
        return { output: "No matches found", receipt: `Glob receipt: directory=${directory}; pattern=${pattern}.` };
      }

      const pageNumbers = matches
        .map((item) => {
          const match = item.match(/page-(\d+)\.md$/);
          return match ? Number.parseInt(match[1], 10) : null;
        })
        .filter((item): item is number => item !== null);

      let output: string;
      if (pageNumbers.length > 0) {
        output = [
          `PAGES for ${directory}:`,
          `range=${Math.min(...pageNumbers)}-${Math.max(...pageNumbers)}; total=${matches.length}`,
          "",
          ...matches.slice(0, 20).map((item) => `- ${item}`),
          ...(matches.length > 20 ? [`- ... (${matches.length - 20} more)`] : []),
        ].join("\n");
      } else {
        output = [
          `MATCHES for ${pattern} in ${resolve(directory)}:`,
          "",
          ...matches.map((item) => `- ${item}`),
        ].join("\n");
      }

      return {
        output,
        receipt: `Glob receipt: directory=${directory}; pattern=${pattern}.`,
      };
    },
  },
};
