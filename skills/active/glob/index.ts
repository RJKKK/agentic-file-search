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
    async glob(input, context) {
      let directory = String(input.directory ?? "");
      const pattern = String(input.pattern ?? "page-*.md") || "page-*.md";
      const scopeMode = !directory || ["scope", "selected_scope"].includes(directory.trim().toLowerCase());
      if (scopeMode) {
        const pageScopes = await context.services.indexSearch?.listPageScopes();
        if (!pageScopes) {
          return { output: "glob scope mode requires an indexed document scope." };
        }
        if (pageScopes.length === 0) {
          return { output: "No page scopes found for the selected documents." };
        }
        const lines = ["PAGE SCOPES for selected documents:", ""];
        for (const item of pageScopes) {
          const pageRange = item.page_range
            ? `${item.page_range.start}-${item.page_range.end}`
            : "-";
          lines.push(
            `- doc_id=${item.doc_id} filename=${item.filename} pages_dir=${item.pages_dir} ` +
              `page_count=${item.page_count} page_range=${pageRange}`,
          );
        }
        context.services.indexSearch?.emit("page_scope_resolved", {
          documents: pageScopes,
        });
        return {
          output: lines.join("\n"),
          receipt:
            `Glob receipt: selected_scope; documents=${pageScopes.length}; ` +
            `pattern=${pattern}; page scopes are ready for cross-document grep/search_candidates/read.`,
        };
      }
      if (!pattern) {
        return { output: "glob requires pattern." };
      }
      const pageScope = context.services.indexSearch?.resolveDocumentPageScope(directory);
      if (pageScope) {
        directory = pageScope.pagesDir;
        context.contextState.setActiveScope({
          documentId: pageScope.document.id,
          filePath: pageScope.pagesDir,
          ranges: [],
        });
        context.services.indexSearch?.emit("page_scope_resolved", {
          document_id: pageScope.document.id,
          original_filename:
            pageScope.document.original_filename ||
            pageScope.document.relative_path ||
            pageScope.document.id,
          pages_dir: pageScope.pagesDir,
          page_count: pageScope.document.page_count || 0,
        });
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
        receipt: pageScope
          ? `Glob receipt: document=${pageScope.document.original_filename || pageScope.document.id}; pages_dir=${directory}; pattern=${pattern}.`
          : `Glob receipt: directory=${directory}; pattern=${pattern}.`,
      };
    },
  },
};
