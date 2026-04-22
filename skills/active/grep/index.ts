/*
Reference: legacy/python/src/fs_explorer/fs.py
Reference: legacy/python/src/fs_explorer/agent.py
*/

import { stat } from "node:fs/promises";

import fg from "fast-glob";

import { parsePageFrontMatter, readTextFile, snippet } from "../../../src/runtime/fs-utils.js";
import type { SkillModule, ToolExecutionContext } from "../../../src/types/skills.js";

async function grepDirectory(
  directory: string,
  pattern: string,
  regex: RegExp,
  context: ToolExecutionContext,
  options: { ingest?: boolean } = {},
): Promise<string> {
  const candidates = await fg("page-*.md", {
    cwd: directory,
    absolute: true,
    onlyFiles: true,
  });
  const hits: string[] = [];
  const structuredHits: Array<{
    doc_id: string;
    absolute_path: string;
    source_unit_no?: number | null;
    score?: number | null;
    text?: string | null;
  }> = [];
  for (const candidate of candidates) {
    let content: string;
    try {
      content = await readTextFile(candidate);
    } catch {
      continue;
    }
    const { header, body } = parsePageFrontMatter(content);
    const matches = [...body.matchAll(regex)];
    if (matches.length === 0) {
      continue;
    }
    const pageNo = header.page_no ?? candidate;
    const docId = String(header.document_id ?? header.original_filename ?? directory);
    const matchIndex = matches[0]?.index ?? 0;
    const snippetText = body.slice(matchIndex, matchIndex + 220);
    hits.push(
      `- page=${pageNo} hits=${matches.length} path=${candidate} heading=${header.heading ?? "-"}\n  snippet: ${snippet(snippetText)}`,
    );
    structuredHits.push({
      doc_id: docId,
      absolute_path: candidate,
      source_unit_no: header.page_no ? Number.parseInt(header.page_no, 10) : null,
      score: matches.length,
      text: snippetText,
    });
  }
  if (hits.length === 0) {
    return "No matches found";
  }
  if (options.ingest !== false) {
    context.contextState.ingestSearchResults({
      query: pattern,
      filters: null,
      hits: structuredHits,
      limit: Math.min(8, Math.max(1, structuredHits.length)),
    });
  }
  return [`CANDIDATE PAGES for ${pattern} in ${directory}:`, "", ...hits.slice(0, 8)].join("\n");
}

export const skillModule: SkillModule = {
  tools: {
    async grep(input, context) {
      const filePath = String(input.file_path ?? "");
      const pattern = String(input.pattern ?? "");
      if (!filePath || !pattern) {
        return { output: "grep requires file_path and pattern." };
      }

      let regex: RegExp;
      try {
        regex = new RegExp(pattern, "gim");
      } catch (error) {
        return { output: `Invalid regex pattern ${pattern}: ${String(error)}` };
      }

      let info;
      try {
        info = await stat(filePath);
      } catch {
        return { output: `No such file: ${filePath}` };
      }

      const indexedScope = await context.services.indexSearch?.searchPagesForTarget(filePath, pattern);
      if (indexedScope) {
        const rendered = await grepDirectory(indexedScope.pagesDir, pattern, regex, context, {
          ingest: false,
        });
        const summary = context.contextState.ingestSearchResults({
          query: pattern,
          filters: null,
          hits: indexedScope.hits.map((hit) => ({
            doc_id: hit.doc_id,
            absolute_path: hit.absolute_path,
            source_unit_no: hit.source_unit_no,
            score: hit.score,
            text: hit.text,
          })),
          limit: Math.min(8, Math.max(1, indexedScope.hits.length)),
        });
        context.services.indexSearch?.emit("candidate_pages_found", {
          document_id: indexedScope.document.id,
          original_filename:
            indexedScope.document.original_filename ||
            indexedScope.document.relative_path ||
            indexedScope.document.id,
          candidate_pages: indexedScope.hits.map((hit) => ({
            page_no: hit.source_unit_no,
            file_path: hit.absolute_path,
            score: hit.score,
          })),
        });
        context.services.indexSearch?.emit("context_scope_updated", {
          context_scope: context.contextState.snapshot().context_scope,
        });
        return {
          output: rendered,
          receipt:
            `Grep receipt: document=${indexedScope.document.original_filename || indexedScope.document.id}; ` +
            `pattern=${JSON.stringify(pattern)}; candidate_pages=` +
            `${indexedScope.hits.length ? indexedScope.hits.map((hit) => hit.source_unit_no).join(", ") : "-"}; ` +
            `${String(summary.summary_for_model ?? "stored candidate pages for later reasoning.")}`,
        };
      }

      if (info.isDirectory()) {
        const rendered = await grepDirectory(filePath, pattern, regex, context);
        return {
          output: rendered,
          receipt: `Grep receipt: target=${filePath}; pattern=${JSON.stringify(pattern)}.`,
        };
      }

      let content: string;
      try {
        content = await readTextFile(filePath);
      } catch (error) {
        return { output: `Error searching ${filePath}: ${String(error)}` };
      }
      const { header, body } = parsePageFrontMatter(content);
      const haystack = Object.keys(header).length > 0 ? body : content;
      const matches = [...haystack.matchAll(regex)];
      if (matches.length === 0) {
        return {
          output: "No matches found",
          receipt: `Grep receipt: target=${filePath}; pattern=${pattern}.`,
        };
      }
      const snippets = matches
        .slice(0, 5)
        .map((match) => snippet(haystack.slice(Math.max(0, (match.index ?? 0) - 40), (match.index ?? 0) + 220)));

      const output =
        header.page_no
          ? `CANDIDATE PAGE page=${header.page_no} path=${filePath} heading=${header.heading ?? "-"} hits=${matches.length}\n\n- ${snippets.join("\n- ")}`
          : `MATCHES for ${pattern} in ${filePath}:\n\n- ${snippets.join("\n- ")}`;

      return {
        output,
        receipt: `Grep receipt: target=${filePath}; pattern=${pattern}.`,
      };
    },
  },
};
