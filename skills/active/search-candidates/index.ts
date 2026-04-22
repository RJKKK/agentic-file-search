/*
Reference: legacy/python/src/fs_explorer/agent.py
Reference: legacy/python/src/fs_explorer/search/query.py
*/

import { snippet } from "../../../src/runtime/fs-utils.js";
import type { SkillModule } from "../../../src/types/skills.js";

export const skillModule: SkillModule = {
  tools: {
    async search_candidates(input, context) {
      const query = String(input.query ?? "").trim();
      if (!query) {
        return { output: "search_candidates requires query." };
      }
      const limit = Math.max(Number(input.limit ?? 24), 1);
      const maxHitsPerDocument = Math.max(Number(input.max_hits_per_document ?? 5), 1);
      try {
        const result = await context.services.indexSearch?.searchPagesAcrossScope(query, {
          maxTotalHits: limit,
          maxHitsPerDocument,
          regex: false,
        });
        if (!result) {
          return { output: "search_candidates requires an indexed document scope." };
        }
        const summary = context.contextState.ingestSearchResults({
          query,
          filters: null,
          hits: result.hits.map((hit) => ({
            doc_id: hit.doc_id,
            absolute_path: hit.absolute_path,
            source_unit_no: hit.source_unit_no,
            score: hit.score,
            text: hit.text,
          })),
          limit: Math.min(limit, Math.max(1, result.hits.length)),
        });
        const candidatePages = result.hits.map((hit) => ({
          document_id: hit.doc_id,
          page_no: hit.source_unit_no,
          file_path: hit.absolute_path,
          score: hit.score,
        }));
        context.services.indexSearch?.emit("candidate_pages_found", {
          query,
          scope: "selected_scope",
          candidate_pages: candidatePages,
          documents: [...new Set(result.hits.map((hit) => hit.doc_id))],
        });
        context.services.indexSearch?.emit("context_scope_updated", {
          context_scope: context.contextState.snapshot().context_scope,
        });
        if (result.hits.length === 0) {
          return {
            output: "No candidate pages found.",
            receipt: `Search candidates receipt: query=${JSON.stringify(query)}; candidate_pages=-.`,
          };
        }
        return {
          output: [
            `CANDIDATE PAGES for ${query} in selected scope:`,
            "",
            ...result.hits.map(
              (hit) =>
                `- doc_id=${hit.doc_id} page=${hit.source_unit_no} score=${hit.score} path=${hit.absolute_path}\n  snippet: ${snippet(hit.text)}`,
            ),
          ].join("\n"),
          receipt:
            `Search candidates receipt: query=${JSON.stringify(query)}; ` +
            `candidate_pages=${candidatePages.length}; documents=${new Set(result.hits.map((hit) => hit.doc_id)).size}; ` +
            `${String(summary.summary_for_model ?? "stored candidate pages for later reasoning.")}`,
        };
      } catch (error) {
        return { output: `Error searching candidates: ${String(error)}` };
      }
    },
  },
};
