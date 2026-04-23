/*
Reference: legacy/python/src/fs_explorer/agent.py
Reference: legacy/python/src/fs_explorer/document_pages.py
*/

import type { SkillModule } from "../../../src/types/skills.js";

export const skillModule: SkillModule = {
  tools: {
    async page_boundary_context(input, context) {
      const filePath = String(input.file_path ?? "").trim();
      const documentId = String(input.document_id ?? "").trim();
      const pageNo = Number(input.page_no ?? 0);
      const startPage = Number(input.start_page ?? 0);
      const endPage = Number(input.end_page ?? 0);
      const directionValue = String(input.direction ?? "").trim();

      if (
        directionValue !== "previous" &&
        directionValue !== "next" &&
        directionValue !== "both"
      ) {
        return {
          output: "page_boundary_context requires direction=previous, next, or both.",
        };
      }
      const direction = directionValue;

      const isSinglePage = !!filePath || (documentId && Number.isFinite(pageNo) && pageNo > 0);
      const isRange =
        !filePath &&
        !!documentId &&
        Number.isFinite(startPage) &&
        startPage > 0 &&
        Number.isFinite(endPage) &&
        endPage >= startPage;

      if ((isSinglePage && isRange) || (!isSinglePage && !isRange)) {
        return {
          output:
            "page_boundary_context requires either file_path, document_id with page_no, or document_id with start_page and end_page.",
        };
      }

      const result = await context.services.indexSearch?.getPageBoundaryContext({
        filePath: filePath || null,
        documentId: documentId || null,
        pageNo: Number.isFinite(pageNo) && pageNo > 0 ? pageNo : null,
        startPage: Number.isFinite(startPage) && startPage > 0 ? startPage : null,
        endPage: Number.isFinite(endPage) && endPage >= startPage && startPage > 0 ? endPage : null,
        direction,
      });
      if (!result) {
        return {
          output: "No page boundary context found for that page or range in the selected scope.",
        };
      }

      const boundaryFilePath =
        result.anchor_page?.file_path ||
        result.start_page?.file_path ||
        result.document.absolute_path ||
        result.document.original_filename ||
        result.document.id;
      const summary = context.contextState.ingestPageBoundaryContext({
        documentId: result.document.id,
        filePath: boundaryFilePath,
        label:
          result.document.original_filename || result.document.relative_path || result.document.id,
        mode: result.mode,
        direction: result.direction,
        pageNo: result.anchor_page?.page_no ?? null,
        startPage: result.start_page?.page_no ?? null,
        endPage: result.end_page?.page_no ?? null,
        rendered: result.rendered,
      });
      context.services.indexSearch?.emit("page_boundary_loaded", {
        mode: result.mode,
        document_id: result.document.id,
        original_filename:
          result.document.original_filename || result.document.relative_path || result.document.id,
        direction: result.direction,
        anchor_page: result.anchor_page?.page_no ?? null,
        start_page: result.start_page?.page_no ?? null,
        end_page: result.end_page?.page_no ?? null,
        previous_page_no: result.previous_page?.page_no ?? null,
        next_page_no: result.next_page?.page_no ?? null,
        missing: result.missing,
      });

      const pageLabel =
        result.mode === "range"
          ? `pages=${result.start_page?.page_no}-${result.end_page?.page_no}`
          : `page=${result.anchor_page?.page_no ?? result.start_page?.page_no}`;

      return {
        output: result.rendered,
        receipt:
          `Page boundary receipt: document=${result.document.original_filename || result.document.id}; ` +
          `mode=${result.mode}; ${pageLabel}; direction=${result.direction}; ` +
          `${String(summary.summary_for_model ?? "stored boundary evidence for later reasoning.")}`,
      };
    },
  },
};
