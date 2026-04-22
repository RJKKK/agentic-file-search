/*
Reference: legacy/python/src/fs_explorer/fs.py
Reference: legacy/python/src/fs_explorer/page_store.py
*/

import { stat } from "node:fs/promises";

import { parsePageFrontMatter, readTextFile } from "../../../src/runtime/fs-utils.js";
import type { SkillModule } from "../../../src/types/skills.js";

export const skillModule: SkillModule = {
  tools: {
    async read(input, context) {
      const filePath = String(input.file_path ?? "");
      if (!filePath) {
        return { output: "read requires file_path." };
      }

      let info;
      try {
        info = await stat(filePath);
      } catch {
        return { output: `No such file: ${filePath}` };
      }
      if (!info.isFile()) {
        return { output: `No such file: ${filePath}` };
      }

      const indexedPage = await context.services.indexSearch?.findPageByPath(filePath);
      let content: string;
      try {
        content = await readTextFile(filePath);
      } catch (error) {
        return { output: `Error reading ${filePath}: ${String(error)}` };
      }
      const { header, body } = parsePageFrontMatter(content);
      const pageNo = header.page_no ? Number.parseInt(header.page_no, 10) : null;
      const documentId = indexedPage?.document.id ?? String(header.document_id ?? header.original_filename ?? filePath);
      const label = String(
        indexedPage?.document.original_filename ??
          indexedPage?.document.relative_path ??
          header.original_filename ??
          filePath,
      );
      const resolvedPageNo = indexedPage?.page.page_no ?? pageNo;

      // Reference: legacy/python/src/fs_explorer/context_state.py page-level evidence ingestion.
      const parseSummary = context.contextState.ingestParseResult({
        documentId,
        filePath: indexedPage?.page.file_path ?? filePath,
        label,
        units: [
          {
            unit_no: resolvedPageNo,
            source_locator: indexedPage?.page.source_locator ?? header.source_locator ?? null,
            heading: indexedPage?.page.heading ?? header.heading ?? null,
            markdown: indexedPage?.page.markdown ?? (body || content),
          },
        ],
        totalUnits: indexedPage?.document.page_count || null,
        focusHint: null,
        anchor: resolvedPageNo,
        window: 0,
        maxUnits: 1,
      });
      if (indexedPage) {
        context.services.indexSearch?.emit("pages_read", {
          document_id: indexedPage.document.id,
          original_filename:
            indexedPage.document.original_filename ||
            indexedPage.document.relative_path ||
            indexedPage.document.id,
          read_pages: [indexedPage.page.page_no],
        });
        context.services.indexSearch?.emit("context_scope_updated", {
          context_scope: context.contextState.snapshot().context_scope,
        });
      }
      const renderedRanges = Array.isArray(parseSummary.returned_ranges)
        ? parseSummary.returned_ranges
            .map((item) => {
              const range = item as { start: number; end: number };
              return range.start === range.end ? `${range.start}` : `${range.start}-${range.end}`;
            })
            .join(", ") || "-"
        : "-";

      if (resolvedPageNo !== null) {
        return {
          output: [
            `=== PAGE ${resolvedPageNo} of ${label} ===`,
            `Heading: ${indexedPage?.page.heading ?? header.heading ?? "-"}`,
            `Source: ${indexedPage?.page.source_locator ?? header.source_locator ?? "-"}`,
            "",
            indexedPage?.page.markdown ?? body,
          ].join("\n").trim(),
          receipt: `Parse receipt: path=${indexedPage?.page.file_path ?? filePath}; returned=${renderedRanges}; new_units=${String(parseSummary.new_units_added ?? 0)}; total_units=${String(parseSummary.total_units ?? "?")}; structured evidence has been stored for the next reasoning step.`,
        };
      }

      return {
        output: content,
        receipt: `Read receipt: path=${filePath}.`,
      };
    },
  },
};
