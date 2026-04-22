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

      let content: string;
      try {
        content = await readTextFile(filePath);
      } catch (error) {
        return { output: `Error reading ${filePath}: ${String(error)}` };
      }
      const { header, body } = parsePageFrontMatter(content);
      const pageNo = header.page_no ? Number.parseInt(header.page_no, 10) : null;
      const documentId = String(header.document_id ?? header.original_filename ?? filePath);
      const label = String(header.original_filename ?? filePath);

      // Reference: legacy/python/src/fs_explorer/context_state.py page-level evidence ingestion.
      const parseSummary = context.contextState.ingestParseResult({
        documentId,
        filePath,
        label,
        units: [
          {
            unit_no: pageNo,
            source_locator: header.source_locator ?? null,
            heading: header.heading ?? null,
            markdown: body || content,
          },
        ],
        totalUnits: null,
        focusHint: null,
        anchor: pageNo,
        window: 0,
        maxUnits: 1,
      });
      const renderedRanges = Array.isArray(parseSummary.returned_ranges)
        ? parseSummary.returned_ranges
            .map((item) => {
              const range = item as { start: number; end: number };
              return range.start === range.end ? `${range.start}` : `${range.start}-${range.end}`;
            })
            .join(", ") || "-"
        : "-";

      if (pageNo !== null) {
        return {
          output: [
            `=== PAGE ${pageNo} of ${header.original_filename ?? filePath} ===`,
            `Heading: ${header.heading ?? "-"}`,
            `Source: ${header.source_locator ?? "-"}`,
            "",
            body,
          ].join("\n").trim(),
          receipt: `Parse receipt: path=${filePath}; returned=${renderedRanges}; new_units=${String(parseSummary.new_units_added ?? 0)}; total_units=${String(parseSummary.total_units ?? "?")}; structured evidence has been stored for the next reasoning step.`,
        };
      }

      return {
        output: content,
        receipt: `Read receipt: path=${filePath}.`,
      };
    },
  },
};
