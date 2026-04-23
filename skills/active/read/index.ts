/*
Reference: legacy/python/src/fs_explorer/fs.py
Reference: legacy/python/src/fs_explorer/page_store.py
*/

import { stat } from "node:fs/promises";

import { parsePageFrontMatter, readTextFile } from "../../../src/runtime/fs-utils.js";
import type { SkillModule } from "../../../src/types/skills.js";

function renderRanges(ranges: unknown): string {
  return Array.isArray(ranges)
    ? ranges
        .map((item) => {
          const range = item as { start: number; end: number };
          return range.start === range.end ? `${range.start}` : `${range.start}-${range.end}`;
        })
        .join(", ") || "-"
    : "-";
}

function toNumberOrNull(value: unknown): number | null {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

export const skillModule: SkillModule = {
  tools: {
    async read(input, context) {
      const batchFilePaths = Array.isArray(input.file_paths)
        ? input.file_paths.map((item) => String(item).trim()).filter(Boolean)
        : [];
      const batchDocumentId = String(input.document_id ?? "").trim();
      if (batchFilePaths.length > 0 || batchDocumentId) {
        const startPage = toNumberOrNull(input.start_page);
        const endPage = toNumberOrNull(input.end_page);
        const requestedRange =
          batchDocumentId && startPage != null
            ? {
                start: startPage,
                end: endPage ?? startPage,
                span: Math.abs((endPage ?? startPage) - startPage) + 1,
              }
            : null;
        const boundaryHint =
          requestedRange != null
            ? "use page_boundary_context(previous|next|both) if the first/last page looks truncated"
            : null;
        const maxPages = Math.min(Math.max(toNumberOrNull(input.max_pages) ?? 3, 1), 5);
        const groups = await context.services.indexSearch?.resolvePageBatch({
          filePaths: batchFilePaths,
          documentId: batchDocumentId || null,
          startPage,
          endPage,
          maxPages,
          maxChars: toNumberOrNull(input.max_chars) ?? 8_000,
        });
        if (!groups) {
          return { output: "batch read requires an indexed document scope." };
        }
        if (groups.every((group) => group.pages.length === 0)) {
          const omitted = groups.flatMap((group) => group.omittedPages);
          return {
            output: "No readable indexed pages found.",
            receipt: `Read receipt: batch; no pages read${omitted.length ? `; omitted=${omitted.join(", ")}` : ""}.`,
          };
        }

        const outputSections: string[] = [];
        const receiptParts: string[] = [];
        for (const group of groups) {
          if (group.pages.length === 0) {
            continue;
          }
          const label =
            group.document.original_filename ||
            group.document.relative_path ||
            group.document.id;
          const parseSummary = context.contextState.ingestParseResult({
            documentId: group.document.id,
            filePath: group.document.absolute_path,
            label,
            units: group.pages.map((page) => ({
              unit_no: page.page_no,
              source_locator: page.source_locator,
              heading: page.heading,
              markdown: page.markdown,
            })),
            totalUnits: group.document.page_count || null,
            focusHint: null,
            anchor: group.pages[0]?.page_no ?? null,
            window: Math.max(0, (group.pages.at(-1)?.page_no ?? group.pages[0]!.page_no) - group.pages[0]!.page_no),
            maxUnits: group.pages.length,
          });
          context.services.indexSearch?.emit("pages_read", {
            document_id: group.document.id,
            original_filename: label,
            read_pages: group.pages.map((page) => page.page_no),
            omitted_pages: group.omittedPages,
            truncated: group.truncated,
            range_start_page: requestedRange?.start ?? null,
            range_end_page: requestedRange?.end ?? null,
            boundary_hint: boundaryHint,
          });
          const returned = renderRanges(parseSummary.returned_ranges);
          const reactivated = Number(parseSummary.new_units_added ?? 0) === 0;
          const wideRangeWarning =
            requestedRange && requestedRange.span > maxPages
              ? `; requested_range=${requestedRange.start}-${requestedRange.end}; WARNING wide range requested (${requestedRange.span} pages). Only ${maxPages} page(s) can be active from one read. Omitted pages were not read and must not be used as evidence. Narrow to the best candidate window (usually <=3 pages), then read more only if the table/text visibly continues`
              : "";
          receiptParts.push(
            `document=${label}; returned=${returned}; new_units=${String(parseSummary.new_units_added ?? 0)}` +
              `${requestedRange ? `; range_start_page=${requestedRange.start}; range_end_page=${requestedRange.end}; boundary_hint=${boundaryHint}` : ""}` +
              `${reactivated ? "; reactivated evidence" : ""}` +
              `${group.omittedPages.length ? `; omitted=${group.omittedPages.join(", ")}` : ""}` +
              wideRangeWarning,
          );
          outputSections.push(
            [
              `=== PAGES ${returned} of ${label} ===`,
              `doc_id: ${group.document.id}`,
              "",
              ...group.pages.map((page) =>
                [
                  `--- PAGE ${page.page_no} ---`,
                  `Heading: ${page.heading ?? "-"}`,
                  `Source: ${page.source_locator ?? "-"}`,
                  "",
                  page.markdown,
                ].join("\n"),
              ),
            ].join("\n").trim(),
          );
        }
        context.services.indexSearch?.emit("context_scope_updated", {
          context_scope: context.contextState.snapshot().context_scope,
        });
        return {
          output: outputSections.join("\n\n"),
          receipt:
            `Parse receipt: batch_read; ${receiptParts.join(" | ")}; ` +
            "structured evidence has been stored for the next reasoning step.",
        };
      }

      const filePath = String(input.file_path ?? "");
      if (!filePath) {
        return { output: "read requires file_path, file_paths, or document_id with start_page/end_page." };
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
      const renderedRanges = renderRanges(parseSummary.returned_ranges);

      if (resolvedPageNo !== null) {
        const singlePageBoundaryHint =
          "use page_boundary_context(previous|next) before reading more pages when this page looks truncated or like part of a continued table/list";
        return {
          output: [
            `=== PAGE ${resolvedPageNo} of ${label} ===`,
            `Heading: ${indexedPage?.page.heading ?? header.heading ?? "-"}`,
            `Source: ${indexedPage?.page.source_locator ?? header.source_locator ?? "-"}`,
            "",
            indexedPage?.page.markdown ?? body,
          ].join("\n").trim(),
          receipt:
            `Parse receipt: path=${indexedPage?.page.file_path ?? filePath}; returned=${renderedRanges}; new_units=${String(parseSummary.new_units_added ?? 0)}; total_units=${String(parseSummary.total_units ?? "?")}; ` +
            `boundary_hint=${singlePageBoundaryHint}; structured evidence has been stored for the next reasoning step.`,
        };
      }

      return {
        output: content,
        receipt: `Read receipt: path=${filePath}.`,
      };
    },
  },
};
