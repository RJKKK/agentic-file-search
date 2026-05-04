/*
Reference: legacy/python/src/fs_explorer/server.py
Reference: legacy/python/src/fs_explorer/document_library.py
*/

import type {
  ParsedBlock,
  ParsedDocument,
  ParsedUnit,
} from "../types/parsing.js";
import type {
  StorageDocumentChunkRecord,
  StorageFixedRetrievalChunkRecord,
  StorageRetrievalChunkRecord,
} from "../types/storage.js";
import {
  adjustSplitEndForProtectedImageLinks,
  classifySize,
  getChunkThresholds,
  mergeContinuedBlockMarkdown,
  normalizeWhitespace,
  parseJsonArray,
  restorePresentProtectedImageLinks,
  sourceLocatorFromPages,
  stableId,
  uniqueOrdered,
  type ImageChunkRendering,
  type SourceChunkRecord,
} from "./traditional-rag-shared.js";
import { protectImageLinks } from "./image-semantic.js";

interface FixedChunkPiece {
  sourceChunkId: string;
  pageNo: number;
  contentMd: string;
  blockType: string;
  mergedPageNosJson: string;
  mergedBboxesJson: string;
  bboxJson: string;
}

function pageDominatedByType(unit: ParsedUnit, blockType: string): boolean {
  if (!unit.blocks.length) {
    return false;
  }
  const normalized = unit.blocks.filter((block) => normalizeWhitespace(block.markdown));
  if (!normalized.length) {
    return false;
  }
  const matching = normalized.filter((block) => block.block_type === blockType);
  if (!matching.length) {
    return false;
  }
  const matchedChars = matching.reduce((sum, block) => sum + block.markdown.length, 0);
  const totalChars = normalized.reduce((sum, block) => sum + block.markdown.length, 0);
  return matchedChars / Math.max(totalChars, 1) >= 0.7;
}

function ruleSummary(content: string, blockType: string): string {
  const text = normalizeWhitespace(content);
  if (!text) {
    return "";
  }
  if (blockType === "table") {
    const lines = text.split("\n").filter(Boolean).slice(0, 12);
    return lines.join("\n").slice(0, 500);
  }
  const sentences = text.split(/(?<=[。！？?!])\s+/).filter(Boolean);
  if (sentences.length <= 3) {
    return text.slice(0, 500);
  }
  const wordScores = new Map<string, number>();
  for (const token of text.toLowerCase().match(/[\u4e00-\u9fff]{1,}|[a-z0-9_]{2,}/g) ?? []) {
    wordScores.set(token, (wordScores.get(token) ?? 0) + 1);
  }
  const middle = sentences
    .slice(1, -1)
    .map((sentence) => ({
      sentence,
      score: (sentence.toLowerCase().match(/[\u4e00-\u9fff]{1,}|[a-z0-9_]{2,}/g) ?? []).reduce(
        (sum, token) => sum + (wordScores.get(token) ?? 0),
        0,
      ),
    }))
    .sort((left, right) => right.score - left.score)
    .slice(0, 3)
    .map((item) => item.sentence);
  const ordered = [sentences[0], ...middle, sentences[sentences.length - 1]].join(" ");
  return ordered.slice(0, 500);
}

function splitOversized(content: string, blockType: string): string[] {
  const thresholds = getChunkThresholds();
  const text = normalizeWhitespace(content);
  if (text.length <= thresholds.normalMaxChars) {
    return [text];
  }
  const units =
    blockType === "table"
      ? text.split("\n").filter(Boolean)
      : text.split(/\n{2,}|(?<=[。！？?!])\s+/).filter(Boolean);
  const parts: string[] = [];
  let current = "";
  for (const piece of units) {
    const candidate = current ? `${current}\n\n${piece}` : piece;
    if (candidate.length <= thresholds.normalMaxChars) {
      current = candidate;
      continue;
    }
    if (current) {
      parts.push(current);
    }
    if (piece.length <= thresholds.normalMaxChars) {
      current = piece;
      continue;
    }
    for (let index = 0; index < piece.length; index += thresholds.normalMaxChars) {
      parts.push(piece.slice(index, index + thresholds.normalMaxChars));
    }
    current = "";
  }
  if (current) {
    parts.push(current);
  }
  return parts;
}

function splitFixedChunkTextSmart(content: string, maxChars: number): string[] {
  const leadingImageLinks: string[] = [];
  let unwrappedContent = normalizeWhitespace(String(content || ""));
  const leadingImageLinkPattern = /^(?:(?:图片链接：)?!\[[^\]]*]\([^)]+\)(?:\n\n+)?)/;
  while (true) {
    const matched = unwrappedContent.match(leadingImageLinkPattern)?.[0];
    if (!matched) {
      break;
    }
    leadingImageLinks.push(normalizeWhitespace(matched));
    unwrappedContent = unwrappedContent.slice(matched.length).trimStart();
  }
  const protectedResult = protectImageLinks(unwrappedContent);
  const text = normalizeWhitespace(
    protectedResult.protectedBlocks.length > 0
      ? protectedResult.placeholderText
      : unwrappedContent,
  );
  if (text.length <= maxChars) {
    const trailingPart =
      protectedResult.protectedBlocks.length > 0
        ? restorePresentProtectedImageLinks(text, protectedResult.protectedBlocks)
        : text;
    return [...leadingImageLinks, trailingPart].filter((part) => part.length > 0);
  }
  const parts: string[] = [];
  let remaining = text;
  while (remaining.length > maxChars) {
    const slice = remaining.slice(0, maxChars);
    const bestBreakAt = Math.max(
      ...[
        "\n## ",
        "\n### ",
        "\n#### ",
        "\n- ",
        "\n1. ",
        "\n2. ",
        "\n3. ",
        "\n\n",
        "\n",
        "。",
        "？",
        "！",
        ". ",
        "; ",
        " ",
      ].map((marker) => slice.lastIndexOf(marker)),
    );
    const breakAt = bestBreakAt >= 0 ? bestBreakAt : maxChars;
    const provisionalEnd = breakAt > Math.floor(maxChars * 0.5) ? breakAt : maxChars;
    const end = adjustSplitEndForProtectedImageLinks(remaining, provisionalEnd);
    parts.push(slice.slice(0, end).trim());
    remaining = remaining.slice(end).trimStart();
  }
  if (remaining.trim()) {
    parts.push(remaining.trim());
  }
  const normalizedParts = parts.filter((part) => part.length > 0);
  if (protectedResult.protectedBlocks.length === 0) {
    return [...leadingImageLinks, ...normalizedParts];
  }
  return [
    ...leadingImageLinks,
    ...normalizedParts.map((part) =>
      restorePresentProtectedImageLinks(part, protectedResult.protectedBlocks),
    ),
  ];
}

function renderUnitBlocks(
  unit: ParsedUnit,
  imageRenderMap: Map<string, ImageChunkRendering>,
): ParsedBlock[] {
  const blocks = unit.blocks.length
    ? [...unit.blocks]
    : [
        {
          index: 0,
          block_type: "text",
          bbox: [0, 0, 0, 0],
          markdown: unit.markdown,
          char_count: unit.markdown.length,
          image_hash: null,
          source_image_index: null,
        },
      ];
  return blocks
    .map((block) => {
      if (block.block_type !== "picture" || !block.image_hash) {
        return block;
      }
      const rendered = imageRenderMap.get(block.image_hash);
      return {
        ...block,
        markdown: rendered?.dropped ? "" : rendered?.markdown || block.markdown,
        char_count: (rendered?.markdown || block.markdown).length,
      };
    })
    .filter((block) => normalizeWhitespace(block.markdown));
}

export function buildSourceChunks(input: {
  documentId: string;
  parsedDocument: ParsedDocument;
  imageRenderMap: Map<string, ImageChunkRendering>;
}): SourceChunkRecord[] {
  let previousNonPictureContent: string | null = null;
  const perPage = input.parsedDocument.units.map((unit) => ({
    unitNo: unit.unit_no,
    blocks: renderUnitBlocks(unit, input.imageRenderMap),
  }));
  const records: SourceChunkRecord[] = [];
  let documentIndex = 0;
  for (let pageIndex = 0; pageIndex < perPage.length; pageIndex += 1) {
    const page = perPage[pageIndex]!;
    for (let blockIndex = 0; blockIndex < page.blocks.length; blockIndex += 1) {
      const block = page.blocks[blockIndex]!;
      if (!normalizeWhitespace(block.markdown)) {
        continue;
      }
      let mergedContent = block.markdown;
      const mergedPageNos = [page.unitNo];
      const mergedBboxes = [block.bbox];
      let scanPageIndex = pageIndex;
      while (
        ["table", "text"].includes(block.block_type) &&
        scanPageIndex + 1 < perPage.length &&
        blockIndex === page.blocks.length - 1
      ) {
        const nextPage = perPage[scanPageIndex + 1]!;
        const nextHead = nextPage.blocks[0];
        const nextUnit = input.parsedDocument.units[scanPageIndex + 1]!;
        if (
          !nextHead ||
          nextHead.block_type !== block.block_type ||
          !pageDominatedByType(nextUnit, block.block_type)
        ) {
          break;
        }
        mergedContent = mergeContinuedBlockMarkdown(
          mergedContent,
          nextHead.markdown,
          block.block_type,
        );
        mergedPageNos.push(nextPage.unitNo);
        mergedBboxes.push(nextHead.bbox);
        nextPage.blocks = nextPage.blocks.slice(1);
        scanPageIndex += 1;
      }
      const sizeClass = classifySize(mergedContent);
      const record: SourceChunkRecord = {
        id: stableId(
          "schunk",
          `${input.documentId}:${documentIndex}:${page.unitNo}:${block.index}:${mergedContent}`,
        ),
        documentId: input.documentId,
        pageNo: page.unitNo,
        documentIndex,
        pageIndex: block.index,
        blockType: block.block_type,
        bboxJson: JSON.stringify(block.bbox),
        contentMd: mergedContent,
        sizeClass,
        summaryText: sizeClass === "oversized" ? ruleSummary(mergedContent, block.block_type) : null,
        mergedPageNosJson: JSON.stringify(mergedPageNos),
        mergedBboxesJson: JSON.stringify(mergedBboxes.map((bbox) => [...bbox])),
        previousContextMd: block.block_type === "picture" ? previousNonPictureContent : null,
      };
      records.push(record);
      if (block.block_type !== "picture") {
        previousNonPictureContent = mergedContent;
      }
      documentIndex += 1;
    }
  }
  return records;
}

export function buildRetrievalChunks(
  documentId: string,
  documentChunks: SourceChunkRecord[],
): StorageRetrievalChunkRecord[] {
  const thresholds = getChunkThresholds();
  const records: StorageRetrievalChunkRecord[] = [];
  let ordinal = 0;
  let pendingSmall: SourceChunkRecord[] = [];
  const createContextChunk = (
    chunks: SourceChunkRecord[],
    contentMd: string,
    input: { sizeClass?: "small" | "normal" | "oversized"; summaryText?: string | null } = {},
  ): StorageRetrievalChunkRecord => {
    const sourceDocumentChunkIds = chunks.map((item) => item.id);
    const pageNos = uniqueOrdered(
      chunks.flatMap((item) => parseJsonArray<number>(item.mergedPageNosJson ?? "[]", [item.pageNo])),
    );
    const bboxes = chunks.flatMap((item) =>
      parseJsonArray<[number, number, number, number]>(
        item.mergedBboxesJson ?? "[]",
        [JSON.parse(item.bboxJson) as [number, number, number, number]],
      ),
    );
    return {
      id: stableId(
        "rchunk",
        `${documentId}:${ordinal}:${sourceDocumentChunkIds.join(",")}:${contentMd}`,
      ),
      documentId,
      ordinal,
      contentMd,
      sizeClass: input.sizeClass ?? classifySize(contentMd),
      summaryText: input.summaryText ?? null,
      sourceDocumentChunkIdsJson: JSON.stringify(sourceDocumentChunkIds),
      pageNosJson: JSON.stringify(pageNos),
      sourceLocator: sourceLocatorFromPages(pageNos),
      bboxesJson: JSON.stringify(bboxes),
    };
  };
  const flushPending = () => {
    if (pendingSmall.length === 0) {
      return;
    }
    const text = normalizeWhitespace(pendingSmall.map((item) => item.contentMd).join("\n\n"));
    records.push(createContextChunk(pendingSmall, text));
    ordinal += 1;
    pendingSmall = [];
  };

  for (const chunk of documentChunks) {
    if (chunk.sizeClass === "small") {
      const candidate = normalizeWhitespace(
        [...pendingSmall.map((item) => item.contentMd), chunk.contentMd].join("\n\n"),
      );
      if (candidate.length > thresholds.normalMaxChars) {
        flushPending();
        pendingSmall = [chunk];
      } else {
        pendingSmall.push(chunk);
      }
      continue;
    }
    flushPending();
    if (chunk.sizeClass === "normal") {
      records.push(createContextChunk([chunk], chunk.contentMd));
      ordinal += 1;
      continue;
    }
    if (chunk.sizeClass === "oversized") {
      records.push(
        createContextChunk([chunk], chunk.contentMd, {
          sizeClass: "oversized",
          summaryText: chunk.summaryText,
        }),
      );
      ordinal += 1;
    }
  }
  flushPending();
  return records;
}

export function buildFixedRetrievalChunks(input: {
  documentId: string;
  sourceChunks: SourceChunkRecord[];
  fixedChunkChars: number;
}): StorageFixedRetrievalChunkRecord[] {
  const maxChars = Math.max(Number(input.fixedChunkChars || 0), 1);
  const pieces: FixedChunkPiece[] = [];
  for (const chunk of input.sourceChunks) {
    const contentParts = splitFixedChunkTextSmart(chunk.contentMd, maxChars);
    for (const part of contentParts) {
      pieces.push({
        sourceChunkId: chunk.id,
        pageNo: chunk.pageNo,
        contentMd: part,
        blockType: chunk.blockType,
        mergedPageNosJson: chunk.mergedPageNosJson,
        mergedBboxesJson: chunk.mergedBboxesJson,
        bboxJson: chunk.bboxJson,
      });
    }
  }

  const records: StorageFixedRetrievalChunkRecord[] = [];
  let ordinal = 0;
  let pendingPieces: FixedChunkPiece[] = [];
  const flushPending = () => {
    if (pendingPieces.length === 0) {
      return;
    }
    const contentMd = normalizeWhitespace(pendingPieces.map((piece) => piece.contentMd).join("\n\n"));
    const sourceDocumentChunkIds = uniqueOrdered(pendingPieces.map((piece) => piece.sourceChunkId));
    const pageNos = uniqueOrdered(
      pendingPieces.flatMap((piece) => parseJsonArray<number>(piece.mergedPageNosJson, [piece.pageNo])),
    );
    const bboxes = pendingPieces.flatMap((piece) =>
      parseJsonArray<[number, number, number, number]>(
        piece.mergedBboxesJson,
        [JSON.parse(piece.bboxJson) as [number, number, number, number]],
      ),
    );
    records.push({
      id: stableId(
        "frchunk",
        `${input.documentId}:${ordinal}:${sourceDocumentChunkIds.join(",")}:${contentMd}`,
      ),
      documentId: input.documentId,
      ordinal,
      contentMd,
      sizeClass: classifySize(contentMd),
      summaryText: null,
      sourceDocumentChunkIdsJson: JSON.stringify(sourceDocumentChunkIds),
      pageNosJson: JSON.stringify(pageNos),
      sourceLocator: sourceLocatorFromPages(pageNos),
      bboxesJson: JSON.stringify(bboxes),
    });
    ordinal += 1;
    pendingPieces = [];
  };

  for (const piece of pieces) {
    const pieceText = normalizeWhitespace(piece.contentMd);
    if (!pieceText) {
      continue;
    }
    const currentText = normalizeWhitespace(pendingPieces.map((item) => item.contentMd).join("\n\n"));
    const candidate = normalizeWhitespace([currentText, pieceText].filter(Boolean).join("\n\n"));
    if (candidate.length > maxChars && pendingPieces.length > 0) {
      flushPending();
    }
    pendingPieces.push(piece);
    const updatedText = normalizeWhitespace(pendingPieces.map((item) => item.contentMd).join("\n\n"));
    if (updatedText.length >= maxChars) {
      flushPending();
    }
  }
  flushPending();
  return records;
}

export function buildIndexedDocumentChunks(
  documentId: string,
  sourceChunks: SourceChunkRecord[],
  retrievalChunks: StorageRetrievalChunkRecord[],
  fixedRetrievalChunks: StorageFixedRetrievalChunkRecord[] = [],
): StorageDocumentChunkRecord[] {
  const thresholds = getChunkThresholds();
  const referenceBySourceChunkId = new Map<string, string>();
  for (const chunk of retrievalChunks) {
    for (const sourceChunkId of parseJsonArray<string>(chunk.sourceDocumentChunkIdsJson, [])) {
      referenceBySourceChunkId.set(sourceChunkId, chunk.id);
    }
  }

  const records: StorageDocumentChunkRecord[] = [];
  const unitIdsBySourceChunkId = new Map<string, string[]>();
  let ordinal = 0;
  for (const chunk of sourceChunks) {
    const referenceRetrievalChunkId = referenceBySourceChunkId.get(chunk.id) ?? null;
    if (chunk.sizeClass === "oversized") {
      const parts = splitOversized(chunk.contentMd, chunk.blockType);
      for (let splitIndex = 0; splitIndex < parts.length; splitIndex += 1) {
        const part = parts[splitIndex]!;
        records.push({
          id: stableId("dchunk", `${documentId}:${chunk.id}:${splitIndex}:${part}`),
          documentId,
          ordinal,
          referenceRetrievalChunkId,
          pageNo: chunk.pageNo,
          documentIndex: chunk.documentIndex,
          pageIndex: chunk.pageIndex,
          blockType: chunk.blockType,
          bboxJson: chunk.bboxJson,
          contentMd: part,
          sizeClass: part.length <= thresholds.smallMaxChars ? "small" : "normal",
          summaryText: chunk.summaryText,
          isSplitFromOversized: true,
          splitIndex,
          splitCount: parts.length,
          mergedPageNosJson: chunk.mergedPageNosJson,
          mergedBboxesJson: chunk.mergedBboxesJson,
        });
        const unitIds = unitIdsBySourceChunkId.get(chunk.id) ?? [];
        unitIds.push(records[records.length - 1]!.id);
        unitIdsBySourceChunkId.set(chunk.id, unitIds);
        ordinal += 1;
      }
      continue;
    }
    records.push({
      id: stableId("dchunk", `${documentId}:${chunk.id}:${ordinal}:${chunk.contentMd}`),
      documentId,
      referenceRetrievalChunkId,
      ordinal,
      pageNo: chunk.pageNo,
      documentIndex: chunk.documentIndex,
      pageIndex: chunk.pageIndex,
      blockType: chunk.blockType,
      bboxJson: chunk.bboxJson,
      contentMd: chunk.contentMd,
      sizeClass: chunk.sizeClass,
      summaryText: chunk.summaryText,
      isSplitFromOversized: false,
      splitIndex: 0,
      splitCount: 1,
      mergedPageNosJson: chunk.mergedPageNosJson,
      mergedBboxesJson: chunk.mergedBboxesJson,
    });
    const unitIds = unitIdsBySourceChunkId.get(chunk.id) ?? [];
    unitIds.push(records[records.length - 1]!.id);
    unitIdsBySourceChunkId.set(chunk.id, unitIds);
    ordinal += 1;
  }
  const remapChunkSourceIds = (chunks: Array<{ sourceDocumentChunkIdsJson: string }>) => {
    for (const chunk of chunks) {
      const sourceChunkIds = parseJsonArray<string>(chunk.sourceDocumentChunkIdsJson, []);
      const unitIds = sourceChunkIds.flatMap(
        (sourceChunkId) => unitIdsBySourceChunkId.get(sourceChunkId) ?? [],
      );
      chunk.sourceDocumentChunkIdsJson = JSON.stringify(unitIds);
    }
  };
  remapChunkSourceIds(retrievalChunks);
  remapChunkSourceIds(fixedRetrievalChunks);
  return records;
}
