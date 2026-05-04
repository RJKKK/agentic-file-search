/*
Reference: legacy/python/src/fs_explorer/server.py
Reference: legacy/python/src/fs_explorer/document_library.py
*/

import type {
  TraditionalRagChunkReference,
  TraditionalRagDetailChunk,
} from "../types/rag.js";
import {
  CER_CONTEXT_EXCERPT_TARGET,
  compressTextPreservingImageLinks,
  extractQueryTokens,
  normalizeWhitespace,
  uniqueOrdered,
  type EvidenceSegment,
  type GroupedEvidence,
} from "./traditional-rag-shared.js";

export function isHeadingLikeShortCenterSegment(segment: EvidenceSegment): boolean {
  if (!segment.isCenter) {
    return false;
  }
  if (segment.blockType && segment.blockType !== "text") {
    return false;
  }
  const text = normalizeWhitespace(segment.mergedExcerpt);
  if (!text || text.length > 160) {
    return false;
  }
  return true
  // const lines = String(segment.mergedExcerpt || "")
  //   .split(/\r?\n/)
  //   .map((line) => line.trim())
  //   .filter(Boolean);
  // if (lines.length === 0) {
  //   return false;
  // }
  // return lines.some((line) => {
  //   const normalizedLine = line.replace(/\*+/g, "").trim();
  //   return (
  //     /^#{1,6}\s*/.test(line) ||
  //     /^(section|chapter|appendix)\b/i.test(normalizedLine) ||
  //     /(利润表|资产负债表|现金流量表|所有者权益变动表|附注|注释)$/.test(normalizedLine)
  //   );
  // });
}

function buildCerSegmentLabels(
  citationNo: number,
  segments: EvidenceSegment[],
): string[] {
  if (segments.length === 0) {
    return [];
  }
  const labels = new Array<string>(segments.length).fill(`chunk[${citationNo}]`);
  const centerIndex = segments.findIndex((segment) => segment.isCenter);
  if (centerIndex < 0) {
    return labels;
  }
  labels[centerIndex] = `chunk[${citationNo}]`;
  let previousCount = 0;
  for (let index = centerIndex - 1; index >= 0; index -= 1) {
    previousCount += 1;
    labels[index] = `chunk[${citationNo}-p-${previousCount}]`;
  }
  let nextCount = 0;
  for (let index = centerIndex + 1; index < segments.length; index += 1) {
    nextCount += 1;
    labels[index] = `chunk[${citationNo}-n-${nextCount}]`;
  }
  return labels;
}

function composeSegmentContent(segment: EvidenceSegment, chunkLabel: string): string {
  return composeEvidenceContent({
    chunkLabel,
    summaryText: segment.summaryText,
    mergedExcerpt: segment.mergedExcerpt,
    sourceLink: segment.sourceLink,
    includeSource: segment.isOversized,
  });
}

function sentenceSplit(text: string): string[] {
  const normalized = normalizeWhitespace(text);
  if (!normalized) {
    return [];
  }
  const sentences = normalized
    .split(/(?<=[\n。！？?!])\s+/)
    .map((value) => value.trim())
    .filter(Boolean);
  return sentences.length ? sentences : [normalized];
}

function isTableSegment(segment: EvidenceSegment): boolean {
  return segment.blockType === "table";
}

function extractRelevantTextSegment(text: string, queryTokens: string[]): string {
  return compressTextPreservingImageLinks(text, (protectedText) => {
    const sentences = sentenceSplit(protectedText);
    if (sentences.length <= 2 || queryTokens.length === 0) {
      return normalizeWhitespace(protectedText);
    }
    const lowered = sentences.map((sentence) => sentence.toLowerCase());
    const hits = lowered
      .map((sentence, index) => ({
        index,
        matched: queryTokens.some((token) => sentence.includes(token)),
      }))
      .filter((item) => item.matched)
      .map((item) => item.index);
    if (hits.length === 0) {
      return compressExcerptAroundKeywords(
        protectedText,
        queryTokens,
        Math.min(Math.max(protectedText.length, 160), CER_CONTEXT_EXCERPT_TARGET),
      );
    }
    const keep = new Set<number>();
    for (const index of hits) {
      keep.add(index);
      if (index > 0) {
        keep.add(index - 1);
      }
      if (index + 1 < sentences.length) {
        keep.add(index + 1);
      }
    }
    return [...keep]
      .sort((left, right) => left - right)
      .map((index) => sentences[index] ?? "")
      .filter(Boolean)
      .join("\n");
  });
}

function extractRelevantTableSegment(text: string, queryTokens: string[]): string {
  return compressTextPreservingImageLinks(text, (protectedText) => {
    const lines = normalizeWhitespace(protectedText)
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean);
    if (lines.length <= 3 || queryTokens.length === 0) {
      return normalizeWhitespace(protectedText);
    }
    const keep = new Set<number>();
    keep.add(0);
    for (let index = 0; index < lines.length; index += 1) {
      const lowered = lines[index]!.toLowerCase();
      if (!queryTokens.some((token) => lowered.includes(token))) {
        continue;
      }
      keep.add(index);
      if (index > 0) {
        keep.add(index - 1);
      }
      if (index + 1 < lines.length) {
        keep.add(index + 1);
      }
    }
    if (keep.size <= 1) {
      return compressExcerptAroundKeywords(
        protectedText,
        queryTokens,
        Math.min(Math.max(protectedText.length, 160), CER_CONTEXT_EXCERPT_TARGET),
      );
    }
    return [...keep]
      .sort((left, right) => left - right)
      .map((index) => lines[index] ?? "")
      .filter(Boolean)
      .join("\n");
  });
}

export function extractRelevantSegment(
  segment: EvidenceSegment,
  queryTokens: string[],
): string {
  if (!segment.mergedExcerpt) {
    return segment.mergedExcerpt;
  }
  return isTableSegment(segment)
    ? extractRelevantTableSegment(segment.mergedExcerpt, queryTokens)
    : extractRelevantTextSegment(segment.mergedExcerpt, queryTokens);
}

export function materializeEvidenceItem(item: GroupedEvidence): GroupedEvidence {
  const cerSegments = item.cerSegments.map((segment) =>
    segment.isCenter
      ? {
          ...segment,
          summaryText: item.summaryText,
          mergedExcerpt: item.mergedExcerpt,
          sourceLink: item.sourceLink,
          sourceLocator: item.sourceLocator,
          pageNos: item.pageNos,
          isOversized: item.hasOversizedSplitChild,
        }
      : segment,
  );
  const baseContent = composeEvidenceContent({
    chunkLabel: `chunk[${item.citationNo}]`,
    summaryText: item.summaryText,
    mergedExcerpt: item.mergedExcerpt,
    sourceLink: item.sourceLink,
    includeSource: item.hasOversizedSplitChild,
  });
  const cerLabels = buildCerSegmentLabels(item.citationNo, cerSegments);
  const enrichedContent = cerSegments
    .map((segment, segmentIndex) =>
      composeSegmentContent(segment, cerLabels[segmentIndex] || `chunk[${item.citationNo}]`),
    )
    .filter((value) => String(value || "").trim())
    .join("\n\n");
  return {
    ...item,
    cerSegments,
    baseContent,
    enrichedContent,
    content: enrichedContent || baseContent,
    cerApplied: cerSegments.length > 1,
    cerContextRefs: cerSegments
      .filter((segment) => !segment.isCenter)
      .map((segment) => ({
        reference_id: segment.referenceId,
        reference_kind: segment.referenceKind,
        role: "context" as const,
        source_locator: segment.sourceLocator,
        source_link: segment.sourceLink,
        page_nos: segment.pageNos,
      })),
  };
}

function toPublicChunkReference(
  item: GroupedEvidence,
  options: { includeCerDebug?: boolean } = {},
): TraditionalRagChunkReference {
  const base = {
    citation_no: item.citationNo,
    reference_id: item.referenceId,
    reference_kind: item.referenceKind,
    document_id: item.documentId,
    document_name: item.documentName,
    source_locator: item.sourceLocator,
    show_full_chunk_detail: false,
    document_chunk_ids: item.documentChunkIds,
    retrieval_unit_ids: item.retrievalUnitIds,
    page_nos: item.pageNos,
    bboxes: item.bboxes,
    score: item.score,
    source_link: item.sourceLink,
    compression_applied: item.compressionApplied,
  };
  if (!options.includeCerDebug) {
    return base;
  }
  return {
    ...base,
    debug_cer_applied: item.cerApplied,
    debug_cer_context_refs: item.cerContextRefs,
    debug_enriched_content: item.content,
  };
}

export function toPublicChunkReferences(
  items: GroupedEvidence[],
  options: { includeCerDebug?: boolean } = {},
): TraditionalRagChunkReference[] {
  const detailIds = new Set(selectHighScoreOversizedChunks(items).map((item) => item.referenceId));
  return items.map((item) => ({
    ...toPublicChunkReference(item, options),
    show_full_chunk_detail: detailIds.has(item.referenceId),
  }));
}

export function toPublicDetailChunks(
  items: GroupedEvidence[],
): TraditionalRagDetailChunk[] {
  const detailMap = new Map<string, TraditionalRagDetailChunk>();
  for (const item of items) {
    const add = (chunk: TraditionalRagDetailChunk) => {
      const key = `${chunk.reference_kind}:${chunk.reference_id}`;
      if (!detailMap.has(key)) {
        detailMap.set(key, chunk);
      }
    };
    if (item.referenceKind === "retrieval_chunk" && item.hasOversizedSplitChild) {
      add({
        ...toPublicChunkReference(item),
        show_full_chunk_detail: true,
      });
    }
    if (item.referenceKind === "document_chunk" && item.hasOversizedSplitChild) {
      add({
        ...toPublicChunkReference(item),
        show_full_chunk_detail: true,
      });
    }
    for (const segment of item.cerSegments) {
      if (!segment.isOversized || !segment.sourceLink) {
        continue;
      }
      add({
        citation_no: item.citationNo,
        reference_id: segment.referenceId,
        reference_kind: segment.referenceKind,
        document_id: item.documentId,
        document_name: item.documentName,
        source_locator: segment.sourceLocator,
        show_full_chunk_detail: true,
        document_chunk_ids:
          segment.referenceKind === "retrieval_chunk" && segment.referenceId === item.referenceId
            ? item.documentChunkIds
            : segment.referenceKind === "document_chunk"
              ? [segment.referenceId]
              : [],
        retrieval_unit_ids: item.retrievalUnitIds,
        page_nos: segment.pageNos,
        bboxes: [],
        score: item.score,
        source_link: segment.sourceLink,
        compression_applied: item.compressionApplied,
      });
    }
  }
  return [...detailMap.values()];
}

export function answerContainsAnyDetailLink(
  answer: string,
  detailChunks: TraditionalRagDetailChunk[],
): boolean {
  const normalized = String(answer || "");
  return detailChunks.some((item) => item.source_link && normalized.includes(item.source_link));
}

export function parseCitationNumbers(value: string): number[] {
  const citations: number[] = [];
  for (const match of String(value || "").matchAll(/\[(\d+(?:\s*,\s*\d+)*)\]/g)) {
    const raw = String(match[1] ?? "");
    for (const piece of raw.split(",")) {
      const parsed = Number.parseInt(piece.trim(), 10);
      if (Number.isFinite(parsed) && parsed > 0) {
        citations.push(parsed);
      }
    }
  }
  return uniqueOrdered(citations);
}

export function selectHighScoreOversizedChunks(
  chunks: GroupedEvidence[],
): GroupedEvidence[] {
  const oversized = chunks.filter((item) => item.hasOversizedSplitChild);
  if (oversized.length === 0) {
    return [];
  }
  const maxScore = Math.max(...oversized.map((item) => item.score), 0);
  const threshold = maxScore * 0.85;
  return oversized.filter((item) => item.score >= threshold);
}

export function appendOversizedChunkLinks(
  answer: string,
  chunks: GroupedEvidence[],
): string {
  const selected = selectHighScoreOversizedChunks(chunks).filter((item) => item.sourceLink);
  if (selected.length === 0) {
    return answer;
  }
  const links = uniqueOrdered(
    selected.map(
      (item) => `- [引用 ${item.citationNo} 对应完整块](${item.sourceLink})`,
    ),
  );
  const suffix = `详情内容如下：\n${links.join("\n")}`;
  const trimmed = String(answer || "").trim();
  if (!trimmed) {
    return suffix;
  }
  if (links.every((link) => trimmed.includes(link))) {
    return trimmed;
  }
  return `${trimmed}\n\n${suffix}`;
}

function composeEvidenceContent(input: {
  chunkLabel: string;
  summaryText: string | null;
  mergedExcerpt: string;
  sourceLink: string;
  includeSource?: boolean;
}): string {
  const summaryBlock = input.summaryText ? `${input.chunkLabel} summary:\n${input.summaryText}` : null;
  const excerptBlock = `${input.chunkLabel} compressed excerpt:\n${input.mergedExcerpt}`;
  const oversizedNote =
    input.includeSource && input.sourceLink
      ? (
          `Note: the ${input.chunkLabel} is shown as a compressed excerpt. ` +
          `Full content is available at ${input.sourceLink}, and you may provide this link directly to the user when it is helpful.`
        )
      : null;
  return [summaryBlock, excerptBlock, oversizedNote]
    .filter((value) => String(value || "").trim())
    .join("\n\n");
}

function fallbackCompressExcerpt(text: string, targetLength: number): string {
  const normalized = normalizeWhitespace(text);
  if (normalized.length <= targetLength) {
    return normalized;
  }
  if (targetLength <= 12) {
    return normalized.slice(0, Math.max(targetLength, 0));
  }
  const head = Math.max(Math.floor(targetLength * 0.35), 4);
  const tail = Math.max(targetLength - head - 3, 4);
  return `${normalized.slice(0, head)}...${normalized.slice(Math.max(normalized.length - tail, head))}`;
}

function compressExcerptAroundKeywords(
  text: string,
  queryTokens: string[],
  targetLength: number,
): string {
  return compressTextPreservingImageLinks(text, (protectedText) => {
    const normalized = normalizeWhitespace(protectedText);
    if (normalized.length <= targetLength) {
      return normalized;
    }
    if (targetLength <= 24) {
      return fallbackCompressExcerpt(normalized, targetLength);
    }
    const lowered = normalized.toLowerCase();
    const anchorIndexes = queryTokens
      .map((token) => lowered.indexOf(token))
      .filter((index) => index >= 0)
      .sort((left, right) => left - right);
    if (anchorIndexes.length === 0) {
      return fallbackCompressExcerpt(normalized, targetLength);
    }
    const center = Math.round(
      anchorIndexes.reduce((sum, index) => sum + index, 0) / anchorIndexes.length,
    );
    const focusLength = Math.max(
      Math.floor(targetLength * 0.55),
      Math.min(240, targetLength),
    );
    const focusStart = Math.max(
      0,
      Math.min(center - Math.floor(focusLength / 2), normalized.length - focusLength),
    );
    const focusEnd = Math.min(normalized.length, focusStart + focusLength);
    const focus = normalized.slice(focusStart, focusEnd).trim();
    let remaining = targetLength - focus.length;
    if (remaining <= 3) {
      return focus.slice(0, targetLength);
    }
    const leftRoom = Math.max(Math.floor(remaining * 0.45), 0);
    const rightRoom = Math.max(remaining - leftRoom - 6, 0);
    const leftPart =
      focusStart > 0
        ? fallbackCompressExcerpt(normalized.slice(0, focusStart), leftRoom).trim()
        : "";
    const rightPart =
      focusEnd < normalized.length
        ? fallbackCompressExcerpt(normalized.slice(focusEnd), rightRoom).trim()
        : "";
    const parts = [
      leftPart ? `${leftPart}...` : "",
      focus,
      rightPart ? `...${rightPart}` : "",
    ].filter(Boolean);
    const result = parts.join("\n");
    return result.length <= targetLength ? result : fallbackCompressExcerpt(result, targetLength);
  });
}

export function compressEvidenceItems(
  items: GroupedEvidence[],
  question: string,
  budget: number,
): GroupedEvidence[] {
  const queryTokens = extractQueryTokens(question);
  const materialized = items.map((item) => materializeEvidenceItem(item));
  const totalLength = materialized.reduce((sum, item) => sum + item.content.length, 0);
  if (totalLength <= budget) {
    return materialized;
  }
  const contextCompressed = materialized.map((item) => {
    let changed = false;
    const cerSegments = item.cerSegments.map((segment) => {
      if (segment.isCenter || !segment.mergedExcerpt) {
        return segment;
      }
      const maxLength =
        segment.referenceKind === "document_chunk" ? CER_CONTEXT_EXCERPT_TARGET : 240;
      if (segment.mergedExcerpt.length <= maxLength) {
        return segment;
      }
      changed = true;
      return {
        ...segment,
        mergedExcerpt: compressExcerptAroundKeywords(segment.mergedExcerpt, queryTokens, maxLength),
      };
    });
    return changed
      ? materializeEvidenceItem({
          ...item,
          cerSegments,
          compressionApplied: true,
        })
      : item;
  });
  const contextCompressedLength = contextCompressed.reduce(
    (sum, item) => sum + item.content.length,
    0,
  );
  if (contextCompressedLength <= budget) {
    return contextCompressed;
  }
  const contextDropped = dropCerContextsBeforeCenterCompression(contextCompressed, budget);
  const contextDroppedLength = contextDropped.reduce(
    (sum, item) => sum + item.content.length,
    0,
  );
  if (contextDroppedLength <= budget) {
    return contextDropped;
  }
  const lockedLength = contextDropped.reduce(
    (sum, item) => sum + Math.max(item.content.length - item.mergedExcerpt.length, 0),
    0,
  );
  const availableForExcerpts = Math.max(budget - lockedLength, 0);
  const totalExcerptLength = contextDropped.reduce(
    (sum, item) => sum + item.mergedExcerpt.length,
    0,
  );
  const compressed = contextDropped.map((item) => {
    if (!item.mergedExcerpt) {
      return item;
    }
    const share =
      totalExcerptLength > 0
        ? item.mergedExcerpt.length / totalExcerptLength
        : 1 / Math.max(contextCompressed.length, 1);
    const targetLength = Math.max(Math.floor(availableForExcerpts * share), 80);
    if (item.mergedExcerpt.length <= targetLength) {
      return item;
    }
    const mergedExcerpt = compressExcerptAroundKeywords(
      item.mergedExcerpt,
      queryTokens,
      targetLength,
    );
    return materializeEvidenceItem({
      ...item,
      mergedExcerpt,
      compressionApplied: true,
    });
  });
  let running = compressed.reduce((sum, item) => sum + item.content.length, 0);
  if (running <= budget) {
    return compressed;
  }
  const sortedByScore = [...compressed].sort((left, right) => left.score - right.score);
  const dropped = new Set<string>();
  for (const item of sortedByScore) {
    if (running <= budget) {
      break;
    }
    dropped.add(item.referenceId);
    running -= item.content.length;
  }
  return compressed.filter((item) => !dropped.has(item.referenceId));
}

export function dropCerContextsBeforeCenterCompression(
  items: GroupedEvidence[],
  budget: number,
): GroupedEvidence[] {
  const working = [...items];
  let totalLength = working.reduce((sum, item) => sum + item.content.length, 0);
  if (totalLength <= budget) {
    return working;
  }
  const dropOrder = working
    .map((item, index) => ({
      index,
      score: item.score,
      contextCount: item.cerSegments.filter((segment) => !segment.isCenter).length,
    }))
    .filter((item) => item.contextCount > 0)
    .sort((left, right) => left.score - right.score || right.contextCount - left.contextCount);
  for (const candidate of dropOrder) {
    if (totalLength <= budget) {
      break;
    }
    const original = working[candidate.index];
    if (!original) {
      continue;
    }
    const centerSegments = original.cerSegments.filter((segment) => segment.isCenter);
    if (centerSegments.length === 0 || centerSegments.length === original.cerSegments.length) {
      continue;
    }
    working[candidate.index] = materializeEvidenceItem({
      ...original,
      cerSegments: centerSegments,
      compressionApplied: true,
    });
    totalLength = working.reduce((sum, item) => sum + item.content.length, 0);
  }
  return working;
}
