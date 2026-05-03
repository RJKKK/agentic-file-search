/*
Reference: E:\projects\esg-rag\src\utils\prompt.js
Reference: legacy/python/src/fs_explorer/server.py
*/

export interface ImageSemanticPayload {
  recognizable: boolean;
  image_kind?: string | null;
  contains_text?: boolean | null;
  retrieval_summary?: string | null;
  detail_markdown?: string | null;
  visible_text?: string | null;
  keywords?: string[];
  entities?: string[];
  qa_hints?: string[];
  detail_truncated?: boolean | null;
  drop_reason?: string | null;
}

export interface RenderedImageSemantic {
  semanticText: string | null;
  semanticDetailText: string | null;
  shortMarkdown: string | null;
  detailTruncated: boolean;
}

const DEFAULT_RETRIEVAL_SUMMARY_MAX_CHARS = 240;
const DEFAULT_VISIBLE_TEXT_MAX_CHARS = 420;
const DEFAULT_DETAIL_MAX_CHARS = 5000;
const DEFAULT_KEYWORD_COUNT = 8;
const DEFAULT_INDEX_SEARCH_IMAGE_TEXT_MAX_CHARS = 280;

function parsePositiveIntegerEnv(name: string, fallback: number): number {
  const raw = Number.parseInt(String(process.env[name] ?? "").trim(), 10);
  return Number.isFinite(raw) && raw > 0 ? raw : fallback;
}

function normalizeWhitespace(value: string): string {
  return String(value || "").replace(/\r\n/g, "\n").replace(/\n{3,}/g, "\n\n").trim();
}

function uniqueStrings(values: Array<string | null | undefined>): string[] {
  const seen = new Set<string>();
  const ordered: string[] = [];
  for (const value of values) {
    const normalized = normalizeWhitespace(String(value || ""));
    if (!normalized || seen.has(normalized)) {
      continue;
    }
    seen.add(normalized);
    ordered.push(normalized);
  }
  return ordered;
}

function splitMarkdownParagraphs(value: string): string[] {
  return normalizeWhitespace(value)
    .split(/\n{2,}/)
    .map((part) => part.trim())
    .filter(Boolean);
}

function splitListAwareLines(value: string): string[] {
  return normalizeWhitespace(value)
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
}

function truncateSmart(value: string, maxChars: number): { text: string; truncated: boolean } {
  const normalized = normalizeWhitespace(value);
  if (!normalized || normalized.length <= maxChars) {
    return { text: normalized, truncated: false };
  }
  const paragraphs = splitMarkdownParagraphs(normalized);
  if (paragraphs.length > 1) {
    const kept: string[] = [];
    let length = 0;
    for (const paragraph of paragraphs) {
      const candidateLength = length === 0 ? paragraph.length : length + 2 + paragraph.length;
      if (candidateLength > maxChars) {
        break;
      }
      kept.push(paragraph);
      length = candidateLength;
    }
    if (kept.length > 0) {
      return { text: kept.join("\n\n"), truncated: true };
    }
  }
  const lines = splitListAwareLines(normalized);
  if (lines.length > 1) {
    const kept: string[] = [];
    let length = 0;
    for (const line of lines) {
      const candidateLength = length === 0 ? line.length : length + 1 + line.length;
      if (candidateLength > maxChars) {
        break;
      }
      kept.push(line);
      length = candidateLength;
    }
    if (kept.length > 0) {
      return { text: kept.join("\n"), truncated: true };
    }
  }
  const slice = normalized.slice(0, maxChars);
  const breakAt = Math.max(slice.lastIndexOf("\n"), slice.lastIndexOf("。"), slice.lastIndexOf("；"), slice.lastIndexOf(". "));
  const end = breakAt >= Math.floor(maxChars * 0.55) ? breakAt : maxChars;
  return {
    text: slice.slice(0, end).trim(),
    truncated: true,
  };
}

function trimKeywordList(values: string[]): string[] {
  return uniqueStrings(values).slice(
    0,
    parsePositiveIntegerEnv("VISION_SHORT_KEYWORD_LIMIT", DEFAULT_KEYWORD_COUNT),
  );
}

function maybePrefixBlock(label: string, value: string | null): string | null {
  const normalized = normalizeWhitespace(String(value || ""));
  return normalized ? `${label}\n${normalized}` : null;
}

export function truncateImageSemanticPreview(value: string, maxChars?: number): string {
  const limit = maxChars ?? parsePositiveIntegerEnv("INDEX_SEARCH_IMAGE_TEXT_MAX_CHARS", DEFAULT_INDEX_SEARCH_IMAGE_TEXT_MAX_CHARS);
  const result = truncateSmart(value, limit);
  return result.truncated ? `${result.text}\n[truncated]` : result.text;
}

export function renderImageSemantic(input: {
  payload: ImageSemanticPayload | null;
  accessibleUrl: string;
}): RenderedImageSemantic {
  if (!input.payload) {
    return {
      semanticText: null,
      semanticDetailText: null,
      shortMarkdown: `![image](${input.accessibleUrl})`,
      detailTruncated: false,
    };
  }

  const retrievalSummaryLimit = parsePositiveIntegerEnv(
    "VISION_RETRIEVAL_SUMMARY_MAX_CHARS",
    DEFAULT_RETRIEVAL_SUMMARY_MAX_CHARS,
  );
  const visibleTextLimit = parsePositiveIntegerEnv(
    "VISION_SHORT_VISIBLE_TEXT_MAX_CHARS",
    DEFAULT_VISIBLE_TEXT_MAX_CHARS,
  );
  const detailLimit = parsePositiveIntegerEnv("VISION_DETAIL_MAX_CHARS", DEFAULT_DETAIL_MAX_CHARS);

  const summaryResult = truncateSmart(String(input.payload.retrieval_summary || ""), retrievalSummaryLimit);
  const visibleTextResult = truncateSmart(String(input.payload.visible_text || ""), visibleTextLimit);
  const keywords = trimKeywordList(input.payload.keywords ?? []);
  const shortBlocks = uniqueStrings([
    summaryResult.text,
    maybePrefixBlock("[Visible Text]", visibleTextResult.text),
    keywords.length > 0 ? `[Keywords]\n- ${keywords.join("\n- ")}` : null,
  ]);
  const semanticText = shortBlocks.join("\n\n").trim() || null;

  const detailSections = uniqueStrings([
    input.payload.detail_markdown,
    summaryResult.text ? `[Retrieval Summary]\n${summaryResult.text}` : null,
    visibleTextResult.text ? `[Visible Text]\n${visibleTextResult.text}` : null,
    keywords.length > 0 ? `[Keywords]\n- ${keywords.join("\n- ")}` : null,
    trimKeywordList(input.payload.entities ?? []).length > 0
      ? `[Entities]\n- ${trimKeywordList(input.payload.entities ?? []).join("\n- ")}`
      : null,
    trimKeywordList(input.payload.qa_hints ?? []).length > 0
      ? `[QA Hints]\n- ${trimKeywordList(input.payload.qa_hints ?? []).join("\n- ")}`
      : null,
  ]);
  const detailResult = truncateSmart(detailSections.join("\n\n"), detailLimit);
  const detailTruncated =
    Boolean(input.payload.detail_truncated) ||
    summaryResult.truncated ||
    visibleTextResult.truncated ||
    detailResult.truncated;
  const semanticDetailText = detailResult.text
    ? detailTruncated
      ? `${detailResult.text}\n\n[detail_truncated=true]`
      : detailResult.text
    : null;
  const shortMarkdown = semanticText
    ? `图片链接：![image](${input.accessibleUrl})\n\n${semanticText}`
    : `图片链接：![image](${input.accessibleUrl})`;

  return {
    semanticText,
    semanticDetailText,
    shortMarkdown,
    detailTruncated,
  };
}

export function buildVisionPromptMessages(): {
  systemPrompt: string;
  userPrompt: string;
} {
  return {
    systemPrompt: [
      "You are a document-image semantic extraction assistant for retrieval and RAG.",
      "Return strict JSON only. Do not wrap JSON in markdown fences.",
      "Required JSON keys: recognizable, image_kind, contains_text, retrieval_summary, detail_markdown, visible_text, keywords, entities, qa_hints, detail_truncated, drop_reason.",
      "Use null for unknown scalar values and [] for unknown arrays.",
      "The output language must follow the image's main language; if the image is primarily Chinese, answer in Chinese.",
    ].join(" "),
    userPrompt: [
      "请分析这张文档图片，目标是同时生成“短检索文本”和“详细语义文本”。",
      "",
      "字段要求：",
      '- `retrieval_summary`：1到4句，面向检索召回，必须高密度、短而准。',
      '- `detail_markdown`：完整语义描述，使用 markdown 分段，可较长，但要保持结构化和可读性。',
      '- `visible_text`：提取图片中最重要的原文，不要无上限抄录全文。',
      '- `keywords` / `entities` / `qa_hints`：提取高价值检索词、实体、问答提示。',
      '- `detail_truncated`：如果为了控制长度而省略了次要细节，设为 true。',
      "",
      "图片类型与详细度策略：",
      "1. 流程图、业务图、时序图、方框图：务必详尽描述节点、分支条件、箭头方向、起止点、循环、异常路径、节点原文、关键数字。",
      "2. 结构图、架构图、关系图、网络拓扑图：务必详尽描述模块、层级、上下游关系、调用链、数据流、依赖、分组、位置关系、关键标识。",
      "3. 统计图表：务必详尽描述图表类型、坐标轴、单位、图例、序列含义、峰值、谷值、趋势、异常点、对比关系，能识别数值时优先保留具体数值。",
      "4. 表格、文本截图：优先保留标题、表头、字段关系、关键行、汇总行、极值行、关键定义、关键数字；如果内容特别大，不要逐字无限展开，要明确哪些内容被省略。",
      "5. 普通照片、装饰图：保持简洁，只保留对检索有意义的信息。",
      "",
      "控长规则：",
      "1. 详细类图片可以写得更详细，但 detail_markdown 仍要优先保留结构、关系、关键文本、关键数字、结论。",
      "2. 如果图片是超大表格、长文本截图或重复性很高的内容，不要完整穷举次要细节；保留关键内容并把 detail_truncated 设为 true。",
      "3. visible_text 只保留最重要的可见文本，例如标题、图例、表头、关键节点文案、关键数字、关键行。",
      "",
      "detail_markdown 推荐结构：",
      "## 图片类型",
      "## 图片摘要",
      "## 关键结构与关系",
      "## 关键文本与数字",
      "## 详细说明",
      "## 省略说明（如有）",
      "",
      "如果图片不可识别或只是无信息装饰图：",
      '- `recognizable=false`，并在 `drop_reason` 中说明原因。',
    ].join("\n"),
  };
}
