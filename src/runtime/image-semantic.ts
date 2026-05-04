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
const IMAGE_LINK_BLOCK_PATTERN = /(?:图片链接：)?!\[[^\]]*]\([^)]+\)/g;

interface ProtectedImageLinkBlock {
  placeholder: string;
  text: string;
}

function parsePositiveIntegerEnv(name: string, fallback: number): number {
  const raw = Number.parseInt(String(process.env[name] ?? "").trim(), 10);
  return Number.isFinite(raw) && raw > 0 ? raw : fallback;
}

function normalizeWhitespace(value: string): string {
  return String(value || "").replace(/\r\n/g, "\n").replace(/\n{3,}/g, "\n\n").trim();
}

export function hasProtectedImageLinks(value: string): boolean {
  return Array.from(String(value || "").matchAll(IMAGE_LINK_BLOCK_PATTERN)).length > 0;
}

export function protectImageLinks(value: string): {
  placeholderText: string;
  protectedBlocks: ProtectedImageLinkBlock[];
} {
  const normalized = normalizeWhitespace(value);
  const protectedBlocks: ProtectedImageLinkBlock[] = [];
  const placeholderText = normalized.replace(IMAGE_LINK_BLOCK_PATTERN, (match) => {
    const placeholder = `@@IMG_LINK_${protectedBlocks.length}@@`;
    protectedBlocks.push({ placeholder, text: match });
    return placeholder;
  });
  return {
    placeholderText,
    protectedBlocks,
  };
}

function stripDanglingImagePlaceholders(value: string): string {
  return String(value || "").replace(/@@IMG_LINK_[0-9]*@@?|@@IMG_LINK_|@@IMG_LINK|@@IMG|@@/g, "");
}

export function restoreProtectedImageLinks(
  value: string,
  protectedBlocks: ProtectedImageLinkBlock[],
): string {
  let restored = String(value || "");
  const missing: string[] = [];
  for (const block of protectedBlocks) {
    if (restored.includes(block.placeholder)) {
      restored = restored.split(block.placeholder).join(block.text);
      continue;
    }
    missing.push(block.text);
  }
  restored = stripDanglingImagePlaceholders(restored).trim();
  if (missing.length === 0) {
    return normalizeWhitespace(restored);
  }
  return normalizeWhitespace([missing.join("\n"), restored].filter(Boolean).join("\n"));
}

function truncateSmartPreservingImageLinks(
  value: string,
  maxChars: number,
): { text: string; truncated: boolean } {
  const normalized = normalizeWhitespace(value);
  if (!normalized) {
    return { text: "", truncated: false };
  }
  const protectedResult = protectImageLinks(normalized);
  if (protectedResult.protectedBlocks.length === 0) {
    return truncateSmart(normalized, maxChars);
  }
  const linkText = normalizeWhitespace(protectedResult.protectedBlocks.map((block) => block.text).join("\n"));
  const remainingText = normalizeWhitespace(
    protectedResult.placeholderText.replace(/@@IMG_LINK_\d+@@/g, ""),
  );
  if (!remainingText) {
    return {
      text: linkText,
      truncated: normalized.length > linkText.length,
    };
  }
  const separatorLength = linkText ? 2 : 0;
  const remainingBudget = Math.max(maxChars - linkText.length - separatorLength, 0);
  const truncated = remainingBudget > 0
    ? truncateSmart(remainingText, remainingBudget)
    : { text: "", truncated: remainingText.length > 0 };
  return {
    text: normalizeWhitespace([linkText, truncated.text].filter(Boolean).join("\n\n")),
    truncated:
      truncated.truncated ||
      normalized.length > normalizeWhitespace([linkText, truncated.text].filter(Boolean).join("\n\n")).length,
  };
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
  const result = truncateSmartPreservingImageLinks(value, limit);
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

  const summaryResult = truncateSmartPreservingImageLinks(
    String(input.payload.retrieval_summary || ""),
    retrievalSummaryLimit,
  );
  const visibleTextResult = truncateSmartPreservingImageLinks(
    String(input.payload.visible_text || ""),
    visibleTextLimit,
  );
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
  const detailResult = truncateSmartPreservingImageLinks(detailSections.join("\n\n"), detailLimit);
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
    ? `图片链接：\n![image](${input.accessibleUrl})\n\n${semanticText}`
    : `图片链接：\n![image](${input.accessibleUrl})`;

  return {
    semanticText,
    semanticDetailText,
    shortMarkdown,
    detailTruncated,
  };
}

export function buildVisionPromptMessages(input: {
  contextText?: string | null;
} = {}): {
  systemPrompt: string;
  userPrompt: string;
} {
  const contextText = normalizeWhitespace(String(input.contextText || ""));
  const contextBlock = contextText
    ? [
        "",
        "邻近上下文（来自图片前一个非图片文本块，仅作为辅助理解，不可机械照抄，不可虚构图中不存在的信息）：",
        truncateSmart(contextText, 900).text,
        "",
        "使用邻近上下文时的规则：",
        "1. 可以借助邻近上下文判断图片标题、主体、关系对象、图表主题。",
        "2. 只有当邻近上下文与图片中可见内容一致时，才能吸收到输出里。",
        "3. 不得仅凭邻近上下文补造图片中看不见的关系边、数字、实体、结论。",
      ].join("\n")
    : "";
  return {
    systemPrompt: [
      "You are a document-image semantic extraction assistant for retrieval and RAG.",
      "Return strict JSON only. Do not wrap JSON in markdown fences.",
      "Required JSON keys: recognizable, image_kind, contains_text, retrieval_summary, detail_markdown, visible_text, keywords, entities, qa_hints, detail_truncated, drop_reason.",
      "Use null for unknown scalar values and [] for unknown arrays.",
      "The output language must follow the image's main language; if the image is primarily Chinese, answer in Chinese.",
    ].join(" "),
    userPrompt: [
      "\u5173\u7cfb\u56fe\u3001\u80a1\u6743\u56fe\u3001\u7a7f\u900f\u56fe\u3001\u7ec4\u7ec7\u5173\u7cfb\u56fe\u7684\u786c\u6027\u8981\u6c42\uff1a",
      "1. \u4e0d\u80fd\u53ea\u5199\u4e00\u53e5\u5408\u5e76\u603b\u7ed3\uff0c\u5fc5\u987b\u628a\u6bcf\u4e00\u6761\u53ef\u89c1\u5173\u7cfb\u8fb9\u5355\u72ec\u5199\u51fa\u6765\u3002",
      "2. \u6bcf\u6761\u5173\u7cfb\u5c3d\u91cf\u5199\u6210\u4e3b\u8c13\u5bbe\u7ed3\u6784\u53e5\u5b50\uff0c\u4f8b\u5982\uff1aA \u6301\u80a1 B 100.00%\u3002A \u901a\u8fc7 B \u5173\u8054 C\u3002B \u6301\u80a1 D 45.00%\u3002",
      "3. \u5982\u679c\u7bad\u5934\u3001\u8fde\u7ebf\u3001\u4e0a\u4e0b\u4f4d\u7f6e\u3001\u767e\u5206\u6bd4\u3001\u6807\u6ce8\u6587\u5b57\u80fd\u8868\u660e\u5173\u7cfb\uff0c\u5fc5\u987b\u9010\u9879\u8f6c\u6210\u53e5\u5b50\uff0c\u4e0d\u8981\u4e22\u6389\u4efb\u4f55\u4e00\u6761\u660e\u786e\u5173\u7cfb\u3002",
      "4. \u5728 `detail_markdown` \u4e2d\u5fc5\u987b\u5355\u72ec\u8bbe\u7acb `## \u9010\u9879\u5173\u7cfb` \u5c0f\u8282\uff0c\u4e00\u884c\u4e00\u6761\u5173\u7cfb\uff0c\u4e0d\u8981\u5408\u5e76\u3002",
      "5. \u5982\u679c\u5173\u7cfb\u6709\u5c42\u7ea7\u6216\u7a7f\u900f\uff0c\u5148\u5199\u76f4\u63a5\u5173\u7cfb\uff0c\u518d\u5199\u7a7f\u900f\u6216\u95f4\u63a5\u5173\u7cfb\uff0c\u5e76\u6ce8\u660e\u4f9d\u636e\u662f\u54ea\u4e2a\u4e2d\u95f4\u8282\u70b9\u3002",
      "6. `retrieval_summary` \u4e5f\u5e94\u5c3d\u53ef\u80fd\u4f18\u5148\u4fdd\u7559\u51e0\u6761\u6700\u5173\u952e\u7684\u8fb9\uff0c\u800c\u4e0d\u662f\u628a\u591a\u6761\u5173\u7cfb\u878d\u6210\u4e00\u53e5\u7b3c\u7edf\u6982\u8ff0\u3002",
      "",
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
      contextBlock,
    ].filter(Boolean).join("\n"),
  };
}
