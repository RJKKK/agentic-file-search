/*
Reference: legacy/python/src/fs_explorer/document_parsing.py
Reference: legacy/python/src/fs_explorer/fs.py
*/

import { spawn } from "node:child_process";
import { access, stat } from "node:fs/promises";
import { constants as fsConstants } from "node:fs";
import { dirname, extname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import {
  DocumentParseError,
  ParsedDocumentSchema,
  PythonBridgeResponseSchema,
  type ParseSelector,
  type ParseSelectorInput,
  type ParsedDocument,
  type ParsedUnit,
  type SupportedExtension,
  isSupportedExtension,
  normalizeParseSelector,
  parsedDocumentMarkdown,
} from "../types/parsing.js";

const DEFAULT_PREVIEW_CHARS = 3000;
const FOCUSED_WINDOW_FALLBACKS = [2, 4, 8] as const;
const PDF_LIKE_EXTENSIONS = new Set<SupportedExtension>([".pdf", ".docx"]);

export interface PythonDocumentParserExecutor {
  parseDocument(filePath: string, selector?: ParseSelector | null): Promise<ParsedDocument>;
}

interface CacheEntry {
  cacheKey: string;
  parsedDocument: ParsedDocument;
}

function repositoryRootFromModule(): string {
  return resolve(dirname(fileURLToPath(import.meta.url)), "../..");
}

function defaultPythonExecutable(repositoryRoot: string): string {
  if (process.env.FS_EXPLORER_PYTHON_BIN?.trim()) {
    return process.env.FS_EXPLORER_PYTHON_BIN.trim();
  }
  if (process.platform === "win32") {
    return resolve(repositoryRoot, ".venv", "Scripts", "python.exe");
  }
  return resolve(repositoryRoot, ".venv", "bin", "python");
}

function defaultBridgeScript(repositoryRoot: string): string {
  return resolve(repositoryRoot, "python", "document_parsing_bridge.py");
}

function pythonLikeRepr(value: string): string {
  return `'${String(value).replace(/\\/g, "\\\\").replace(/'/g, "\\'")}'`;
}

export function parsePythonBridgeJson(raw: string): unknown {
  const trimmed = raw.trim();
  try {
    return JSON.parse(trimmed);
  } catch (firstError) {
    const jsonStart = trimmed.lastIndexOf('{"ok"');
    const jsonEnd = trimmed.lastIndexOf("}");
    if (jsonStart >= 0 && jsonEnd > jsonStart) {
      try {
        return JSON.parse(trimmed.slice(jsonStart, jsonEnd + 1));
      } catch {
        // Fall through to the original parse error so callers see the real failure.
      }
    }
    throw firstError;
  }
}

function coerceExtension(filePath: string): string {
  return extname(filePath).toLowerCase();
}

async function ensureReadableFile(filePath: string): Promise<void> {
  try {
    await access(filePath, fsConstants.F_OK);
  } catch {
    throw new DocumentParseError(filePath, "missing_file", `No such file: ${filePath}`);
  }

  let info;
  try {
    info = await stat(filePath);
  } catch {
    throw new DocumentParseError(filePath, "missing_file", `No such file: ${filePath}`);
  }
  if (!info.isFile()) {
    throw new DocumentParseError(filePath, "missing_file", `No such file: ${filePath}`);
  }
}

function coerceSupportedExtension(filePath: string): SupportedExtension {
  const extension = coerceExtension(filePath);
  if (!isSupportedExtension(extension)) {
    throw new DocumentParseError(
      filePath,
      "unsupported_extension",
      `Unsupported file extension: ${extension}. Supported: ${[".doc", ".docx", ".html", ".md", ".pdf", ".pptx", ".xlsx"].join(", ")}`,
    );
  }
  return extension;
}

function queryTerms(query: string | null | undefined): string[] {
  if (!query) {
    return [];
  }
  const lowered = query.trim().toLowerCase();
  if (!lowered) {
    return [];
  }
  const terms = lowered.match(/[\u4e00-\u9fff]{2,}|[a-zA-Z0-9_]{3,}/g) ?? [];
  const ordered: string[] = [];
  const seen = new Set<string>();
  for (const term of terms) {
    const normalized = term.trim();
    if (!normalized || seen.has(normalized)) {
      continue;
    }
    seen.add(normalized);
    ordered.push(normalized);
  }
  return ordered;
}

function queryScores(units: ParsedUnit[], query: string | null | undefined): Map<number, number> {
  if (!query) {
    return new Map<number, number>();
  }
  const lowered = query.trim().toLowerCase();
  if (!lowered) {
    return new Map<number, number>();
  }
  const terms = queryTerms(lowered);
  const scores = new Map<number, number>();
  for (const unit of units) {
    const haystack = unit.markdown.toLowerCase();
    let score = 0;
    if (haystack.includes(lowered)) {
      score += 100;
    }
    for (const term of terms) {
      if (haystack.includes(term)) {
        score += Math.max(1, Math.min(term.length, 8));
      }
    }
    if (score > 0) {
      scores.set(unit.unit_no, score);
    }
  }
  return scores;
}

function explicitRequestedUnitNos(selector: ParseSelector): number[] | null {
  const requested = new Set<number>();
  for (const unitNo of selector.unit_nos ?? []) {
    if (unitNo > 0) {
      requested.add(unitNo);
    }
  }
  if (selector.anchor != null) {
    const window = Math.max(Number(selector.window ?? 1), 0);
    const anchor = Number(selector.anchor);
    for (let unitNo = anchor - window; unitNo <= anchor + window; unitNo += 1) {
      if (unitNo > 0) {
        requested.add(unitNo);
      }
    }
  }
  if (requested.size === 0) {
    return null;
  }
  return [...requested].sort((left, right) => left - right);
}

export function selectParsedDocument(
  parsedDocument: ParsedDocument,
  selectorInput: ParseSelectorInput | ParseSelector | null | undefined,
): ParsedDocument {
  const selector = normalizeParseSelector(selectorInput);
  if (selector == null) {
    return parsedDocument;
  }

  const units = [...parsedDocument.units];
  if (units.length === 0) {
    return parsedDocument;
  }

  const scoreByUnit = queryScores(units, selector.query);
  const selectedNumbers = new Set<number>();

  for (const unitNo of explicitRequestedUnitNos(selector) ?? []) {
    if (units.some((unit) => unit.unit_no === unitNo)) {
      selectedNumbers.add(unitNo);
    }
  }

  if (selector.query && scoreByUnit.size > 0) {
    const ranked = [...scoreByUnit.entries()].sort(
      (left, right) => right[1] - left[1] || left[0] - right[0],
    );
    const queryPick = Math.max(1, Math.min(4, ranked.length));
    for (const [unitNo] of ranked.slice(0, queryPick)) {
      selectedNumbers.add(unitNo);
    }
  }

  if (selectedNumbers.size === 0) {
    for (const unit of units) {
      selectedNumbers.add(unit.unit_no);
    }
  }

  let selected = units.filter((unit) => selectedNumbers.has(unit.unit_no));
  if (selected.length === 0) {
    selected = units;
  }

  if (selector.max_units != null && selector.max_units > 0) {
    const anchor = selector.anchor;
    selected = [...selected]
      .sort((left, right) => {
        const leftDistance = anchor != null ? Math.abs(left.unit_no - anchor) : 0;
        const rightDistance = anchor != null ? Math.abs(right.unit_no - anchor) : 0;
        const leftScore = scoreByUnit.get(left.unit_no) ?? 0;
        const rightScore = scoreByUnit.get(right.unit_no) ?? 0;
        return leftDistance - rightDistance || rightScore - leftScore || left.unit_no - right.unit_no;
      })
      .slice(0, selector.max_units);
  }

  selected.sort((left, right) => left.unit_no - right.unit_no);
  return ParsedDocumentSchema.parse({
    parser_name: parsedDocument.parser_name,
    parser_version: parsedDocument.parser_version,
    units: selected,
  });
}

export class PythonDocumentParserBridge implements PythonDocumentParserExecutor {
  private readonly repositoryRoot: string;

  private readonly pythonExecutable: string;

  private readonly bridgeScript: string;

  constructor(options: {
    repositoryRoot?: string;
    pythonExecutable?: string;
    bridgeScript?: string;
  } = {}) {
    this.repositoryRoot = options.repositoryRoot ?? repositoryRootFromModule();
    this.pythonExecutable = options.pythonExecutable ?? defaultPythonExecutable(this.repositoryRoot);
    this.bridgeScript = options.bridgeScript ?? defaultBridgeScript(this.repositoryRoot);
  }

  async parseDocument(filePath: string, selector?: ParseSelector | null): Promise<ParsedDocument> {
    await ensureReadableFile(filePath);
    const extension = coerceSupportedExtension(filePath);
    void extension;

    const payload = JSON.stringify({
      operation: "parse_document",
      file_path: resolve(filePath),
      selector: selector ?? null,
    });

    const raw = await new Promise<string>((resolvePromise, rejectPromise) => {
      const child = spawn(this.pythonExecutable, [this.bridgeScript], {
        cwd: this.repositoryRoot,
        stdio: ["pipe", "pipe", "pipe"],
      });

      let stdout = "";
      let stderr = "";
      child.stdout.setEncoding("utf8");
      child.stderr.setEncoding("utf8");
      child.stdout.on("data", (chunk: string) => {
        stdout += chunk;
      });
      child.stderr.on("data", (chunk: string) => {
        stderr += chunk;
      });
      child.on("error", (error) => {
        rejectPromise(
          new DocumentParseError(
            resolve(filePath),
            "python_bridge_failed",
            `Failed to start Python bridge with ${this.pythonExecutable}: ${String(error)}`,
          ),
        );
      });
      child.on("close", (code) => {
        if (code !== 0) {
          rejectPromise(
            new DocumentParseError(
              resolve(filePath),
              "python_bridge_failed",
              stderr.trim() ||
                `Python bridge exited with code ${code} using ${this.pythonExecutable}.`,
            ),
          );
          return;
        }
        resolvePromise(stdout);
      });
      child.stdin.end(payload);
    });

    let parsedResponse;
    try {
      parsedResponse = PythonBridgeResponseSchema.parse(parsePythonBridgeJson(raw));
    } catch (error) {
      throw new DocumentParseError(
        resolve(filePath),
        "python_bridge_invalid_response",
        `Could not decode Python bridge response: ${String(error)}`,
      );
    }

    if (!parsedResponse.ok) {
      throw new DocumentParseError(
        parsedResponse.error.file_path,
        parsedResponse.error.code,
        `${parsedResponse.error.message} Python executable: ${this.pythonExecutable}`,
      );
    }
    return parsedResponse.document;
  }
}

export class DocumentParsingRuntime {
  private readonly cache = new Map<string, CacheEntry>();

  constructor(
    private readonly executor: PythonDocumentParserExecutor = new PythonDocumentParserBridge(),
  ) {}

  clearCache(): void {
    this.cache.clear();
  }

  async parseDocument(
    filePath: string,
    selectorInput?: ParseSelectorInput | null,
  ): Promise<ParsedDocument> {
    const resolvedPath = resolve(filePath);
    await ensureReadableFile(resolvedPath);
    coerceSupportedExtension(resolvedPath);
    const selector = normalizeParseSelector(selectorInput);
    const parsed = await this.executor.parseDocument(resolvedPath, selector);
    return ParsedDocumentSchema.parse(parsed);
  }

  async previewFile(filePath: string, maxChars = DEFAULT_PREVIEW_CHARS): Promise<string> {
    const resolvedPath = resolve(filePath);
    try {
      await ensureReadableFile(resolvedPath);
      coerceSupportedExtension(resolvedPath);
      const parsed = await this.getCachedOrParse(resolvedPath);
      const fullContent = parsedDocumentMarkdown(parsed);
      let preview = fullContent.slice(0, maxChars);
      if (fullContent.length > maxChars) {
        preview += `\n\n[... PREVIEW TRUNCATED. Full document has ${fullContent.length.toLocaleString("en-US")} characters. Use parse_file() to read the complete document ...]`;
      }
      return `=== PREVIEW of ${resolvedPath} ===\n\n${preview}`;
    } catch (error) {
      return this.renderParsingError("previewing", resolvedPath, error);
    }
  }

  async parseFile(input: {
    filePath: string;
    focusHint?: string | null;
    anchor?: number | null;
    window?: number;
    maxUnits?: number | null;
  }): Promise<string> {
    const resolvedPath = resolve(input.filePath);
    try {
      await ensureReadableFile(resolvedPath);
      const extension = coerceSupportedExtension(resolvedPath);
      const selector =
        (input.focusHint != null && input.focusHint.trim()) ||
        input.anchor != null ||
        input.maxUnits != null
          ? normalizeParseSelector({
              query: input.focusHint?.trim() || null,
              anchor: input.anchor ?? null,
              window: Math.max(input.window ?? 1, 0),
              max_units: input.maxUnits ?? null,
            })
          : null;

      let parsed: ParsedDocument | null = null;
      if (selector == null || !PDF_LIKE_EXTENSIONS.has(extension)) {
        parsed = await this.getCachedOrParse(resolvedPath);
      }

      let focused =
        parsed ??
        ParsedDocumentSchema.parse({
          parser_name: "unknown",
          parser_version: "unknown",
          units: [],
        });
      let totalUnitsDisplay: number | string = parsed ? parsed.units.length : "?";

      if (selector != null) {
        if (PDF_LIKE_EXTENSIONS.has(extension)) {
          focused = await this.parseDocument(resolvedPath, selector);
          totalUnitsDisplay = "?";
        } else {
          focused = selectParsedDocument(parsed!, selector);
        }

        focused = selectParsedDocument(focused, selector);
        if (focused.units.length === 0) {
          if (input.anchor != null) {
            for (const expandWindow of FOCUSED_WINDOW_FALLBACKS) {
              const fallbackSelector = normalizeParseSelector({
                query: input.focusHint?.trim() || null,
                anchor: input.anchor,
                window: expandWindow,
                max_units: input.maxUnits ?? null,
              });
              focused = PDF_LIKE_EXTENSIONS.has(extension)
                ? await this.parseDocument(resolvedPath, fallbackSelector)
                : selectParsedDocument(parsed!, fallbackSelector);
              if (focused.units.length > 0) {
                break;
              }
            }
          }
          if (focused.units.length === 0) {
            if (parsed != null) {
              focused = parsed;
            } else if (PDF_LIKE_EXTENSIONS.has(extension)) {
              focused = await this.parseDocument(resolvedPath);
              totalUnitsDisplay = focused.units.length;
            }
          }
        }
      }

      if (focused === parsed) {
        return parsedDocumentMarkdown(focused);
      }

      const lines = [
        `=== FOCUSED PARSE of ${resolvedPath} ===`,
        `Units returned: ${focused.units.length} / ${totalUnitsDisplay} (anchor=${input.anchor ?? null}, window=${input.window ?? 1}, max_units=${input.maxUnits ?? null})`,
        "",
      ];
      for (const unit of focused.units) {
        lines.push(
          `[UNIT ${unit.unit_no} | source=${unit.source_locator ?? `unit-${unit.unit_no}`} | heading=${unit.heading ?? "-"}]`,
        );
        lines.push(unit.markdown);
        lines.push("");
      }
      return lines.join("\n").trim();
    } catch (error) {
      return this.renderParsingError("parsing", resolvedPath, error);
    }
  }

  private async getCachedOrParse(filePath: string): Promise<ParsedDocument> {
    const resolvedPath = resolve(filePath);
    const info = await stat(resolvedPath);
    const cacheKey = `${resolvedPath}:${info.mtimeMs}`;
    const cached = this.cache.get(resolvedPath);
    if (cached?.cacheKey === cacheKey) {
      return cached.parsedDocument;
    }
    const parsedDocument = await this.parseDocument(resolvedPath);
    this.cache.set(resolvedPath, {
      cacheKey,
      parsedDocument,
    });
    return parsedDocument;
  }

  private renderParsingError(
    verb: "previewing" | "parsing",
    filePath: string,
    error: unknown,
  ): string {
    if (error instanceof DocumentParseError) {
      if (error.code === "missing_file") {
        return `No such file: ${filePath}`;
      }
      if (error.code === "unsupported_extension") {
        return error.detail;
      }
      return `Error ${verb} ${filePath}: [${error.code}] ${error.detail}`;
    }
    return `Error ${verb} ${filePath}: ${String(error)}`;
  }
}

export function createDocumentParsingRuntime(
  executor?: PythonDocumentParserExecutor,
): DocumentParsingRuntime {
  return new DocumentParsingRuntime(executor);
}
