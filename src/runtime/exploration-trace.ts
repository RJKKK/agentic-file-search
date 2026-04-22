/*
Reference: legacy/python/src/fs_explorer/exploration_trace.py
*/

import { isAbsolute, join, resolve } from "node:path";

const FILE_TOOLS = new Set(["read", "grep", "preview_file", "parse_file"]);
const SOURCE_CITATION_RE = /\[Source:\s*([^,\]]+)/g;

function normalizePath(path: string, rootDirectory: string): string {
  if (path.startsWith("/")) {
    return path.replace(/\\/g, "/").replace(/\/+/g, "/");
  }
  if (isAbsolute(path)) {
    return resolve(path).replace(/\\/g, "/");
  }
  return resolve(join(rootDirectory, path)).replace(/\\/g, "/");
}

export function extractCitedSources(finalResult: string | null | undefined): string[] {
  if (!finalResult) {
    return [];
  }
  const seen = new Set<string>();
  const orderedSources: string[] = [];
  for (const match of finalResult.matchAll(SOURCE_CITATION_RE)) {
    const source = String(match[1] ?? "").trim();
    if (source && !seen.has(source)) {
      seen.add(source);
      orderedSources.push(source);
    }
  }
  return orderedSources;
}

export class ExplorationTrace {
  readonly stepPath: string[] = [];

  private readonly referencedDocuments = new Set<string>();

  constructor(private readonly rootDirectory: string) {}

  recordToolCall(input: {
    stepNumber: number;
    toolName: string;
    toolInput: Record<string, unknown>;
    resolvedDocumentPath?: string | null;
  }): void {
    const pathEntries: string[] = [];
    const directory = input.toolInput.directory;
    if (typeof directory === "string" && directory) {
      pathEntries.push(`directory=${normalizePath(directory, this.rootDirectory)}`);
    }

    const filePath = input.toolInput.file_path;
    if (typeof filePath === "string" && filePath) {
      const normalizedFilePath = normalizePath(filePath, this.rootDirectory);
      pathEntries.push(`file=${normalizedFilePath}`);
      if (FILE_TOOLS.has(input.toolName)) {
        this.referencedDocuments.add(normalizedFilePath);
      }
    }

    if (input.resolvedDocumentPath) {
      const normalizedDocPath = normalizePath(input.resolvedDocumentPath, this.rootDirectory);
      pathEntries.push(`document=${normalizedDocPath}`);
      this.referencedDocuments.add(normalizedDocPath);
    }

    const parameters = pathEntries.length ? pathEntries.join(", ") : "no-path-args";
    this.stepPath.push(`${input.stepNumber}. tool:${input.toolName} (${parameters})`);
  }

  recordGoDeeper(input: { stepNumber: number; directory: string }): void {
    this.stepPath.push(
      `${input.stepNumber}. godeeper (directory=${normalizePath(input.directory, this.rootDirectory)})`,
    );
  }

  sortedDocuments(): string[] {
    return [...this.referencedDocuments].sort();
  }
}
