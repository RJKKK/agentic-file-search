/*
Reference: legacy/python/src/fs_explorer/document_parsing.py
Reference: legacy/python/src/fs_explorer/fs.py
*/

import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { parsePythonBridgeJson } from "./document-parsing.js";

function repositoryRootFromModule(): string {
  return resolve(dirname(fileURLToPath(import.meta.url)), "../..");
}

function defaultPythonExecutable(repositoryRoot: string): string {
  if (process.env.FS_EXPLORER_PYTHON_BIN?.trim()) {
    return process.env.FS_EXPLORER_PYTHON_BIN.trim();
  }
  if (process.platform === "win32") {
    const venvPython = resolve(repositoryRoot, ".venv", "Scripts", "python.exe");
    return existsSync(venvPython) ? venvPython : "python";
  }
  const venvPython = resolve(repositoryRoot, ".venv", "bin", "python");
  return existsSync(venvPython) ? venvPython : "python";
}

function defaultBridgeScript(repositoryRoot: string): string {
  return resolve(repositoryRoot, "python", "document_parsing_bridge.py");
}

export interface ExtractedPdfImageAsset {
  image_hash: string;
  page_no: number;
  image_index: number;
  mime_type: string | null;
  width: number | null;
  height: number | null;
  bytes_base64: string;
  byte_size: number;
}

export interface PythonImageInspection {
  supported: boolean;
  has_text: boolean;
  interference_score: number;
  width?: number | null;
  height?: number | null;
  pixel_area?: number | null;
  aspect_ratio?: number | null;
  grayscale_stddev?: number | null;
  edge_density?: number | null;
  grayscale_entropy?: number | null;
  grayscale_mean?: number | null;
  hue_stddev?: number | null;
  compressed_bytes_base64: string;
  compressed_byte_size: number;
  output_mime_type: string;
}

export class PythonDocumentAssetBridge {
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

  async extractPdfImages(filePath: string, pageNos?: number[] | null): Promise<ExtractedPdfImageAsset[]> {
    const raw = await this.invoke({
      operation: "extract_pdf_images",
      file_path: resolve(filePath),
      page_nos: pageNos ?? null,
    });
    const parsed = parsePythonBridgeJson(raw) as { ok?: boolean; images?: ExtractedPdfImageAsset[] };
    return parsed.ok ? parsed.images ?? [] : [];
  }

  async inspectImage(
    bytesBase64: string,
    mimeType?: string | null,
  ): Promise<PythonImageInspection | null> {
    const raw = await this.invoke({
      operation: "inspect_image",
      bytes_base64: bytesBase64,
      mime_type: mimeType ?? null,
    });
    const parsed = parsePythonBridgeJson(raw) as { ok?: boolean; result?: PythonImageInspection };
    return parsed.ok ? parsed.result ?? null : null;
  }

  private async invoke(payload: Record<string, unknown>): Promise<string> {
    return await new Promise<string>((resolvePromise, rejectPromise) => {
      const child = spawn(this.pythonExecutable, [this.bridgeScript], {
        cwd: this.repositoryRoot,
        env: {
          ...process.env,
          PYTHONIOENCODING: "utf-8",
          PYTHONUTF8: "1",
        },
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
      child.on("error", (error) => rejectPromise(error));
      child.on("close", (code) => {
        if (code !== 0) {
          rejectPromise(new Error(stderr.trim() || `Python asset bridge exited with code ${code}.`));
          return;
        }
        resolvePromise(stdout);
      });
      child.stdin.end(JSON.stringify(payload));
    });
  }
}
