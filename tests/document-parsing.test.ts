import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { describe, it } from "node:test";

import {
  DocumentParsingRuntime,
  parsePythonBridgeJson,
  selectParsedDocument,
  type PythonDocumentParserExecutor,
} from "../src/runtime/document-parsing.js";
import type { ParseSelector, ParsedDocument } from "../src/types/parsing.js";

class FakeParserExecutor implements PythonDocumentParserExecutor {
  readonly calls: Array<{ filePath: string; selector: ParseSelector | null | undefined }> = [];

  constructor(private readonly document: ParsedDocument) {}

  async parseDocument(filePath: string, selector?: ParseSelector | null): Promise<ParsedDocument> {
    this.calls.push({ filePath, selector });
    return this.document;
  }
}

function buildDocument(): ParsedDocument {
  return {
    parser_name: "fake",
    parser_version: "m2-v2",
    units: [
      {
        unit_no: 1,
        markdown: "# Overview\nGeneral summary.",
        content_hash: "u1",
        heading: "Overview",
        source_locator: "unit-1",
        images: [],
      },
      {
        unit_no: 2,
        markdown: "# Purchase Price\nThe purchase price is $45,000,000.",
        content_hash: "u2",
        heading: "Purchase Price",
        source_locator: "unit-2",
        images: [],
      },
      {
        unit_no: 3,
        markdown: "# Adjustments\nWorking capital adjustments apply.",
        content_hash: "u3",
        heading: "Adjustments",
        source_locator: "unit-3",
        images: [],
      },
    ],
  };
}

function configuredPythonExecutable(): string {
  if (process.env.FS_EXPLORER_PYTHON_BIN?.trim()) {
    return process.env.FS_EXPLORER_PYTHON_BIN.trim();
  }
  if (process.platform === "win32") {
    return resolve(process.cwd(), ".venv", "Scripts", "python.exe");
  }
  return resolve(process.cwd(), ".venv", "bin", "python");
}

function runPythonTableNormalization(markdown: string): string | null {
  const pythonExecutable = configuredPythonExecutable();
  const pythonEnv = {
    ...process.env,
    PYTHONUTF8: "1",
    PYTHONIOENCODING: "utf-8",
  };
  const probe = spawnSync(pythonExecutable, ["-c", "print('ok')"], {
    cwd: process.cwd(),
    encoding: "utf8",
    env: pythonEnv,
  });
  if (probe.error || probe.status !== 0) {
    return null;
  }
  const script = [
    "import json",
    "import sys",
    "from pathlib import Path",
    "sys.path.insert(0, str(Path.cwd()))",
    "from python.document_parsing import _normalize_pdf_table_markdown",
    "print(json.dumps({'text': _normalize_pdf_table_markdown(sys.stdin.read())}, ensure_ascii=False))",
  ].join("\n");
  const completed = spawnSync(pythonExecutable, ["-c", script], {
    cwd: process.cwd(),
    input: markdown,
    encoding: "utf8",
    env: pythonEnv,
  });
  if (completed.error) {
    throw completed.error;
  }
  assert.equal(completed.status, 0, completed.stderr || completed.stdout || "python normalization failed");
  return (JSON.parse(completed.stdout) as { text: string }).text;
}

describe("document parsing runtime", () => {
  it("reuses cached full parses for preview_file on non-pdf inputs", async () => {
    const root = await mkdtemp(join(tmpdir(), "afs-preview-"));
    const filePath = join(root, "sample.md");
    await writeFile(filePath, "# Sample\nBody");

    const executor = new FakeParserExecutor(buildDocument());
    const runtime = new DocumentParsingRuntime(executor);

    const first = await runtime.previewFile(filePath, 18);
    const second = await runtime.previewFile(filePath, 18);

    assert.match(first, /=== PREVIEW of/);
    assert.match(first, /PREVIEW TRUNCATED/);
    assert.equal(second, first);
    assert.equal(executor.calls.length, 1);
  });

  it("keeps non-pdf focused parse selection local after one cached parse", async () => {
    const root = await mkdtemp(join(tmpdir(), "afs-focused-md-"));
    const filePath = join(root, "sample.md");
    await writeFile(filePath, "# Sample\nBody");

    const executor = new FakeParserExecutor(buildDocument());
    const runtime = new DocumentParsingRuntime(executor);

    const result = await runtime.parseFile({
      filePath,
      focusHint: "purchase price",
      anchor: 2,
      window: 1,
      maxUnits: 2,
    });

    assert.match(result, /=== FOCUSED PARSE of/);
    assert.match(result, /\[UNIT 1 \| source=unit-1 \| heading=Overview\]/);
    assert.match(result, /\[UNIT 2 \| source=unit-2 \| heading=Purchase Price\]/);
    assert.equal(executor.calls.length, 1);
    assert.equal(executor.calls[0]?.selector ?? null, null);
  });

  it("sends selector through the external parser for docx focused parses", async () => {
    const root = await mkdtemp(join(tmpdir(), "afs-focused-docx-"));
    const filePath = join(root, "sample.docx");
    await writeFile(filePath, "placeholder");

    const executor = new FakeParserExecutor(buildDocument());
    const runtime = new DocumentParsingRuntime(executor);

    await runtime.parseFile({
      filePath,
      focusHint: "purchase price",
      anchor: 2,
      window: 1,
      maxUnits: 2,
    });

    assert.equal(executor.calls.length, 1);
    assert.equal(executor.calls[0]?.selector?.query, "purchase price");
    assert.equal(executor.calls[0]?.selector?.anchor, 2);
    assert.equal(executor.calls[0]?.selector?.window, 1);
    assert.equal(executor.calls[0]?.selector?.max_units, 2);
  });

  it("matches legacy select_parsed_document scoring behavior", () => {
    const selected = selectParsedDocument(buildDocument(), {
      query: "purchase price",
      max_units: 1,
    });

    assert.equal(selected.units.length, 1);
    assert.equal(selected.units[0]?.unit_no, 2);
  });

  it("decodes Python bridge JSON even when dependency warnings precede stdout", () => {
    const parsed = parsePythonBridgeJson('Warning - optional module noise\n{"ok":true,"document":{"units":[]}}');
    assert.deepEqual(parsed, { ok: true, document: { units: [] } });
  });

  it("normalizes unstable PDF table markdown into caption plus flat headers", (t) => {
    const normalized = runPythonTableNormalization([
      "|钁ｄ簨鍑哄腑钁ｄ簨浼氬強鑲′笢澶т細鐨勬儏鍐祙---|---|---|---|---|---|---|",
      "|---|---|---|---|---|---|---|---|",
      "|钁ｄ簨濮撳悕|鏈姤鍛婃湡搴?br>鍙傚姞钁ｄ簨浼?br>娆℃暟|鐜板満鍑哄腑钁?br>浜嬩細娆℃暟|浠ラ€氳鏂瑰紡<br>鍙傚姞钁ｄ簨浼?br>娆℃暟|濮旀墭鍑哄腑钁?br>浜嬩細娆℃暟|缂哄腑钁ｄ簨浼?br>娆℃暟|鏄惁杩炵画涓?br>娆℃湭浜茶嚜鍙?br>鍔犺懀浜嬩細浼?br>璁畖鍑哄腑鑲′笢澶?br>浼氭鏁皘",
      "|鏇炬瘬缇7|1|6|0|0|鍚2|",
    ].join("\n"));
    if (normalized == null) {
      t.skip("Python executable is not available for PDF normalization checks.");
      return;
    }
    assert.doesNotMatch(normalized, /琛ㄦ牸鏍囬锛?/);
    const mergedPlaceholder = runPythonTableNormalization([
      "|椤圭洰|Col1|2024|",
      "|---|---|---|",
      "|浜у搧|鍔ㄥ姏鐢垫睜绯荤粺|321|",
      "|鍦板尯|涓浗|88|",
    ].join("\n"));
    assert.equal(mergedPlaceholder?.includes("Col1"), false);
    assert.match(mergedPlaceholder ?? "", /\| 椤圭洰 \| 2024 \|/);
    assert.match(mergedPlaceholder ?? "", /\| 浜у搧鍔ㄥ姏鐢垫睜绯荤粺 \| 321 \|/);

    const plainTable = runPythonTableNormalization([
      "|鍒桝|鍒桞|",
      "|---|---|",
      "|A|B|",
    ].join("\n"));
    assert.equal(plainTable, "| 鍒桝 | 鍒桞 |\n| --- | --- |\n| A | B |");

    const plainText = runPythonTableNormalization("鏅€氭钀絓n\n涓嶆槸琛ㄦ牸");
    assert.equal(plainText, "鏅€氭钀絓n\n涓嶆槸琛ㄦ牸");
  });
});
