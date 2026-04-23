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
    parser_version: "m2-v3",
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

  it("normalizes unstable PDF table markdown without synthetic captions or Col placeholders", (t) => {
    const normalized = runPythonTableNormalization([
      "|董事出席董事会及股东大会的情况|---|---|---|---|---|---|---|",
      "|---|---|---|---|---|---|---|---|",
      "|董事姓名|本报告期应<br>参加董事会<br>次数|现场出席董<br>事会次数|以通讯方式<br>参加董事会<br>次数|委托出席董<br>事会次数|缺席董事会<br>次数|是否连续两<br>次未亲自参<br>加董事会会<br>议|出席股东大<br>会次数|",
      "|曾毓群|7|1|6|0|0|否|2|",
    ].join("\n"));
    if (normalized == null) {
      t.skip("Python executable is not available for PDF normalization checks.");
      return;
    }
    assert.equal(normalized.includes("表格标题："), false);
    assert.match(normalized, /\| 董事出席董事会及股东大会的情况 \| --- \| --- \| --- \| --- \| --- \| --- \| --- \|/);
    assert.match(
      normalized,
      /\| 董事姓名 \| 本报告期应参加董事会次数 \| 现场出席董事会次数 \| 以通讯方式参加董事会次数 \| 委托出席董事会次数 \| 缺席董事会次数 \| 是否连续两次未亲自参加董事会会议 \| 出席股东大会次数 \|/,
    );

    const mergedPlaceholder = runPythonTableNormalization([
      "|项目|Col1|2024|",
      "|---|---|---|",
      "|产品|动力电池系统|321|",
      "|地区|中国|88|",
    ].join("\n"));
    assert.equal(mergedPlaceholder?.includes("Col1"), false);
    assert.equal(
      mergedPlaceholder,
      "| 项目 |  | 2024 |\n| --- | --- | --- |\n| 产品 | 动力电池系统 | 321 |\n| 地区 | 中国 | 88 |",
    );

    const plainTable = runPythonTableNormalization([
      "|列A|列B|",
      "|---|---|",
      "|A|B|",
    ].join("\n"));
    assert.equal(plainTable, "| 列A | 列B |\n| --- | --- |\n| A | B |");

    const plainText = runPythonTableNormalization("普通段落\n\n不是表格");
    assert.equal(plainText, "普通段落\n\n不是表格");
  });
});
