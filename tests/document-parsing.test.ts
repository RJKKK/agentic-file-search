import assert from "node:assert/strict";
import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
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
});
