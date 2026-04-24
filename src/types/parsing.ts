/*
Reference: legacy/python/src/fs_explorer/document_parsing.py
Reference: legacy/python/src/fs_explorer/fs.py
*/

import { z } from "zod";

export const SUPPORTED_EXTENSIONS = [
  ".pdf",
  ".docx",
  ".doc",
  ".pptx",
  ".xlsx",
  ".html",
  ".md",
] as const;

export const ParseSelectorSchema = z.object({
  unit_nos: z.array(z.number().int().positive()).optional().nullable(),
  query: z.string().optional().nullable(),
  anchor: z.number().int().positive().optional().nullable(),
  window: z.number().int().min(0).optional().nullable(),
  max_units: z.number().int().positive().optional().nullable(),
});

export const ParsedImageSchema = z.object({
  image_hash: z.string().min(1),
  page_no: z.number().int().positive(),
  image_index: z.number().int().positive(),
  mime_type: z.string().nullable().optional(),
  width: z.number().int().positive().nullable().optional(),
  height: z.number().int().positive().nullable().optional(),
  bbox: z.array(z.number()).length(4).nullable().optional(),
  placeholder: z.string().nullable().optional(),
});

export const ParsedBlockSchema = z.object({
  index: z.number().int().min(0),
  block_type: z.string().min(1),
  bbox: z.array(z.number()).length(4),
  markdown: z.string(),
  char_count: z.number().int().min(0).optional().default(0),
  image_hash: z.string().nullable().optional(),
  source_image_index: z.number().int().min(0).nullable().optional(),
});

export const ParsedUnitSchema = z.object({
  unit_no: z.number().int().positive(),
  markdown: z.string(),
  content_hash: z.string().min(1),
  heading: z.string().nullable().optional(),
  source_locator: z.string().nullable().optional(),
  images: z.array(ParsedImageSchema).default([]),
  blocks: z.array(ParsedBlockSchema).default([]),
});

export const ParsedDocumentSchema = z.object({
  parser_name: z.string().min(1),
  parser_version: z.string().min(1),
  units: z.array(ParsedUnitSchema),
});

export const PythonBridgeErrorSchema = z.object({
  file_path: z.string().min(1),
  code: z.string().min(1),
  message: z.string().min(1),
});

export const PythonBridgeSuccessSchema = z.object({
  ok: z.literal(true),
  document: ParsedDocumentSchema,
});

export const PythonBridgeFailureSchema = z.object({
  ok: z.literal(false),
  error: PythonBridgeErrorSchema,
});

export const PythonBridgeResponseSchema = z.union([
  PythonBridgeSuccessSchema,
  PythonBridgeFailureSchema,
]);

export type SupportedExtension = (typeof SUPPORTED_EXTENSIONS)[number];
export type ParseSelectorInput = z.input<typeof ParseSelectorSchema>;
export type ParseSelector = z.infer<typeof ParseSelectorSchema>;
export type ParsedImage = z.infer<typeof ParsedImageSchema>;
export type ParsedBlock = z.infer<typeof ParsedBlockSchema>;
export type ParsedUnit = z.infer<typeof ParsedUnitSchema>;
export type ParsedDocument = z.infer<typeof ParsedDocumentSchema>;
export type PythonBridgeResponse = z.infer<typeof PythonBridgeResponseSchema>;

export class DocumentParseError extends Error {
  constructor(
    readonly filePath: string,
    readonly code: string,
    readonly detail: string,
  ) {
    super(`${code}: ${detail}`);
    this.name = "DocumentParseError";
  }
}

export function isSupportedExtension(extension: string): extension is SupportedExtension {
  return SUPPORTED_EXTENSIONS.includes(extension.toLowerCase() as SupportedExtension);
}

export function normalizeParseSelector(
  selector: ParseSelectorInput | null | undefined,
): ParseSelector | null {
  if (selector == null) {
    return null;
  }
  const parsed = ParseSelectorSchema.parse(selector);
  const unitNos = parsed.unit_nos
    ? [...new Set(parsed.unit_nos.map((value) => Number(value)).filter((value) => value > 0))].sort(
        (left, right) => left - right,
      )
    : null;
  return {
    unit_nos: unitNos && unitNos.length > 0 ? unitNos : null,
    query: parsed.query?.trim() || null,
    anchor: parsed.anchor ?? null,
    window: Math.max(Number(parsed.window ?? 1), 0),
    max_units: parsed.max_units ?? null,
  };
}

export function parsedDocumentMarkdown(document: ParsedDocument): string {
  return document.units
    .map((unit) => unit.markdown)
    .filter((markdown) => markdown.trim().length > 0)
    .join("\n\n");
}
